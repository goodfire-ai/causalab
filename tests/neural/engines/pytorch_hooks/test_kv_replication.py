"""KV-head replication on real processes over ``gloo``
(``docs/model_parallelism.md`` §6.6, §10.6): ``tiny-random/qwen3.5-moe``
(8 heads, 4 KV heads, one full-attention layer) at ``tp=8`` — twice the KV
heads, so ``k_proj`` / ``v_proj`` are replicated and each rank holds KV head
``rank // 2`` — beside ``tp=4``, which divides the KV heads and is unchanged.

Three scenarios, each spawning a world the way ``test_sharded_load.py`` and
``test_tensor_expert_parallel_run.py`` do:

* **the load** — on every rank ``k_proj.weight`` and ``v_proj.weight`` are
  plain parameters equal to the whole weight, ``q_proj.weight`` a DTensor
  eighth, the mixer's ``num_key_value_groups`` is 1 (one query head, one KV
  head per rank; 2 at ``tp=4``), the reader was asked for the whole of the
  K/V bytes and an eighth of the Q bytes, and the logits agree with the
  one-process forward within the band;
* **the run** — the corpus interchange through the real CLI at ``--parallel
  tp=8 --device cpu`` with reads and swaps at ``attention_query`` (a head
  shard), ``attention_key`` (the repeated shard made whole: the four KV
  heads), ``attention_probs`` (the pattern, written through the softmax) and
  ``block_output`` (the stream), compared to the world-1 run of the same
  document: the same digests and stamps, every float within the band;
* **the mutation** — a rank holding its neighbour's KV head (``held_chunk``
  shifted by one) lands outside the band on the logits.

**The band.** A replicated K/V projection computes the whole GEMM and a
gather of equal copies drops all but one, so the replication adds no
rounding of its own; what remains is the colwise ``q`` shard's GEMM and the
rowwise all-reduce over eight partial sums, as at ``tp=2`` over two.
Measured on the fixture in fp32 (2026-09-14, torch 2.9.0, gloo, CPU, this
file's own programs): the loader's logits ``4.6e-7`` at ``tp=8`` and
``6.0e-7`` at ``tp=4``; the run's stream and gathered reads ``7.2e-7``, the
logits after a swap ``1.0e-6`` (`MEASURED`); pinned at `BAND`,
the loader's rule (twenty times the largest maximum) and five orders below
the mutation, which lands at ``1.96``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.protocol.parallel import ParallelGeometry

from tests._helpers.sharded_world import run_world
from tests.neural.engines.pytorch_hooks.conftest import TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

REPO = Path(__file__).resolve().parents[4]
CORPUS = REPO / "tests" / "protocols" / "02_interchange_im.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"

#: The fixture's attention facts and its one full-attention layer.
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, LAYER = 8, 4, 32, 3
MIXER = f"model.layers.{LAYER}.self_attn"

#: The measured maxima (module docstring): the loader's logits at ``tp=8``
#: and ``tp=4``; the run's classes — ``stream`` (the residual-stream reads,
#: the logits and the metric tables), ``sharded_read`` (the gathered query,
#: key and pattern), ``sharded_write`` (the logits after a swap landed at
#: each of the four sites).
MEASURED: dict[str, float] = {
    "load tp=8": 4.7e-7,
    "load tp=4": 6.0e-7,
    "run stream": 7.2e-7,
    "run sharded_read": 7.2e-7,
    "run sharded_write": 1.1e-6,
    # the embeddings' gradient at ``tp=8`` against world 1, relative to its
    # largest entry (the gradient scenario below)
    "gradient tp=8": 4.3e-7,
}
#: Twenty-plus times the largest measured maximum (the loader's rule), and
#: five orders below the mutation: a rank holding its neighbour's KV head
#: moves the logits by ``1.96``.
BAND = 2.5e-5
assert BAND >= 20 * max(MEASURED.values())

IDS = torch.tensor([[5, 17, 23, 42, 8, 91, 3], [12, 4, 77, 61, 30, 2, 9]])

#: The mutation: every rank holds its neighbour's KV head.
SHIFTED_HEAD = "shifted_head"


# --------------------------------------------------------------------------- #
# the load
# --------------------------------------------------------------------------- #


def _shift_held_chunk() -> None:
    from causalab.neural.engines.pytorch_hooks import kv_replication

    def shifted(rank: int, repeat: int) -> int:
        return (rank // repeat + 1) % NUM_KV_HEADS

    kv_replication.held_chunk = shifted  # type: ignore[assignment]


def _program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    from torch.distributed.tensor import DTensor

    tensor, mutation = payload
    if mutation == SHIFTED_HEAD:
        _shift_held_chunk()
    geometry = ParallelGeometry(tensor=tensor)
    sharding = Sharding.from_mesh(Mesh.from_environment(geometry))
    bundle = load_model(TINY_QWEN35_MOE, sharding=sharding)
    model = bundle.model
    assert bundle.load_report is not None
    mixer = dict(model.named_modules())[MIXER]
    parameters = dict(model.named_parameters())

    def held(name: str) -> torch.Tensor:
        """What this rank holds of a parameter: a DTensor's local shard, a
        plain parameter whole (tensors cross a process boundary as values)."""
        parameter = parameters[f"{MIXER}.{name}.weight"]
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        return local.detach().clone()

    out: dict[str, Any] = {
        "dtensors": sorted(
            name for name, p in parameters.items() if isinstance(p, DTensor)
        ),
        "k_proj": held("k_proj"),
        "v_proj": held("v_proj"),
        "q_proj_local": held("q_proj"),
        "groups": int(mixer.num_key_value_groups),
        "k_out_features": int(mixer.k_proj.out_features),
        "requested": dict(bundle.load_report.bytes_requested),
        "on_disk": dict(bundle.load_report.bytes_on_disk),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    handle = mixer.k_proj.register_forward_hook(
        lambda _m, _i, o: shapes.__setitem__("k_proj", tuple(o.shape))
    )
    try:
        with torch.no_grad():
            out["logits"] = model(input_ids=IDS).logits.detach().clone()
    finally:
        handle.remove()
    out["k_proj_out_shape"] = shapes["k_proj"]
    return out


def _load(tensor: int, mutation: str | None = None) -> dict[int, dict[str, Any]]:
    return run_world(tensor, _program, (tensor, mutation))


@pytest.fixture(scope="module")
def moe() -> Any:
    return load_model(TINY_QWEN35_MOE)


@pytest.fixture(scope="module")
def reference_logits(moe: Any) -> torch.Tensor:
    with torch.no_grad():
        return moe.model(input_ids=IDS).logits


def _record(label: str, worst: float) -> None:
    print(f"\nmeasured {label}: {worst:.3e} (pinned {MEASURED[label]:.1e})")
    assert worst <= BAND, (label, worst)
    # a drift past the record is a re-measurement, not a silent pass
    assert worst <= max(MEASURED[label] * 2, 1e-7), (label, worst)


def test_tp8_replicates_the_kv_projections_and_agrees_within_the_band(
    moe: Any, reference_logits: torch.Tensor
) -> None:
    ranks = _load(8)
    assert set(ranks) == set(range(8))
    whole = dict(moe.model.named_parameters())
    worst = 0.0
    for rank, out in ranks.items():
        # the K/V weights are plain parameters, whole on every rank
        assert f"{MIXER}.k_proj.weight" not in out["dtensors"]
        assert f"{MIXER}.v_proj.weight" not in out["dtensors"]
        assert torch.equal(out["k_proj"], whole[f"{MIXER}.k_proj.weight"])
        assert torch.equal(out["v_proj"], whole[f"{MIXER}.v_proj.weight"])
        # q_proj is the colwise eighth
        assert f"{MIXER}.q_proj.weight" in out["dtensors"]
        assert torch.equal(
            out["q_proj_local"], whole[f"{MIXER}.q_proj.weight"].chunk(8)[rank]
        )
        # one query head per rank, reading one KV head: the library's repeat is 1
        assert out["groups"] == 1
        # the module keeps its whole width; its forward emits the held head
        assert out["k_out_features"] == NUM_KV_HEADS * HEAD_DIM
        assert out["k_proj_out_shape"] == (2, IDS.shape[1], HEAD_DIM)
        # the reader was asked for the whole of K/V and an eighth of Q
        for leaf in ("k_proj", "v_proj"):
            name = f"{MIXER}.{leaf}.weight"
            assert out["requested"][name] == out["on_disk"][name], leaf
        q = f"{MIXER}.q_proj.weight"
        assert out["requested"][q] * 8 == out["on_disk"][q]
        diff = (out["logits"] - reference_logits).abs().max().item()
        assert torch.isfinite(out["logits"]).all()
        worst = max(worst, diff)
    # every rank computes the same replicated logits
    for rank in range(1, 8):
        assert torch.equal(ranks[0]["logits"], ranks[rank]["logits"])
    _record("load tp=8", worst)


def test_tp4_divides_the_kv_heads_and_is_unchanged(
    moe: Any, reference_logits: torch.Tensor
) -> None:
    ranks = _load(4)
    whole = dict(moe.model.named_parameters())
    worst = 0.0
    for rank, out in ranks.items():
        # the KV heads are sharded colwise: a DTensor quarter, the library's
        # repeat the model's own
        assert f"{MIXER}.k_proj.weight" in out["dtensors"]
        assert out["groups"] == NUM_HEADS // NUM_KV_HEADS
        assert out["k_proj_out_shape"] == (2, IDS.shape[1], HEAD_DIM)
        q = f"{MIXER}.q_proj.weight"
        assert out["requested"][q] * 4 == out["on_disk"][q]
        k = f"{MIXER}.k_proj.weight"
        assert out["requested"][k] * 4 == out["on_disk"][k]
        assert torch.equal(out["q_proj_local"], whole[q].chunk(4)[rank])
        assert torch.equal(out["k_proj"], whole[k].chunk(4)[rank])
        worst = max(worst, (out["logits"] - reference_logits).abs().max().item())
    _record("load tp=4", worst)


def test_a_rank_holding_its_neighbours_kv_head_leaves_the_band(
    reference_logits: torch.Tensor,
) -> None:
    ranks = _load(8, mutation=SHIFTED_HEAD)
    nearest = math.inf
    for rank, out in ranks.items():
        diff = (out["logits"] - reference_logits).abs().max().item()
        assert diff > BAND, (rank, diff)
        nearest = min(nearest, diff)
    print(f"\nthe shifted KV head lands no nearer than {nearest:.3e} (band {BAND:.0e})")


# --------------------------------------------------------------------------- #
# the run: reads and swaps at the four sites through the CLI
# --------------------------------------------------------------------------- #

#: ``(name, component, pos)``: the pattern and the key are whole-tensor
#: reads (the key's position axis runs over the positions attended to).
SITES: tuple[tuple[str, str, object], ...] = (
    ("query", "attention_query", -1),
    ("key", "attention_key", "all"),
    ("probs", "attention_probs", "all"),
    ("block", "block_output", -1),
)
CLASSES: dict[str, str] = {
    "r_query": "sharded_read",
    "r_key": "sharded_read",
    "r_probs": "sharded_read",
    "r_block": "stream",
    "logits": "stream",
    "logits_clean": "stream",
    "logits_query": "sharded_write",
    "logits_key": "sharded_write",
    "logits_probs": "sharded_write",
    "logits_block": "sharded_write",
}
TABLES = ("iia.json", "logit_diff.json")


def _saved(read: str, model: str) -> dict[str, str]:
    return {"read": read, "model": model, "file_path": f"{read}.safetensors"}


def _document(tmp: Path) -> Path:
    """Corpus 02 on the MoE fixture at its full-attention layer: a read of
    each site on the original model, and a swap at each site with the
    patched logits saved."""
    doc = json.loads(CORPUS.read_text())
    doc["model"] = {"key": TINY_QWEN35_MOE, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    sites, reads, writes = method["sites"], method["reads"], method["writes"]
    models, save = method["intervened_models"], method["save"]
    sites["target"]["layers"] = [LAYER]
    save.append(_saved("logits", "patched"))
    # the un-intervened model on base is a declared model with no writes (§2.9)
    original = models.setdefault("original_base", {"input": "base", "reads": []})
    counterfactual = models["original_counterfactual"]
    # the unpatched logits: what every swap below is shown to move
    reads["logits_clean"] = {"site": "lm_head", "pos": -1}
    original["reads"].append("logits_clean")
    save.append(_saved("logits_clean", "original_base"))
    for name, component, pos in SITES:
        sites[name] = {"component": component, "layers": [LAYER]}
        reads[f"r_{name}"] = {"site": name, "pos": pos}
        original["reads"].append(f"r_{name}")
        save.append(_saved(f"r_{name}", "original_base"))
        reads[f"{name}_cf"] = {"site": name, "pos": pos}
        counterfactual["reads"].append(f"{name}_cf")
        writes[f"patch_{name}"] = {
            "site": name,
            "pos": pos,
            "do": {"swap": f"{name}_cf"},
        }
        reads[f"logits_{name}"] = {"site": "lm_head", "pos": -1}
        models[f"patched_{name}"] = {
            "input": "base",
            "reads": [f"logits_{name}"],
            "writes": [f"patch_{name}"],
        }
        save.append(_saved(f"logits_{name}", f"patched_{name}"))
    target = tmp / "kv.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _argv(document: Path, out: Path, *extra: str) -> list[str]:
    return [
        "run",
        str(document),
        "--engine",
        "pytorch_hooks",
        "--record",
        "--data-root",
        str(DATA),
        "--artifacts-root",
        str(document.parent),
        "--out",
        str(out),
        "--device",
        "cpu",
        *extra,
    ]


def _max_diffs(solo: Path, parallel: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    names = sorted(p.name for p in solo.glob("*.safetensors"))
    assert names == sorted(p.name for p in parallel.glob("*.safetensors"))
    for name in names:
        a, b = load_file(str(solo / name)), load_file(str(parallel / name))
        assert set(a) == set(b), name
        worst = 0.0
        for key in a:
            if a[key].shape != b[key].shape:
                worst = math.inf
            elif a[key].dtype.is_floating_point:
                worst = max(
                    worst, (a[key].double() - b[key].double()).abs().max().item()
                )
            elif not torch.equal(a[key], b[key]):
                worst = math.inf
        out[name.removesuffix(".safetensors")] = worst
    return out


def _table_diff(solo: Path, parallel: Path, name: str) -> float:
    from causalab.io.tables import read_table

    a_rows, b_rows = read_table(solo / name), read_table(parallel / name)
    assert len(a_rows) == len(b_rows), name
    worst = 0.0
    for a, b in zip(a_rows, b_rows):
        for key in a:
            if isinstance(a[key], float) or isinstance(b[key], float):
                worst = max(worst, abs(float(a[key]) - float(b[key])))
            else:
                assert a[key] == b[key], (name, key)
    return worst


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawn parent never loads a model; the children are fresh
    interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("kv-solo")
    document = _document(tmp)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


def test_tp8_run_agrees_with_world_one_at_every_site(
    solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo_out = solo
    out = tmp_path / "tp8"
    assert main(_argv(document, out, "--parallel", "tp=8")) == 0
    a = json.loads((solo_out / "protocol.json").read_text())
    b = json.loads((out / "protocol.json").read_text())
    assert b["execution"]["parallel"] == {
        "data": 1,
        "pipeline": 1,
        "context": 1,
        "tensor": 8,
        "expert": 1,
        "data_mode": "points",
        "world": 8,
        "launcher": "spawned",
    }
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    for name in sorted(p.name for p in solo_out.glob("*.safetensors")):
        with (
            safe_open(str(solo_out / name), "pt") as x,
            safe_open(str(out / name), "pt") as y,
        ):
            assert x.metadata() == y.metadata(), name
    diffs = _max_diffs(solo_out, out)
    assert set(diffs) == set(CLASSES), diffs
    by_class: dict[str, float] = {}
    for name, worst in diffs.items():
        by_class[CLASSES[name]] = max(by_class.get(CLASSES[name], 0.0), worst)
    for table in TABLES:
        by_class["stream"] = max(by_class["stream"], _table_diff(solo_out, out, table))
    for kind, worst in by_class.items():
        _record(f"run {kind}", worst)
    # anti-vacuity: at world 1 every swap moved the logits
    (clean,) = load_file(str(solo_out / "logits_clean.safetensors")).values()
    for name, _, _ in SITES:
        (patched,) = load_file(str(solo_out / f"logits_{name}.safetensors")).values()
        assert not torch.allclose(clean, patched), f"the swap at {name} did nothing"


# --------------------------------------------------------------------------- #
# the gradient: the residual's gradient through a replicated K/V projection
# --------------------------------------------------------------------------- #

#: A fixed readout of the logits, so the loss is one scalar on every rank.
READOUT_SEED = 7


def _readout_loss(logits: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(READOUT_SEED)
    weights = torch.randn(logits.shape[-1], generator=generator)
    return (logits[:, -1, :] * weights).sum()


def _gradient_program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    """The embeddings' gradient through the whole sharded model at ``tp``."""
    (tensor,) = payload
    geometry = ParallelGeometry(tensor=tensor)
    sharding = Sharding.from_mesh(Mesh.from_environment(geometry))
    model = load_model(TINY_QWEN35_MOE, sharding=sharding).model
    return {"grad": _embedding_gradient(model)}


def _embedding_gradient(model: Any) -> torch.Tensor:
    embeds = model.get_input_embeddings()(IDS).detach().requires_grad_()
    logits = model(inputs_embeds=embeds).logits
    _readout_loss(logits).backward()
    assert embeds.grad is not None
    return embeds.grad.detach().clone()


def test_tp8_backpropagates_the_world_one_gradient_on_every_rank(moe: Any) -> None:
    """§6.6, §7: a replicated K/V projection's *input* gradient is a partial
    sum — this rank's query heads' path through its one held KV head — and
    must be summed over the tensor group, as the colwise query's is by
    DTensor and the router's by ``sum_router_gradient``. Every rank must
    match the world-1 gradient within `MEASURED`'s ``gradient tp=8``."""
    reference = _embedding_gradient(moe.model)
    ranks = run_world(NUM_HEADS, _gradient_program, (NUM_HEADS,))
    grads = [ranks[rank]["grad"] for rank in range(NUM_HEADS)]
    for rank in range(1, NUM_HEADS):
        assert torch.equal(grads[rank], grads[0]), rank
    scale = float(reference.abs().max())
    worst = float((grads[0] - reference).abs().max()) / scale
    _record("gradient tp=8", worst)


def _style_program(rank: int, world: int, payload: Any) -> dict[str, Any]:
    """The style object alone on a one-KV-head projection at ``tp=2``: both
    ranks hold the one head (``repeat = 2``); each weights the held output by
    a rank-specific readout, so the two partial input gradients differ, and
    the summed gradient is the one-process gradient of the summed loss."""
    from causalab.neural.engines.pytorch_hooks.kv_replication import KvReplicated
    from causalab.neural.engines.pytorch_hooks.styles import Group
    from causalab.neural.shared.parallel.collective import TorchCollective

    (seed,) = payload
    generator = torch.Generator().manual_seed(seed)
    projection = torch.nn.Linear(6, HEAD_DIM)
    with torch.no_grad():
        projection.weight.copy_(torch.randn(HEAD_DIM, 6, generator=generator))
        projection.bias.copy_(torch.randn(HEAD_DIM, generator=generator))
    x = torch.randn(3, 6, generator=generator)
    readouts = torch.randn(world, HEAD_DIM, generator=generator)
    collective = TorchCollective(Mesh.from_environment(ParallelGeometry(tensor=world)))
    KvReplicated(world, collective).install(
        projection, Group("tensor", rank, world), expert_parallel=False
    )
    local = x.clone().requires_grad_()
    (projection(local) * readouts[rank]).sum().backward()
    assert local.grad is not None
    return {"grad": local.grad.clone(), "x": x, "readouts": readouts}


def test_the_style_sums_the_input_gradient_over_the_group() -> None:
    ranks = run_world(2, _style_program, (READOUT_SEED,))
    x, readouts = ranks[0]["x"], ranks[0]["readouts"]
    generator = torch.Generator().manual_seed(READOUT_SEED)
    projection = torch.nn.Linear(6, HEAD_DIM)
    with torch.no_grad():
        projection.weight.copy_(torch.randn(HEAD_DIM, 6, generator=generator))
        projection.bias.copy_(torch.randn(HEAD_DIM, generator=generator))
    whole = x.clone().requires_grad_()
    (projection(whole) * readouts.sum(0)).sum().backward()
    assert whole.grad is not None
    assert torch.equal(ranks[0]["grad"], ranks[1]["grad"])
    torch.testing.assert_close(ranks[0]["grad"], whole.grad, rtol=1e-6, atol=1e-6)
