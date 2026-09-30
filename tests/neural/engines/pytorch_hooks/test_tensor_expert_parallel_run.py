"""Tensor and expert parallelism through the real CLI (``docs/model_parallelism.md``
§6.1, §8.2, §10.6): ``--parallel tp=2 --device cpu`` on the tiny Llama and
``ep=2``, ``ep=4``, ``tp=2,ep=4`` on ``tiny-random/qwen3.5-moe``, over ``gloo``.

The corpus interchange (``02_interchange_im.json``) retargeted to a tiny
fixture, extended to **read and write every module-boundary class the
placement table names** (§4): the residual stream (``block_output`` swapped,
``attention_output`` / ``mlp_output`` / ``lm_head`` read — replicated), a
colwise output (``attention_query_pre_rope`` read *and* swapped — on the MoE
fixture this is Qwen's ``q_norm``, sharded on the head axis) and a colwise
value projection (``attention_value_states``), a rowwise input
(``attention_premix`` read and swapped — an input tap), and under ``ep`` the
router's outputs (``expert_idx`` read and swapped — the remapped table
reconstructed and re-remapped; ``router_scores``, expert-local) and the
``expert:``-scoped face of the experts interior (``expert_output`` at one
expert, ragged). Every parallel run is compared to the world-1 run of the
same document: **the same digests and stamps, the receipt equal but for
``execution.parallel``, every integral tensor exactly, every float within the
band below**, through the spawn parent (the pytest process never loads a
model).

**The band.** A sharded forward is the one-process computation up to the
reduction order its collectives introduce — a rowwise projection's
all-reduce over ``tp`` partial sums, the expert outputs' all-reduce over
``ep`` ranks — and a read at a sharded boundary adds nothing (an all-gather
is exact); a *write* at one lands the rank's chunk of the globally edited
tensor and the layers above re-reduce from it. Measured on the fixtures in
fp32 (2026-09-14, torch 2.9.0, gloo, CPU, this file's own program): see
`MEASURED` — maximum absolute difference per output class against the
world-1 run. The band is pinned at `BAND`, at least twenty times the
largest measured maximum — room for gloo's reduction order to vary with the
rank count — and orders below what a wrong shard makes: the two mutations
below (an all-gather concatenated in reversed rank order; an expert group
that thinks it is the next rank) leave the band by construction.

The ``joined`` entry presets the environment the way ``torchrun`` does and
is where a mutation of the collective is applied before ``main`` runs — the
production spawn path is left untouched.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import torch
import torch.multiprocessing as mp
from safetensors import safe_open
from safetensors.torch import load_file

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.shared.parallel import launcher
from causalab.io.tables import read_table
from causalab.neural.shared.parallel.spawn import reserve_port

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [pytest.mark.smoke, pytest.mark.parallel_world]

REPO = Path(__file__).resolve().parents[4]
CORPUS = REPO / "tests" / "protocols" / "02_interchange_im.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"

RECEIPT = "protocol.json"
TABLES = ("iia.json", "logit_diff.json", "logit_diff_q.json")

#: The measured maxima (module docstring), per geometry and output class:
#: ``stream`` — the replicated residual-stream reads and the metric tables;
#: ``sharded_read`` — a read at a colwise output or rowwise input, gathered
#: (under ``ep`` alone the attention boundaries are replicated, and the class
#: measures the same read whole);
#: ``sharded_write`` — the logits after a swap landed at a sharded boundary
#: (the colwise query, the rowwise premix, and under ``ep`` the routing
#: table — reconstructed, swapped, remapped, the scores re-masked);
#: ``experts`` — the expert-local face and the routing scores made whole.
#: Maximum absolute difference against the world-1 run, identical on every
#: publishing rank (the parent compares what rank 0 wrote).
MEASURED: dict[str, dict[str, float]] = {
    "llama tp=2": {"stream": 4.5e-8, "sharded_read": 1.5e-8, "sharded_write": 7.5e-8},
    "moe ep=2": {
        "stream": 7.1e-7,
        "sharded_read": 7.2e-7,
        "sharded_write": 9.9e-7,
        "experts": 3.0e-8,
    },
    "moe ep=4": {
        "stream": 6.0e-7,
        "sharded_read": 9.6e-7,
        "sharded_write": 8.1e-7,
        "experts": 3.8e-8,
    },
    "moe tp=2,ep=4": {
        "stream": 5.4e-7,
        "sharded_read": 9.6e-7,
        "sharded_write": 6.6e-7,
        "experts": 3.0e-8,
    },
}
#: Twenty times the largest measured maximum (the loader's rule,
#: ``test_sharded_load.py``: ``1e-5`` over ``4.8e-7``); the mutations below
#: move a sharded read by ``0.16`` and a swapped-boundary logit by ``0.04``
#: on the Llama, and the routing table by whole expert ids on the MoE.
BAND = 2e-5
assert BAND >= 20 * max(max(v.values()) for v in MEASURED.values())

#: The full-attention layer the attention boundaries are tapped at, and the
#: routed expert the ragged face reads.
LAYER = {TINY_LLAMA: 1, TINY_QWEN35_MOE: 3}
EXPERT = 3

#: Which output class each saved tensor file belongs to (the band is
#: reported per class); integral files are compared exactly.
CLASSES: dict[str, str] = {
    "v_cf": "stream",
    "r_attn": "stream",
    "r_mlp": "stream",
    "logits": "stream",
    "r_q": "sharded_read",
    "r_v": "sharded_read",
    "r_premix": "sharded_read",
    "logits_q": "sharded_write",
    "logits_premix": "sharded_write",
    "r_scores": "experts",
    "r_face": "experts",
    "logits_reroute": "sharded_write",
}
INTEGRAL = ("r_idx",)


def _document(tmp: Path, key: str) -> Path:
    """Corpus 02 on ``key`` at its full-attention layer, plus the boundary
    reads and writes the module docstring lists."""
    doc = json.loads(CORPUS.read_text())
    layer = LAYER[key]
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    sites, reads, writes = method["sites"], method["reads"], method["writes"]
    models, save = method["intervened_models"], method["save"]
    sites["target"]["layers"] = [layer]
    for name, component in (
        ("attn", "attention_output"),
        ("mlp", "mlp_output"),
        ("q", "attention_query_pre_rope"),
        ("v", "attention_value_states"),
        ("premix", "attention_premix"),
    ):
        sites[name] = {"component": component, "layers": [layer]}
        _read(reads, models, f"r_{name}", {"site": name, "pos": -1}, "patched", "base")
        save.append(_saved(f"r_{name}", "patched"))
    save.append(_saved("v_cf", "original", "counterfactual"))
    save.append(_saved("logits", "patched"))
    # a swap landing at a colwise output, and one at a rowwise input
    for name in ("q", "premix"):
        _read(
            reads,
            models,
            f"{name}_cf",
            {"site": name, "pos": -1},
            "original",
            "counterfactual",
        )
        writes[f"patch_{name}"] = {
            "site": name,
            "pos": -1,
            "do": {"swap": f"{name}_cf"},
        }
        models[f"patched_{name}"] = {
            "input": "base",
            "reads": [],
            "writes": [f"patch_{name}"],
        }
        _read(
            reads,
            models,
            f"logits_{name}",
            {"site": "lm_head", "pos": -1},
            f"patched_{name}",
            "base",
        )
        save.append(_saved(f"logits_{name}", f"patched_{name}"))
    # the corpus's logit difference, over the logits after the swap (§2.10)
    (logit_diff,) = [e for e in save if e.get("file_path") == "logit_diff.json"]
    save.append(
        {
            "read": "logits_q",
            "model": "patched_q",
            "aggregation": dict(logit_diff["aggregation"]),
            "file_path": "logit_diff_q.json",
        }
    )
    if key == TINY_QWEN35_MOE:
        sites["idx"] = {"component": "expert_idx", "layers": [layer]}
        sites["scores"] = {"component": "router_scores", "layers": [layer]}
        sites["face"] = {
            "component": "expert_output",
            "layers": [layer],
            "expert": EXPERT,
        }
        for name in ("idx", "scores", "face"):
            _read(
                reads,
                models,
                f"r_{name}",
                {"site": name, "pos": "all"},
                "original",
                "base",
            )
            save.append(_saved(f"r_{name}", "original"))
        # the routing table swapped at the answer slot: the remapped table
        # made whole, swapped, and remapped back on every rank
        _read(
            reads,
            models,
            "idx_cf",
            {"site": "idx", "pos": -1},
            "original",
            "counterfactual",
        )
        writes["reroute"] = {"site": "idx", "pos": -1, "do": {"swap": "idx_cf"}}
        models["rerouted"] = {"input": "base", "reads": [], "writes": ["reroute"]}
        _read(
            reads,
            models,
            "logits_reroute",
            {"site": "lm_head", "pos": -1},
            "rerouted",
            "base",
        )
        save.append(_saved("logits_reroute", "rerouted"))
    target = tmp / "parity.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _saved(value: str, model: str, input_role: str = "base") -> dict[str, str]:
    """A tensor save entry of read ``value`` on ``model`` (§2.12). The
    un-intervened model on a role is the declared model with no writes:
    ``original`` on ``base`` is spelled ``original_base`` here, the corpus's
    ``original_counterfactual`` on ``counterfactual``."""
    return {
        "read": value,
        "model": _unwritten(model, input_role),
        "file_path": f"{value}.safetensors",
    }


def _unwritten(model: str, input_role: str) -> str:
    return f"original_{input_role}" if model == "original" else model


def _read(
    reads: dict[str, Any],
    models: dict[str, Any],
    name: str,
    spec: dict[str, Any],
    model: str,
    input_role: str,
) -> None:
    """Declare read ``name`` at ``spec`` and list it on ``model`` (§2.9),
    declaring the un-intervened model on ``input_role`` when it is not yet."""
    reads[name] = spec
    owner = _unwritten(model, input_role)
    entry = models.setdefault(owner, {"input": input_role, "reads": []})
    entry.setdefault("reads", []).append(name)


def _argv(document: Path, out: Path, *extra: str, device: str = "cpu") -> list[str]:
    """The CLI ``run`` of ``document`` into ``out`` on ``device`` (``cpu``: the
    gloo tier; the CUDA twin, ``tests/golden/test_parallel_worlds.py``, passes
    ``cuda``)."""
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
        device,
        *extra,
    ]


def _block(**axes: int) -> dict[str, Any]:
    """The receipt's ``execution.parallel`` block for a model-parallel geometry."""
    geometry = {"data": 1, "pipeline": 1, "context": 1, "tensor": 1, "expert": 1}
    geometry.update(axes)
    return {
        **geometry,
        "data_mode": "points",
        "world": max(geometry["tensor"], geometry["expert"]),
    }


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _max_diffs(solo: Path, parallel: Path) -> dict[str, float]:
    """Per saved tensor file, the maximum absolute difference of every float
    entry; an integral entry that differs is ``inf`` (labels are exact or
    wrong), a shape that differs too."""
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
            elif a[key].numel() == 0:
                # Empty entries agree; max() has no reduction over no values.
                continue
            elif a[key].dtype.is_floating_point:
                worst = max(
                    worst, (a[key].double() - b[key].double()).abs().max().item()
                )
            elif not torch.equal(a[key], b[key]):
                worst = math.inf
        out[name.removesuffix(".safetensors")] = worst
    return out


def _table_diff(solo: Path, parallel: Path, name: str) -> float:
    a_rows, b_rows = read_table(solo / name), read_table(parallel / name)
    assert len(a_rows) == len(b_rows), name
    worst = 0.0
    for a, b in zip(a_rows, b_rows):
        assert set(a) == set(b), name
        for key in a:
            if isinstance(a[key], float) or isinstance(b[key], float):
                worst = max(worst, abs(float(a[key]) - float(b[key])))
            else:
                assert a[key] == b[key], (name, key)
    return worst


def _assert_receipts_and_stamps(
    solo: Path, parallel: Path, block: dict[str, Any]
) -> None:
    a, b = _receipt(solo), _receipt(parallel)
    assert b["execution"]["parallel"] == block
    assert a["execution"]["parallel"]["launcher"] == "solo"
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["document_digest"] == b["document_digest"]
    for name in sorted(p.name for p in solo.glob("*.safetensors")):
        with (
            safe_open(str(solo / name), "pt") as x,
            safe_open(str(parallel / name), "pt") as y,
        ):
            assert x.metadata() == y.metadata(), name


def _assert_parity(
    solo: Path, parallel: Path, block: dict[str, Any], *, band: float
) -> dict[str, float]:
    """Every artifact of ``parallel`` against ``solo``'s: receipts and stamps
    identical, integral tensors exact, floats within ``band``; the maxima per
    output class are returned for the record."""
    _assert_receipts_and_stamps(solo, parallel, block)
    diffs = _max_diffs(solo, parallel)
    for name in INTEGRAL:
        if name in diffs:
            assert diffs.pop(name) == 0.0, name
    by_class: dict[str, float] = {}
    for name, worst in diffs.items():
        by_class[CLASSES[name]] = max(by_class.get(CLASSES[name], 0.0), worst)
        assert worst <= band, (name, worst)
    for table in TABLES:
        worst = _table_diff(solo, parallel, table)
        by_class["stream"] = max(by_class.get("stream", 0.0), worst)
        assert worst <= band, (table, worst)
    return by_class


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawned ranks inherit the environment: offline, and one thread
    each so four ranks on one CPU do not oversubscribe it."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model: in *this* process the
    loader raises; the children are fresh interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def llama_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("tp-llama")
    document = _document(tmp, TINY_LLAMA)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def moe_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("ep-moe")
    document = _document(tmp, TINY_QWEN35_MOE)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


# --------------------------------------------------------------------------- #
# parity through the spawn parent
# --------------------------------------------------------------------------- #


def test_tp2_llama_agrees_with_world_one_within_the_band(
    llama_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = llama_solo
    out = tmp_path / "tp2"
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    measured = _assert_parity(
        solo, out, {**_block(tensor=2), "launcher": "spawned"}, band=BAND
    )
    _assert_measured("llama tp=2", measured)


@pytest.mark.parametrize("geometry", ["ep=2", "ep=4", "tp=2,ep=4"])
def test_moe_agrees_with_world_one_within_the_band(
    moe_solo: tuple[Path, Path], tmp_path: Path, never_load: None, geometry: str
) -> None:
    document, solo = moe_solo
    out = tmp_path / geometry.replace(",", "_").replace("=", "")
    assert main(_argv(document, out, "--parallel", geometry)) == 0
    axes = {
        {"tp": "tensor", "ep": "expert"}[k]: int(v)
        for k, v in (item.split("=") for item in geometry.split(","))
    }
    measured = _assert_parity(
        solo, out, {**_block(**axes), "launcher": "spawned"}, band=BAND
    )
    _assert_measured(f"moe {geometry}", measured)


def _assert_measured(label: str, measured: dict[str, float]) -> None:
    """Every output class the record pins was exercised, and none exceeds
    its recorded maximum by more than the band leaves room for — a drift
    past the record is a re-measurement, not a silent pass."""
    print(label, measured)
    recorded = MEASURED[label]
    assert set(measured) == set(recorded), (label, measured)
    for kind, worst in measured.items():
        assert worst <= max(recorded[kind] * 2, 1e-7), (label, kind, worst)


# --------------------------------------------------------------------------- #
# the mutations: a wrong shard leaves the band
# --------------------------------------------------------------------------- #


def _reversed_gather() -> None:
    """An all-gather concatenated in reversed rank order: the head chunks of
    every sharded boundary land in the wrong place."""
    from causalab.neural.shared.parallel import collective

    original = collective.TorchCollective.all_gather

    def all_gather(
        self: Any, tensor: torch.Tensor, dim: int, axis: Any
    ) -> torch.Tensor:
        gathered = original(self, tensor, dim, axis)
        size = self.size(axis)
        if size == 1:
            return gathered
        return torch.cat(gathered.chunk(size, dim=dim)[::-1], dim=dim)

    collective.TorchCollective.all_gather = all_gather  # type: ignore[method-assign]


def _shifted_expert_rank() -> None:
    """An expert group whose ranks each think they are the next one: the
    routing table reconstructs to the wrong global ids and the wrong slots
    are kept."""
    from causalab.neural.shared.parallel import collective

    original = collective.TorchCollective.rank

    def rank(self: Any, axis: Any) -> int:
        local = original(self, axis)
        return (local + 1) % self.size(axis) if axis == "expert" else local

    collective.TorchCollective.rank = rank  # type: ignore[method-assign]


MUTATIONS: dict[str, Callable[[], None]] = {
    "reversed_gather": _reversed_gather,
    "shifted_expert_rank": _shifted_expert_rank,
}


def _mutated_entry(
    rank: int, argv: Sequence[str], world: int, port: int, mutation: str
) -> None:
    """One ``torchrun``-style rank with the named mutation of the collective
    applied before the CLI runs."""
    os.environ.update(
        {
            "WORLD_SIZE": str(world),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    os.environ.pop(launcher.LAUNCHER_VARIABLE, None)
    MUTATIONS[mutation]()
    code = main(list(argv))
    if code:
        sys.exit(code)


def _run_mutated(
    document: Path, out: Path, geometry: str, world: int, mutation: str
) -> None:
    with reserve_port() as hold:
        mp.spawn(
            _mutated_entry,
            args=(
                _argv(document, out, "--parallel", geometry),
                world,
                hold.port,
                mutation,
            ),
            nprocs=world,
            join=True,
        )


def test_a_reversed_gather_leaves_the_band_on_the_llama(
    llama_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = llama_solo
    out = tmp_path / "mutated"
    _run_mutated(document, out, "tp=2", 2, "reversed_gather")
    diffs = _max_diffs(solo, out)
    # the residual stream is untouched by the gather; every sharded boundary is not
    assert diffs["r_q"] > BAND and diffs["r_premix"] > BAND, diffs
    assert diffs["logits_q"] > BAND, diffs


def test_a_shifted_expert_rank_leaves_the_band_on_the_moe(
    moe_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = moe_solo
    out = tmp_path / "mutated"
    _run_mutated(document, out, "ep=4", 4, "shifted_expert_rank")
    diffs = _max_diffs(solo, out)
    # the reconstructed routing table names the wrong experts
    assert diffs["r_idx"] == math.inf, diffs
