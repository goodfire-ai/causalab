"""Context parallelism through the real CLI (``docs/model_parallelism.md``
§6.4, §8.4, §10.6): ``--parallel cp=2 --device cpu`` on the tiny Llama and on
``tiny-random/qwen3.5-moe`` (three DeltaNet layers, one attention layer — the
state and conv-history handoffs exercised), and ``cp=2,tp=2`` on the MoE,
over ``gloo``.

The corpus interchange (``02_interchange_im.json``) retargeted to a tiny
fixture, extended so that reads and writes fall in **different chunks** of
the padded frame: the answer-slot swap and the reads at ``pos: -1`` land in
the last chunk, a read and a swap at ``pos: 0`` in the first, an
all-positions read spans both; the attention interior is read at the
gathered key (``attention_key``, this rank's chunk made whole), the scores
and the pattern (``(b, H, q_local, k_full)``, the query axis chunked, the
key axis whole); and on the MoE the DeltaNet conv output and the per-step
state at the answer slot (``delta_state``, which threads the state received
from the rank below through the recurrent path). Every parallel run is
compared to the world-1 run of the same document: **the same digests and
stamps, the receipt equal but for ``execution.parallel``, every integral
tensor exactly, every float within the band below**, through the spawn
parent (the pytest process never loads a model).

**The band.** A context-parallel forward is the one-process computation
with the attention GEMMs run over a shorter query axis (the mask, the
gathered keys and values, the rotary tables and every position-local layer
are the same tensors), and on the MoE with the chunked gated-delta kernel's
64-position blocks falling differently on each chunk — its sums
re-associate, the recurrence's value is the same. Measured on the fixtures
in fp32 (2026-09-14, torch 2.9.0, gloo, CPU, this file's own program): see
`MEASURED` — maximum absolute difference per output class against the
world-1 run. The band is pinned at `BAND`, at least twenty times the
largest measured maximum, and orders below what a wrong chunk makes: the
mutation below (the mask's query rows one position late — a chunk boundary
off by one) leaves the band by construction.

A decoding document is refused by name on every rank before any weights
(``check_context``), and ``dry-run`` reports the same refusal without torch.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

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
TABLES = ("iia.json", "logit_diff.json", "logit_diff_first.json")

#: The measured maxima (module docstring), per geometry and output class:
#: ``stream`` — the residual-stream reads in both chunks, the logits and the
#: metric tables; ``interior`` — the attention interior reads (the gathered
#: key, the scores and the pattern at the chunk's query rows) and, on the
#: MoE, the DeltaNet conv output and the per-step state; ``chunked_write`` —
#: the logits after a swap landed in the first chunk.
MEASURED: dict[str, dict[str, float]] = {
    "llama cp=2": {"stream": 6.0e-8, "interior": 1.5e-8, "chunked_write": 6.0e-8},
    "moe cp=2": {"stream": 9.6e-7, "interior": 7.2e-7, "chunked_write": 1.2e-6},
    "moe cp=2,tp=2": {"stream": 5.4e-7, "interior": 9.6e-7, "chunked_write": 7.8e-7},
}
#: At least twenty times the largest measured maximum (the loader's rule,
#: ``test_sharded_load.py``: ``1e-5`` over ``4.8e-7``): the Llama, whose only
#: re-association is the attention GEMM over a shorter query axis, sits at
#: the fp32 ulp of its values; the MoE adds the chunked kernel's blocks. The
#: mutation below moves the scores and the logits by orders more.
BAND = 3e-5
assert BAND >= 20 * max(max(v.values()) for v in MEASURED.values())

#: The full-attention layer the attention interior is tapped at.
LAYER = {TINY_LLAMA: 1, TINY_QWEN35_MOE: 3}
#: The DeltaNet layer the kernel boundary is tapped at on the MoE.
DELTA_LAYER = 1

#: Which output class each saved tensor file belongs to.
CLASSES: dict[str, str] = {
    "v_cf": "stream",
    "first": "stream",
    "patched_all": "stream",
    "logits": "stream",
    "r_key": "interior",
    "r_scores": "interior",
    "r_probs": "interior",
    "r_conv": "interior",
    "r_state": "interior",
    "logits_first": "chunked_write",
    "logits_state": "chunked_write",
}


def _document(tmp: Path, key: str) -> Path:
    """Corpus 02 on ``key`` at its full-attention layer, plus the reads and
    writes the module docstring lists."""
    doc = json.loads(CORPUS.read_text())
    layer = LAYER[key]
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    sites, reads, writes = method["sites"], method["reads"], method["writes"]
    models, save = method["intervened_models"], method["save"]
    sites["target"]["layers"] = [layer]
    save.append(_saved("v_cf", "original", "counterfactual"))
    save.append(_saved("logits", "patched"))
    # a read and a swap in the first chunk, the logits after it
    _read(reads, models, "first", {"site": "target", "pos": 0}, "original", "base")
    _read(
        reads,
        models,
        "first_cf",
        {"site": "target", "pos": 0},
        "original",
        "counterfactual",
    )
    writes["patch_first"] = {"site": "target", "pos": 0, "do": {"swap": "first_cf"}}
    models["patched_first"] = {"input": "base", "reads": [], "writes": ["patch_first"]}
    _read(
        reads,
        models,
        "logits_first",
        {"site": "lm_head", "pos": -1},
        "patched_first",
        "base",
    )
    save.append(_saved("first", "original"))
    save.append(_saved("logits_first", "patched_first"))
    # the corpus's logit difference, over the logits after the swap (§2.10)
    (logit_diff,) = [e for e in save if e.get("file_path") == "logit_diff.json"]
    save.append(
        {
            "read": "logits_first",
            "model": "patched_first",
            "aggregation": dict(logit_diff["aggregation"]),
            "file_path": "logit_diff_first.json",
        }
    )
    # every position of the patched residual: both chunks in one read
    _read(
        reads,
        models,
        "patched_all",
        {"site": "target", "pos": "all"},
        "patched",
        "base",
    )
    save.append(_saved("patched_all", "patched"))
    # the attention interior: the gathered key, the scores and the pattern
    for name, component, pos in (
        ("key", "attention_key", -1),
        # two position axes: the pattern is read whole (query rows chunked,
        # keys whole, gathered by the placement)
        ("scores", "attention_scores", "all"),
        ("probs", "attention_probs", "all"),
    ):
        sites[name] = {"component": component, "layers": [layer]}
        _read(
            reads, models, f"r_{name}", {"site": name, "pos": pos}, "original", "base"
        )
        save.append(_saved(f"r_{name}", "original"))
    if key == TINY_QWEN35_MOE:
        for name, component in (("conv", "delta_conv"), ("state", "delta_state")):
            sites[name] = {"component": component, "layers": [DELTA_LAYER]}
            _read(
                reads,
                models,
                f"r_{name}",
                {"site": name, "pos": -1},
                "original",
                "base",
            )
            save.append(_saved(f"r_{name}", "original"))
        # a state write at the answer slot: the stepwise substitution runs
        # on every rank from the state the rank below handed over, and the
        # write fires on the rank whose chunk holds the step alone — its
        # fire count made whole over the context group
        _read(
            reads,
            models,
            "state_cf",
            {"site": "state", "pos": -1},
            "original",
            "counterfactual",
        )
        writes["patch_state"] = {"site": "state", "pos": -1, "do": {"swap": "state_cf"}}
        models["patched_state"] = {
            "input": "base",
            "reads": [],
            "writes": ["patch_state"],
        }
        _read(
            reads,
            models,
            "logits_state",
            {"site": "lm_head", "pos": -1},
            "patched_state",
            "base",
        )
        save.append(_saved("logits_state", "patched_state"))
    target = tmp / "context.json"
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
    """The receipt's ``execution.parallel`` block for a geometry."""
    geometry = {"data": 1, "pipeline": 1, "context": 1, "tensor": 1, "expert": 1}
    geometry.update(axes)
    world = (
        geometry["data"]
        * geometry["pipeline"]
        * geometry["context"]
        * max(geometry["tensor"], geometry["expert"])
    )
    return {**geometry, "data_mode": "points", "world": world}


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


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
    _assert_receipts_and_stamps(solo, parallel, block)
    diffs = _max_diffs(solo, parallel)
    by_class: dict[str, float] = {}
    for name, worst in diffs.items():
        by_class[CLASSES[name]] = max(by_class.get(CLASSES[name], 0.0), worst)
        assert worst <= band, (name, worst)
    for table in TABLES:
        worst = _table_diff(solo, parallel, table)
        by_class["stream"] = max(by_class.get("stream", 0.0), worst)
        assert worst <= band, (table, worst)
    return by_class


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
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    # the tiny MoE is a hybrid tower: cp is refused on it unless waived (§8.4)
    monkeypatch.setenv("CAUSALAB_EXPERIMENTAL_CONTEXT", "1")


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


@pytest.fixture(scope="module")
def llama_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("cp-llama")
    document = _document(tmp, TINY_LLAMA)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def moe_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("cp-moe")
    document = _document(tmp, TINY_QWEN35_MOE)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


# --------------------------------------------------------------------------- #
# parity through the spawn parent
# --------------------------------------------------------------------------- #


def test_cp2_llama_agrees_with_world_one_within_the_band(
    llama_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = llama_solo
    out = tmp_path / "cp2"
    assert main(_argv(document, out, "--parallel", "cp=2")) == 0
    measured = _assert_parity(
        solo, out, {**_block(context=2), "launcher": "spawned"}, band=BAND
    )
    _assert_measured("llama cp=2", measured)


@pytest.mark.parametrize("geometry", ["cp=2", "cp=2,tp=2"])
def test_moe_agrees_with_world_one_within_the_band(
    moe_solo: tuple[Path, Path], tmp_path: Path, never_load: None, geometry: str
) -> None:
    document, solo = moe_solo
    out = tmp_path / geometry.replace(",", "_").replace("=", "")
    assert main(_argv(document, out, "--parallel", geometry)) == 0
    axes = {
        {"cp": "context", "tp": "tensor"}[k]: int(v)
        for k, v in (item.split("=") for item in geometry.split(","))
    }
    measured = _assert_parity(
        solo, out, {**_block(**axes), "launcher": "spawned"}, band=BAND
    )
    _assert_measured(f"moe {geometry}", measured)


# --------------------------------------------------------------------------- #
# the refusal: a decoding document
# --------------------------------------------------------------------------- #


def _decoding(document: Path) -> Path:
    doc = json.loads(document.read_text())
    doc["method"]["reads"]["logits"]["pos"] = {
        "generated": {"max_new_tokens": 2},
        "index": -1,
    }
    target = document.parent / "decode.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def test_a_decoding_document_is_refused_by_name_on_every_rank(
    llama_solo: tuple[Path, Path], tmp_path: Path, never_load: None, capsys: Any
) -> None:
    """Refused before any weights (``check_context``): the spawned ranks
    exit non-zero and the parent reports the failed rank; ``dry-run``
    reports the same refusal in this process, torch-free."""
    document, _ = llama_solo
    decoding = _decoding(document)
    assert main(_argv(decoding, tmp_path / "cp2", "--parallel", "cp=2")) != 0
    code = main(
        [
            "dry-run",
            str(decoding),
            "--engine",
            "pytorch_hooks",
            "--data-root",
            str(DATA),
            "--artifacts-root",
            str(decoding.parent),
            "--parallel",
            "cp=2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "--parallel.context" in captured.err and "cp=2" in captured.err
    # the same document at cp=1 passes the check (and would run)
    assert (
        main(
            [
                "dry-run",
                str(decoding),
                "--engine",
                "pytorch_hooks",
                "--data-root",
                str(DATA),
                "--artifacts-root",
                str(decoding.parent),
                "--parallel",
                "cp=1",
            ]
        )
        == 0
    )


# --------------------------------------------------------------------------- #
# the mutation: a chunk boundary off by one leaves the band
# --------------------------------------------------------------------------- #


def _mask_rows_one_late() -> None:
    """The whole frame's mask at query rows one position after this rank's
    chunk: the causal boundary moves by one on every row of every chunk."""
    from causalab.neural.shared.parallel import context

    original = context.local_causal_mask

    def late(
        attention_mask: torch.Tensor, chunk: range, dtype: torch.dtype
    ) -> torch.Tensor:
        padded_len = attention_mask.shape[1]
        shifted = range(
            min(chunk.start + 1, padded_len - 1), min(chunk.stop + 1, padded_len)
        )
        rows = original(attention_mask, shifted, dtype)
        if rows.shape[2] < len(chunk):
            rows = torch.cat(
                [rows, rows[:, :, -1:].expand(-1, -1, len(chunk) - rows.shape[2], -1)],
                dim=2,
            )
        return rows

    context.local_causal_mask = late  # type: ignore[assignment]
    context.SequenceFrame.attention_mask = (  # type: ignore[method-assign]
        lambda self, dtype: late(self.mask, self.chunk, dtype)
    )


MUTATIONS: dict[str, Callable[[], None]] = {"mask_rows_one_late": _mask_rows_one_late}


def _mutated_entry(
    rank: int, argv: list[str], world: int, port: int, mutation: str
) -> None:
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


def test_a_chunk_boundary_off_by_one_leaves_the_band_on_the_llama(
    llama_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = llama_solo
    out = tmp_path / "mutated"
    with reserve_port() as hold:
        mp.spawn(
            _mutated_entry,
            args=(
                _argv(document, out, "--parallel", "cp=2"),
                2,
                hold.port,
                "mask_rows_one_late",
            ),
            nprocs=2,
            join=True,
        )
    diffs = _max_diffs(solo, out)
    # the attention scores and everything downstream of the attention move
    assert diffs["r_scores"] > BAND, diffs
    assert diffs["logits"] > BAND or diffs["logits_first"] > BAND, diffs
