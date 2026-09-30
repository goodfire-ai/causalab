"""CLI training parity for tensor and expert parallelism over gloo on CPU.

Compare the corpus DAS fit on tiny Llama (``tp=2``) and tiny Qwen MoE
(``ep=2``) with world 1. The parent loads no model; children hold shards.
Each fit exercises OOM agreement, gradient averaging and, for MoE, expert
all-reduces in both forward and backward passes.

At replicated sites, gradients agree bit for bit across ranks. Differences
from world 1 arise from reduction order through the fp32 AdamW updates.
`MEASURED` records reference differences from torch 2.9.0 on CPU;
`BAND` is at least twenty times their maximum. Bundle differences are
absolute; table differences are relative to a magnitude of at least one.
Router gradients must sum the expert ranks' contributions before reaching
the residual. The parity assertion detects a missing sum.

Sharded-site fits cover ``attention_query`` under tensor parallelism and
``expert_activation`` under expert parallelism. Two successive attention
taps check that gradients survive gather and fragment. The recorded
pre-mean gradients must agree across ranks and match world 1 within
`GRADIENT_BAND`, relative to each step's largest entry. Expert-site
fits use a separate ``5e-4`` bundle band for small-gradient sensitivity.
Fixed step counts avoid early-stop differences at metric ties; vectorised
CPU kernels still round chunked and whole tensors differently.

Every fit enables the runtime gradient-agreement check at the gloo band
from ``conftest.GRADIENT_AGREEMENT_GLOO``. A fragment backward that retains
only its local slice must raise ``GradientDisagreement`` before averaging;
a detached gather must move the two-tap fit outside its band. Malformed
check settings must fail on every rank before a forward.

Mutation children also average in the wrong direction on one rank, zeroing
its update. With the agreement check disabled, this must fail fit parity;
with it enabled, the mismatch must be refused before the mean. Dropping
the mean at a replicated site is the identity, so that mutation alone
cannot test the averaging contract.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.multiprocessing.spawn import ProcessRaisedException

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine
from causalab.neural.engines.pytorch_hooks.loading import (
    LOAD_REPORT_VARIABLE,
    load_report_path,
)
from causalab.neural.shared.parallel import launcher
from causalab.neural.shared.parallel.agreements import AGREEMENT_VARIABLE
from causalab.io.tables import read_table
from causalab.neural.shared.parallel.spawn import reserve_port

from tests._helpers.paths import PROTOCOLS_DIR
from tests.neural.engines.pytorch_hooks.conftest import (
    GRADIENT_AGREEMENT_GLOO,
    TINY_LLAMA,
    TINY_QWEN35_MOE,
)
from tests._helpers.parity_band import fit_band

# every fit of this file runs with the §7 check on at the gloo band (module
# docstring, ``conftest.py``)
#
# `parallel_world`: every test here runs a document or a fit across a spawned
# multi-rank process world — minutes on the CI runner. The PR gate deselects the
# marker; the nightly CPU job runs it (docs/TESTS.md).
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.parallel_world,
    pytest.mark.usefixtures("checked_gradients_gloo"),
]

REPO = Path(__file__).resolve().parents[4]
DAS = REPO / "tests" / "protocols" / "04_das_im.json"
DBM_EXPERTS = PROTOCOLS_DIR / "dbm_expert_neuron.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"

RECEIPT = "protocol.json"
EVENTS = "events.jsonl"
TABLES = ("iia.json", "ce.json")

#: The layer the subspace is fitted at: the tiny Llama's first block; the
#: MoE fixture's second, a Gated DeltaNet block with routed experts.
LAYER = {TINY_LLAMA: 0, TINY_QWEN35_MOE: 1}
#: The subspace width, retargeted to the fixture like the layer: the corpus
#: document's ``k: 8`` is the **whole** residual of the hidden-8 MoE fixture,
#: and a full-space swap is the same swap under every rotation — the loss
#: does not depend on the parameters and the gradient is fp32 noise
#: (``|g| ≈ 6e-8``, measured), so the fit would be a random walk on
#: rounding. At ``k: 4`` the gradient is real (``|g| ≥ 7e-5`` at layer 1).
#: The tiny Llama's hidden 16 keeps the document's 8.
K = {TINY_LLAMA: 8, TINY_QWEN35_MOE: 4}

#: The measured maxima (module docstring), per geometry and output class:
#: ``bundle`` — the fitted rotation, absolute; ``tables`` — the saved
#: metrics, relative to a magnitude of at least one.
MEASURED: dict[str, dict[str, float]] = {
    "llama tp=2": {"bundle": 3.0e-8, "tables": 1.5e-8},
    "moe ep=2": {"bundle": 3.0e-7, "tables": 9.0e-8},
    # run by the golden tier (``test_train_parallel_run_tp8.py``)
    "moe tp=8": {"bundle": 3.9e-7, "tables": 1.8e-7},
    # the sharded sites (module docstring)
    "llama attention_query tp=2": {"bundle": 1.5e-8, "tables": 7.5e-9},
    "llama attention_query x2 tp=2": {"bundle": 6.0e-8, "tables": 9.2e-8},
    "moe expert_activation ep=2": {"bundle": 2.4e-5, "tables": 3.2e-7},
}
#: At least twenty times the largest measured maximum (the loader's rule);
#: the wrong-direction mutation lands at ``2.7e-3`` / ``1.1e-4`` on the
#: Llama and ``1.8e-2`` / ``5.1e-3`` on the MoE, three orders outside.
BAND = 1e-5
#: The expert-neuron DBM fit at ``expert_activation`` has a band of its own
#: (module docstring): twenty times its measured maximum.
BANDS: dict[str, float] = {"moe expert_activation ep=2": 5e-4}
for _label, _measured in MEASURED.items():
    assert BANDS.get(_label, BAND) >= 20 * max(_measured.values()), _label
#: The featurizer's pre-mean gradient against world 1, relative to its
#: largest entry, at every step (module docstring): measured ``8.7e-7`` and
#: ``5.3e-5``; pinned at more than sixteen times the larger.
GRADIENT_MEASURED: dict[str, float] = {
    "llama attention_query x2 tp=2": 8.7e-7,
    "moe expert_activation ep=2": 5.3e-5,
}
GRADIENT_BAND = 1e-3
assert GRADIENT_BAND >= 16 * max(GRADIENT_MEASURED.values())
#: Where a recording guard (`_recording_guard`) writes each rank's
#: gradients; set by the parent for the run, read in the children.
GRADIENTS_VARIABLE = "CAUSALAB_TEST_GRADIENTS_DIR"
#: The recording guards' saves, run at exit in a child and by hand in the
#: parent after its own world-1 fit.
_RECORDER_SAVES: list[Callable[[], None]] = []


def _document(tmp: Path, key: str, *, component: str = "block_output") -> Path:
    """The corpus DAS fit retargeted to ``key``: its layer and its width,
    fp32; ``component`` moves the site (``attention_query``: the head-major
    ``(b, s, H·d)`` contract, the tiny Llama's four heads of four)."""
    doc = json.loads(DAS.read_text())
    doc["model"] = {"key": key, "revision": "main", "dtype": "fp32"}
    doc["method"]["sites"]["target"] = {"component": component, "layers": [LAYER[key]]}
    doc["method"]["featurizers"]["rot"]["k"] = K[key]
    target = tmp / "das.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _two_site_document(tmp: Path) -> Path:
    """The tiny Llama's DAS fit at ``attention_query`` on layers 0 **and** 1
    through one featurizer: a featurizer on a band is refused by the parser
    (one map per layer is the document's choice to make), so the band is two
    sites, two reads and two writes sharing ``rot``, the second tap
    downstream of the first."""
    doc = json.loads(DAS.read_text())
    doc["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    method = doc["method"]
    method["featurizers"]["rot"]["k"] = K[TINY_LLAMA]
    method["sites"] = {
        "target0": {"component": "attention_query", "layers": [0]},
        "target1": {"component": "attention_query", "layers": [1]},
        "lm_head": {"component": "lm_head"},
    }
    source = method["reads"]["v_cf"]
    method["reads"] = {
        "v_cf0": {**source, "site": "target0"},
        "v_cf1": {**source, "site": "target1"},
        "logits": method["reads"]["logits"],
    }
    patch = method["writes"]["patch"]
    method["writes"] = {
        "patch0": {**patch, "site": "target0", "do": {"swap": "v_cf0"}},
        "patch1": {**patch, "site": "target1", "do": {"swap": "v_cf1"}},
    }
    method["intervened_models"] = {
        "original_counterfactual": {
            "input": "counterfactual",
            "reads": ["v_cf0", "v_cf1"],
        },
        "patched": {
            "input": "base",
            "reads": ["logits"],
            "writes": ["patch0", "patch1"],
        },
    }
    method["save"][-1] = {
        "value": "rot",
        "site": "target0",
        "file_path": "rot.safetensors",
    }
    # a fixed step count: the corpus fit's early stop sat at a tie here
    # (module docstring)
    method["train"].pop("early_stop")
    target = tmp / "das_two.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


def _experts_document(tmp: Path) -> Path:
    """The expert-neuron DBM preset on the tiny MoE: the corpus data, the
    fixture's layer, the DAS fit's step count, fp32."""
    doc = json.loads(DBM_EXPERTS.read_text())
    das = json.loads(DAS.read_text())
    doc["model"] = {"key": TINY_QWEN35_MOE, "revision": "main", "dtype": "fp32"}
    doc["data"] = das["data"]
    method = doc["method"]
    for name in ("routed", "shared"):
        method["sites"][name]["layers"] = [LAYER[TINY_QWEN35_MOE]]
    method["train"]["steps"] = das["method"]["train"]["steps"]
    method["train"]["eval"]["split"] = das["method"]["train"]["eval"]["split"]
    target = tmp / "dbm_experts.json"
    target.write_text(json.dumps(doc, indent=2))
    return target


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
        # Set the row budget explicitly: free memory can differ across ranks,
        # but the receipts must compare exactly.
        "--fit-rows",
        "16",
        *extra,
    ]


def _block(**axes: int) -> dict[str, Any]:
    geometry = {"data": 1, "pipeline": 1, "context": 1, "tensor": 1, "expert": 1}
    geometry.update(axes)
    return {
        **geometry,
        "data_mode": "points",
        "world": max(geometry["tensor"], geometry["expert"]),
        "launcher": "spawned",
    }


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _bundle_diff(solo: Path, other: Path) -> float:
    """The largest absolute difference over every fitted bundle's entries."""
    bundles = sorted(p.name for p in solo.glob("*.safetensors"))
    assert bundles, "a fit saves at least one bundle"
    worst = 0.0
    for bundle in bundles:
        a, b = load_file(str(solo / bundle)), load_file(str(other / bundle))
        assert set(a) == set(b)
        for name in a:
            if a[name].shape != b[name].shape:
                return math.inf
            worst = max(worst, float((a[name].double() - b[name].double()).abs().max()))
    return worst


def _tables_diff(solo: Path, other: Path) -> float:
    """The largest relative difference over every float cell of every table
    (to a magnitude of at least one); a non-float cell that differs is
    ``inf``."""
    worst = 0.0
    for name in TABLES:
        a_rows, b_rows = read_table(solo / name), read_table(other / name)
        assert len(a_rows) == len(b_rows), name
        for a, b in zip(a_rows, b_rows):
            assert set(a) == set(b), name
            for key in a:
                if isinstance(a[key], float) or isinstance(b[key], float):
                    x, y = float(a[key]), float(b[key])
                    worst = max(worst, abs(x - y) / max(1.0, abs(x)))
                elif a[key] != b[key]:
                    return math.inf
    return worst


def _diffs(solo: Path, other: Path) -> dict[str, float]:
    assert sorted(p.name for p in solo.iterdir()) == sorted(
        p.name for p in other.iterdir()
    )
    return {"bundle": _bundle_diff(solo, other), "tables": _tables_diff(solo, other)}


def _assert_parity(
    solo: Path, parallel: Path, block: dict[str, Any], label: str
) -> dict[str, float]:
    """The parallel fit against the world-1 fit: the receipt equal but for
    ``execution.parallel`` — the measured bounds ``fit_rows_resolved`` /
    ``fit_rows_shrinks`` included, key by key — and every output within the
    band; the maxima are returned for the record."""
    a, b = _receipt(solo), _receipt(parallel)
    assert b["execution"]["parallel"] == block
    assert a["execution"]["parallel"]["launcher"] == "solo"
    for key in ("fit_rows_resolved", "fit_rows_shrinks"):
        assert a["execution"].get(key) == b["execution"].get(key), key
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    measured = _diffs(solo, parallel)
    print(label, measured)
    for kind, worst in measured.items():
        assert worst <= BANDS.get(label, BAND), (label, kind, worst)
        # a drift past the record is a re-measurement, not a silent pass
        assert worst <= fit_band(MEASURED[label][kind]), (label, kind, worst)
    return measured


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawned ranks inherit the environment: offline, one thread each."""
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
    tmp = tmp_path_factory.mktemp("train-llama")
    document = _document(tmp, TINY_LLAMA)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def moe_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("train-moe")
    document = _document(tmp, TINY_QWEN35_MOE)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def llama_query_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("train-llama-query")
    document = _document(tmp, TINY_LLAMA, component="attention_query")
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def llama_two_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("train-llama-two")
    document = _two_site_document(tmp)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


@pytest.fixture(scope="module")
def moe_experts_solo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("train-moe-experts")
    document = _experts_document(tmp)
    out = tmp / "solo"
    assert main(_argv(document, out)) == 0
    return document, out


# --------------------------------------------------------------------------- #
# parity through the spawn parent
# --------------------------------------------------------------------------- #


def test_tp2_fit_on_the_llama_lands_within_the_band(
    llama_solo: tuple[Path, Path],
    tmp_path: Path,
    never_load: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, solo = llama_solo
    out = tmp_path / "tp2"
    reports = tmp_path / "reports"
    monkeypatch.setenv(LOAD_REPORT_VARIABLE, str(reports))
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    _assert_parity(solo, out, _block(tensor=2), "llama tp=2")
    _assert_load_reports(reports, world=2)


def test_ep2_fit_on_the_moe_lands_within_the_band(
    moe_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = moe_solo
    out = tmp_path / "ep2"
    assert main(_argv(document, out, "--parallel", "ep=2")) == 0
    _assert_parity(solo, out, _block(expert=2), "moe ep=2")


# --------------------------------------------------------------------------- #
# parity at the sharded sites (module docstring)
# --------------------------------------------------------------------------- #


def test_tp2_fit_at_attention_query_lands_within_the_band(
    llama_query_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = llama_query_solo
    out = tmp_path / "tp2"
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    _assert_parity(solo, out, _block(tensor=2), "llama attention_query tp=2")


def test_tp2_fit_through_two_attention_query_taps_lands_within_the_band(
    llama_two_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = llama_two_solo
    out = tmp_path / "tp2"
    assert main(_argv(document, out, "--parallel", "tp=2")) == 0
    _assert_parity(solo, out, _block(tensor=2), "llama attention_query x2 tp=2")


def test_ep2_fit_at_expert_activation_lands_within_the_band(
    moe_experts_solo: tuple[Path, Path], tmp_path: Path, never_load: None
) -> None:
    document, solo = moe_experts_solo
    out = tmp_path / "ep2"
    assert main(_argv(document, out, "--parallel", "ep=2")) == 0
    _assert_parity(solo, out, _block(expert=2), "moe expert_activation ep=2")


def _assert_load_reports(reports: Path, *, world: int) -> None:
    """Every rank wrote its report (``CAUSALAB_LOAD_REPORT_DIR``): the same
    parameter set, each sharded parameter requested at exactly ``1 / world``
    of its bytes on disk and each replicated one whole, at least one of
    each, and the sharded set the same on every rank."""
    records = [
        json.loads(load_report_path(reports, rank).read_text()) for rank in range(world)
    ]
    sharded_sets: list[set[str]] = []
    for rank, record in enumerate(records):
        assert record["rank"] == rank and record["world"] == world
        requested, on_disk = record["bytes_requested"], record["bytes_on_disk"]
        assert set(requested) == set(on_disk) and requested
        sharded = {n for n in requested if requested[n] != on_disk[n]}
        for name in sharded:
            assert on_disk[name] % world == 0, name
            assert requested[name] == on_disk[name] // world, name
        assert sharded and sharded != set(requested)
        sharded_sets.append(sharded)
    assert all(s == sharded_sets[0] for s in sharded_sets)


# --------------------------------------------------------------------------- #
# the mutations, in a torchrun-style child
# --------------------------------------------------------------------------- #


def _dropped_guard() -> None:
    """§7's guard skipped on every rank: the identity at a replicated site,
    which the measurement below shows rather than assumes."""
    from causalab.neural.engines.pytorch_hooks import train as train_module

    def none(parameters: Any, collective: Any, **kwargs: Any) -> None:
        return None

    train_module.average_gradients = none  # type: ignore[assignment]


def _wrong_direction_guard() -> None:
    """A guard whose one rank contributes its gradient negated: the mean of
    ``g`` and ``-g`` is zero, so no update ever moves the featurizer. Run
    with the check off — with it on, the negated rank is refused before the
    mean (a relative disagreement of two) and there is no fit to compare;
    the statement here is the guard's arithmetic alone."""
    from causalab.neural.engines.pytorch_hooks import train as train_module

    _unchecked_guard()

    real = train_module.average_gradients

    def wrong(parameters: Any, collective: Any, **kwargs: Any) -> None:
        parameters = list(parameters)
        if collective.rank("model") == 1:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.neg_()
        real(parameters, collective, **kwargs)

    train_module.average_gradients = wrong  # type: ignore[assignment]


def _recording_guard() -> None:
    """Not a mutation: the §7 guard recording every parameter's gradient
    before the mean, per step, to ``GRADIENTS_VARIABLE/rank<r>.pt`` at exit
    — what the gradient statement compares across the ranks and to world 1."""
    import atexit
    import os

    import torch

    from causalab.neural.engines.pytorch_hooks import train as train_module

    real = train_module.average_gradients
    records: list[list[torch.Tensor]] = []

    def recording(parameters: Any, collective: Any, **kwargs: Any) -> None:
        parameters = list(parameters)
        records.append(
            [p.grad.detach().clone() for p in parameters if p.grad is not None]
        )
        real(parameters, collective, **kwargs)

    train_module.average_gradients = recording  # type: ignore[assignment]
    rank = int(os.environ.get("RANK", "0"))  # as the launcher sets it; 0 at world 1
    target = Path(os.environ[GRADIENTS_VARIABLE]) / f"rank{rank}.pt"

    def save() -> None:
        torch.save(records, target)

    _RECORDER_SAVES.append(save)
    atexit.register(save)


def _checked_guard() -> None:
    """Not a mutation: the §7 check at ``0`` — bit identity across the
    ranks before the mean, tighter than the band the fixture sets — through
    the variable the child reads ([`AGREEMENT_VARIABLE`][causalab.neural.shared.parallel.environment.AGREEMENT_VARIABLE])."""
    import os

    os.environ[AGREEMENT_VARIABLE] = "0"


def _unchecked_guard() -> None:
    """The variable unset in the child: the guard is the plain mean, as a
    production run without the setting has it."""
    import os

    os.environ.pop(AGREEMENT_VARIABLE, None)


def _malformed_setting() -> None:
    """The variable holding something that is not a tolerance."""
    import os

    os.environ[AGREEMENT_VARIABLE] = "half"


def _own_slice_fragment() -> None:
    """The pairing's first mutation: a ``fragment`` that is plain torch math
    — the code before ``parallel/autograd.py`` — whose backward scatters
    this rank's slice into zeros, a partial gradient. Nothing else: whether
    the guard sees it is the variable's doing (module docstring)."""
    from causalab.neural.shared.parallel import fragments as fragments_module

    def own_slice(
        edited: Any, dim: int, axis: Any, collective: Any, **kwargs: Any
    ) -> Any:
        rank, size = collective.rank(axis), collective.size(axis)
        return fragments_module._chunk(
            edited, dim, rank, size, fragments_module.Sharded(dim, axis)
        )

    fragments_module.edit_fragment = own_slice  # type: ignore[assignment]


def _own_slice_fragment_unchecked() -> None:
    _own_slice_fragment()
    _unchecked_guard()


def _detaching_whole() -> None:
    """The pairing's second mutation: a ``whole`` that is the raw all-gather,
    fresh buffers off the graph."""
    from causalab.neural.shared.parallel import fragments as fragments_module

    def detaching(
        tensor: Any, dim: int, axis: Any, collective: Any, **kwargs: Any
    ) -> Any:
        return collective.all_gather(tensor, dim, axis)

    fragments_module.gather_for_edit = detaching  # type: ignore[assignment]


MUTATIONS: dict[str, Callable[[], None]] = {
    "dropped_guard": _dropped_guard,
    "wrong_direction_guard": _wrong_direction_guard,
    "checked_guard": _checked_guard,
    "malformed_setting": _malformed_setting,
    "recording_guard": _recording_guard,
    "own_slice_fragment": _own_slice_fragment,
    "own_slice_fragment_unchecked": _own_slice_fragment_unchecked,
    "detaching_whole": _detaching_whole,
}


def _mutated_entry(
    rank: int, argv: Sequence[str], world: int, port: int, mutation: str
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


def _run_mutated(document: Path, out: Path, geometry: str, mutation: str) -> None:
    with reserve_port() as hold:
        mp.spawn(
            _mutated_entry,
            args=(_argv(document, out, "--parallel", geometry), 2, hold.port, mutation),
            nprocs=2,
            join=True,
        )


def test_dropping_the_guard_is_the_identity_at_a_replicated_site(
    llama_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    """§7, measured: the featurizer's gradient is already the same tensor on
    every rank, so a fit without the guard lands where the guarded fit does
    — which is why the guard is held by the mutations below, not by this one."""
    document, solo = llama_solo
    out = tmp_path / "dropped"
    _run_mutated(document, out, "tp=2", "dropped_guard")
    diffs = _diffs(solo, out)
    assert all(worst <= BAND for worst in diffs.values()), diffs


@pytest.mark.parametrize(
    "fixture, geometry", [("llama_solo", "tp=2"), ("moe_solo", "ep=2")]
)
def test_a_guard_averaging_in_the_wrong_direction_leaves_the_band(
    request: pytest.FixtureRequest, tmp_path: Path, fixture: str, geometry: str
) -> None:
    document, solo = request.getfixturevalue(fixture)
    out = tmp_path / "wrong"
    _run_mutated(document, out, geometry, "wrong_direction_guard")
    diffs = _diffs(solo, out)
    assert diffs["bundle"] > BAND and diffs["tables"] > BAND, diffs


# --------------------------------------------------------------------------- #
# the pairing at a sharded site, held by its gradient and its mutations
# --------------------------------------------------------------------------- #


def test_the_ranks_gradients_agree_before_the_mean_at_a_sharded_site(
    llama_query_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    """§7's invariant for real: the check at ``0`` — bit identity across
    the ranks before the mean, through the variable — passes at
    ``attention_query`` under ``tp=2`` and the checked fit lands where the
    band-checked one does."""
    document, solo = llama_query_solo
    out = tmp_path / "checked"
    _run_mutated(document, out, "tp=2", "checked_guard")
    diffs = _diffs(solo, out)
    assert all(worst <= BAND for worst in diffs.values()), diffs


def test_a_fragment_keeping_its_own_slice_is_refused_through_the_variable(
    llama_query_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    """The hole closed (module docstring): nothing but the fixture's
    ``CAUSALAB_GRADIENT_AGREEMENT`` stands between the mutation and a quiet
    half-gradient fit, and the real loop refuses it by name."""
    document, _ = llama_query_solo
    assert os.environ[AGREEMENT_VARIABLE] == GRADIENT_AGREEMENT_GLOO
    with pytest.raises(ProcessRaisedException) as err:
        _run_mutated(document, tmp_path / "own_slice", "tp=2", "own_slice_fragment")
    assert "GradientDisagreement" in str(err.value)
    assert AGREEMENT_VARIABLE in str(err.value)


def test_the_same_mutation_runs_through_with_the_variable_unset(
    llama_query_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    """The hole itself, measured: without the setting the guard averages
    two partials and the fit completes — AdamW normalises the half gradient
    away, so the fit even lands near the band — which is why the check is a
    runtime setting and on in every smoke, not a test-only seam."""
    document, solo = llama_query_solo
    out = tmp_path / "own_slice_unchecked"
    _run_mutated(document, out, "tp=2", "own_slice_fragment_unchecked")
    diffs = _diffs(solo, out)
    print("llama attention_query tp=2 own-slice unchecked", diffs)
    assert math.isfinite(diffs["bundle"]), "the fit ran to completion and saved"


def test_a_malformed_setting_is_refused_by_name_on_every_rank(
    llama_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, _ = llama_solo
    with pytest.raises(ProcessRaisedException) as err:
        _run_mutated(document, tmp_path / "malformed", "tp=2", "malformed_setting")
    assert "AgreementSetting" in str(err.value)
    assert f"{AGREEMENT_VARIABLE}='half'" in str(err.value)


def test_a_whole_that_detaches_leaves_the_band_through_two_taps(
    llama_two_solo: tuple[Path, Path], tmp_path: Path
) -> None:
    document, solo = llama_two_solo
    out = tmp_path / "detached"
    _run_mutated(document, out, "tp=2", "detaching_whole")
    diffs = _diffs(solo, out)
    assert diffs["bundle"] > BAND, diffs


# --------------------------------------------------------------------------- #
# the gradient statement: every step, every rank, against world 1
# --------------------------------------------------------------------------- #


def _recorded(directory: Path, rank: int) -> list[list[Any]]:
    import torch

    # Map every rank's recorded tensors to CPU before comparing devices.
    return torch.load(directory / f"rank{rank}.pt", map_location="cpu")


def _solo_gradients(document: Path, tmp: Path) -> list[list[Any]]:
    """The world-1 fit run in this process with the recording guard installed
    for its duration; the loader is the same one the fixtures used."""
    from causalab.neural.engines.pytorch_hooks import train as train_module

    real = train_module.average_gradients
    os.environ[GRADIENTS_VARIABLE] = str(tmp)
    try:
        _recording_guard()
        assert main(_argv(document, tmp / "solo")) == 0
        # a child saves at exit; the parent saves now
        for save in _RECORDER_SAVES:
            save()
        _RECORDER_SAVES.clear()
    finally:
        train_module.average_gradients = real  # type: ignore[assignment]
    return _recorded(tmp, 0)


def _assert_gradient_parity(
    solo: list[list[Any]], ranks: list[list[list[Any]]], label: str
) -> None:
    """Every step: the ranks' gradients bit-identical, and each within
    `GRADIENT_BAND` of world 1 relative to the step's largest entry."""
    assert len(solo) > 0 and all(len(r) == len(solo) for r in ranks), (
        label,
        len(solo),
        [len(r) for r in ranks],
    )
    worst = 0.0
    for step, grads in enumerate(solo):
        for index, grad in enumerate(grads):
            mine = [r[step][index] for r in ranks]
            for other in mine[1:]:
                assert bool((other == mine[0]).all()), (label, step, index, "ranks")
            scale = max(float(grad.abs().max()), 1e-30)
            worst = max(worst, float((mine[0] - grad).abs().max()) / scale)
    print(label, "gradient", worst)
    assert worst <= GRADIENT_BAND, (label, worst)
    assert worst <= GRADIENT_MEASURED[label] * 2, (label, worst)


@pytest.mark.parametrize(
    "fixture, geometry, label",
    [
        ("llama_two_solo", "tp=2", "llama attention_query x2 tp=2"),
        ("moe_experts_solo", "ep=2", "moe expert_activation ep=2"),
    ],
)
def test_the_featurizer_gradient_is_the_world_one_gradient_on_every_rank(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: str,
    geometry: str,
    label: str,
) -> None:
    document, _ = request.getfixturevalue(fixture)
    solo_dir, parallel_dir = tmp_path / "solo", tmp_path / "parallel"
    solo_dir.mkdir()
    parallel_dir.mkdir()
    solo = _solo_gradients(document, solo_dir)
    monkeypatch.setenv(GRADIENTS_VARIABLE, str(parallel_dir))
    _run_mutated(document, tmp_path / "fit", geometry, "recording_guard")
    _assert_gradient_parity(solo, [_recorded(parallel_dir, r) for r in range(2)], label)
