"""The parallel smokes on CUDA over NCCL at worlds 2, 4 and 8
(``docs/model_parallelism.md`` §2, §8, §10.6) — the CUDA twin of the gloo
tier, on the tiny fixtures, every case gated on the devices the node shows.

The gloo smokes hard-code world 2 (and ``ep=4``, ``tp=2,ep=4``, ``pp=4`` on
the MoE); this module runs their documents through the real CLI at
``--device cuda`` and the geometries a two-GPU node refuses by name
(``launcher.check_spawn_devices``): ``tp=2,ep=4``, ``pp=4``, ``dp=2,tp=2``,
``pp=2,ep=2``, ``cp=2,tp=2``, ``dp=2:rows,pp=2``, ``dp=4`` and ``tp=4`` at
world 4 (and ``dp=4:rows``, which the corpus fit's two-row minibatch refuses
by name, §8.3); ``tp=8`` — above the tiny MoE's four KV heads, so
the K/V projections are **replicated** and each rank keeps the one KV head
its query heads read (``kv_replicated``, §6.6) — and ``ep=8`` at world 8;
``tp=2,ep=2`` at world 2, which the gloo tier never spelled. Inference and
fits: the tensor/expert boundary document, the pipeline document (through
the CLI's spawn, where the gloo module drives ``run_protocol`` itself), the
context document and the data scan; the DAS fit under every model axis and
under a pipeline, the context fit, and the ``rows`` mode.

Every run is compared to the world-1 run of the same document on the same
device the way its smoke compares: **byte for byte** where §8 says exact
(``dp`` over points, ``pp``), within the smoke's pinned band otherwise, and
against the maxima `MEASURED` pins for this node so a drift past
them fails rather than passes — the smoke tier's own drift rule. The boundary
document's target is swept over two layers (``_parallel_worlds.swept``), so
``dp=2`` has two points to shard and a pipeline a write on each stage. The
pytest process runs the world-1 oracles and never loads a model for a spawn
(``never_load``); each geometry's children hold the shards, one geometry at
a time.

`MEASURED` contains fp32 NCCL reference differences captured on H100
GPUs with torch 2.9.0+cu128. Fit references use the masked grouped-MM
expert path; reduction order can differ across combined tensor/expert
geometries. The pipeline pre-flight uses the full model table, with a
CPU regression test in ``test_memory_preflight.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

from tests.golden import _parallel as par
from tests.golden import _parallel_worlds as worlds
from tests.neural.engines.pytorch_hooks import (
    test_context_parallel_run as context,
    test_context_parallel_train_run as context_train,
    test_data_parallel_rows_run as rows,
    test_data_parallel_run as data,
    test_pipeline_run as pipeline,
    test_tensor_expert_parallel_run as boundary,
    test_train_parallel_run as train,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

DEVICE = "cuda"

#: The boundary document's target sweep per fixture: the attention taps'
#: layer (the fixture's full-attention layer) and one below it.
SWEEP = {TINY_LLAMA: (0, 1), TINY_QWEN35_MOE: (1, 3)}

#: The measured maxima per label and output class (module docstring), the
#: classes each smoke's; pinned by the smoke tier's rule (twice the
#: maximum, a floor) through `_parallel_worlds.pin_problems`.
MEASURED: dict[str, dict[str, float]] = {
    # inference: the boundary document (the tensor/expert smoke's classes)
    "moe tp=2,ep=2": {
        "stream": 5.4e-7,
        "sharded_read": 6.0e-7,
        "sharded_write": 5.9e-7,
        "experts": 2.3e-8,
    },
    "moe tp=2,ep=4": {
        "stream": 5.4e-7,
        "sharded_read": 4.8e-7,
        "sharded_write": 6.9e-7,
        "experts": 3.0e-8,
    },
    "moe dp=2,tp=2": {
        "stream": 5.6e-7,
        "sharded_read": 7.2e-7,
        "sharded_write": 6.0e-7,
        "experts": 3.0e-8,
    },
    "moe pp=2,ep=2": {
        "stream": 4.8e-7,
        "sharded_read": 7.2e-7,
        "sharded_write": 4.8e-7,
        "experts": 2.3e-8,
    },
    "moe tp=8": {
        "stream": 6.0e-7,
        "sharded_read": 7.2e-7,
        "sharded_write": 8.4e-7,
        "experts": 3.0e-8,
    },
    "moe ep=8": {
        "stream": 4.8e-7,
        "sharded_read": 4.8e-7,
        "sharded_write": 7.2e-7,
        "experts": 2.3e-8,
    },
    "llama tp=4": {"stream": 6.0e-8, "sharded_read": 1.5e-8, "sharded_write": 9.0e-8},
    # inference: the context document (the context smoke's classes)
    "moe cp=2,tp=2": {"stream": 5.1e-7, "interior": 7.2e-7, "chunked_write": 6.0e-7},
    "moe cp=2,tp=4": {"stream": 6.1e-7, "interior": 7.2e-7, "chunked_write": 6.0e-7},
    # fits: the DAS bundle (absolute) and the tables (relative)
    # Combined tensor/expert reduction order has its own fit reference.
    "moe fit tp=2,ep=2": {"bundle": 3.6e-7, "tables": 1.8e-7},
    "moe fit ep=4": {"bundle": 1.4e-7, "tables": 1.1e-7},
    "moe fit tp=2,ep=4": {"bundle": 2.6e-7, "tables": 7.9e-8},
    "moe fit tp=8": {"bundle": 3.8e-8, "tables": 1.2e-7},
    "moe fit ep=8": {"bundle": 1.9e-7, "tables": 1.8e-7},
    "llama fit tp=4": {"bundle": 3.8e-9, "tables": 0.0},
    "moe context fit cp=2,tp=2": {"bundle": 3.0e-8, "tables": 1.8e-7},
}
#: The floor under the pins: the smokes' own (``1e-7`` for a boundary read,
#: ``1e-8`` for a fit).
FLOORS = {"boundary": 1e-7, "context": 1e-7, "fit": 1e-8}


def _needs(text: str) -> Any:
    return pytest.param(
        text, marks=worlds.needs(worlds.geometry_of(text).world), id=text
    )


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The spawned ranks inherit the environment: offline, one thread each."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    # the tiny MoE is a hybrid tower: cp is refused on it unless waived (§8.4)
    monkeypatch.setenv("CAUSALAB_EXPERIMENTAL_CONTEXT", "1")


@pytest.fixture
def never_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent of a spawn never loads a model: in *this* process the
    loader raises; the children are fresh interpreters and load normally."""

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the spawn parent entered load_model")

    monkeypatch.setattr(hooks_engine, "load_model", never)


def _solo(document: Path, argv: Any) -> Path:
    out = document.parent / f"solo_{document.stem}"
    assert main(argv(document, out, device=DEVICE)) == 0
    return out


@pytest.fixture(scope="module")
def moe_boundary(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("boundary-moe")
    document = worlds.swept(
        boundary._document(tmp, TINY_QWEN35_MOE),  # pyright: ignore[reportPrivateUsage]
        SWEEP[TINY_QWEN35_MOE],
    )
    return document, _solo(document, boundary._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def llama_boundary(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("boundary-llama")
    document = worlds.swept(
        boundary._document(tmp, TINY_LLAMA),  # pyright: ignore[reportPrivateUsage]
        SWEEP[TINY_LLAMA],
    )
    return document, _solo(document, boundary._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def moe_pipeline(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The pipeline smoke's MoE document: the write on layer 1, the probe on
    layer 3 — every stage of four holds a module the document touches."""
    tmp = tmp_path_factory.mktemp("pipeline-moe")
    document = pipeline._document(  # pyright: ignore[reportPrivateUsage]
        tmp, TINY_QWEN35_MOE, write_layer=1, read_layer=3
    )
    return document, _solo(document, boundary._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def moe_context(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("context-moe")
    document = context._document(tmp, TINY_QWEN35_MOE)  # pyright: ignore[reportPrivateUsage]
    return document, _solo(document, context._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def llama_scan(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("scan-llama")
    document = data._document(tmp)  # pyright: ignore[reportPrivateUsage]
    return document, _solo(document, data._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def moe_fit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("fit-moe")
    document = train._document(tmp, TINY_QWEN35_MOE)  # pyright: ignore[reportPrivateUsage]
    return document, _solo(document, train._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def llama_fit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("fit-llama")
    document = train._document(tmp, TINY_LLAMA)  # pyright: ignore[reportPrivateUsage]
    return document, _solo(document, train._argv)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def moe_context_fit(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("context-fit-moe")
    document = context_train._document(tmp, TINY_QWEN35_MOE)  # pyright: ignore[reportPrivateUsage]
    return document, _solo(document, train._argv)  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- #
# the comparison
# --------------------------------------------------------------------------- #


def _run(document: Path, argv: Any, out: Path, text: str) -> Path:
    assert main(argv(document, out, "--parallel", text, device=DEVICE)) == 0
    return out


def _out(tmp_path: Path, text: str) -> Path:
    return tmp_path / text.replace(",", "_").replace("=", "").replace(":", "-")


def _pinned(
    label: str, measured: dict[str, float], *, band: float, floor: float
) -> None:
    print(label, measured)
    assert worlds.pin_problems(label, measured, MEASURED, band=band, floor=floor) == []


def _exact(solo: Path, out: Path, text: str) -> None:
    assert worlds.receipt_problems(solo, out, text) == []
    assert par.exact_differences(solo, out) == []


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        _needs("tp=2,ep=2"),
        _needs("tp=2,ep=4"),
        _needs("dp=2,tp=2"),
        _needs("pp=2,ep=2"),
        _needs("tp=8"),
        _needs("ep=8"),
    ],
)
def test_the_moe_boundary_document_lands_within_the_band(
    moe_boundary: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """Every module-boundary class read and written, the routing table
    swapped, the ``expert:`` face — within the tensor/expert smoke's band;
    ``tp=8`` through the replicated K/V projections."""
    document, solo = moe_boundary
    out = _run(document, boundary._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    measured = boundary._assert_parity(  # pyright: ignore[reportPrivateUsage]
        solo, out, worlds.block(text), band=boundary.BAND
    )
    _pinned(f"moe {text}", measured, band=boundary.BAND, floor=FLOORS["boundary"])


@pytest.mark.parametrize("text", [_needs("tp=4")])
def test_the_llama_boundary_document_lands_within_the_band(
    llama_boundary: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """The tiny Llama's four heads over four ranks: one head a rank."""
    document, solo = llama_boundary
    out = _run(document, boundary._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    measured = boundary._assert_parity(  # pyright: ignore[reportPrivateUsage]
        solo, out, worlds.block(text), band=boundary.BAND
    )
    _pinned(f"llama {text}", measured, band=boundary.BAND, floor=FLOORS["boundary"])


@pytest.mark.parametrize("text", [_needs("pp=4")])
def test_the_moe_pipeline_document_is_byte_identical_through_the_cli(
    moe_pipeline: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """Four stages, one layer each, spawned by the CLI (§6.5, §8.3: placement,
    no reduction moves — byte identity)."""
    document, solo = moe_pipeline
    out = _run(document, boundary._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    _exact(solo, out, text)


@pytest.mark.parametrize("text", [_needs("cp=2,tp=2"), _needs("cp=2,tp=4")])
def test_the_moe_context_document_lands_within_the_band(
    moe_context: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """Reads and writes in different chunks, the attention interior at the
    gathered key, the DeltaNet handoffs — under a tensor group too (§8.4)."""
    document, solo = moe_context
    out = _run(document, context._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    measured = context._assert_parity(  # pyright: ignore[reportPrivateUsage]
        solo, out, worlds.block(text), band=context.BAND
    )
    _pinned(f"moe {text}", measured, band=context.BAND, floor=FLOORS["context"])


@pytest.mark.parametrize("text", [_needs("dp=4")])
def test_the_llama_scan_is_byte_identical_over_four_replicas(
    llama_scan: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """Four points, one a replica, joined by point digest (§8.3)."""
    document, solo = llama_scan
    out = _run(document, data._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    data._assert_byte_identical(solo, out, worlds.block(text))  # pyright: ignore[reportPrivateUsage]
    assert len(worlds.receipt(out)["points"]) == 4


# --------------------------------------------------------------------------- #
# fits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        _needs("tp=2,ep=2"),
        _needs("ep=4"),
        _needs("tp=2,ep=4"),
        _needs("tp=8"),
        _needs("ep=8"),
    ],
)
def test_the_moe_das_fit_lands_within_the_band(
    moe_fit: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """The §7 agreements for real over the model group: the out-of-memory
    ``any``, the gradient mean, the experts' all-reduce in every forward and
    backward, the router's summed input gradient."""
    document, solo = moe_fit
    out = _run(document, train._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    assert worlds.receipt_problems(solo, out, text) == []
    measured = train._diffs(solo, out)  # pyright: ignore[reportPrivateUsage]
    _pinned(f"moe fit {text}", measured, band=train.BAND, floor=FLOORS["fit"])


@pytest.mark.parametrize("text", [_needs("pp=4")])
def test_the_moe_das_fit_is_byte_identical_over_four_stages(
    moe_fit: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """The featurizer on stage 1, the head on stage 3, the publisher on stage
    0 holding the trained parameters through the post-step sync (§7)."""
    document, solo = moe_fit
    out = _run(document, train._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    _exact(solo, out, text)


@pytest.mark.parametrize("text", [_needs("tp=4")])
def test_the_llama_das_fit_lands_within_the_band(
    llama_fit: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    document, solo = llama_fit
    out = _run(document, train._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    assert worlds.receipt_problems(solo, out, text) == []
    measured = train._diffs(solo, out)  # pyright: ignore[reportPrivateUsage]
    _pinned(f"llama fit {text}", measured, band=train.BAND, floor=FLOORS["fit"])


@pytest.mark.parametrize("text", [_needs("dp=2:rows,pp=2")])
def test_the_llama_das_fit_over_rows_lands_within_the_rows_band(
    llama_fit: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """Each step's minibatch split two rows to each two-stage pipeline
    replica, the joiner alone publishing (§8.3), within the rows smoke's
    band."""
    document, solo = llama_fit
    out = _run(document, train._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    assert worlds.receipt_problems(solo, out, text) == []
    rows._assert_within_band(solo, out)  # pyright: ignore[reportPrivateUsage]
    assert len(worlds.receipt(out)["points"]) == 1


@pytest.mark.parametrize("text", [_needs("dp=4:rows")])
def test_four_replicas_over_a_two_row_minibatch_are_refused_by_name(
    llama_fit: tuple[Path, Path],
    tmp_path: Path,
    never_load: None,
    text: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """§8.3's rule at world 4: every minibatch — an epoch's remainder
    included — holds at least ``dp`` rows, and the corpus fit meets a
    two-row minibatch (``rows.py``, once the row count is known); every rank
    refuses by name (their stderr is the parent's file descriptor), the
    parent names a failed rank, and nothing is published — the receipt the
    run opened stays the skeleton every fit-time refusal leaves at world 1
    too."""
    document, _ = llama_fit
    out = _out(tmp_path, text)
    assert main(train._argv(document, out, "--parallel", text, device=DEVICE)) == 1  # pyright: ignore[reportPrivateUsage]
    err = capfd.readouterr().err
    assert "--parallel.data" in err and text in err and "rank" in err, err[-2000:]
    assert not list(out.glob("*.safetensors")) and not (out / "ce.json").exists()


@pytest.mark.parametrize("text", [_needs("cp=2,tp=2")])
def test_the_moe_context_fit_lands_within_the_band(
    moe_context_fit: tuple[Path, Path], tmp_path: Path, never_load: None, text: str
) -> None:
    """The write at the weekday token, the loss at the last; the KV gather's
    backward and the DeltaNet handoffs with their gradient, under a tensor
    group (§7, §8.4)."""
    document, solo = moe_context_fit
    out = _run(document, train._argv, _out(tmp_path, text), text)  # pyright: ignore[reportPrivateUsage]
    assert worlds.receipt_problems(solo, out, text) == []
    measured = train._diffs(solo, out)  # pyright: ignore[reportPrivateUsage]
    _pinned(
        f"moe context fit {text}",
        measured,
        band=context_train.BAND,
        floor=FLOORS["fit"],
    )
