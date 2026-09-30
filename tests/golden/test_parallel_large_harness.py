"""The oracle path of the parallel golden end to end on the tiny Llama over
``gloo`` (``docs/model_parallelism.md`` §10.6 "The large model"): the large
model's inference document (``tests/golden/_parallel/large.py``) with the
tiny dense fixture in fp32 on the CPU in place of the 70B, ``pp=2`` its
oracle (the fixture has two layers and an untied head), ``tp=2`` banded
(four heads over two ranks) — run and **captured** exactly as the GPU
script captures the 70B: no world-1 run is made, the oracle's run is the
reference, the receipts agree under the oracle rule, every rank's load
report passes the several-axes loader rule, the record block names the
oracle and replays against itself, and a byte that differs between the
oracle and an exact geometry is refused by name.

This is the harness's own smoke, not a numerics tier (the fixture's ``tp``
band is the smoke tier's business): what it pins is that a document with
no world-1 run goes through ``run_all``, ``capture`` and the record on the
production spawn path. The recorder is off here (no CUDA: an inference
rank on the CPU writes no memory, and the memory rule needs a device), so
the memory half of the rule stays with its CPU guard on hand-built peaks
(``test_parallel_large_rules.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.golden import _parallel as par
from tests.golden._parallel import inference, large
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke

#: The tiny Llama in fp32 on the CPU at its first layer: the sweep is its
#: two layers (``inference.SWEEP_STRIDES``), one on each stage of ``pp=2``.
TINY = par.Realization(TINY_LLAMA, "fp32", "cpu", layer=0)

DOCUMENT = dataclasses.replace(
    large.INFERENCE,
    exact=("pp=2",),
    banded=("tp=2",),
    oracle="pp=2",
    realization=TINY,
    recorded=False,
    estimate=None,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, list[str]]:
    root = tmp_path_factory.mktemp("large-harness")
    block, refusals = par.capture(root, DOCUMENT, TINY, with_context=False)
    return root, block, refusals


def test_the_capture_is_certifiable_with_no_world_one_run(captured) -> None:
    root, block, refusals = captured
    assert refusals == []
    base = root / DOCUMENT.name
    assert not (base / "solo").exists()
    assert (base / "pp2" / par.RECEIPT).exists() and (
        base / "tp2" / par.RECEIPT
    ).exists()
    assert par.receipt(base / "pp2")["execution"]["parallel"] == par.parallel_block(
        "pp=2"
    )
    assert len(par.receipt(base / "pp2")["points"]) == 2
    assert inference.sweep(TINY) == (0, 1)


def test_the_block_names_the_oracle_and_the_dense_classes_and_replays(captured) -> None:
    _, block, _ = captured
    assert block["oracle"] == "pp=2"
    assert block["exact"] == {}  # the oracle alone is exact: compared to nothing
    assert set(block["geometries"]) == {"tp=2"}
    assert set(block["geometries"]["tp=2"]) == set(inference.DENSE_CLASSES)
    assert block["realization"] == {
        "model": TINY_LLAMA,
        "dtype": "fp32",
        "device": "cpu",
    }
    assert set(block["load"]) == {"pp=2", "tp=2"}
    assert "memory" not in block and "estimate" not in block
    record = par.make_record(None, par.A3B, {DOCUMENT.name: block})
    assert par.compare_records(record, record, DOCUMENT.name) == []


def test_every_rank_passes_the_several_axes_loader_rule(captured) -> None:
    root, _, _ = captured
    for geometry in DOCUMENT.geometries:
        reports = par.load_reports(root / DOCUMENT.name / par.out_name(geometry), 2)
        assert DOCUMENT.loader_problems(geometry, reports) == [], geometry
    stages = par.load_reports(root / DOCUMENT.name / "pp2", 2)
    whole = par.load_reports(root / DOCUMENT.name / "tp2", 2)[0]["bytes_on_disk"]
    assert set(stages[0]["bytes_on_disk"]) | set(stages[1]["bytes_on_disk"]) == set(
        whole
    )


def test_a_byte_that_differs_from_the_oracle_is_refused_by_name(
    captured, tmp_path: Path
) -> None:
    """A second exact geometry whose output differs from the oracle's by a
    byte: the capture names it. The oracle's kept run is reused and the
    'other' exact geometry is staged as a copy with one table altered."""
    import shutil

    root, _, _ = captured
    staged = tmp_path / "staged"
    (staged / DOCUMENT.name).mkdir(parents=True)
    for geometry in DOCUMENT.geometries:
        shutil.copytree(
            root / DOCUMENT.name / par.out_name(geometry),
            staged / DOCUMENT.name / par.out_name(geometry),
        )
    # ``dp=2`` is exact by design and its run is staged as a copy of the
    # oracle's with one metric altered — the harness never launches it
    shutil.copytree(root / DOCUMENT.name / "pp2", staged / DOCUMENT.name / "dp2")
    receipt = json.loads((staged / DOCUMENT.name / "dp2" / par.RECEIPT).read_text())
    receipt["execution"]["parallel"] = par.parallel_block("dp=2")
    (staged / DOCUMENT.name / "dp2" / par.RECEIPT).write_text(json.dumps(receipt))
    table = staged / DOCUMENT.name / "dp2" / "iia.json"
    rows = json.loads(table.read_text())
    (rows[0] if isinstance(rows, list) else rows)["iia"] = 0.123456
    table.write_text(json.dumps(rows))
    for rank in range(2):
        src = root / DOCUMENT.name / "pp2" / par.REPORTS
        dst = staged / DOCUMENT.name / "dp2" / par.REPORTS
        if not dst.exists():
            shutil.copytree(src, dst)
    widened = dataclasses.replace(DOCUMENT, exact=("pp=2", "dp=2"))
    _, refusals = par.capture(staged, widened, TINY, with_context=False)
    assert any("dp=2: not byte-identical" in r and "iia.json" in r for r in refusals), (
        refusals
    )
