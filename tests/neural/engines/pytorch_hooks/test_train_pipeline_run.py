"""A fit across pipeline stages on real processes over ``gloo``
(``docs/model_parallelism.md`` §6.5, §7, §8.3, §10.6): the corpus DAS fit
(``tests/protocols/04_das_im.json``) retargeted to the tiny Llama, through
the real CLI at ``--parallel pp=2 --device cpu``, against the world-1 fit of
the same document.

The assertion is **byte identity** (§8.3: pipeline parallelism is placement
— the forward moves no reduction, and the gradient crosses the stage
boundary as one tensor): the fitted bundle, every table and the receipt
minus its ``execution.parallel`` block — ``fit_rows_resolved`` and
``fit_rows_shrinks`` included, the ``RowBudget`` agreeing over the stages
under an authored ``--fit-rows`` — equal to the world-1 run's bytes. Twice:
the write on stage 0 with the head on stage 1, where the gradient crosses
the boundary down to the featurizer; and the write on stage 1, where the
publisher (stage 0, rank 0 of the data replica) holds the trained
parameters through the post-step sync alone. And once more as a
**cohort** (spec §4 "Cohorts"): the layer swept over both blocks, two
points fitted together as one forward per step with a member on each
stage — the six-step workflow's DAS step at ``pp=2`` — each member's fire
count summed over the pipeline before its check and each member synced
from its own owner (``cohort.py``, §6.5, §8.3), the fitted bundle's two
entries and every table byte-identical to the world-1 cohort. The parent
of the spawn never loads a model; the two children hold the stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks import engine as hooks_engine

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

# the §7 gradient agreement check on at the measured band (conftest.py):
# the model group is one here, so the check is the identity — set all
# the same, so every training smoke runs under one rule
pytestmark = [pytest.mark.smoke, pytest.mark.usefixtures("checked_gradients_gloo")]

REPO = Path(__file__).resolve().parents[4]
DAS = REPO / "tests" / "protocols" / "04_das_im.json"
DATA = REPO / "tests" / "protocol" / "fixtures" / "data"

RECEIPT = "protocol.json"
EVENTS = "events.jsonl"
#: the corpus document's ``k: 8`` on the tiny Llama's residual of 16
K = 8


#: the cohort: the layer swept over the tiny Llama's two blocks, one point
#: a block, fitted together (spec §4 "Cohorts") with a member on each stage
COHORT = "cohort"
Fit = int | str


def _document(tmp: Path, fit: Fit) -> Path:
    """The corpus DAS fit retargeted to the tiny Llama, fp32: the write at
    block ``fit``, or — `COHORT` — swept over both blocks."""
    doc = json.loads(DAS.read_text())
    doc["model"] = {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"}
    layers: Any = {"sweep": [0, 1]} if fit == COHORT else [fit]
    doc["method"]["sites"]["target"]["layers"] = layers
    doc["method"]["featurizers"]["rot"]["k"] = K
    target = tmp / f"das_{fit}.json"
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
        # an authored row budget, as `test_train_parallel_run.py`: the measured
        # one depends on the device's free memory at the probe, and the
        # receipts are compared byte for byte
        "--fit-rows",
        "16",
        *extra,
    ]


def _receipt(out: Path) -> dict[str, Any]:
    return json.loads((out / RECEIPT).read_text())


def _assert_byte_identical(solo: Path, staged: Path) -> None:
    """Every file but the event stream (timestamps) byte-identical, and the
    receipt equal minus ``execution.parallel``."""
    names = sorted(p.name for p in solo.iterdir())
    assert names == sorted(p.name for p in staged.iterdir())
    for name in names:
        if name in (RECEIPT, EVENTS):
            continue
        assert (staged / name).read_bytes() == (solo / name).read_bytes(), name
    a, b = _receipt(solo), _receipt(staged)
    assert a["execution"]["parallel"]["launcher"] == "solo"
    assert b["execution"]["parallel"] == {
        "data": 1,
        "data_mode": "points",
        "pipeline": 2,
        "context": 1,
        "tensor": 1,
        "expert": 1,
        "world": 2,
        "launcher": "spawned",
    }
    for key in ("fit_rows_resolved", "fit_rows_shrinks"):
        assert a["execution"].get(key) == b["execution"].get(key), key
    del a["execution"]["parallel"], b["execution"]["parallel"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


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
def solo_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[Fit, tuple[Path, Path]]:
    """The world-1 fits — the write at block 0, at block 1, and the cohort
    over both — run in this process (which is then never the spawn parent's
    loader)."""
    runs: dict[Fit, tuple[Path, Path]] = {}
    for fit in (0, 1, COHORT):
        tmp = tmp_path_factory.mktemp(f"train-pp-{fit}")
        document = _document(tmp, fit)
        out = tmp / "solo"
        assert main(_argv(document, out)) == 0
        runs[fit] = (document, out)
    return runs


@pytest.mark.parametrize("fit", [0, 1, COHORT])
def test_pp2_fit_is_byte_identical_to_world_one(
    solo_runs: dict[Fit, tuple[Path, Path]],
    tmp_path: Path,
    fit: Fit,
    never_load: None,
) -> None:
    document, solo = solo_runs[fit]
    staged = tmp_path / "pp2"
    assert main(_argv(document, staged, "--parallel", "pp=2")) == 0
    _assert_byte_identical(solo, staged)
    receipt = _receipt(solo)
    points = 2 if fit == COHORT else 1
    assert len(receipt["fires"]) == points
    assert all(
        "patch" in fires
        for point in receipt["fires"].values()
        for fires in point.values()
    )
