"""The real worker keeps deferred external inputs fixed across repeated loads."""

from pathlib import Path

import pytest

import causalab
from causalab.measurement.census import CensusError, collect_pins
from causalab.measurement.runtime.worker import Worker
from causalab.io.env import ResolutionEnv
from causalab.workflow.runner import OverlayArtifacts
from tests.measurement.runtime.test_pin_contract import (
    deferred_replay as deferred_replay,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arm", ["before", "after", "eager"])
def test_worker_reloads_and_evaluation_hold_concrete_deferred_file_pins(request, arm):
    full, _, env, external = request.getfixturevalue("deferred_replay")
    root = external.parent
    worker = Worker.__new__(Worker)
    worker.path = root / "workflow.json"
    worker.raw = {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "fit": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.select"},
                "inputs": {},
                "outputs": {"weights": "weights.safetensors"},
            },
            "replay": {
                "type": "intervention_protocol",
                "document": "methods/locate.json",
            },
        },
        "pins": full,
    }
    worker.env = env
    worker.fitting = set()
    worker.pin_contract = None
    worker.config = {
        "arm": arm,
        "package_root": str(Path(causalab.__file__).resolve().parent.parent),
    }
    first = collect_pins(worker.load(7), env.datasets)
    expected = env.artifacts.file_digest("external.safetensors")
    assert first["files"]["external.safetensors"] == "0" * 64
    assert worker.raw["pins"] == full
    assert worker.pin_contract.pins["files"]["external.safetensors"] == expected
    assert collect_pins(worker.load(11), env.datasets) == first
    assert worker.pin_contract.shared["files"] == {"external.safetensors": expected}

    fit_root = root / "run"
    overlay = ResolutionEnv(
        datasets=env.datasets,
        artifacts=OverlayArtifacts(
            fit_root, env.artifacts, frozenset(worker.raw["steps"])
        ),
        model_info=env.model_info,
    )
    selected = {**worker.raw, "steps": {"replay": worker.raw["steps"]["replay"]}}
    replay = collect_pins(
        worker.load_subset(selected, overlay, fit_root=fit_root), overlay.datasets
    )
    assert replay["files"]["external.safetensors"] == expected
    assert "fit/weights.safetensors" in replay["files"]

    external.write_bytes(external.read_bytes() + b"changed")
    with pytest.raises(CensusError, match="pins.files.external"):
        worker.load(13)
    with pytest.raises(CensusError, match="pins.files.external"):
        worker.load_subset(selected, overlay, fit_root=fit_root)
