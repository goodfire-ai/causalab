"""Capture artifacts remain useful without optional numerical evidence."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from causalab.measurement.capture import controller, worker as child
from causalab.measurement.collection import write_record
from causalab.measurement.study.controller import ProcessSession
from tests.measurement.test_single_runtime import make_worker, forbidden

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["warm", "cold"])
def test_capture_worker_does_not_extract_observations(tmp_path, monkeypatch, mode):
    from causalab.measurement.runtime import worker
    from causalab.measurement.capture import ranges

    fake = make_worker(tmp_path)
    fake.engine, fake.bundles = None, {}
    monkeypatch.setattr(worker, "Worker", lambda config: fake)

    @contextmanager
    def phases(*args, **kwargs):
        yield {}

    monkeypatch.setattr(ranges, "phase_ranges", phases)
    config = {
        "device": "cpu",
        "plan": fake.plan,
        "capture": {
            "backend": "torch",
            "mode": mode,
            "case": "workflow",
            "seed": 0,
            "directory": str(tmp_path),
            "options": {},
        },
    }
    result = child.capture(config)
    assert result["status"] == "completed"
    assert result["observation_status"] == "not_requested"
    assert "observations" not in result
    assert "observation_specs" not in result
    assert len(result["output_files"]) == 1
    assert (tmp_path / "trace.json").is_file()
    assert not list(tmp_path.rglob("*.safetensors"))


@pytest.mark.parametrize("downgrade", [False, True])
def test_controller_uses_authored_observation_policy(tmp_path, monkeypatch, downgrade):
    identity = {"models": {}, "execution_probe": {}}
    policy = "required" if downgrade else "not_requested"
    config = {
        "device": "cpu",
        "controller_root": str(tmp_path),
        "package_root": str(tmp_path),
        "plan": {
            "mode": "single",
            "observation_policy": policy,
            "cases": {"workflow": {"cold_process": False}},
            "profile": {"backends": {"torch": {}}},
        },
    }
    receipt = tmp_path / "measurement.json"
    write_record(receipt, {"samples": [{"seconds": 1}]})

    def execute(command, **kwargs):
        capture = json.loads(Path(command[-1]).read_text())["capture"]
        target = Path(capture["directory"])
        (target / "trace.json").write_text('{"traceEvents": []}')
        write_record(
            Path(capture["result"]),
            {
                "status": "completed",
                "identity": identity,
                "observation_status": "not_requested",
                "output_files": {},
                "coverage": {},
                "instrumentation": {},
            },
        )

    monkeypatch.setattr(controller, "execute", execute)
    controller.run_captures(
        sys.executable, config, identity, "workflow", 0, tmp_path, receipt
    )
    record = json.loads(receipt.read_text())
    capture = record["captures"][0]
    assert record["samples"] == [{"seconds": 1}]
    assert capture["status"] == ("failed" if downgrade else "completed")
    if not downgrade:
        assert capture["observation_status"] == "not_requested"
        assert "observations" not in capture


def test_cold_timing_only_skips_diagnostic_process_and_observation_exports(
    tmp_path, monkeypatch
):
    from causalab.measurement.study import controller as study
    from causalab.measurement.runtime import observations

    worker = make_worker(tmp_path)
    directory = tmp_path / "sample"
    receipt = worker.sample(
        {"case": "workflow", "seed": 0, "directory": str(directory)}
    )
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"output_dir": "outputs"}))
    session = object.__new__(ProcessSession)
    session.identity = {"models": {}}
    session.python, session.logs = sys.executable, tmp_path
    session.config = {
        "plan": worker.plan,
        "workflow": str(workflow),
        "controller_root": str(tmp_path),
        "package_root": str(tmp_path),
    }
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        config = json.loads(Path(command[-1]).read_text())
        output = Path(config["cold"]["directory"]) / "outputs"
        output.mkdir(parents=True)
        (output / "saved.txt").write_text("cold output")
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "identity": session.identity,
                    "status": "completed",
                    "cache_policy": {"kind": "cold"},
                }
            )
        )

    monkeypatch.setattr(study.subprocess, "run", run)
    monkeypatch.setattr(observations, "observations", forbidden)
    session._cold("workflow", 0, directory, receipt)
    record = json.loads(receipt.read_text())
    sample = record["samples"][0]
    assert len(calls) == 1
    assert sample["peak_memory"] is None
    assert sample["workflow_outputs"] == "cold/outputs"
    assert sample["observation_status"] == "not_requested"
    assert any("cold/outputs/saved.txt" == name for name in sample["output_files"])
    assert not list(directory.rglob("*.safetensors"))
