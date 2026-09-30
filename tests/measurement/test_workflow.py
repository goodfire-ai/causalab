"""Real offline tiny-model workflow, collector, traces, and analysis via the CLI."""

import json

import pytest

from examples.measurements.smoke import run

pytestmark = pytest.mark.smoke


def test_operation_and_actual_workflow_reports(tmp_path):
    root = run(tmp_path / "experiment")
    manifest = json.loads((root / "workflow.json").read_text())
    assert all(step["status"] == "completed" for step in manifest["steps"].values())
    for name in ("subspace_apply", "workflow"):
        report = json.loads((root / name / "summary.json").read_text())
        assert len(report["timing"]["per_seed"]) == 2
        assert report["observations"]
        assert (root / name / "report.html").is_file()
        for trace in report["traces"].values():
            assert trace["status"] == "completed"
            assert trace["observation_check"]["status"] == "compared"
    # Every repeated workflow has its own published outputs and actual engine receipt.
    engine_record = (
        tmp_path
        / "experiment/workflow_before/seed_0_repeat_0_timing/observed/observe/_step.json"
    )
    assert engine_record.is_file()
