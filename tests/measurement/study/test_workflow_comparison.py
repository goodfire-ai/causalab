"""Real same-commit workflows produce different configured outputs in isolated workers."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from causalab.measurement.study import controller
from tests.measurement.study.test_controller import _local_inputs

# `measurement_study`: this module builds and installs source arms and runs the
# study controller with cold worker processes — minutes each.
# `-m "not measurement_study"` deselects it for a quick run (docs/TESTS.md lists
# the markers).
pytestmark = [pytest.mark.smoke, pytest.mark.measurement_study]


def _documents(tmp_path, checkpoint, data):
    specification = {
        "header": {"protocol_version": "4"},
        "model": {"key": str(checkpoint), "revision": "local", "dtype": "fp32"},
        "data": {"base": {"dataset": "prompts", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "head", "pos": -1}},
            "save": [
                {
                    "read": "logits",
                    "model": "original",
                    "file_path": "logits.safetensors",
                }
            ],
        },
    }
    (tmp_path / "baseline.json").write_text(json.dumps(specification))
    candidate = deepcopy(specification)
    candidate["method"]["reads"]["logits"]["pos"] = -2
    (tmp_path / "candidate.json").write_text(json.dumps(candidate))
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    study = {
        "version": "1",
        "output_dir": "research",
        "steps": {
            "inference": {"type": "intervention_protocol", "document": "baseline.json"}
        },
        "measurement": {
            "version": 1,
            "comparison": "workflow",
            "arms": {
                "before": {"revision": revision},
                "after": {"revision": revision, "workflow": "after.json"},
                "eager": {"revision": revision},
            },
            "seeds": [0],
            "repeats": 1,
            "warmups": 0,
            "bootstrap_draws": 100,
            "cases": {
                "operation": {"kind": "operation", "step": "inference"},
                "resident": {"kind": "workflow", "cold_process": False},
                "cold": {"kind": "workflow", "cold_process": True},
            },
            "observations": {
                "logits": {
                    "step": "inference",
                    "file": "logits.safetensors",
                    "kind": "tensor",
                }
            },
            "profile": {"cases": ["operation", "cold"]},
        },
    }
    after = {
        key: deepcopy(value) for key, value in study.items() if key != "measurement"
    }
    after["steps"]["inference"]["document"] = "candidate.json"
    (tmp_path / "after.json").write_text(json.dumps(after))
    document = tmp_path / "study.json"
    document.write_text(json.dumps(study))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repository), "python": sys.executable}
                    for arm in study["measurement"]["arms"]
                },
                "device": "cpu",
                "data_root": str(data),
                "artifacts_root": str(tmp_path),
            }
        )
    )
    return document, bindings, revision


def test_same_commit_distinct_workflows_cold_profile_control_and_resume(
    tmp_path, monkeypatch
):
    checkpoint, data = _local_inputs(tmp_path, monkeypatch)
    document, bindings, revision = _documents(tmp_path, checkpoint, data)
    output = tmp_path / "run"
    controller.run(document, bindings, output)
    for case in ("operation", "resident", "cold"):
        report = json.loads((output / "reports" / f"{case}.json").read_text())
        assert report["observations"]
        assert any(
            row["max_abs"] > 0
            for value in report["observations"].values()
            for row in value["paired_drift"]
        )
        control = json.loads(
            (output / "reports" / f"{case}.eager_before.json").read_text()
        )
        assert all(
            row["max_abs"] == 0
            for value in control["observations"].values()
            for row in value["paired_drift"]
        )
        before = json.loads(
            (output / "collections" / f"{case}.before.json").read_text()
        )
        after = json.loads((output / "collections" / f"{case}.after.json").read_text())
        assert before["input_identity"] == after["input_identity"]
        workers = [record["context"]["worker"] for record in (before, after)]
        assert workers[0]["benchmark_identity"] != workers[1]["benchmark_identity"]
        assert {worker["source_commit"] for worker in workers} == {revision}
        rendered = (output / "reports" / f"{case}.html").read_text()
        assert "Configuration differences" in rendered
        assert "configured outputs" in rendered
        for arm in ("before", "after"):
            captures = report["captures"][arm]
            assert len(captures) == {"operation": 1, "resident": 0, "cold": 2}[case]
            for capture in captures:
                assert capture["status"] == "completed", capture
                assert capture["observation_check"]["exactly_equal"]
    ledger = (output / "collections/study.json").read_bytes()
    controller.run(document, bindings, output, resume=True)
    assert (output / "collections/study.json").read_bytes() == ledger
    candidate = tmp_path / "candidate.json"
    raw = json.loads(candidate.read_text())
    raw["method"]["reads"]["logits"]["pos"] = -1
    candidate.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="authored study or deployment changed"):
        controller.run(document, bindings, output, resume=True)


def test_differing_commits_refused_before_build(tmp_path, monkeypatch):
    # Own both revisions so shallow checkouts exercise the same pre-build guard.
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    for message in ("baseline", "candidate"):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Measurement test",
                "-c",
                "user.email=measurement-test@example.com",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-qm",
                message,
            ],
            cwd=repository,
            check=True,
        )
    document, bindings, _ = _documents(
        tmp_path, tmp_path / "checkpoint", tmp_path / "data"
    )
    raw = json.loads(document.read_text())
    for arm in raw["measurement"]["arms"].values():
        arm["revision"] = "HEAD"
    raw["measurement"]["arms"]["after"]["revision"] = "HEAD~1"
    document.write_text(json.dumps(raw))
    deployment = json.loads(bindings.read_text())
    for arm in deployment["arms"].values():
        arm["repository"] = str(repository)
    bindings.write_text(json.dumps(deployment))

    def no_build(*args, **kwargs):
        raise AssertionError("revision mismatch must be rejected before build")

    monkeypatch.setattr(controller, "build_arm", no_build)
    with pytest.raises(ValueError, match="same resolved code commit"):
        controller.run(document, bindings, tmp_path / "run")
