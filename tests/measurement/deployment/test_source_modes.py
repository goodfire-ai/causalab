"""Deployment retains pin anchors across single, code, and workflow studies."""

import json

from hypothesis import given, settings, strategies as st
import pytest

from causalab.measurement.deployment import remote
from causalab.measurement.census import CensusError
from causalab.measurement.runtime.pins import PinContract
from tests.measurement.deployment.test_single_deployment import launch_args
from tests.measurement.deployment.test_workflow_comparison_remote import _workflow_study
from tests.measurement.deployment.test_source_pins import _revisions
from tests.measurement.runtime.test_pin_contract import MODULE, PACKAGE_ROOT, census

pytestmark = pytest.mark.unit


@settings(max_examples=12, deadline=None)
@given(
    comparison=st.sampled_from(["code", "workflow"]),
    anchor=st.sampled_from(["before", "source"]),
    arm=st.sampled_from(["before", "after", "source"]),
)
def test_pin_policy_combines_workflow_strictness_with_source_anchor(
    comparison, anchor, arm
):
    actual = census()
    authored = {**actual, "code": {MODULE: "0" * 64}}
    options = {
        "arm": arm,
        "source_pin_anchor": anchor,
        "package_root": PACKAGE_ROOT,
        "comparison": comparison,
    }
    if comparison == "workflow" or arm == anchor:
        with pytest.raises(CensusError, match="pins.code"):
            PinContract.resolve(authored, actual, **options)
    else:
        assert PinContract.resolve(authored, actual, **options).pins == actual


def test_workflow_launch_selects_baseline_interpreter(tmp_path, monkeypatch):
    document = _workflow_study(tmp_path)
    repository = tmp_path / "repository"
    revision, _ = _revisions(repository)
    raw = json.loads(document.read_text())
    for arm in raw["measurement"]["arms"].values():
        arm["revision"] = revision
    raw["pins"]["code"] = {}
    document.write_text(json.dumps(raw))
    candidate = tmp_path / "candidate/workflow.json"
    raw = json.loads(candidate.read_text())
    raw["pins"]["code"] = {}
    candidate.write_text(json.dumps(raw))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {
                        "repository": str(repository),
                        "python": f"/remote/{arm}/python",
                    }
                    for arm in ("before", "after")
                },
                "device": "cpu",
                "data_root": "/data",
                "artifacts_root": "/artifacts",
            }
        )
    )
    jobs = []

    class Transport:
        def __init__(self, host):
            pass

        def call(self, command, **kwargs):
            if "stdin" in kwargs:
                return b""
            if "data" in kwargs:
                jobs.append(json.loads(kwargs["data"]))
                return b""
            return json.dumps("/remote/home").encode()

    monkeypatch.setattr(remote, "SSH", Transport)
    monkeypatch.setattr(remote, "query", lambda *args: {"status": "submitted"})
    result = remote.launch(launch_args(document, bindings, tmp_path / "job.json"))
    assert result["status"] == "submitted"
    assert jobs[0]["command"][0] == "/remote/before/python"
    assert set(jobs[0]["sources"]) == {"before", "after"}
