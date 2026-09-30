"""Scientific choices enter the workflow; deployment paths do not."""

import copy

import pytest

from causalab.workflow.document import WorkflowError, parse_workflow
from causalab.measurement.spec import parse_measurement

pytestmark = pytest.mark.unit


def study():
    return {
        "version": "1",
        "output_dir": "research",
        "steps": {"fit": {"type": "intervention_protocol", "document": "fit.json"}},
        "measurement": {
            "version": 1,
            "arms": {
                "before": {"revision": "main"},
                "after": {"revision": "candidate"},
            },
            "seeds": [10, 20],
            "repeats": 3,
            "cases": {
                "fit": {"kind": "operation", "step": "fit"},
                "end_to_end": {"kind": "workflow"},
            },
            "observations": {
                "basis": {"step": "fit", "file": "rot.safetensors", "kind": "subspace"}
            },
        },
    }


def test_normalized_round_trip_and_scopes():
    raw = study()
    parsed = parse_workflow(raw)
    plan = parsed.measurement
    assert plan["cases"]["fit"]["cold_process"] is False
    assert plan["cases"]["end_to_end"]["cold_process"] is True
    assert plan["profile"]["cases"] == ["fit", "end_to_end"]
    assert plan == parse_measurement(plan, parsed.steps)
    ordinary = copy.deepcopy(raw)
    del ordinary["measurement"]
    assert parse_workflow(ordinary).measurement is None


def test_profile_false_normalizes_to_no_cases():
    raw = study()
    raw["measurement"]["profile"] = False
    disabled = parse_workflow(raw)
    assert disabled.measurement["profile"]["cases"] == []
    assert (
        parse_measurement(disabled.measurement, disabled.steps) == disabled.measurement
    )
    raw["measurement"]["profile"] = {"cases": []}
    assert parse_workflow(raw).measurement == disabled.measurement


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(host="remote"),
        lambda p: p.update(seeds=[0, 0]),
        lambda p: p.update(repeats=True),
        lambda p: p["arms"]["after"].update(python="/usr/bin/python"),
        lambda p: p["arms"]["before"].update(revision="--help"),
        lambda p: p["arms"]["after"].update(execution={"engine": "missing"}),
        lambda p: p["cases"]["fit"].update(step="missing"),
        lambda p: p["cases"]["fit"].update(cold_process=True),
        lambda p: p["observations"]["basis"].update(file="../rot.safetensors"),
        lambda p: p["observations"]["basis"].update(kind="table"),
        lambda p: p.update(profile={"cases": ["missing"]}),
        lambda p: p.update(profile=True),
        lambda p: p.update(profile=0),
        lambda p: p.update(evaluation={"arm": "missing"}),
        lambda p: p.update(
            acceptance=[{"name": "x", "path": ["x"], "maximum": float("inf")}]
        ),
    ],
)
def test_invalid_authorship_refused(change):
    raw = study()
    change(raw["measurement"])
    with pytest.raises(
        WorkflowError, match="measurement|acceptance|observation|cold_process|engine"
    ):
        parse_workflow(raw)


def test_backend_options_round_trip_without_tool_probes(monkeypatch):
    from causalab.profiling import get_backend

    def no_probe(*args, **kwargs):
        raise AssertionError("authoring must not probe the compute host")

    monkeypatch.setattr("subprocess.run", no_probe)
    raw = study()
    raw["measurement"]["profile"] = {
        "backends": {"torch": {"with_stack": True}, "nsys": {}},
        "timeout_seconds": 120,
    }
    parsed = parse_workflow(raw)
    profile = parsed.measurement["profile"]
    assert profile["with_stack"] is True
    assert profile["backends"]["nsys"] == get_backend("nsys").normalize_options({})
    assert profile["timeout_seconds"] == 120
    assert parse_measurement(parsed.measurement, parsed.steps) == parsed.measurement


@pytest.mark.parametrize(
    "profile",
    [
        {"backends": {}},
        {"backends": {"missing": {}}},
        {"backends": {"torch": []}},
        {"backends": {"torch": {"with_stack": True}}, "with_stack": False},
        {"backends": {"nsys": {}}, "record_shapes": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
    ],
)
def test_bad_profiler_configuration_is_refused(profile):
    raw = study()
    raw["measurement"]["profile"] = profile
    with pytest.raises(WorkflowError, match="measurement.profile"):
        parse_workflow(raw)
