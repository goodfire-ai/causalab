"""Single-source authoring and canonical identity contracts."""

import copy

import pytest
from hypothesis import given, strategies as st

from causalab.measurement.spec import MeasurementSpecError, parse_measurement
from causalab.workflow.document import parse_workflow
from tests.measurement.test_spec import study

pytestmark = pytest.mark.unit


def single_study():
    raw = study()
    plan = raw["measurement"]
    plan["mode"] = "single"
    plan["source"] = plan.pop("arms")["before"]
    del plan["observations"]
    return raw


def test_single_defaults_are_timing_and_torch_without_observations():
    raw = single_study()
    parsed = parse_workflow(raw)
    plan = parsed.measurement
    assert plan["mode"] == "single"
    assert "arms" not in plan
    assert "observations" not in plan
    assert plan["profile"]["backends"] == {
        "torch": {"record_shapes": False, "with_stack": False}
    }
    assert parse_measurement(plan, parsed.steps) == plan


@pytest.mark.parametrize(
    "field,value",
    [
        ("arms", {"before": {"revision": "HEAD"}}),
        ("observations", {}),
        ("observations", None),
        ("evaluation", None),
        ("acceptance", []),
        ("bootstrap_draws", 100),
        ("mode", "unknown"),
    ],
)
def test_single_rejects_invalid_or_comparison_only_options(field, value):
    raw = single_study()
    raw["measurement"][field] = value
    with pytest.raises(MeasurementSpecError):
        parse_measurement(raw["measurement"], raw["steps"])


def test_single_source_is_required_and_comparison_cannot_author_one():
    raw = single_study()
    del raw["measurement"]["source"]
    with pytest.raises(MeasurementSpecError):
        parse_measurement(raw["measurement"], raw["steps"])
    raw = study()
    raw["measurement"]["source"] = {"revision": "HEAD"}
    with pytest.raises(MeasurementSpecError):
        parse_measurement(raw["measurement"], raw["steps"])


@given(
    st.lists(
        st.integers(min_value=0, max_value=10000), min_size=1, max_size=5, unique=True
    )
)
def test_single_normalization_is_idempotent(seeds):
    raw = single_study()
    raw["measurement"]["seeds"] = seeds
    parsed = parse_measurement(raw["measurement"], raw["steps"])
    assert parse_measurement(parsed, raw["steps"]) == parsed


def test_observation_policy_and_source_enter_workflow_identity():
    from causalab.measurement.study.scheduler import digest

    raw = single_study()
    first = parse_workflow(raw).measurement
    observed = copy.deepcopy(raw)
    observed["measurement"]["observations"] = study()["measurement"]["observations"]
    second = parse_workflow(observed).measurement
    assert digest(first) != digest(second)
    changed = copy.deepcopy(raw)
    changed["measurement"]["source"]["revision"] = "candidate"
    assert digest(first) != digest(parse_workflow(changed).measurement)


def test_typed_view_normalizes_targets_without_weakening_pin_policy():
    from causalab.measurement.plan import (
        ComparisonMeasurementPlan,
        SingleMeasurementPlan,
        execution_plan,
        plan_view,
    )

    raw = single_study()
    single = parse_measurement(raw["measurement"], raw["steps"])
    view = plan_view(single)
    assert isinstance(view, SingleMeasurementPlan)
    assert view.observations.status == "not_requested"
    internal = execution_plan(single)
    assert set(internal["arms"]) == {"source"}
    assert internal["source_pin_anchor"] == "source"
    assert internal["observation_policy"] == "not_requested"
    assert internal["observations"] == {}
    raw = study()
    comparison = parse_measurement(raw["measurement"], raw["steps"])
    assert isinstance(plan_view(comparison), ComparisonMeasurementPlan)
    assert execution_plan(comparison)["source_pin_anchor"] == "before"
    assert execution_plan(comparison)["observation_policy"] == "required"
