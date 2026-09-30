"""Execution normalization preserves observation intent across repeated calls."""

from copy import deepcopy

from hypothesis import given, strategies as st
import pytest

from causalab.measurement.plan import execution_plan
from causalab.measurement.spec import parse_measurement
from tests.measurement.study.test_definitions import _study as workflow_study
from tests.measurement.test_single_spec import single_study
from tests.measurement.test_spec import study

pytestmark = pytest.mark.unit


@given(
    mode=st.sampled_from(["single", "single_observed", "code", "workflow"]),
    passes=st.integers(min_value=2, max_value=5),
    seeds=st.lists(st.integers(0, 100), min_size=1, max_size=4, unique=True),
)
def test_execution_normalization_is_idempotent_without_mutating_authorship(
    mode, passes, seeds
):
    raw = workflow_study() if mode == "workflow" else study()
    if mode.startswith("single"):
        raw = single_study()
        if mode == "single_observed":
            raw["measurement"]["observations"] = study()["measurement"]["observations"]
    raw["measurement"]["seeds"] = seeds
    authored = parse_measurement(raw["measurement"], raw["steps"])
    original = deepcopy(authored)
    expected = execution_plan(authored)
    current = deepcopy(expected)
    for _ in range(passes):
        current = execution_plan(current)
        assert current == expected
    assert authored == original
    assert current["observation_policy"] == (
        "not_requested" if mode == "single" else "required"
    )


@pytest.mark.parametrize("observations", [{}, None])
def test_authored_empty_observations_are_not_reinterpreted_as_timing_only(observations):
    raw = single_study()
    authored = parse_measurement(raw["measurement"], raw["steps"])
    authored["observations"] = observations
    with pytest.raises(ValueError, match="nonempty"):
        execution_plan(authored)


@pytest.mark.parametrize("mode", ["single", "code"])
def test_execution_policy_cannot_discard_requested_observations(mode):
    raw = study()
    if mode == "single":
        observed = raw["measurement"]["observations"]
        raw = single_study()
        raw["measurement"]["observations"] = observed
    plan = execution_plan(parse_measurement(raw["measurement"], raw["steps"]))
    plan["observation_policy"] = "not_requested"
    with pytest.raises(ValueError, match="observation"):
        execution_plan(plan)
