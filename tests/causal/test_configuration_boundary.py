"""All equation configuration is captured at model construction."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism, submodel

pytestmark = pytest.mark.unit


def test_direct_and_helper_closures_share_the_construction_boundary():
    offset = 1

    def helper(x):
        return x + offset

    @mechanism
    def equations(x: Dom(range(2))):
        direct = V(x + offset)
        indirect = V(helper(x))
        result = V(direct == indirect)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    offset = 2
    first = CausalModel(equations)
    offset = 3
    second = CausalModel(equations)
    offset = 4
    for model, expected in [(first, 2), (second, 3)]:
        trace = model.new_trace({"x": 0})
        assert trace["direct"] == trace["indirect"] == expected
        assert trace["result"] is True
        assert model.definition._global_namespace is None
        assert not model.definition._closure_cells


_GLOBAL_OFFSET = 1


def _global_helper(x):
    return x + _GLOBAL_OFFSET


@mechanism
def _global_equations(x: Dom(range(2))):
    direct = V(x + _GLOBAL_OFFSET)
    indirect = V(_global_helper(x))
    result = V(direct == indirect)
    raw_input = V(str(x))  # noqa: F841
    raw_output = V(str(result))  # noqa: F841
    return result


def test_global_rebinding_before_construction_is_seen_everywhere(monkeypatch):
    monkeypatch.setitem(globals(), "_GLOBAL_OFFSET", 8)
    model = CausalModel(_global_equations)
    monkeypatch.setitem(globals(), "_GLOBAL_OFFSET", 9)
    trace = model.new_trace({"x": 0})
    assert trace["direct"] == trace["indirect"] == 8


def test_mutable_configuration_rebinding_and_mutation_are_frozen_together():
    configuration = {"offset": 1}

    def helper(x):
        return x + configuration["offset"]

    @mechanism
    def equations(x: Dom([0])):
        direct = V(x + configuration["offset"])
        indirect = V(helper(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(direct + indirect))
        return raw_output

    configuration = {"offset": 3}
    model = CausalModel(equations)
    configuration["offset"] = 10
    assert model.new_trace({"x": 0})["raw_output"] == "6"


def test_submodel_configuration_uses_the_same_construction_boundary():
    offset = 1

    def helper(x):
        return x + offset

    @submodel
    def child(x):
        direct = V(x + offset)
        indirect = V(helper(x))
        result = V(direct == indirect)
        return result

    @mechanism
    def equations(x: Dom([0])):
        comparison = child(x)
        direct = V(x + offset)  # noqa: F841
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(comparison))
        return raw_output

    offset = 7
    model = CausalModel(equations)
    trace = model.new_trace({"x": 0})
    assert (
        trace["comparison.direct"]
        == trace["comparison.indirect"]
        == trace["direct"]
        == 7
    )
    assert trace["raw_output"] == "True"


ANNOTATION_CHOICES = [99]


def test_string_annotation_preserves_factory_local_over_a_same_named_global():
    def factory():
        ANNOTATION_CHOICES = [1, 2]  # noqa: F841

        @mechanism
        def equations(x: "Dom(ANNOTATION_CHOICES)"):
            result = V(x)
            raw_input = V(str(x))  # noqa: F841
            raw_output = V(str(result))  # noqa: F841
            return result

        return equations

    model = CausalModel(factory())
    assert model.values["x"] == [1, 2]
    assert model.new_trace({"x": 2})["result"] == 2
