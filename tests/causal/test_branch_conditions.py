"""Private branch merges preserve predicates without undoing interventions."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism

pytestmark = pytest.mark.unit


def test_identical_private_assignments_keep_condition_reads_and_atomic_failure():
    @mechanism
    def equations(x: Dom([5, 6])):
        gate = V(1, domain=Dom([0, 1]))
        if 1 // gate:
            tmp = x + 1
        else:
            tmp = x + 1
        result = V(tmp)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"gate", "x"}
    assert model.parents["raw_input"] == ["x"]
    trace = model.new_trace({"x": 5})
    with pytest.raises(ZeroDivisionError):
        trace["gate"] = 0
    assert trace["gate"] == 1
    assert trace["result"] == 6


def test_reassigning_an_existing_identical_private_value_still_reads_condition():
    @mechanism
    def equations(x: Dom([5]), gate: Dom([0, 1])):
        tmp = x + 1
        if 1 // gate:
            tmp = x + 1
        result = V(tmp)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"gate", "x"}
    with pytest.raises(ZeroDivisionError):
        model.new_trace({"x": 5, "gate": 0})


def test_identical_exposed_equations_can_still_be_overridden_as_a_whole():
    @mechanism
    def equations(gate: Dom([0, 1])):
        if 1 // gate:
            result = V(1, domain=Dom([1, 2]))
        else:
            result = V(1, domain=Dom([1, 2]))
        raw_input = V(str(gate))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["raw_output"] == ["result"]
    trace = model.new_trace({"gate": 1})
    trace.intervene_many({"result": 2, "gate": 0})
    assert trace["raw_output"] == "2"


def test_nested_identical_private_branches_preserve_each_condition():
    @mechanism
    def equations(outer: Dom(bool), gate: Dom([0, 1])):
        if outer:
            if 1 // gate:
                tmp = 1
            else:
                tmp = 1
        else:
            tmp = 1
        result = V(tmp)
        raw_input = V(str(outer))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.new_trace({"outer": False, "gate": 0})
    assert trace["result"] == 1
    with pytest.raises(ZeroDivisionError):
        trace["outer"] = True
    assert trace["outer"] is False


def test_passthrough_submodel_alias_keeps_its_call_site_condition():
    from causalab.causal import submodel

    @submodel
    def passthrough(x):
        return x

    @mechanism
    def equations(x: Dom([5]), gate: Dom([0, 1])):
        if 1 // gate:
            tmp = passthrough(x)
        else:
            tmp = passthrough(x)
        result = V(tmp)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"gate", "x"}
    trace = model.new_trace({"x": 5, "gate": 1})
    with pytest.raises(ZeroDivisionError):
        trace["gate"] = 0
    assert trace["gate"] == 1


def test_new_submodel_equation_owns_its_call_site_condition():
    from causalab.causal import submodel

    @submodel
    def child(x):
        result = V(x)
        return result

    @mechanism
    def equations(x: Dom([5, 6]), gate: Dom([0, 1])):
        if 1 // gate:
            tmp = child(x)
        else:
            tmp = child(x)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(tmp))
        return raw_output

    model = CausalModel(equations)
    assert model.parents["raw_output"] == ["tmp.result"]
    trace = model.new_trace({"x": 5, "gate": 1})
    trace.intervene_many({"tmp.result": 6, "gate": 0})
    assert trace["raw_output"] == "6"


def test_forward_family_passthrough_keeps_its_call_site_condition():
    from causalab.causal import family, submodel

    @submodel
    def passthrough(x):
        return x

    @mechanism
    def equations(x: Dom([5]), gate: Dom([0, 1])):
        steps = family(size=2, domain=Dom([5]))
        if 1 // gate:
            tmp = passthrough(steps[1])
        else:
            tmp = passthrough(steps[1])
        steps[0] = tmp
        steps[1] = x
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(steps[0]))
        return raw_output

    model = CausalModel(equations)
    assert set(model.parents["steps[0]"]) == {"gate", "steps[1]"}
    trace = model.new_trace({"x": 5, "gate": 1})
    with pytest.raises(ZeroDivisionError):
        trace["gate"] = 0
    assert trace["gate"] == 1
