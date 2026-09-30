"""Requirements evaluate independently on the available parts of a trace."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism, require

pytestmark = pytest.mark.unit


def test_unrelated_missing_roots_do_not_disable_ready_requirements():
    @mechanism
    def equations(x: Dom([-1, 0, 1]), unrelated: Dom([0, 1])):
        checked = V(x, lazy=True)
        require(unrelated == 1, error="unrelated must be one")
        require(checked >= 0, error="negative x")
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(x))
        return raw_output

    model = CausalModel(equations)
    with pytest.raises(ValueError, match="negative x"):
        model.new_trace({"x": -1})
    trace = model.new_trace({"x": 0})
    with pytest.raises(ValueError, match="negative x"):
        trace["x"] = -1
    assert trace["x"] == 0
    assert trace["checked"] == 0
    with pytest.raises(ValueError, match="unrelated must be one"):
        trace["unrelated"] = 0
    assert "unrelated" not in trace


def test_requirement_short_circuit_decides_without_unread_missing_operands():
    @mechanism
    def equations(x: Dom([-1, 0, 1]), other: Dom([0, 1])):
        require(x >= 0 and (x == 0 or other == 1), error="invalid pair")
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(x))
        return raw_output

    model = CausalModel(equations)
    trace = model.new_trace({"x": 0})
    assert trace["raw_output"] == "0"
    with pytest.raises(ValueError, match="invalid pair"):
        trace["x"] = -1
    trace["x"] = 1  # The constraint is now waiting on the missing other input.
    with pytest.raises(ValueError, match="invalid pair"):
        trace["other"] = 0
    assert "other" not in trace
    trace["other"] = 1
    assert trace["raw_output"] == "1"


def test_real_key_errors_in_constraints_propagate_and_roll_back():
    def lookup(x):
        return {0: True}[x]

    @mechanism
    def equations(x: Dom([0, 1]), unrelated: Dom([0, 1])):
        require(lookup(x), error="lookup rejected x")
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(x))
        return raw_output

    model = CausalModel(equations)
    trace = model.new_trace({"x": 0})
    with pytest.raises(KeyError) as exc:
        trace["x"] = 1
    assert exc.value.args == (1,)
    assert trace["x"] == 0
    assert trace["raw_output"] == "0"


def test_missing_input_reads_still_report_key_errors():
    @mechanism
    def equations(x: Dom([0, 1])):
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(x))
        return raw_output

    trace = CausalModel(equations).new_trace()
    with pytest.raises(KeyError, match="has not been supplied"):
        trace["x"]
