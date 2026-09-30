"""Unpacking preserves Python iteration, arity and equation-local evaluation."""

import math

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.compiler import _unpack_exact

pytestmark = pytest.mark.unit


def test_dictionary_unpack_uses_keys_and_preserves_reads():
    @mechanism
    def equations(x: Dom([2, 3])):
        a, b = {1: x, 0: x + 1}
        result = V(a * 10 + b)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["x"]
    trace = model.new_trace({"x": 2})
    assert trace["result"] == 10
    trace["x"] = 3
    assert trace["result"] == 10


def test_unpack_evaluates_rhs_once_per_equation_and_after_intervention(monkeypatch):
    original = math.modf
    calls = []

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(math, "modf", observed)

    @mechanism
    def equations(x: Dom([1.5, 2.5])):
        fraction, whole = math.modf(x)
        result = V(fraction + whole, domain=Dom(float))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    calls.clear()
    trace = model.new_trace({"x": 1.5})
    assert trace["result"] == 1.5
    assert calls == [1.5]
    trace["x"] = 2.5
    assert trace["result"] == 2.5
    assert calls == [1.5, 2.5]


@pytest.mark.parametrize("items", [(1,), (1, 2, 3)])
def test_unpack_checks_arity_even_when_one_target_is_unused(items):
    @mechanism
    def equations(x: Dom(tuple)):
        first, second = x
        result = V(first, domain=Dom(int))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    with pytest.raises(ValueError, match="values to unpack"):
        model.new_trace({"x": items})


def test_unused_nested_target_is_still_unpacked_and_validated():
    @mechanism
    def equations(x: Dom(tuple)):
        first, (_second, _third) = (1, x)
        result = V(first, domain=Dom(int))
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["x"]
    assert model.new_trace({"x": (2, 3)})["result"] == 1
    with pytest.raises(ValueError, match="not enough values"):
        model.new_trace({"x": (2,)})


def test_unpack_bindings_are_distinct_across_reassignment():
    @mechanism
    def equations(x: Dom([2, 3])):
        a, b = (x, x + 1)
        old = a
        a, b = (x + 2, x + 3)
        result = V((old, a, b))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": 2})["result"] == (2, 4, 5)


def test_unpack_keeps_conditional_execution_lazy():
    @mechanism
    def equations(enabled: Dom(bool), x: Dom(tuple)):
        if enabled:
            a, b = x
            chosen = a + b
        else:
            chosen = 0
        result = V(chosen, domain=Dom(int))
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.new_trace({"enabled": False, "x": (1,)})
    assert trace["result"] == 0
    with pytest.raises(ValueError, match="not enough values"):
        trace["enabled"] = True
    assert trace["enabled"] is False


def test_unpack_in_fixed_loops_and_comprehensions_uses_iteration():
    @mechanism
    def equations(x: Dom([2, 3])):
        for a, b in [{1: 9, 0: 8}]:
            offset = a * 10 + b
        result = V([x + a * 10 + b + offset for a, b in [{1: 9, 0: 8}]])
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 2})["result"] == [22]


def test_unpack_infinite_iterable_reads_only_one_extra_element():
    visited = []

    def values():
        while True:
            visited.append(len(visited))
            yield visited[-1]

    with pytest.raises(ValueError, match="too many values"):
        _unpack_exact(values(), (None, None))
    assert visited == [0, 1, 2]


def test_empty_targets_cannot_silently_drop_an_arity_check():
    from causalab.causal import DefinitionError

    @mechanism
    def equations(x: Dom(tuple)):
        () = x
        result = V(1)
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Unpacking needs a named target"):
        CausalModel(equations)
