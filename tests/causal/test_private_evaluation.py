"""Private initializers and comprehension steps retain Python evaluation order."""

import math

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, mechanism, submodel

pytestmark = pytest.mark.unit


def test_private_nan_initializer_is_shared_within_an_equation():
    @mechanism
    def equations():
        nan = float("nan")
        result = V(nan in [nan], domain=Dom(bool))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.variables) == {"result", "raw_input", "raw_output"}
    assert model.new_trace()["result"] is True


def test_private_initializer_runs_once_and_recomputes_after_intervention(monkeypatch):
    original = math.modf
    calls = []

    def observed(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(math, "modf", observed)

    @mechanism
    def equations(x: Dom([1.5, 2.5])):
        parts = math.modf(x)
        result = V(parts[0] + parts[1], domain=Dom(float))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["x"]
    calls.clear()
    trace = model.new_trace({"x": 1.5})
    assert trace["result"] == 1.5
    assert calls == [1.5]
    trace["x"] = 2.5
    assert trace["result"] == 2.5
    assert calls == [1.5, 2.5]


def test_private_assignment_versions_keep_distinct_initializers():
    @mechanism
    def equations():
        nan = float("nan")
        old = nan
        nan = float("nan")
        result = V(old in [old] and nan in [nan] and old not in [nan])
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace()["result"] is True


def test_private_cache_lifetime_is_one_equation():
    @mechanism
    def equations():
        nan = float("nan")
        left = V(nan, domain=Dom(float))
        right = V(nan, domain=Dom(float))
        result = V(left not in [right])
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace()["result"] is True


def test_private_bindings_keep_submodel_scopes_and_exposed_return_aliases():
    @submodel
    def check():
        nan = float("nan")
        result = V(nan in [nan])
        alias = result
        return alias

    @mechanism
    def equations():
        first = check()
        second = check()
        result = V(first and second)
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        alias = result
        return alias

    model = CausalModel(equations)
    assert model.return_variable == "result"
    assert set(model.variables) == {
        "first.result",
        "second.result",
        "result",
        "raw_input",
        "raw_output",
    }
    assert model.new_trace()["result"] is True


def test_private_local_iterator_requires_a_pure_helper():
    @mechanism
    def equations(x: Dom([1, 2])):
        values = iter([x, x + 1])
        result = V((next(values), next(values)))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(
        DefinitionError, match=r"Stateful iter\(\)/next\(\).*pure helper"
    ):
        CausalModel(equations)


@pytest.mark.parametrize("operation", [iter, next])
def test_private_iterator_aliases_require_a_pure_helper(operation):
    @mechanism
    def equations():
        alias = operation
        result = V(alias([True, False]), domain=Dom(list))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(
        DefinitionError, match=r"Stateful iter\(\)/next\(\).*pure helper"
    ):
        CausalModel(equations)


def test_private_iterator_algorithm_inside_a_helper_is_supported():
    def consume(x):
        values = iter([x, x + 1])
        return next(values), next(values)

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(consume(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == (1, 2)


def test_comprehension_items_share_runtime_private_nan_references():
    @mechanism
    def equations():
        nan = float("nan")
        values = (nan, nan)
        result = V([item in [nan] for item in values], domain=Dom(list))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace()["result"] == [True, True]


def test_comprehension_items_share_runtime_private_container_references():
    @mechanism
    def equations():
        values = [float("nan")]
        result = V([item in values for item in values], domain=Dom(list))
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace()["result"] == [True]


def test_comprehension_unpack_precedes_its_own_filter():
    @mechanism
    def equations(enabled: Dom(bool)):
        result = V([a for a, b in [(1, 2, 3)] if enabled])
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    for enabled in (False, True):
        with pytest.raises(ValueError, match="too many values to unpack"):
            model.new_trace({"enabled": enabled})


def test_comprehension_unpack_precedes_a_constant_false_filter():
    @mechanism
    def equations():
        result = V([1 for a, b in [(1, 2, 3)] if False])
        raw_input = V("input")  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    with pytest.raises(ValueError, match="too many values to unpack"):
        model.new_trace()


def test_unused_inner_unpack_is_guarded_by_the_outer_filter():
    @mechanism
    def equations(enabled: Dom(bool)):
        result = V([1 for i in range(1) if enabled for a, b in [(1, 2, 3)]])
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert "enabled" in model.parents["result"]
    trace = model.new_trace({"enabled": False})
    assert trace["result"] == []
    with pytest.raises(ValueError, match="too many values to unpack"):
        trace["enabled"] = True
    assert trace["enabled"] is False
    assert trace["result"] == []


def test_comprehension_outer_filter_runs_once_per_outer_iteration(monkeypatch):
    calls = []

    def observed(value):
        calls.append(value)
        return value > 0

    monkeypatch.setattr(math, "isfinite", observed)

    @mechanism
    def equations(x: Dom([-1.0, 1.0])):
        result = V(
            [j for i in range(2) if math.isfinite(x) for j in range(3)],
            domain=Dom(list),
        )
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    model = CausalModel(equations)
    calls.clear()
    trace = model.new_trace({"x": 1.0})
    assert trace["result"] == [0, 1, 2, 0, 1, 2]
    assert calls == [1.0, 1.0]
    trace["x"] = -1.0
    assert trace["result"] == []
    assert calls == [1.0, 1.0, -1.0, -1.0]


@pytest.mark.parametrize("bad_iterable", [False, True])
def test_fixed_inner_failures_remain_guarded_by_runtime_filters(bad_iterable):
    @mechanism
    def equations(enabled: Dom(bool)):
        if bad_iterable:
            result = V([j for i in range(1) if enabled for j in range(1 // 0)])
        else:
            result = V([j for i in range(1) if enabled for j in range(1) if 1 // 0])
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.new_trace({"enabled": False})
    assert trace["result"] == []
    with pytest.raises(ZeroDivisionError):
        trace["enabled"] = True
    assert trace["enabled"] is False


def test_comprehension_unpacks_before_a_consumed_filter_and_element():
    @mechanism
    def equations(enabled: Dom(bool)):
        result = V([a + b for a, b in [(1, 2), (3, 4)] if enabled and a < 3])
        raw_input = V(str(enabled))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"enabled": True})["result"] == [3]
    assert model.new_trace({"enabled": False})["result"] == []
