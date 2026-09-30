"""Domain representatives must preserve branch outcomes and intervention reads."""

import math

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, DomainError, V, mechanism

pytestmark = pytest.mark.unit


def test_nested_types_keep_both_branches_under_intervention():
    @mechanism
    def equations(x: Dom([True, 1]), y: Dom([10, 20])):
        packed = V((x,))
        result = V(y if packed[0] is True else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"packed", "y"}
    trace = model.new_trace({"x": 1, "y": 10})
    assert trace["result"] == 99
    trace["packed"] = (True,)
    assert trace["result"] == 10
    trace["y"] = 20
    assert trace["result"] == 20
    trace["packed"] = (1,)
    assert trace["result"] == 99


def test_dictionary_order_remains_a_distinct_inferred_value():
    @mechanism
    def equations(reverse: Dom(bool), y: Dom([10, 20])):
        mapping = V({"b": 2, "a": 1} if reverse else {"a": 1, "b": 2})
        result = V(y if list(mapping)[0] == "a" else 99)
        raw_input = V(str(reverse))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    trace = model.new_trace({"reverse": True, "y": 10})
    assert trace["result"] == 99
    trace["reverse"] = False
    assert trace["result"] == 10
    assert model.domains["mapping"].cardinality() == 2


def test_signed_zero_keeps_the_negative_branch_and_its_parent():
    @mechanism
    def equations(x: Dom([0.0, -0.0]), y: Dom([10, 20])):
        intermediate = V(x)
        result = V(y if math.copysign(1.0, intermediate) < 0 else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"intermediate", "y"}
    trace = model.new_trace({"x": 0.0, "y": 10})
    assert trace["result"] == 99
    trace["x"] = -0.0
    assert trace["result"] == 10
    trace["y"] = 20
    assert trace["result"] == 20


def test_explicit_finite_domain_rejects_unrepresented_dictionary_order():
    @mechanism
    def equations(mapping: Dom([{"a": 1, "b": 2}])):
        result = V(list(mapping)[0])
        raw_input = V(str(mapping))  # noqa: F841
        raw_output = V(result)  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"mapping": {"a": 1, "b": 2}})["result"] == "a"
    with pytest.raises(DomainError, match="mapping"):
        model.new_trace({"mapping": {"b": 2, "a": 1}})


def test_unsupported_inferred_values_request_an_explicit_type_domain():
    @mechanism
    def equations(x: Dom([1, 2])):
        result = V({x})
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match=r"explicit type domain Dom\(set\)"):
        CausalModel(equations)


def test_explicit_set_type_domain_still_supports_set_comprehensions():
    @mechanism
    def equations(x: Dom([1, 2])):
        result = V({x + i for i in range(2)}, domain=Dom(set))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result), domain=Dom(str))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == {1, 2}


def test_object_identity_requires_value_semantics():
    @mechanism
    def equations(x: Dom([1000]), y: Dom([1000])):
        result = V(x is y)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Object identity"):
        CausalModel(equations)


def test_object_address_cannot_be_inferred_from_representatives():
    @mechanism
    def equations(x: Dom([1000])):
        result = V(id(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Object identity"):
        CausalModel(equations)


def test_known_singleton_identity_checks_are_supported():
    @mechanism
    def equations(x: Dom([None, True, 1])):
        result = V((x is None, x is True, type(x) is int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": 1})["result"] == (False, False, True)


def test_shadowed_builtin_type_is_not_an_identity_singleton():
    @mechanism
    def equations(x: Dom([1000]), int: Dom([1000]), y: Dom([10, 20])):
        result = V(y if x is int else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Object identity"):
        CausalModel(equations)


def test_private_alias_cannot_hide_object_address_lookup():
    @mechanism
    def equations(x: Dom([1000])):
        identify = id
        result = V(identify(x), domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="Object identity"):
        CausalModel(equations)
