"""Helpers must declare the configuration whose snapshot they consume."""

import builtins

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, mechanism

pytestmark = pytest.mark.unit

_namespace_alias = globals


def _dynamic_helper(value):
    return value + globals().get("_offset", 100)


def _alias_helper(value):
    return value + _namespace_alias().get("_offset", 100)


def _module_helper(value):
    return value + builtins.globals().get("_offset", 100)


def _default_helper(value, namespace=globals):
    return value + namespace().get("_offset", 100)


def _nested_helper(value):
    def nested():
        return globals().get("_offset", 100)

    return value + nested()


@pytest.mark.parametrize(
    "helper",
    [_dynamic_helper, _alias_helper, _module_helper, _default_helper, _nested_helper],
)
def test_dynamic_global_helpers_fail_with_source_and_configuration_guidance(helper):
    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(
        DefinitionError, match=r"test_dynamic_namespace.py:\d+:.*globals.*explicitly"
    ):
        CausalModel(equations)


def test_dynamic_global_closure_alias_is_rejected():
    namespace = globals

    def helper(value):
        return value + namespace().get("_offset", 100)

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="globals.*explicitly"):
        CausalModel(equations)


def test_an_innocent_local_named_globals_is_supported():
    def helper(value):
        globals = {"offset": 10}
        return value + globals["offset"]

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == 11


def test_a_captured_function_named_globals_is_supported():
    def globals():
        return {"offset": 10}

    def helper(value):
        return value + globals()["offset"]

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == 11
