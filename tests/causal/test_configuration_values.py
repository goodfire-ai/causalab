"""Configuration snapshots retain helper state and own NumPy metadata."""

from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    NoReturn,
    Optional,
    ParamSpec,
    TypeVar,
)

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, mechanism
from causalab.causal.compiler import ConfigurationCopier

pytestmark = pytest.mark.unit


def test_helper_attributes_annotations_and_defaults_share_an_owned_snapshot():
    offset = [10]

    def helper(value, config=offset):
        return value + config[0] + helper.offset[0] + helper.__annotations__["bias"][0]

    helper.offset = offset
    helper.__annotations__["bias"] = offset
    helper.recursive = helper

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x), domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    public = model.definition.environment["helper"]
    assert public.offset is public.__annotations__["bias"] is public.__defaults__[0]
    assert public.recursive is public
    offset[0] = 20
    public.offset[0] = 30
    assert model.new_trace({"x": 1})["result"] == 31
    assert CausalModel(equations).new_trace({"x": 1})["result"] == 61


def test_helper_preserves_common_type_annotations_and_annotation_metadata():
    metadata = {"offset": [1]}

    def helper(value: list[int] | tuple[int, ...]) -> Optional[int]:
        return value[0]

    helper.__annotations__["configured"] = Annotated[int, metadata]
    copied = ConfigurationCopier()(helper)
    assert copied.__annotations__["value"] == helper.__annotations__["value"]
    assert copied.__annotations__["return"] == Optional[int]
    metadata["offset"][0] = 2
    assert copied.__annotations__["configured"].__metadata__[0]["offset"] == [1]
    assert copied([3]) == 3


def test_typing_code_symbols_and_forward_references_preserve_annotations():
    parameter = TypeVar("T", bound="int")
    signature = ParamSpec("P")

    def helper(value):
        return value

    helper.__annotations__ = {
        "value": Optional["int"],
        "return": Any,
        "unreachable": NoReturn,
        "generic": Callable[signature, parameter],
        "args": signature.args,
        "kwargs": signature.kwargs,
        "list": List,
        "dict": Dict,
        "callable": Callable,
    }
    copied = ConfigurationCopier()(helper)
    assert copied.__annotations__ == helper.__annotations__
    assert copied.__annotations__["generic"].__args__ == (signature, parameter)
    original_ref = helper.__annotations__["value"].__args__[0]
    copied_ref = copied.__annotations__["value"].__args__[0]
    assert isinstance(copied_ref, ForwardRef)
    assert copied_ref is not original_ref
    assert copied(3) == 3


def test_forward_reference_evaluation_cache_owns_annotation_metadata():
    metadata = {"offset": [1]}
    reference = ForwardRef("Configured", module=__name__, is_class=True)
    reference.__forward_evaluated__ = True
    reference.__forward_value__ = Annotated[int, metadata]
    copied = ConfigurationCopier()(reference)
    metadata["offset"][0] = 2
    assert copied.__forward_evaluated__
    assert copied.__forward_is_class__
    assert copied.__forward_module__ == __name__
    assert copied.__forward_value__.__metadata__[0]["offset"] == [1]


def test_model_can_capture_helper_with_typing_leaf_and_forward_annotations():
    def helper(value: Any) -> Optional["int"]:
        return value + 1

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(helper(x), domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    assert CausalModel(equations).new_trace({"x": 1})["result"] == 2


def test_numpy_metadata_cannot_mutate_existing_compiled_configuration():
    np = pytest.importorskip("numpy")
    config = np.array([1], dtype=np.dtype(np.longlong, metadata={"offset": [1]}))

    @mechanism
    def equations(x: Dom([1, 2])):
        result = V(x + config.dtype.metadata["offset"][0], domain=Dom(int))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    public = model.definition.environment["config"]
    assert public.dtype.type is np.longlong
    config.dtype.metadata["offset"][0] = 20
    public.dtype.metadata["offset"][0] = 10
    assert model.new_trace({"x": 1})["result"] == 2


def test_numpy_field_and_subarray_metadata_are_copied_recursively():
    np = pytest.importorskip("numpy")
    metadata = {"offset": [1]}
    element = np.dtype(np.longlong, metadata=metadata)
    dtype = np.dtype([("field", element, (2,))], align=True)
    config = np.zeros(1, dtype=dtype)
    config["field"] = [[2, 3]]
    copied = ConfigurationCopier()({"dtype": dtype, "array": config})
    assert copied["dtype"] is copied["array"].dtype
    actual = copied["dtype"].fields["field"][0].subdtype[0]
    assert actual.type is np.longlong
    metadata["offset"][0] = 10
    assert actual.metadata["offset"] == [1]
    assert copied["array"]["field"].tolist() == [[2, 3]]
    assert copied["dtype"].isalignedstruct


def test_object_dtype_metadata_is_not_a_backdoor_to_unsnapshotted_objects():
    np = pytest.importorskip("numpy")
    with pytest.raises(DefinitionError, match="Object arrays"):
        ConfigurationCopier()(np.dtype(object))


def test_cyclic_numpy_metadata_fails_clearly():
    np = pytest.importorskip("numpy")
    owner = []
    dtype = np.dtype("i8", metadata={"owner": owner})
    owner.append(dtype)
    with pytest.raises(DefinitionError, match="Cyclic NumPy metadata"):
        ConfigurationCopier()(dtype)
