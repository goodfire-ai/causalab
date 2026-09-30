"""NaN identity variants cannot justify exhaustive value or read proofs."""

import struct

import pytest

from causalab.causal import CausalModel, DefinitionError, Dom, V, mechanism

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("negate", [False, True])
def test_nan_membership_keeps_both_outcomes_and_intervention_reads(negate):
    nan = float("nan")

    @mechanism
    def equations(x: Dom([nan]), y: Dom([10, 20])):
        result = V(y if (x not in [nan] if negate else x in [nan]) else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.parents["result"]) == {"x", "y"}
    assert set(model.domains["result"].enumerated()) == {10, 20, 99}
    for x in (nan, float("nan")):
        selected = x not in [nan] if negate else x in [nan]
        trace = model.new_trace({"x": x, "y": 10})
        assert trace["result"] == (10 if selected else 99)
        trace["y"] = 20
        assert trace["result"] == (20 if selected else 99)


def test_nan_membership_result_infers_a_boolean_domain():
    nan = float("nan")

    @mechanism
    def equations(x: Dom([nan]), y: Dom([10, 20])):
        member = V(x in [nan])
        result = V(y if member else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert set(model.domains["member"].enumerated()) == {False, True}
    trace = model.new_trace({"x": float("nan"), "y": 10})
    assert trace["result"] == 99
    trace["x"] = nan
    assert trace["result"] == 10


def test_nested_nan_container_equality_preserves_python_semantics():
    nan = float("nan")

    @mechanism
    def equations(x: Dom([{"n": [(nan,)]}]), y: Dom([10, 20])):
        result = V(y if x == {"n": [(nan,)]} else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert "y" in model.parents["result"]
    for value in (nan, float("nan")):
        x = {"n": [(value,)]}
        expected = 10 if x == {"n": [(nan,)]} else 99
        assert model.new_trace({"x": x, "y": 10})["result"] == expected


def test_nan_witness_search_includes_helper_configuration():
    nan = float("nan")

    def member(x):
        return x in [nan]

    @mechanism
    def equations(x: Dom([float("nan")]), y: Dom([10, 20])):
        result = V(y if member(x) else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": nan, "y": 10})["result"] == 10
    assert model.new_trace({"x": float("nan"), "y": 10})["result"] == 99


@pytest.mark.parametrize("explicit", [False, True])
def test_nan_dependent_opaque_results_need_an_explicit_domain(explicit):
    nan = float("nan")
    result_domain = Dom([10, 99]) if explicit else None

    def outcome(x):
        return 10 if x in [nan] else 99

    @mechanism
    def equations(x: Dom([nan])):
        result = V(outcome(x), domain=result_domain)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    if not explicit:
        with pytest.raises(DefinitionError, match="Cannot infer a sound domain"):
            CausalModel(equations)
    else:
        model = CausalModel(equations)
        assert model.new_trace({"x": nan})["result"] == 10
        assert model.new_trace({"x": float("nan")})["result"] == 99


def test_multiple_causal_nans_can_share_a_new_identity():
    nan = float("nan")

    @mechanism
    def equations(a: Dom([nan]), b: Dom([nan]), y: Dom([10, 20])):
        result = V(y if a in [b] and a not in [nan] else 99)
        raw_input = V(str((a, b)))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    shared = float("nan")
    trace = model.new_trace({"a": shared, "b": shared, "y": 10})
    assert trace["result"] == 10
    trace["y"] = 20
    assert trace["result"] == 20
    trace["b"] = float("nan")
    assert trace["result"] == 99


def test_nested_nan_witnesses_can_mix_shared_and_fresh_identities():
    nan = float("nan")

    @mechanism
    def equations(x: Dom([(nan, nan)]), y: Dom([10, 20])):
        result = V(y if x[0] in [nan] and x[1] not in [nan] else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": (nan, float("nan")), "y": 10})["result"] == 10
    assert model.new_trace({"x": (nan, nan), "y": 10})["result"] == 99


def test_nan_witnesses_preserve_sign_and_payload_bits():
    bits = bytes.fromhex("fff8000000000007")
    nan = struct.unpack("!d", bits)[0]

    @mechanism
    def equations(x: Dom([nan]), y: Dom([10, 20])):
        result = V(y if x not in [nan] else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    fresh = struct.unpack("!d", bits)[0]
    assert model.new_trace({"x": fresh, "y": 10})["result"] == 10
    assert model.new_trace({"x": nan, "y": 10})["result"] == 99


def test_nan_elsewhere_does_not_prevent_a_boolean_guard_proof():
    nan = float("nan")

    @mechanism
    def equations(x: Dom([nan]), enabled: Dom([False]), y: Dom([10, 20])):
        result = V(y if enabled else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.parents["result"] == ["enabled"]
    assert model.new_trace({"x": nan, "enabled": False, "y": 10})["result"] == 99


def test_nan_representatives_cannot_prove_a_conditional_read_absent():
    nan = float("nan")

    @mechanism
    def equations(x: Dom([nan]), y: Dom([10])):
        result = V(y if x == x else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    with pytest.raises(DefinitionError, match="cannot establish whether it reads 'y'"):
        CausalModel(equations)


@pytest.mark.parametrize("captured", [False, True])
def test_numpy_nan_witnesses_respect_separate_trace_value_copies(captured):
    import numpy as np

    nan = np.float64("nan")

    def member(a, b):
        return a in [nan] if captured else a in [b]

    @mechanism
    def equations(a: Dom([nan]), b: Dom([nan]), y: Dom([10])):
        result = V(y if member(a, b) else 99)
        raw_input = V(str((a, b)))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    # A NumPy scalar is copied into each trace value. Neither a configuration
    # atom nor another parent's scalar can supply an identical runtime object.
    with pytest.raises(DefinitionError, match="cannot establish whether it reads 'y'"):
        CausalModel(equations)


def test_numpy_nan_aliases_inside_one_parent_remain_valid_witnesses():
    import numpy as np

    nan = np.float64("nan")

    @mechanism
    def equations(x: Dom([(nan, nan)]), y: Dom([10, 20])):
        result = V(y if x[0] in [x[1]] else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    shared = np.float64("nan")
    assert model.new_trace({"x": (shared, shared), "y": 10})["result"] == 10
    distinct = (np.float64("nan"), np.float64("nan"))
    assert model.new_trace({"x": distinct, "y": 10})["result"] == 99


def test_unsupported_finite_configuration_nans_do_not_poison_witnesses():
    import numpy as np

    nan = float("nan")
    # Numeric configuration may contain NumPy shapes that cannot themselves
    # serve as exhaustive finite domain values (here, dtype metadata).
    configuration = np.array(
        [float("nan")], dtype=np.dtype("float64", metadata={"label": "config"})
    )

    def member(x):
        return x in [nan] and bool(configuration[0])

    @mechanism
    def equations(x: Dom([nan]), y: Dom([10])):
        result = V(y if member(x) else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert model.new_trace({"x": nan, "y": 10})["result"] == 10
    assert model.new_trace({"x": float("nan"), "y": 10})["result"] == 99


def test_nan_witness_search_reads_configuration_storage_without_running_properties():
    """The witness walk reads a slot's stored value, as ConfigurationCopier
    does; a property shadowing that slot is user code and must not run while
    the model compiles."""
    from dataclasses import dataclass

    calls = []

    @dataclass
    class Stored:
        __slots__ = ("tag",)
        tag: int

    class Shadowed(Stored):
        __slots__ = ()

        @property
        def tag(self):
            calls.append("tag")
            return 0

    holder = Stored.__new__(Shadowed)
    Stored.__dict__["tag"].__set__(holder, 1)
    nan = float("nan")

    def member_reading_nothing(x):
        _ = holder  # the closure reference puts holder in the witness walk
        return x in [nan]

    @mechanism
    def equations(x: Dom([float("nan")]), y: Dom([10, 20])):
        result = V(y if member_reading_nothing(x) else 99)
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    assert calls == []
    assert model.new_trace({"x": nan, "y": 10})["result"] == 10
