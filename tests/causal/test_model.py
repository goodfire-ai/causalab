"""Tests for ``causalab.causal.model.CausalModel``.

``CausalModel`` is the symbolic ground-truth oracle that drives causalab's
counterfactual labeling pipeline: task setup wires ``mechanisms``/``values``
into a ``CausalModel``; counterfactual generation calls ``new_trace``,
``enumerate_inputs``, ``sample_input``, and ``label_counterfactual_data`` to
produce ``(input, counterfactual_inputs, label)`` tuples; downstream
intervention experiments (DAS, manifold, ...) consume the labels and the
``run_interchange`` semantics. Wrong values here silently invalidate every
intervention/featurization metric downstream.
"""

from __future__ import annotations

import pytest

from causalab.causal import Dom, V, mechanism
from causalab.causal.counterfactuals import CounterfactualExample
from causalab.causal.model import (
    CausalModel,
    CausalTrace,
)


@pytest.fixture
def arithmetic_model() -> CausalModel:
    """Two-digit arithmetic model: A1A0 + B1B0 (sum) and A1A0 * B1B0 (product).

    Used by the init / new_trace / sample tests to exercise a model with a
    non-trivial parent/child graph and string-valued raw_input/raw_output.
    """

    @mechanism
    def equations(
        A1: Dom(list(range(10))),
        A0: Dom(list(range(10))),
        B1: Dom(list(range(10))),
        B0: Dom(list(range(10))),
    ):
        CARRY_SUM = V(1 if A0 + B0 >= 10 else 0, domain=Dom([0, 1]))
        SUM0 = V((A0 + B0) % 10, domain=Dom(list(range(10))))
        PROD0 = V(A0 * B0 % 10, domain=Dom(list(range(10))))
        PROD1 = V((A0 * B1 + A1 * B0 + A0 * B0 // 10) % 10, domain=Dom(list(range(10))))
        PROD2 = V(
            (A1 * B1 + (A0 * B1 + A1 * B0 + A0 * B0 // 10) // 10) % 10,
            domain=Dom(list(range(10))),
        )
        PROD3 = V(
            (A1 * B1 + (A0 * B1 + A1 * B0 + A0 * B0 // 10) // 10) // 10,
            domain=Dom(list(range(10))),
        )
        raw_input = V(  # noqa: F841
            f"{A1}{A0} + {B1}{B0} = ?, {A1}{A0} * {B1}{B0} = ?", domain=Dom(str)
        )
        SUM1 = V((A1 + B1 + CARRY_SUM) % 10, domain=Dom(list(range(10))))
        raw_output = V(  # noqa: F841
            f"Sum: {SUM1}{SUM0}, Product: {str(PROD3) + str(PROD2) + str(PROD1) + str(PROD0).lstrip('0') or '0'}",
            domain=Dom(str),
        )
        return PROD3

    return CausalModel(equations, id="arithmetic_model")


@pytest.fixture
def two_chain_model() -> CausalModel:
    """Two independent chains X->Y->Z and A->B->C.

    Used by run_interchange tests (including arrow-syntax cross-variable
    interchange) because the chains let us swap a single variable from one
    chain into the other without aliasing.
    """

    @mechanism
    def equations(X: Dom([0, 1, 2]), A: Dom([0, 1, 2])):
        Y = V(X + 1, domain=Dom([0, 1, 2, 3]))
        B = V(A + 1, domain=Dom([0, 1, 2, 3]))
        raw_input = V(f"X={X}, A={A}", domain=Dom(str))  # noqa: F841
        Z = V(Y + 1, domain=Dom([0, 1, 2, 3, 4]))
        C = V(B + 1, domain=Dom([0, 1, 2, 3, 4]))
        raw_output = V(f"Z={Z}, C={C}", domain=Dom(str))  # noqa: F841
        return C

    return CausalModel(equations, id="two_chain_model")


@pytest.fixture
def doubling_model() -> CausalModel:
    """Single-input doubling model used by label_counterfactual_data tests."""

    @mechanism
    def equations(input_val: Dom([1, 2, 3])):
        processed_val = V(input_val * 2, domain=Dom([2, 4, 6]))
        raw_input = V(f"Input: {input_val}", domain=Dom(str))  # noqa: F841
        raw_output = V(f"Output: {processed_val}", domain=Dom(str))  # noqa: F841
        return processed_val

    return CausalModel(equations, id="doubling_model")


class TestCausalModelInitUnit:
    """Construction of CausalModel: derived structure, validation, sorting."""

    pytestmark = pytest.mark.unit

    def test_variables_derived_from_mechanisms(self, arithmetic_model: CausalModel):
        expected_vars = {
            "A1",
            "A0",
            "B1",
            "B0",
            "CARRY_SUM",
            "SUM0",
            "SUM1",
            "PROD0",
            "PROD1",
            "PROD2",
            "PROD3",
            "raw_input",
            "raw_output",
        }
        assert set(arithmetic_model.variables) == expected_vars

    def test_inputs_are_parentless_variables(self, arithmetic_model: CausalModel):
        assert set(arithmetic_model.inputs) == {"A1", "A0", "B1", "B0"}

    def test_inputs_have_timestep_zero(self, arithmetic_model: CausalModel):
        for var in arithmetic_model.inputs:
            assert arithmetic_model.timesteps[var] == 0

    def test_raw_input_and_raw_output_required(self, arithmetic_model: CausalModel):
        assert "raw_input" in arithmetic_model.variables
        assert "raw_output" in arithmetic_model.variables

    def test_missing_raw_input_raises(self):
        @mechanism
        def equations():
            raw_output = V("x", domain=Dom(str))
            return raw_output

        with pytest.raises(ValueError):
            CausalModel(equations)

    def test_missing_raw_output_raises(self):
        @mechanism
        def equations():
            raw_input = V("x", domain=Dom(str))
            return raw_input

        with pytest.raises(ValueError):
            CausalModel(equations)

    def test_missing_input_domain_raises(self):
        @mechanism
        def equations(A):
            raw_input = V(f"A={A}", domain=Dom(str))  # noqa: F841
            raw_output = V(f"A={A}", domain=Dom(str))
            return raw_output

        with pytest.raises(ValueError, match="needs a Dom"):
            CausalModel(equations)

    def test_print_pos_defaults_to_dict(self, arithmetic_model: CausalModel):
        assert isinstance(arithmetic_model.print_pos, dict)
        # raw_input defaults to (0, -2) per __init__.
        assert arithmetic_model.print_pos["raw_input"] == (0, -2)
        for var in arithmetic_model.variables:
            assert var in arithmetic_model.print_pos


class TestNewTraceUnit:
    """``CausalModel.new_trace`` returns a fresh ``CausalTrace`` with derived values."""

    pytestmark = pytest.mark.unit

    def test_basic_arithmetic(self, arithmetic_model: CausalModel):
        # 25 + 37 = 62, 25 * 37 = 925.
        setting = arithmetic_model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 7})
        assert setting["SUM0"] == 2
        assert setting["SUM1"] == 6
        assert setting["CARRY_SUM"] == 1
        assert setting["PROD0"] == 5
        assert setting["PROD1"] == 2
        assert setting["PROD2"] == 9
        assert setting["PROD3"] == 0
        assert setting["raw_input"] is not None
        assert setting["raw_output"] is not None
        assert "25 + 37" in setting["raw_input"]
        assert "Sum: 62" in setting["raw_output"]

    def test_zero_inputs(self, arithmetic_model: CausalModel):
        # 20 + 09 = 29.
        setting = arithmetic_model.new_trace({"A1": 2, "A0": 0, "B1": 0, "B0": 9})
        assert setting["SUM0"] == 9
        assert setting["SUM1"] == 2
        assert setting["CARRY_SUM"] == 0

    def test_carrying_sum(self, arithmetic_model: CausalModel):
        # 95 + 17 = 112.
        setting = arithmetic_model.new_trace({"A1": 9, "A0": 5, "B1": 1, "B0": 7})
        assert setting["SUM0"] == 2
        assert setting["SUM1"] == 1
        assert setting["CARRY_SUM"] == 1

    def test_large_product_with_carry(self, arithmetic_model: CausalModel):
        # 95 * 95 = 9025.
        setting = arithmetic_model.new_trace({"A1": 9, "A0": 5, "B1": 9, "B0": 5})
        assert setting["PROD0"] == 5
        assert setting["PROD1"] == 2
        assert setting["PROD2"] == 0
        assert setting["PROD3"] == 9

    def test_intervention_overrides_mechanism(self, arithmetic_model: CausalModel):
        # Forcing CARRY_SUM=0 on a setting where the natural carry is 1
        # should propagate to SUM1.
        base = arithmetic_model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 7})
        intervened = arithmetic_model.new_trace(
            {"A1": 2, "A0": 5, "B1": 3, "B0": 7, "CARRY_SUM": 0}
        )
        assert intervened["SUM1"] == base["SUM1"] - 1
        assert intervened["SUM0"] == base["SUM0"]
        assert intervened["PROD0"] == base["PROD0"]


class TestRunInterchangeUnit:
    """``CausalModel.run_interchange`` — same-name and ``"<-"`` cross-variable syntax.

    Absorbs the historical ``test_run_interchange_arrow_syntax.py``: the
    arrow syntax is the documented raison d'être of ``run_interchange``, so
    direct coverage of both branches lives in this class.
    """

    pytestmark = pytest.mark.unit

    def test_same_name_interchange_arithmetic(self, arithmetic_model: CausalModel):
        # Original: 25 + 37. Counterfactual: 25 + 38 (B0 changed).
        original = arithmetic_model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 7})
        cf = arithmetic_model.new_trace({"A1": 2, "A0": 5, "B1": 3, "B0": 8})

        result = arithmetic_model.run_interchange(original, {"B0": cf})

        assert result["B0"] == 8
        assert result["A1"] == 2
        assert result["A0"] == 5
        assert result["B1"] == 3
        # 5 + 8 = 13 -> SUM0=3, CARRY_SUM=1, SUM1 = 2+3+1 = 6.
        assert result["SUM0"] == 3
        assert result["CARRY_SUM"] == 1
        assert result["SUM1"] == 6

    def test_original_syntax_still_works(self, two_chain_model: CausalModel):
        # X=1,A=2 -> Y=2,B=3,Z=3,C=4 vs X=0,A=0 -> Y=1,B=1,Z=2,C=2.
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y": cf})

        assert result["Y"] == 1
        assert result["Z"] == 2
        assert result["C"] == 4  # other chain untouched

    def test_arrow_syntax_basic(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y<-B": cf})

        assert result["Y"] == 1  # from cf B=1
        assert result["Z"] == 2  # Y+1
        assert result["B"] == 3  # original chain
        assert result["C"] == 4

    def test_arrow_syntax_with_spaces(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y <- B": cf})

        assert result["Y"] == 1
        assert result["Z"] == 2
        assert result["B"] == 3
        assert result["C"] == 4

    def test_arrow_syntax_swap_chains(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y<-B": cf, "B<-Y": cf})

        assert result["Y"] == 1
        assert result["B"] == 1
        assert result["Z"] == 2
        assert result["C"] == 2

    def test_arrow_syntax_mixed_with_regular(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y<-B": cf, "B": cf})

        assert result["Y"] == 1
        assert result["B"] == 1
        assert result["Z"] == 2
        assert result["C"] == 2

    def test_arrow_syntax_different_counterfactuals(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf1 = two_chain_model.new_trace({"X": 0, "A": 0})  # Y=1, B=1
        cf2 = two_chain_model.new_trace({"X": 2, "A": 2})  # Y=3, B=3

        result = two_chain_model.run_interchange(original, {"Y<-B": cf1, "B<-Y": cf2})

        assert result["Y"] == 1
        assert result["B"] == 3
        assert result["Z"] == 2
        assert result["C"] == 4

    def test_arrow_syntax_input_variable(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"X<-A": cf})

        assert result["X"] == 0  # cf A=0
        assert result["A"] == 2
        assert result["Y"] == 1
        assert result["B"] == 3
        assert result["Z"] == 2
        assert result["C"] == 4

    def test_arrow_syntax_same_variable_name_is_no_op_form(
        self, two_chain_model: CausalModel
    ):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        result = two_chain_model.run_interchange(original, {"Y<-Y": cf})

        # Behaves like the regular "Y" syntax.
        assert result["Y"] == 1
        assert result["Z"] == 2


class TestEnumerateInputsUnit:
    """``CausalModel.enumerate_inputs`` and ``n_unique_inputs``."""

    pytestmark = pytest.mark.unit

    def test_cardinality_no_filter(self, two_chain_model: CausalModel):
        traces = two_chain_model.enumerate_inputs()
        # |X| * |A| = 3 * 3 = 9.
        assert len(traces) == 9
        assert two_chain_model.n_unique_inputs == 9
        for trace in traces:
            assert isinstance(trace, CausalTrace)
            for var in two_chain_model.inputs:
                assert trace[var] in two_chain_model.values[var]

    def test_input_filter_drops_combinations(self, two_chain_model: CausalModel):
        # Drop combinations where X == A.
        two_chain_model.input_filter = lambda t: t["X"] != t["A"]
        traces = two_chain_model.enumerate_inputs()

        # 9 total combinations, 3 with X == A, so 6 remain.
        assert len(traces) == 6
        assert two_chain_model.n_unique_inputs == 6
        for trace in traces:
            assert trace["X"] != trace["A"]


class TestSampleInputUnit:
    """``CausalModel.sample_input`` — random input draw with optional predicates."""

    pytestmark = pytest.mark.unit

    def test_returns_trace_with_inputs_in_values(self, arithmetic_model: CausalModel):
        trace = arithmetic_model.sample_input()
        for var in arithmetic_model.inputs:
            assert var in trace
            assert trace[var] in arithmetic_model.values[var]

    def test_filter_func_honored(self, two_chain_model: CausalModel):
        trace = two_chain_model.sample_input(filter_func=lambda t: t["X"] == 2)
        assert trace["X"] == 2

    def test_input_filter_honored(self, two_chain_model: CausalModel):
        two_chain_model.input_filter = lambda t: t["A"] == 0
        trace = two_chain_model.sample_input()
        assert trace["A"] == 0

    def test_combined_predicates(self, two_chain_model: CausalModel):
        two_chain_model.input_filter = lambda t: t["A"] == 0
        trace = two_chain_model.sample_input(filter_func=lambda t: t["X"] == 2)
        assert trace["A"] == 0
        assert trace["X"] == 2


class TestLabelCounterfactualDataUnit:
    """``CausalModel.label_counterfactual_data`` — interchange + label extraction."""

    pytestmark = pytest.mark.unit

    def test_basic_labeling(self, doubling_model: CausalModel):
        input1 = doubling_model.new_trace({"input_val": 1})
        cf1 = doubling_model.new_trace({"input_val": 2})
        input2 = doubling_model.new_trace({"input_val": 2})
        cf2 = doubling_model.new_trace({"input_val": 3})

        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1]},
            {"input": input2, "counterfactual_inputs": [cf2]},
        ]
        result = doubling_model.label_counterfactual_data(examples, ["processed_val"])

        assert len(result) == 2
        assert result[0]["label"] == "Output: 4"
        assert result[1]["label"] == "Output: 6"

    def test_single_cf_broadcast_across_target_vars(self, two_chain_model: CausalModel):
        # One counterfactual, multiple target variables -> cf is reused for each.
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf]},
        ]

        result = two_chain_model.label_counterfactual_data(examples, ["Y", "B"])

        # Y<-cf.Y=1, B<-cf.B=1 -> Z=2, C=2.
        assert result[0]["label"] == "Z=2, C=2"
        assert "setting" in result[0]

    def test_list_element_target_variables(self, two_chain_model: CausalModel):
        # target_variables element can itself be a list of variables sharing one cf.
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf]},
        ]

        result = two_chain_model.label_counterfactual_data(examples, [["Y", "B"]])

        assert result[0]["label"] == "Z=2, C=2"

    def test_setting_serialized_via_to_dict(self, doubling_model: CausalModel):
        input1 = doubling_model.new_trace({"input_val": 1})
        cf1 = doubling_model.new_trace({"input_val": 2})
        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1]},
        ]
        result = doubling_model.label_counterfactual_data(examples, ["processed_val"])
        # to_dict returns a plain dict, not a CausalTrace.
        assert isinstance(result[0]["setting"], dict)
        assert result[0]["setting"]["processed_val"] == 4

    def test_default_label_variable_is_raw_output(self, doubling_model: CausalModel):
        input1 = doubling_model.new_trace({"input_val": 1})
        cf1 = doubling_model.new_trace({"input_val": 2})
        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1]},
        ]
        result_default = doubling_model.label_counterfactual_data(
            examples, ["processed_val"]
        )
        result_explicit = doubling_model.label_counterfactual_data(
            examples, ["processed_val"], label_variable="raw_output"
        )
        assert result_default[0]["label"] == result_explicit[0]["label"]

    def test_custom_label_variable(self, doubling_model: CausalModel):
        input1 = doubling_model.new_trace({"input_val": 1})
        cf1 = doubling_model.new_trace({"input_val": 2})
        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1]},
        ]
        result = doubling_model.label_counterfactual_data(
            examples, ["processed_val"], label_variable="processed_val"
        )
        assert result[0]["label"] == 4


class TestCanDistinguishWithDatasetUnit:
    """``CausalModel.can_distinguish_with_dataset`` — interchange-vs-baseline counts."""

    pytestmark = pytest.mark.unit

    def test_baseline_branch_counts_changed_outputs(self, two_chain_model: CausalModel):
        # target_variables2 is None: count examples where intervening on Y
        # changes raw_output relative to the unintervened input trace.
        original = two_chain_model.new_trace({"X": 1, "A": 2})  # Z=3, C=4
        cf = two_chain_model.new_trace({"X": 0, "A": 0})  # Y=1
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf]},
        ]

        result = two_chain_model.can_distinguish_with_dataset(
            examples, target_variables1=["Y"], target_variables2=None
        )

        # Intervening on Y -> Z=2 -> output differs from baseline.
        assert result == {"proportion": 1.0, "count": 1}

    def test_two_target_sets_branch(self, two_chain_model: CausalModel):
        # Compare interchange-on-Y vs interchange-on-B.
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf]},
        ]

        result = two_chain_model.can_distinguish_with_dataset(
            examples, target_variables1=["Y"], target_variables2=["B"]
        )

        # Two different interchanges change two different chains -> outputs differ.
        assert result["count"] == 1
        assert result["proportion"] == 1.0

    def test_identical_target_sets_yield_zero_count(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf]},
        ]
        result = two_chain_model.can_distinguish_with_dataset(
            examples, target_variables1=["Y"], target_variables2=["Y"]
        )
        assert result == {"proportion": 0.0, "count": 0}

    def test_requires_single_counterfactual(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf1 = two_chain_model.new_trace({"X": 0, "A": 0})
        cf2 = two_chain_model.new_trace({"X": 2, "A": 2})
        examples: list[CounterfactualExample] = [
            {"input": original, "counterfactual_inputs": [cf1, cf2]},
        ]
        with pytest.raises(AssertionError):
            two_chain_model.can_distinguish_with_dataset(
                examples, target_variables1=["Y"], target_variables2=None
            )


class TestCausalModelProperty:
    """Structural invariants of CausalModel that must hold for every model."""

    pytestmark = pytest.mark.property

    def test_new_trace_does_not_mutate_inputs(self, arithmetic_model: CausalModel):
        inputs = {"A1": 2, "A0": 5, "B1": 3, "B0": 7, "CARRY_SUM": 1}
        snapshot = inputs.copy()
        arithmetic_model.new_trace(inputs)
        assert inputs == snapshot

    @pytest.mark.parametrize(
        "model_fixture", ["arithmetic_model", "two_chain_model", "doubling_model"]
    )
    def test_timesteps_form_valid_topological_order(
        self, model_fixture: str, request: pytest.FixtureRequest
    ):
        model: CausalModel = request.getfixturevalue(model_fixture)
        for var in model.variables:
            for parent in model.parents[var]:
                assert model.timesteps[parent] < model.timesteps[var], (
                    f"parent {parent} not strictly before child {var}"
                )

    @pytest.mark.parametrize(
        "model_fixture", ["arithmetic_model", "two_chain_model", "doubling_model"]
    )
    def test_parents_and_children_are_mutual_inverses(
        self, model_fixture: str, request: pytest.FixtureRequest
    ):
        model: CausalModel = request.getfixturevalue(model_fixture)
        for var in model.variables:
            for parent in model.parents[var]:
                assert var in model.children[parent]
            for child in model.children[var]:
                assert var in model.parents[child]

    def test_enumerate_inputs_matches_n_unique_inputs_unfiltered(
        self, two_chain_model: CausalModel
    ):
        assert two_chain_model.input_filter is None
        assert (
            len(two_chain_model.enumerate_inputs()) == two_chain_model.n_unique_inputs
        )

    def test_enumerate_inputs_matches_n_unique_inputs_filtered(
        self, two_chain_model: CausalModel
    ):
        # Pick a filter that drops some combinations to exercise the filtered branch.
        two_chain_model.input_filter = lambda t: t["X"] >= t["A"]
        enumerated = two_chain_model.enumerate_inputs()
        assert len(enumerated) == two_chain_model.n_unique_inputs
        # Sanity: the filter actually dropped something.
        assert two_chain_model.n_unique_inputs < 9

    def test_interchange_with_self_is_no_op(self, two_chain_model: CausalModel):
        trace = two_chain_model.new_trace({"X": 1, "A": 2})
        original_y = trace["Y"]
        result = two_chain_model.run_interchange(trace, {"Y": trace})
        assert result["Y"] == original_y

    def test_arrow_self_equivalent_to_regular(self, two_chain_model: CausalModel):
        original = two_chain_model.new_trace({"X": 1, "A": 2})
        cf = two_chain_model.new_trace({"X": 0, "A": 0})

        arrow_result = two_chain_model.run_interchange(original, {"Y<-Y": cf})
        plain_result = two_chain_model.run_interchange(original, {"Y": cf})

        # Compare every variable in the trace.
        for var in two_chain_model.variables:
            assert arrow_result[var] == plain_result[var]

    def test_mechanism_determinism(self, arithmetic_model: CausalModel):
        # Same inputs (with all inputs pinned) produce the same raw_output
        # across two independent new_trace calls.
        inputs = {"A1": 4, "A0": 6, "B1": 1, "B0": 9}
        trace_a = arithmetic_model.new_trace(inputs)
        trace_b = arithmetic_model.new_trace(inputs)
        assert trace_a["raw_output"] == trace_b["raw_output"]
