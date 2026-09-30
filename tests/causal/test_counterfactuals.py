"""Tests for ``causalab.causal.counterfactuals``.

Defines the two ``TypedDict`` schemas (``CounterfactualExample``,
``LabeledCounterfactualExample``) that every counterfactual sample conforms
to. The base dict pairs a ``CausalTrace`` input with a list of counterfactual
``CausalTrace`` inputs and is produced by
``causalab.causal.counterfactuals.generate_counterfactual_samples`` and consumed
by ``io.counterfactuals`` (save/load), ``methods.filter.filter_dataset``,
``methods.metric.*``, ``methods.pca``, and
``CausalModel.label_counterfactual_data``. Wrong/missing keys break
serialization round-trips, DAS/DBM training loss (needs ``label``), and every
runner that ingests a counterfactual dataset.

NOTE: ``TypedDict`` is runtime-permissive (extra keys accepted; missing keys
not raised on construction). The tests below assert *structural* invariants
that consumers depend on, not schema enforcement.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from causalab.causal.counterfactuals import (
    CounterfactualExample,
    LabeledCounterfactualExample,
    generate_counterfactual_samples,
    get_partial_filter,
    label_data_with_variables,
    sample_intervention,
)
from causalab.causal.model import CausalModel, CausalTrace
from tests._helpers.tiny import tiny_chain_model


@pytest.fixture
def example_pair() -> tuple[CausalTrace, CausalTrace]:
    """Base/counterfactual trace pair from the 3-node A→B→C model."""
    model = tiny_chain_model()
    base = model.new_trace({"A": 0})
    cf = model.new_trace({"A": 1})
    return base, cf


class TestCounterfactualExampleUnit:
    """Schema for the base ``CounterfactualExample`` consumed by all downstream code."""

    pytestmark = pytest.mark.unit

    def test_dict_literal_constructs(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        assert isinstance(example, dict)

    def test_required_keys_reachable(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        assert example["input"] is base
        assert example["counterfactual_inputs"] == [cf]

    def test_type_annotations_declared(self):
        # Static type contract — consumers rely on `input: CausalTrace` and
        # `counterfactual_inputs: list[CausalTrace]`.
        # `CausalTrace` is a TYPE_CHECKING-only forward ref in the module; pass
        # it via localns so get_type_hints can resolve.
        hints = get_type_hints(
            CounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "input" in hints
        assert "counterfactual_inputs" in hints

    def test_extra_keys_accepted_at_runtime(self, example_pair):
        # TypedDict is runtime-permissive; consumers must not break if extra
        # keys appear (e.g. "label", "setting").
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "extra_field": "anything",  # type: ignore[typeddict-unknown-key]
        }
        assert example["extra_field"] == "anything"  # type: ignore[typeddict-item]


class TestLabeledCounterfactualExampleUnit:
    """``LabeledCounterfactualExample`` extends the base schema with ``label``."""

    pytestmark = pytest.mark.unit

    def test_dict_literal_with_label_constructs(self, example_pair):
        base, cf = example_pair
        example: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": "any-value",
        }
        assert example["label"] == "any-value"

    def test_label_key_is_in_type_hints(self):
        hints = get_type_hints(
            LabeledCounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "label" in hints

    def test_inherits_base_keys(self):
        # Static-type inheritance: the labeled schema must include the base
        # keys, so downstream code typed against `CounterfactualExample`
        # accepts labeled examples too.
        hints = get_type_hints(
            LabeledCounterfactualExample, localns={"CausalTrace": CausalTrace}
        )
        assert "input" in hints
        assert "counterfactual_inputs" in hints


class TestCounterfactualExampleProperty:
    """Structural invariants that downstream consumers (``filter_dataset``,
    ``save_counterfactual_examples``, ``label_counterfactual_data``) depend on."""

    pytestmark = pytest.mark.property

    def test_counterfactual_inputs_is_indexable_list(self, example_pair):
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf, cf, cf],
        }
        # Must be a list (consumers index it via [0], [i], etc.).
        assert isinstance(example["counterfactual_inputs"], list)
        # Indexable.
        assert example["counterfactual_inputs"][0] is cf
        assert example["counterfactual_inputs"][2] is cf

    def test_input_and_cf_share_node_set(self, example_pair):
        # Every consumer assumes the cf traces have the same variables as the
        # input trace (otherwise interchange dies). Build both from the same
        # model and confirm.
        base, cf = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
        }
        input_vars = set(example["input"].to_dict())
        for cf_trace in example["counterfactual_inputs"]:
            assert set(cf_trace.to_dict()) == input_vars

    def test_empty_counterfactual_list_allowed(self, example_pair):
        # No consumer crashes on an empty list (it just no-ops).
        base, _ = example_pair
        example: CounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [],
        }
        assert example["counterfactual_inputs"] == []


class TestLabeledCounterfactualExampleProperty:
    """A ``LabeledCounterfactualExample`` must be drop-in compatible with every
    consumer typed against the base ``CounterfactualExample``."""

    pytestmark = pytest.mark.property

    def test_labeled_satisfies_base_schema(self, example_pair):
        base, cf = example_pair
        labeled: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": 42,
        }
        # Re-narrow to the base type — consumers that only read input /
        # counterfactual_inputs must not break.
        as_base: CounterfactualExample = labeled  # type: ignore[assignment]
        assert as_base["input"] is base
        assert as_base["counterfactual_inputs"] == [cf]

    @pytest.mark.parametrize(
        "label_value", [0, 1, "string", None, [1, 2, 3], {"k": "v"}]
    )
    def test_label_accepts_arbitrary_types(self, example_pair, label_value):
        # `label: Any` — no runtime constraint.
        base, cf = example_pair
        example: LabeledCounterfactualExample = {
            "input": base,
            "counterfactual_inputs": [cf],
            "label": label_value,
        }
        assert example["label"] == label_value


@pytest.fixture
def simple_chain_model() -> CausalModel:
    """3-node A→B→C ``CausalModel`` from ``tests/_helpers/tiny.py``."""
    return tiny_chain_model()


class TestGenerateCounterfactualSamplesUnit:
    """``generate_counterfactual_samples`` — bounded loop sampler with optional filter."""

    pytestmark = pytest.mark.unit

    def test_basic_generation(self):
        def sampler() -> CounterfactualExample:
            trace = CausalTrace.from_values({"raw_input": "original input", "var": 1})
            cf1 = CausalTrace.from_values({"raw_input": "cf 1", "var": 2})
            cf2 = CausalTrace.from_values({"raw_input": "cf 2", "var": 3})
            return {"input": trace, "counterfactual_inputs": [cf1, cf2]}

        samples = generate_counterfactual_samples(5, sampler)

        assert len(samples) == 5
        for sample in samples:
            assert "input" in sample
            assert "counterfactual_inputs" in sample
            assert sample["input"]["raw_input"] == "original input"
            assert len(sample["counterfactual_inputs"]) == 2

    def test_generation_with_filter(self):
        counter = [0]

        def sampler() -> CounterfactualExample:
            counter[0] += 1
            trace = CausalTrace.from_values(
                {"raw_input": f"input {counter[0]}", "var": counter[0]},
            )
            cf = CausalTrace.from_values(
                {"raw_input": f"cf {counter[0]}", "var": counter[0]},
            )
            return {"input": trace, "counterfactual_inputs": [cf]}

        def filter_fn(sample: CounterfactualExample) -> bool:
            return sample["input"]["var"] % 2 == 0

        samples = generate_counterfactual_samples(3, sampler, filter=filter_fn)

        assert len(samples) == 3
        for sample in samples:
            assert sample["input"]["var"] % 2 == 0


class TestGenerateCounterfactualSamplesProperty:
    """Invariances of the bounded sampler."""

    pytestmark = pytest.mark.property

    @pytest.mark.parametrize("size", [1, 3, 10])
    def test_returned_count_equals_size(self, size):
        def sampler() -> CounterfactualExample:
            trace = CausalTrace.from_values({"raw_input": "x", "var": 1})
            return {"input": trace, "counterfactual_inputs": []}

        samples = generate_counterfactual_samples(size, sampler)
        assert len(samples) == size

    def test_every_sample_satisfies_filter(self):
        def sampler() -> CounterfactualExample:
            value = 1 + len(samples_so_far) % 5
            trace = CausalTrace.from_values({"raw_input": "x", "var": value})
            return {"input": trace, "counterfactual_inputs": []}

        samples_so_far: list = []

        def keep_odd(s: CounterfactualExample) -> bool:
            samples_so_far.append(s)
            return s["input"]["var"] % 2 == 1

        samples = generate_counterfactual_samples(5, sampler, filter=keep_odd)
        for s in samples:
            assert s["input"]["var"] % 2 == 1


class TestSampleInterventionUnit:
    """``sample_intervention`` — random partial intervention over non-input,
    non-output variables, satisfying an optional filter."""

    pytestmark = pytest.mark.unit

    def test_sample_excludes_inputs_and_outputs(self, simple_chain_model):
        # Loop a few times to amortize the random sampling.
        for _ in range(50):
            intervention = sample_intervention(simple_chain_model)
            if intervention:
                for var in intervention:
                    assert var not in simple_chain_model.inputs
                    assert var not in simple_chain_model.outputs
                return
        pytest.skip(
            "sample_intervention produced no non-empty intervention in 50 tries"
        )


class TestSampleInterventionProperty:
    """Invariances of the random intervention sampler."""

    pytestmark = pytest.mark.property

    def test_values_drawn_from_value_lists(self, simple_chain_model):
        # Every sampled value must be in the model's permitted value list.
        for _ in range(50):
            intervention = sample_intervention(simple_chain_model)
            for var, val in intervention.items():
                assert val in simple_chain_model.values[var]


class TestLabelDataWithVariablesUnit:
    """``label_data_with_variables`` — concatenated-target-value → integer label."""

    pytestmark = pytest.mark.unit

    def test_label_keys_and_mapping(self, simple_chain_model):
        model = simple_chain_model
        test_inputs = [{"A": 0}, {"A": 1}, {"A": 0}]
        traces = [model.new_trace(inp) for inp in test_inputs]
        data_list = [{"input": t} for t in traces]

        labeled_dataset, label_mapping = label_data_with_variables(
            model, data_list, ["C"]
        )

        assert len(labeled_dataset) == 3
        assert len(label_mapping) == 2  # C=0 and C=1
        assert "0" in label_mapping
        assert "1" in label_mapping

        labels = [item["label"] for item in labeled_dataset]
        assert labels[0] == labels[2]  # both A=0 -> C=0
        assert labels[0] != labels[1]


class TestLabelDataWithVariablesProperty:
    """Invariances: determinism and order-stability of label assignment."""

    pytestmark = pytest.mark.property

    def test_determinism_same_input_same_labels(self, simple_chain_model):
        # Same input list, called twice, yields identical labels and mapping.
        model = simple_chain_model
        traces = [model.new_trace({"A": v}) for v in [0, 1, 0, 1]]
        data = [{"input": t} for t in traces]

        labeled_a, mapping_a = label_data_with_variables(model, data, ["C"])
        labeled_b, mapping_b = label_data_with_variables(model, data, ["C"])

        assert [d["label"] for d in labeled_a] == [d["label"] for d in labeled_b]
        assert mapping_a == mapping_b


class TestGetPartialFilterUnit:
    """``get_partial_filter`` — closure that checks a setting against a partial spec."""

    pytestmark = pytest.mark.unit

    def test_matches_partial(self, simple_chain_model):
        partial = get_partial_filter({"A": 1, "B": 1})
        assert partial(
            {"A": 1, "B": 1, "C": 1, "raw_input": "...", "raw_output": "..."}
        )

    def test_rejects_mismatch(self, simple_chain_model):
        partial = get_partial_filter({"A": 1, "B": 1})
        assert not partial(
            {"A": 0, "B": 1, "C": 1, "raw_input": "...", "raw_output": "..."}
        )
