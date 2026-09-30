"""Test how counterfactual examples are saved, loaded, and displayed."""

import json
import os
import random
import tempfile

import pytest

from causalab.causal import Dom, DomainError, V, mechanism
from causalab.causal.counterfactuals import CounterfactualExample
from causalab.causal.model import CausalModel, CausalTrace
from causalab.io.counterfactuals import (
    deserialize_counterfactual_examples,
    display_counterfactual_examples,
    load_counterfactual_examples,
    save_counterfactual_examples,
)

pytestmark = pytest.mark.unit


def test_loaded_values_recompute_after_an_intervention(tmp_path):
    @mechanism
    def equations(x: Dom(range(3))):
        middle = V(x + 1, domain=Dom(range(10)))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(2 * middle))
        return raw_output

    model = CausalModel(equations)
    path = str(tmp_path / "examples.json")
    save_counterfactual_examples(
        [{"input": model.new_trace({"x": 1}), "counterfactual_inputs": []}], path
    )
    trace = load_counterfactual_examples(path, model)[0]["input"]
    trace["middle"] = 4
    assert trace["raw_output"] == "8"


def test_json_roundtrip_restores_tuple_inputs(tmp_path):
    @mechanism
    def equations(pair: Dom([(1, 2), (3, 4)])):
        result = V(pair[0] + pair[1])
        raw_input = V(str(pair))  # noqa: F841
        raw_output = V(str(result))  # noqa: F841
        return result

    model = CausalModel(equations)
    original = model.new_trace({"pair": (1, 2)})
    path = str(tmp_path / "examples.json")
    save_counterfactual_examples(
        [{"input": original, "counterfactual_inputs": []}], path
    )
    trace = load_counterfactual_examples(path, model)[0]["input"]
    assert trace.to_dict() == original.to_dict()
    trace["pair"] = (3, 4)
    assert trace["raw_output"] == "7"
    with pytest.raises(DomainError):
        model.new_trace({"pair": [1, 2]})


def test_saved_interventions_remain_active_after_loading(tmp_path):
    @mechanism
    def equations(x: Dom(range(5))):
        middle = V(x + 1, domain=Dom(range(10)))
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(2 * middle))
        return raw_output

    model = CausalModel(equations)
    original = model.new_trace({"x": 1})
    original.intervene_many({"x": 2, "middle": 4})
    path = str(tmp_path / "examples.json")
    save_counterfactual_examples(
        [{"input": original, "counterfactual_inputs": [original.copy()]}], path
    )
    example = load_counterfactual_examples(path, model)[0]
    for trace in [example["input"], *example["counterfactual_inputs"]]:
        del trace["x"]
        assert trace["x"] == 2
        trace["x"] = 3
        assert trace["raw_output"] == "8"
        trace["middle"] = 5
        assert trace["raw_output"] == "10"


def test_saved_values_preserve_nested_containers_and_mapping_keys(tmp_path):
    @mechanism
    def equations(value: Dom(tuple)):
        raw_input = V(str(value))  # noqa: F841
        raw_output = V(str(value))
        return raw_output

    model = CausalModel(equations)
    value = ({"tuple": [1], (2, 3): [({"mapping": (4,)},)]}, [5])
    original = model.new_trace({"value": value})
    path = str(tmp_path / "examples.json")
    save_counterfactual_examples(
        [{"input": original, "counterfactual_inputs": []}], path
    )
    trace = load_counterfactual_examples(path, model)[0]["input"]
    assert trace.to_dict() == original.to_dict()


@pytest.mark.parametrize("task", ["entity_binding", "graph_walk", "grouped_arithmetic"])
@pytest.mark.parametrize("legacy", [False, True])
def test_task_json_roundtrip_preserves_values_and_interventions(tmp_path, task, legacy):
    if task == "entity_binding":
        from causalab.tasks.entity_binding.causal_models import CAUSAL_MODEL

        model = CAUSAL_MODEL
    elif task == "graph_walk":
        from causalab.tasks.graph_walk.causal_models import create_causal_model
        from causalab.tasks.graph_walk.config import GraphWalkConfig

        model = create_causal_model(
            GraphWalkConfig(graph_type="ring", graph_size=5, context_length=6)
        )
    else:
        from causalab.tasks.natural_domains_arithmetic.causal_models import (
            create_causal_model,
        )
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

        model = create_causal_model(
            NaturalDomainConfig(
                domain_type="weekdays", number_range=6, number_groups=[[1, 3], [4, 6]]
            )
        )
    original = model.sample_input(rng=random.Random(2))
    donor = model.sample_input(rng=random.Random(3))
    path = tmp_path / "examples.json"
    if legacy:
        path.write_text(
            json.dumps(
                [
                    {
                        "input": original.to_dict(),
                        "counterfactual_inputs": [donor.to_dict()],
                    }
                ]
            )
        )
    else:
        save_counterfactual_examples(
            [{"input": original, "counterfactual_inputs": [donor]}], str(path)
        )
    example = load_counterfactual_examples(str(path), model)[0]
    loaded = example["input"]
    assert loaded.to_dict() == original.to_dict()
    assert example["counterfactual_inputs"][0].to_dict() == donor.to_dict()
    for name in model.inputs:
        expected = original.copy().intervene(name, donor[name])
        actual = loaded.copy().intervene(name, donor[name])
        assert actual.to_dict() == expected.to_dict()


def test_legacy_values_recompute_and_unknown_variables_raise():
    @mechanism
    def equations(x: Dom(range(3))):
        raw_input = V(str(x))  # noqa: F841
        raw_output = V(str(x + 1))
        return raw_output

    model = CausalModel(equations)
    rows = [{"input": {"x": 1, "raw_output": "old cache"}, "counterfactual_inputs": []}]
    trace = deserialize_counterfactual_examples(rows, model)[0]["input"]
    assert trace["raw_output"] == "2"
    trace["x"] = 2
    assert trace["raw_output"] == "3"
    rows[0]["input"]["unknown"] = 1
    with pytest.raises(ValueError, match="Unknown saved variables"):
        deserialize_counterfactual_examples(rows, model)


class TestSaveLoadCounterfactualExamples:
    """Tests for save and load counterfactual examples functions."""

    @pytest.fixture
    def simple_causal_model(self):
        """Create a simple CausalModel for testing."""

        @mechanism
        def equations():
            raw_input = V("test", domain=Dom(str))  # noqa: F841
            var_a = V(0, domain=Dom([0, 1, 2, 3, 10, 20]))
            var_b = V("", domain=Dom(["", "hello", "world", "test", "foo", "bar"]))
            raw_output = V(f"{var_a}_{var_b}", domain=Dom(str))  # noqa: F841
            return var_b

        return CausalModel(equations)

    def test_roundtrip(self, simple_causal_model: CausalModel) -> None:
        """Test saving and loading counterfactual examples."""
        # Create traces using the causal model
        input1 = simple_causal_model.new_trace(
            {"raw_input": "input1", "var_a": 1, "var_b": "hello"}
        )
        cf1_1 = simple_causal_model.new_trace(
            {"raw_input": "cf1_1", "var_a": 2, "var_b": "world"}
        )
        cf1_2 = simple_causal_model.new_trace(
            {"raw_input": "cf1_2", "var_a": 3, "var_b": "test"}
        )
        input2 = simple_causal_model.new_trace(
            {"raw_input": "input2", "var_a": 10, "var_b": "foo"}
        )
        cf2_1 = simple_causal_model.new_trace(
            {"raw_input": "cf2_1", "var_a": 20, "var_b": "bar"}
        )

        examples: list[CounterfactualExample] = [
            {"input": input1, "counterfactual_inputs": [cf1_1, cf1_2]},
            {"input": input2, "counterfactual_inputs": [cf2_1]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_examples.json")

            # Save
            save_counterfactual_examples(examples, path)
            assert os.path.exists(path)

            # Load (returns CausalTrace objects)
            loaded = load_counterfactual_examples(path, simple_causal_model)

            # Verify
            assert len(loaded) == 2

            # Check first example (accessed via CausalTrace)
            assert loaded[0]["input"]["raw_input"] == "input1"
            assert loaded[0]["input"]["var_a"] == 1
            assert loaded[0]["input"]["var_b"] == "hello"
            assert len(loaded[0]["counterfactual_inputs"]) == 2

            # Check second example
            assert loaded[1]["input"]["raw_input"] == "input2"
            assert len(loaded[1]["counterfactual_inputs"]) == 1

            # Verify loaded items are CausalTrace objects
            assert isinstance(loaded[0]["input"], CausalTrace)
            assert isinstance(loaded[0]["counterfactual_inputs"][0], CausalTrace)


class TestDisplayCounterfactualExamplesUnit:
    """``display_counterfactual_examples`` — pretty-printer; returns selected entries."""

    pytestmark = pytest.mark.unit

    def test_display_verbose(self, capsys):
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [
                    {"raw_input": "cf1_1", "var": 2},
                    {"raw_input": "cf1_2", "var": 3},
                ],
            },
            {
                "input": {"raw_input": "input2", "var": 4},
                "counterfactual_inputs": [{"raw_input": "cf2_1", "var": 5}],
            },
        ]

        result = display_counterfactual_examples(examples, num_examples=2)

        captured = capsys.readouterr()
        assert captured.out  # something was printed
        assert "input1" in captured.out
        assert len(result) == 2
        assert 0 in result and 1 in result

    def test_display_quiet(self, capsys):
        examples = [
            {
                "input": {"raw_input": "input1", "var": 1},
                "counterfactual_inputs": [{"raw_input": "cf1", "var": 2}],
            },
        ]

        result = display_counterfactual_examples(examples, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert len(result) == 1

    def test_display_limited_examples(self, capsys):
        examples = [
            {
                "input": {"raw_input": f"input{i}", "var": i},
                "counterfactual_inputs": [],
            }
            for i in range(10)
        ]

        result = display_counterfactual_examples(
            examples, num_examples=3, verbose=False
        )
        assert len(result) == 3
