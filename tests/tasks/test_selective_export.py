"""Scoring and explicit exports request only the lazy values they consume."""

import pytest

from causalab.causal import CausalModel, Dom, V, mechanism
from causalab.causal.scoring import ScoringSpec
from causalab.tasks.serialize import serialize_examples

pytestmark = pytest.mark.unit


def _model(*, scoring=None):
    @mechanism
    def equations(x: Dom([0, 1])):
        bad = V(1 // x, domain=Dom(int), lazy=True)  # noqa: F841
        intermediate = V(x + 1, lazy=True)
        answer = V(x, lazy=True)
        extra = V(intermediate + 1, lazy=True)  # noqa: F841
        raw_input = V(str(x), domain=Dom(str))  # noqa: F841
        raw_output = V(str(x), domain=Dom(str))  # noqa: F841
        return answer

    return CausalModel(equations, scoring=scoring)


def _examples(model):
    return [
        {
            "input": model.new_trace({"x": 0}),
            "counterfactual_inputs": [model.new_trace({"x": 1})],
        }
    ]


def test_snapshot_requested_values_include_dependencies_without_mutating_source():
    trace = _model().new_trace({"x": 0})
    original = trace.snapshot()
    exported = trace.snapshot(required=["extra"])
    assert exported["extra"] == 2
    assert exported["intermediate"] == 1
    assert "bad" not in exported
    assert trace.snapshot() == original
    with pytest.raises(ZeroDivisionError):
        trace.snapshot(required=["extra", "bad"])
    assert trace.snapshot() == original
    with pytest.raises(TypeError, match="sequence"):
        trace.snapshot(required="extra")


def test_label_settings_export_explicit_extras_as_plain_dicts():
    model = _model()
    examples = _examples(model)
    result = model.label_counterfactual_data(
        examples, ["x"], setting_variables=["answer", "extra"]
    )[0]
    assert isinstance(result["setting"], dict)
    assert result["label"] == "1"
    assert result["setting"]["answer"] == 1
    assert result["setting"]["extra"] == 3
    assert "bad" not in result["setting"]
    assert "extra" not in examples[0]["input"]


def test_selected_lazy_scoring_answer_and_extras_are_exported():
    model = _model(
        scoring=ScoringSpec(
            forms={"answer": {0: ["zero"], 1: ["one"]}},
            answer_variable="answer",
        )
    )
    examples = _examples(model)
    exported = serialize_examples(
        model,
        examples,
        split="train",
        target_variables=["x"],
        extra_variables=["extra"],
    )
    row = exported.rows[0]
    assert row["label_forms"] == ["one"]
    assert row["base_answer_forms"] == ["zero"]
    assert row["cf_answer_forms"] == ["one"]
    assert row["answer"] == "0"
    assert row["extra"] == "2"
    assert row["counterfactual_inputs_variables"][0]["answer"] == "1"
    assert row["counterfactual_inputs_variables"][0]["extra"] == "3"
    assert "bad" not in row
    assert "bad" not in row["counterfactual_inputs_variables"][0]
    assert "answer" not in examples[0]["input"]


def test_answer_override_does_not_force_the_unused_default_scoring_variable():
    model = _model(
        scoring=ScoringSpec(
            forms={"bad": {1: ["unused"]}, "answer": {0: ["zero"], 1: ["one"]}},
            answer_variable="bad",
        )
    )
    result = serialize_examples(
        model,
        _examples(model),
        split="train",
        target_variables=["x"],
        answer_variable="answer",
    )
    assert result.answer_variable == "answer"
    assert result.rows[0]["label_forms"] == ["one"]
    assert "bad" not in result.rows[0]


def test_explicit_extra_must_be_available_on_each_side():
    model = _model()
    with pytest.raises(ZeroDivisionError):
        serialize_examples(
            model,
            _examples(model),
            split="train",
            target_variables=["x"],
            extra_variables=["bad"],
        )
