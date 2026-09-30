"""Exact-pair handoff: reordering, wrong answers, and unavailable evidence."""

import json
import runpy
from pathlib import Path

import pytest

from causalab.analysis.compare_hypotheses import compare_saved_outputs
from causalab.analysis.hypothesis_artifacts import export_hypotheses
from causalab.io.tables import read_table

pytestmark = pytest.mark.unit


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [int(text.strip())]


def records():
    pairs = [
        dict(
            example_id=str(i),
            pair_id=f"p{i}",
            family="broad",
            split="test",
        )
        for i in range(4)
    ]
    predictions = [
        dict(pair, hypothesis_id=name, answer_forms=[str(value)])
        for pair, values in zip(pairs, [(1, 0), (1, 1), (0, 1), (0, 1)])
        for name, value in zip(["target", "alternative"], values)
    ]
    neural = [
        dict(
            example_id=str(i),
            metric="top1",
            value=json.dumps({"indices": [value]}),
            eligible=True,
        )
        for i, value in enumerate([1, 1, 0, 9])
    ]
    return pairs, predictions, neural


def compare(pairs, predictions, neural):
    return compare_saved_outputs(
        pairs,
        predictions,
        neural,
        tokenizer=Tokenizer(),
        target="target",
        alternatives=["alternative"],
        metric="top1",
    )


def test_four_pairs_preserve_agreement_and_distinguishing_denominators():
    pairs, predictions, neural = records()
    rows = compare(pairs, predictions[::-1], neural[::-1])
    assert sum(r["target_score"] for r in rows) / 4 == 0.75
    assert sum(r["alternative_score"] for r in rows) / 4 == 0.25
    assert sum(r["value"] for r in rows) / 4 == 0.5
    different = [r for r in rows if r["distinguishing"]]
    assert len(different) == 3
    assert sum(r["target_score"] for r in different) / 3 == 2 / 3
    assert rows[0]["neural_token_id"] == 9  # Outside the answer vocabulary.
    assert rows[0]["target_score"] == rows[0]["alternative_score"] == 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "identity", "null_id"])
def test_bad_handoffs_are_refused(defect):
    pairs, predictions, neural = records()
    if defect == "missing":
        neural.pop()
    elif defect == "duplicate":
        neural.append(neural[0])
    elif defect == "identity":
        predictions[0]["pair_id"] = "other"
    else:
        pairs[0]["example_id"] = None
    with pytest.raises(ValueError):
        compare(pairs, predictions, neural)


def test_points_are_their_coordinate_columns():
    """Two sweep points share one pair table; each is keyed by its coordinates
    and each must cover the table on its own."""
    pairs, predictions, neural = records()
    swept = [
        {**row, "sites.target.layers": layer} for layer in (4, 8) for row in neural
    ]
    rows = compare(pairs, predictions, swept)
    assert len(rows) == 8
    assert {row["sites.target.layers"] for row in rows} == {4, 8}
    with pytest.raises(ValueError, match="duplicate neural output"):
        compare(pairs, predictions, [*swept, swept[-1]])
    with pytest.raises(ValueError, match="cover the exact pair table"):
        compare(pairs, predictions, swept[:-1])


def test_unavailable_output_retains_its_reason_and_identity():
    pairs, predictions, neural = records()
    neural[0].update(eligible=False, value=None, reason_code="alignment_missing")
    row = compare(pairs, predictions, neural)[0]
    assert row["value"] is None
    assert row["eligible"] is False
    assert row["reason_code"] == "alignment_missing"
    assert row["pair_id"] == "p0"


def test_indistinguishable_predictions_are_identified_without_a_false_gap():
    pairs, predictions, neural = records()
    for index in range(0, len(predictions), 2):
        predictions[index + 1]["answer_forms"] = predictions[index]["answer_forms"]
    rows = compare(pairs, predictions, neural)
    assert not any(row["distinguishing"] for row in rows)
    assert all(row["value"] == 0 for row in rows)


def test_export_to_saved_neural_comparison(tmp_path):
    model_module = runpy.run_path(
        str(Path(__file__).parents[2] / "demos/hypothesis_testing/models.py")
    )
    examples = [
        {
            "input": {"a": a, "b": b, "context": i * 2},
            "counterfactual_inputs": [{"a": da, "b": db, "context": i * 2 + 1}],
        }
        for i, (a, b, da, db) in enumerate(
            [(0, 0, 1, 0), (0, 0, 1, 1), (1, 0, 0, 0), (1, 0, 0, 0)]
        )
    ]
    export_hypotheses(
        model_module["MODELS"],
        model_module["HYPOTHESES"],
        ["swap_a"],
        {"broad": examples},
        {"broad": {"family": "broad", "split": "test"}},
        tmp_path,
    )
    pairs = read_table(tmp_path / "swap_a/pairs.json")
    predictions = read_table(tmp_path / "swap_a/predictions.json")
    assert pairs[1]["label"] == "1"
    assert pairs[1]["cf_answer"] == "2"
    neural = [
        dict(
            example_id=p["example_id"],
            metric="top1",
            value=json.dumps({"indices": [value]}),
            eligible=True,
        )
        for p, value in zip(pairs, [1, 1, 0, 9])
    ]
    rows = compare_saved_outputs(
        pairs,
        predictions[::-1],
        neural[::-1],
        tokenizer=Tokenizer(),
        target="swap_a",
        alternatives=["swap_b"],
        metric="top1",
    )
    assert sum(row["value"] for row in rows) / 4 == 0.5
    assert sum(row["distinguishing"] for row in rows) == 3
