"""Check native logits, pair eligibility and protocol mixture reductions."""

import copy
import json

import pytest

from causalab.analysis import hypothesis_metrics as METRICS
from causalab.neural.shared.results import MetricTable
from causalab.protocol.results import Unavailable

pytestmark = pytest.mark.unit


def _save(label, aggregation, read="logits", model="patched"):
    """One protocol-4 save entry: ``aggregation`` over ``read`` on ``model``,
    tabled under ``label``."""
    return {
        "read": read,
        "model": model,
        "aggregation": aggregation,
        "file_path": f"{label}.json",
    }


def _entry(run, label):
    (entry,) = [e for e in run["method"]["save"] if e["file_path"] == f"{label}.json"]
    return entry


def family_score(n, total, target, logit, logit_n):
    return {
        "target": target,
        "alternative": 1 - target if target is not None else None,
        "n": n,
        "total": total,
        "excluded": {},
        "logit_diff": {
            "value": logit,
            "n": logit_n,
            "total": total,
            "eligible_total": logit_n,
            "missing": 0,
            "unknown_eligibility": 0,
            "excluded": {"answer_unchanged": total - logit_n},
        },
    }


def test_mixture_conditions_on_pair_filter_and_metric_eligibility():
    scores = {
        "broad": family_score(100, 100, 1.0, 2.0, 50),
        "narrow": family_score(10, 10, 0.0, -2.0, 10),
    }
    mixed = METRICS.mix_scores(
        scores, {"broad": 0.1, "narrow": 0.9}, {"broad": 100, "narrow": 100}
    )
    assert mixed["target"] == pytest.approx(0.1 / 0.19)
    assert mixed["effective_family_weights"]["narrow"] == pytest.approx(0.09 / 0.19)
    assert mixed["n"] == mixed["total"] == 110
    assert mixed["family_counts"]["narrow"] == {
        "population": 100,
        "selected": 10,
        "scored": 10,
    }
    assert mixed["logit_diff"]["value"] == pytest.approx((0.05 * 2 - 0.09 * 2) / 0.14)
    assert mixed["logit_diff"]["n"] == 60
    assert mixed["logit_diff"]["effective_family_weights"]["broad"] == pytest.approx(
        0.05 / 0.14
    )


def test_empty_eligible_family_has_zero_mass_but_missing_logits_do_not_disappear():
    scores = {
        "broad": family_score(10, 10, 0.0, 0.0, 10),
        "narrow": family_score(10, 10, 1.0, None, 0),
    }
    weights, populations = {"broad": 0.1, "narrow": 0.9}, {"broad": 10, "narrow": 10}
    mixed = METRICS.mix_scores(scores, weights, populations)
    assert mixed["target"] == pytest.approx(0.9)
    assert mixed["logit_diff"]["value"] == 0.0
    assert mixed["logit_diff"]["effective_family_weights"] == {
        "broad": 1.0,
        "narrow": 0.0,
    }
    scores["narrow"]["logit_diff"].update(
        eligible_total=10, missing=10, excluded={"token_logits_not_saved": 10}
    )
    mixed = METRICS.mix_scores(scores, weights, populations)
    assert mixed["logit_diff"]["value"] is None
    assert mixed["logit_diff"]["missing"] == 10
    assert "unavailable" in mixed["logit_diff"]["reason"]
    assert mixed["logit_diff"]["effective_family_weights"]["narrow"] == pytest.approx(
        0.9
    )


def test_zero_weight_and_absent_populations_remain_distinct():
    scores = {
        "broad": family_score(10, 10, 0.0, 0.0, 10),
        "unused": family_score(0, 0, None, None, 0),
    }
    mixed = METRICS.mix_scores(scores, {"broad": 1.0, "unused": 0.0}, {"broad": 10})
    assert mixed["target"] == mixed["logit_diff"]["value"] == 0.0
    mixed = METRICS.mix_scores(scores, {"broad": 0.1, "unused": 0.9}, {"broad": 10})
    assert mixed["target"] is None and mixed["logit_diff"]["value"] is None
    assert "No saved pairs" in mixed["reason"]


def test_reducer_distinguishes_no_pairs_missing_logits_and_true_zero():
    empty = METRICS.score_rows([])
    assert empty["target"] is None and empty["logit_diff"]["missing"] == 0
    row = {"eligible": True, "target_score": 0.0, "alternative_score": 1.0}
    missing = METRICS.score_rows([row])
    assert missing["target"] == 0.0 and missing["logit_diff"]["value"] is None
    assert (
        missing["logit_diff"]["unknown_eligibility"]
        == missing["logit_diff"]["missing"]
        == 1
    )
    row["logit_diff"] = {
        "eligible": True,
        "value": 0.0,
        "population_eligible": True,
        "missing": False,
        "reason_code": None,
    }
    measured = METRICS.score_rows([row])
    assert measured["logit_diff"]["value"] == 0.0 and measured["logit_diff"]["n"] == 1


class Tokenizer:
    def encode(self, text, **kwargs):
        return [{"A": 1, "B": 2, "C": 3}[text.strip()]]


#: The native point and one more point of the same sweep. A saved table holds
#: every point's rows; the coordinate columns place each row on its point.
NATIVE_COORDS = {"layer": 2, "positions.tap": {"index": -1}}
OTHER_COORDS = {"layer": 3, "positions.tap": {"index": -1}}
IDENTITY = {"unit": "logit", "estimand_version": "token_logits/v1"}


def token_logit_table(pairs, coords):
    """The rows the engine writes for ``answers`` at ``coords``: the fourth
    pair's position is missing, so its row is an excluded measurement."""
    table = MetricTable()
    table.add(
        "answers",
        [
            {
                "indices": [1, 2, 3],
                "tokens": ["A", "B", "C"],
                "values": [2.0, -1.0, 2.0],
            }
            if index != 3
            else Unavailable("alignment_missing", "no answer slot", "p3")
            for index in range(len(pairs))
        ],
        coords,
        identity=IDENTITY,
        labels=[pair["example_id"] for pair in pairs],
    )
    return table.rows


@pytest.fixture
def native(tmp_path):
    pairs = [
        {"example_id": f"p{index}", "label": label, "base_answer": original}
        for index, (label, original) in enumerate(
            (("A", "B"), ("B", "B"), ("C", "A"), ("A", "B"))
        )
    ]
    rows = token_logit_table(pairs, NATIVE_COORDS)
    other = token_logit_table(pairs, OTHER_COORDS)
    for row in other:
        if row["eligible"]:
            row["value"] = json.dumps(
                {"indices": [1, 2, 3], "tokens": ["A", "B", "C"], "values": [0.0] * 3}
            )
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(rows + other))
    run = {
        "digest": "point",
        "coords": NATIVE_COORDS,
        "rows": pairs,
        "receipt": {
            "points": [
                {"digest": "point", "coords": NATIVE_COORDS},
                {"digest": "other-point", "coords": OTHER_COORDS},
            ]
        },
        "paths": {"run_dir": str(tmp_path)},
        "method": {
            "save": [
                _save("top1", {"kind": "top_k", "k": 1, "by": "prob"}),
                _save("iia", {"kind": "match", "expected": "label"}),
                _save("answers", {"kind": "token_logits", "tokens": ["A", "B", "C"]}),
            ],
        },
    }
    return run, {"source_metric": "top1"}, path, rows


def test_native_token_logits_keep_changed_zero_and_exclude_unchanged(native):
    run, item, path, _ = native
    paths = []
    values, source = METRICS.read_token_logits(run, item, Tokenizer(), paths)
    assert source["status"] == "available" and paths == [path]
    assert values["p0"]["value"] == 3.0
    assert values["p1"]["reason_code"] == "answer_unchanged"
    assert values["p2"]["value"] == 0.0 and values["p2"]["eligible"] is True
    assert values["p3"]["population_eligible"] is False
    scored = METRICS.score_rows(
        [
            {
                "eligible": True,
                "target_score": 1,
                "alternative_score": 0,
                "logit_diff": value,
            }
            for value in values.values()
        ]
    )
    assert scored["logit_diff"]["value"] == 1.5
    assert scored["logit_diff"]["n"] == scored["logit_diff"]["eligible_total"] == 2
    assert scored["logit_diff"]["excluded"] == {
        "answer_unchanged": 1,
        "alignment_missing": 1,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "point",
        "metric",
        "readout",
        "token_form",
        "ids",
        "infinite",
        "duplicate",
        "pair",
        "coords",
        "eligibility",
    ],
)
def test_relabelled_or_corrupt_native_logits_are_refused(native, defect):
    run, item, path, rows = native
    if defect == "point":
        rows[0]["layer"] = 5
    elif defect == "metric":
        rows[0]["metric"] = "another-metric"
    elif defect == "readout":
        _entry(run, "answers")["read"] = "unpatched-logits"
        item["logit_metric"] = "answers"
    elif defect == "token_form":
        _entry(run, "answers")["aggregation"]["token_form"] = "id"
    elif defect in {"ids", "infinite"}:
        value = json.loads(rows[0]["value"])
        value["indices" if defect == "ids" else "values"][0] = (
            99 if defect == "ids" else float("inf")
        )
        rows[0]["value"] = value
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "pair":
        rows[0]["example_id"] = "unknown"
    elif defect == "coords":
        del rows[0]["positions.tap"]
    else:
        rows[0]["eligible"] = None
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        METRICS.read_token_logits(run, item, Tokenizer(), [])


def test_optional_logits_remain_unavailable_and_partial_rows_keep_reasons(native):
    run, item, path, rows = native
    path.write_text(json.dumps(rows[1:]))
    values, source = METRICS.read_token_logits(run, item, Tokenizer(), [])
    assert source["status"] == "partial"
    assert (
        values["p0"]["population_eligible"] is True and values["p0"]["missing"] is True
    )
    run["method"]["save"].remove(_entry(run, "answers"))
    values, source = METRICS.read_token_logits(run, item, Tokenizer(), [])
    assert source["status"] == "unavailable" and source["metric"] is None
    assert values["p0"]["value"] is None and values["p1"]["missing"] is False


def test_missing_answer_tokens_and_unknown_original_answers_are_not_zero(native):
    run, item, path, rows = native
    _entry(run, "answers")["aggregation"]["tokens"] = ["A", "C"]
    for row in rows:
        row["value"] = {"indices": [1, 3], "values": [2.0, 2.0]}
    path.write_text(json.dumps(rows))
    run["rows"][2]["base_answer"] = None
    values, source = METRICS.read_token_logits(run, item, Tokenizer(), [])
    assert source["status"] == "partial"
    assert values["p0"]["value"] is None and values["p0"]["population_eligible"] is True
    assert values["p0"]["reason_code"] == "answer_token_not_saved"
    assert values["p2"]["value"] is None and values["p2"]["population_eligible"] is None
    assert values["p2"]["reason_code"] == "original_answer_not_saved"


def test_multiple_saved_token_metrics_require_an_explicit_choice(native):
    run, item, _, _ = native
    other = copy.deepcopy(_entry(run, "answers"))
    other["file_path"] = "other.json"
    run["method"]["save"].append(other)
    with pytest.raises(ValueError, match="Specify logit_metric"):
        METRICS.read_token_logits(run, item, Tokenizer(), [])
    item["logit_metric"] = "answers"
    values, _ = METRICS.read_token_logits(run, item, Tokenizer(), [])
    assert values["p0"]["value"] == 3.0
