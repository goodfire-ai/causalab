"""T1 without a model (spec §2.10 "Eligibility", §4.1): a cohort
with *k* of *n* rows structurally unobservable — here, rows the table carries
no answer for — produces *n − k* scored values and *k* typed ``unavailable``
values in their places, and the table row of each excluded measurement
carries ``eligible: false`` with its reason code.

*Mutation:* score ``None`` as the string ``"None"`` (what a bare ``str`` over
the column did) or take the mean over *n* — both fail
`test_t1_k_of_n_rows_without_an_answer_are_excluded_measurements`.

Module-level imports are names that existed before this PR on purpose, so the
fails-without witness on the base export fails *behaviourally* — a value over
*n* rows where *n − k* is right — and not by an ``ImportError`` at collection.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.metrics import compute_metric
from causalab.neural.shared.results import MetricTable
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.results import Unavailable
from causalab.protocol.schema import AggregationSpec

pytestmark = pytest.mark.unit

VOCAB = 16


class FakeTokenizer:
    """One id per distinct string, deterministic, single-token always."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text not in self._ids:
            self._ids[text] = len(self._ids) + 1
        return [self._ids[text]]

    def decode(self, ids: list[int]) -> str:
        by_id = {v: k for k, v in self._ids.items()}
        return "".join(by_id.get(int(i), "?") for i in ids)


def _token_logit(**extra: Any) -> AggregationSpec:
    return AggregationSpec(
        kind="token_logit",
        fields={"token": "entity"},
        **extra,
    )


def _logits(n: int) -> torch.Tensor:
    # (n, 1, VOCAB): row i holds i*VOCAB + j at entry j, so every value is
    # distinct and a wrong row or a wrong token is visible in the number
    return torch.arange(n * VOCAB, dtype=torch.float32).reshape(n, 1, VOCAB)


ROWS = [
    {"entity": "one"},
    {"entity": None},  # no answer: an excluded measurement
    {"entity": "three"},
    {"entity": "four"},
]


def test_t1_k_of_n_rows_without_an_answer_are_excluded_measurements():
    tok = FakeTokenizer()
    values = compute_metric(_token_logit(), _logits(4), ROWS, tok)
    scored = [v for v in values if isinstance(v, float)]
    excluded = [v for v in values if isinstance(v, Unavailable)]
    # n − k values, k excluded — not n values with "None" scored as a token
    assert (len(scored), len(excluded)) == (3, 1), values
    assert isinstance(values[1], Unavailable)
    assert values[1].reason == "alignment_missing"
    assert "row 1" in values[1].detail and "'entity'" in values[1].detail
    assert values[1].denominator_key == "token_logit"  # the kind, uncoordinated
    # the eligible rows are scored at their own row of the read
    want = [
        float(_logits(4)[i, 0, tok.encode(ROWS[i]["entity"])[0]]) for i in (0, 2, 3)
    ]
    assert scored == want
    # the mean is over the eligible rows only
    mean_eligible = sum(scored) / 3
    assert mean_eligible != sum(scored) / 4  # the mutation: a mean over n


def test_t1_the_table_row_of_an_excluded_measurement_carries_the_record():
    tok = FakeTokenizer()
    values = compute_metric(_token_logit(), _logits(4), ROWS, tok)
    table = MetricTable()
    identity = metric_record_identity("token_logit", unit=None, estimand_version=None)
    table.add("tl", values, {}, identity=identity)
    assert [row["eligible"] for row in table.rows] == [True, False, True, True]
    excluded = table.rows[1]
    assert excluded["value"] is None
    assert excluded["reason_code"] == "alignment_missing"
    # an eligible row records nothing but `eligible: true` (§4.1: an
    # available cell records nothing new)
    for row in (table.rows[0], table.rows[2], table.rows[3]):
        assert "reason_code" not in row
        assert isinstance(row["value"], float)
    # the pre-existing columns are untouched
    assert set(table.rows[0]) == {
        "example_id",
        "metric",
        "value",
        "unit",
        "estimand_version",
        "eligible",
    }


def test_twin_a_table_with_every_answer_scores_every_row():
    tok = FakeTokenizer()
    rows = [{"entity": a} for a in ("one", "two", "three", "four")]
    values = compute_metric(_token_logit(), _logits(4), rows, tok)
    assert all(isinstance(v, float) for v in values) and len(values) == 4
    table = MetricTable()
    identity = metric_record_identity("token_logit", unit=None, estimand_version=None)
    table.add("tl", values, {}, identity=identity)
    assert all(row["eligible"] is True for row in table.rows)
    assert not any("reason_code" in row for row in table.rows)


def test_the_denominator_key_is_the_callers_cell_key():
    from causalab.protocol.answers import excluded_rows

    excluded = excluded_rows(_token_logit(), ROWS, "tl[layer=3]")
    assert set(excluded) == {1}
    assert excluded[1].denominator_key == "tl[layer=3]"
    values = compute_metric(
        _token_logit(), _logits(4), ROWS, FakeTokenizer(), denominator_key="tl[layer=3]"
    )
    assert values[1].denominator_key == "tl[layer=3]"


def test_an_empty_form_group_is_an_excluded_row_not_a_refusal():
    from causalab.protocol.answers import excluded_rows

    metric = AggregationSpec(kind="match", fields={"expected": "entity"})
    rows = [{"entity": ["one", "uno"]}, {"entity": []}, {}]
    excluded = excluded_rows(metric, rows, "iia")
    assert set(excluded) == {1, 2}
    assert all(cell.reason == "alignment_missing" for cell in excluded.values())
    values = compute_metric(metric, _logits(3), rows, FakeTokenizer())
    assert isinstance(values[0], float)
    assert isinstance(values[1], Unavailable) and isinstance(values[2], Unavailable)


def test_every_row_excluded_returns_only_unavailables_and_raises_nothing():
    rows = [{"entity": None}, {"entity": None}]
    values = compute_metric(_token_logit(), _logits(2), rows, FakeTokenizer())
    assert all(isinstance(v, Unavailable) for v in values) and len(values) == 2


def test_a_kind_naming_no_column_excludes_nothing():
    from causalab.protocol.answers import excluded_rows

    metric = AggregationSpec(kind="top_k", fields={"k": 2, "by": "value"})
    assert excluded_rows(metric, ROWS, "tk") == {}


def test_a_windowed_metric_carries_the_record_per_position():
    """An unmatched continuation row is excluded under ``alignment_missing``
    with the ``null`` value it always had; a matched one is eligible."""
    table = MetricTable()
    identity = metric_record_identity("match", unit=None, estimand_version=None)
    table.add_windowed(
        "said",
        [[1.0, 0.0], []],
        {},
        identity=identity,
        steps=[[0, 1], []],
        matched=[True, False],
    )
    assert [row["eligible"] for row in table.rows] == [True, True, False]
    assert table.rows[2]["reason_code"] == "alignment_missing"
    assert table.rows[2]["value"] is None and table.rows[2]["matched"] is False
    assert not any("reason_code" in row for row in table.rows[:2])
