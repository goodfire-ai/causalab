"""Vocabulary queries are per metric call, including grouped match targets."""

import pytest
import torch

from causalab.neural.shared.metrics import compute_metric
from causalab.protocol.answers import (
    column_token_id,
    column_token_ids,
    restrict_token_ids,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import AggregationSpec

pytestmark = pytest.mark.unit


class Vocabulary:
    def __init__(self, size=8):
        self.size = size
        self.calls = 0

    def __len__(self):
        self.calls += 1
        return self.size


def test_metric_reuses_bound_across_rows_and_columns():
    tokenizer = Vocabulary()
    metric = AggregationSpec(
        kind="logit_diff", fields={"a": "a", "b": "b"}, token_form="id"
    )
    logits = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    rows = [{"a": 7, "b": 2}] * 8
    assert compute_metric(metric, logits, rows, tokenizer) == [5.0] * 8
    assert tokenizer.calls == 1


def test_grouped_match_reuses_bound_and_preserves_scores():
    tokenizer = Vocabulary()
    metric = AggregationSpec(
        kind="match", fields={"expected": "answer"}, token_form="id"
    )
    logits = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    rows = [{"answer": [1, 7]}, {"answer": [2, 3]}] * 4
    assert compute_metric(metric, logits, rows, tokenizer) == [1.0, 0.0] * 4
    assert tokenizer.calls == 1


@pytest.mark.parametrize("value", [-1, 8, True, 1.0, "1"])
def test_scalar_and_column_keep_integer_bounds(value):
    tokenizer = Vocabulary()
    for resolve, argument in [(column_token_id, value), (column_token_ids, [value])]:
        with pytest.raises(ProtocolError, match="integer token ID"):
            resolve(tokenizer, argument, token_form="id")


def test_next_call_sees_added_tokens():
    tokenizer = Vocabulary()
    with pytest.raises(ProtocolError):
        column_token_ids(tokenizer, [8], token_form="id")
    tokenizer.size = 9
    assert column_token_ids(tokenizer, [8] * 100, token_form="id") == [8] * 100
    assert tokenizer.calls == 2


def test_tokenizer_bound_is_not_model_output_width():
    tokenizer = Vocabulary(9)
    metric = AggregationSpec(
        kind="token_logit", fields={"token": "answer"}, token_form="id"
    )
    # A tokenizer-added ID is valid for tokenization but does not invent a model row.
    with pytest.raises(IndexError):
        compute_metric(metric, torch.zeros(1, 8), [{"answer": 8}], tokenizer)


def test_js_restrict_takes_integer_ids_and_the_metric_call_queries_once():
    tokenizer = Vocabulary()
    metric = AggregationSpec(
        kind="js",
        fields={"target": "other", "restrict": "answers"},
        token_form="id",
    )
    logits = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    rows = [{"answers": [1, 7]}, {"answers": [2, 3]}] * 8
    # identical distributions: the divergence is zero whatever the restriction,
    # so the assertion is about resolution (integers accepted) and the query count
    values = compute_metric(metric, logits, rows[:8], tokenizer, target_value=logits)
    assert values == pytest.approx([0.0] * 8, abs=1e-6)
    assert tokenizer.calls == 1


def test_js_restrict_literal_list_and_the_objective_path_query_once():
    tokenizer = Vocabulary()
    metric = AggregationSpec(
        kind="js",
        fields={"target": "other", "restrict": [1, 7]},
        token_form="id",
    )
    rows = [{"x": 0}] * 32
    # the training objective resolves without an enclosing snapshot: one query, not one per row
    assert restrict_token_ids(metric, rows, tokenizer) == [[1, 7]] * 32
    assert tokenizer.calls == 1


@pytest.mark.parametrize("bad", [[1, 8], [1, "7"], [1, True]])
def test_js_restrict_keeps_the_integer_bound(bad):
    tokenizer = Vocabulary()
    metric = AggregationSpec(
        kind="js",
        fields={"target": "other", "restrict": bad},
        token_form="id",
    )
    with pytest.raises(ProtocolError, match="integer token ID"):
        restrict_token_ids(metric, [{"x": 0}], tokenizer)
