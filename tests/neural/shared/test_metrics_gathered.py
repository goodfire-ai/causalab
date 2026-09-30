"""A gathered metric is ``compute_metric`` to the bit, without the vocabulary
leaving the device (``metrics.GATHERED_KINDS``).

``logit_diff``, ``soft_accuracy`` and ``token_logit`` read the projection at
the answer token ids and nowhere else; selecting those entries first and
upcasting after is the same fp32 value as upcasting the whole row and
selecting, and the per-example operation is the same 0-d float op, so
``gathered_metric`` reproduces ``compute_metric`` entry for entry — excluded
rows included — over random bf16 logits. The kinds with a softmax, a
log-sum-exp or an argmax over the vocabulary are not in the set.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.shared.metrics import (
    GATHERED_KINDS,
    compute_metric,
    gathered_metric,
)
from causalab.protocol.answers import metric_token_ids
from causalab.protocol.results import Unavailable
from causalab.protocol.schema import METRIC_KINDS

pytestmark = pytest.mark.property


class _Vocabulary:
    """What ``token_form: id`` needs of a tokenizer: its size."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _metric(kind: str) -> Any:
    fields = {"token": "tok"} if kind == "token_logit" else {"a": "ans_a", "b": "ans_b"}
    return SimpleNamespace(kind=kind, fields=fields, token_form="id", of="logits")


@st.composite
def _cases(draw: Any) -> tuple[str, torch.Tensor, list[dict[str, Any]], int]:
    kind = draw(st.sampled_from(sorted(GATHERED_KINDS)))
    rows = draw(st.integers(1, 6))
    vocab = draw(st.integers(2, 40))
    values = draw(
        st.lists(
            st.lists(
                st.floats(-40.0, 40.0, allow_nan=False, allow_infinity=False),
                min_size=vocab,
                max_size=vocab,
            ),
            min_size=rows,
            max_size=rows,
        )
    )
    logits = torch.tensor(values).to(torch.bfloat16).unsqueeze(1)  # (rows, 1, vocab)
    table = []
    for _ in range(rows):
        row: dict[str, Any] = {
            "tok": draw(st.integers(0, vocab - 1)),
            "ans_a": draw(st.integers(0, vocab - 1)),
            "ans_b": draw(st.integers(0, vocab - 1)),
        }
        if draw(st.booleans()) and draw(st.booleans()):
            # an excluded row: no answer in one of the metric's columns
            row[draw(st.sampled_from(sorted(row)))] = None
        table.append(row)
    return kind, logits, table, vocab


@given(case=_cases())
@settings(max_examples=300, deadline=None)
def test_a_gathered_metric_is_compute_metric_entry_for_entry(case) -> None:
    kind, logits, rows, vocab = case
    metric = _metric(kind)
    tokenizer = _Vocabulary(vocab)
    want = compute_metric(metric, logits, rows, tokenizer)
    got = gathered_metric(metric, logits, rows, tokenizer)
    assert len(got) == len(want) == len(rows)
    for a, b in zip(got, want, strict=True):
        if isinstance(b, Unavailable):
            assert isinstance(a, Unavailable)
            assert a.denominator_key == b.denominator_key
        else:
            assert isinstance(a, float) and a == b  # to the bit, no tolerance
    # resolved once, handed in: the same again
    ids = metric_token_ids(metric, rows, tokenizer)
    assert gathered_metric(metric, logits, rows, tokenizer, token_ids=ids) == got


def test_the_gathered_kinds_are_metric_kinds_that_only_select() -> None:
    assert GATHERED_KINDS <= set(METRIC_KINDS)
    # a softmax, a log-sum-exp or an argmax over the vocabulary is not a selection
    assert not GATHERED_KINDS & {
        "cross_entropy",
        "kl",
        "js",
        "match",
        "top_k",
        "class_probs",
    }
    with pytest.raises(ValueError):
        gathered_metric(
            _metric("cross_entropy"), torch.zeros(1, 1, 4), [{}], _Vocabulary(4)
        )


def test_the_ids_are_resolved_over_the_kept_rows_only() -> None:
    metric = _metric("logit_diff")
    rows = [
        {"ans_a": 1, "ans_b": 2},
        {"ans_a": None, "ans_b": 2},
        {"ans_a": 3, "ans_b": 0},
    ]
    ids = metric_token_ids(metric, rows, _Vocabulary(4))
    assert ids == {"a": [1, 3], "b": [2, 0]}
    logits = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
    got = gathered_metric(metric, logits, rows, _Vocabulary(4), token_ids=ids)
    assert got[0] == 1.0 - 2.0 and got[2] == 11.0 - 8.0
    assert isinstance(got[1], Unavailable)
