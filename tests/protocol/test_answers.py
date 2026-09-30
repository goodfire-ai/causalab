"""``answers.metric_token_ids`` is the one answer resolution
(``causalab/protocol/answers.py``): the run door's pass before the weights
(``pipeline.resolve_answers``) and every score path call it, so a value the
door accepts is a value the score resolves to the same id, and a refusal
reads the same wherever it is raised.

On the base the function lived in ``neural/shared/metrics.py``, served the
three gathered kinds only and raised ``ValueError`` for every other kind,
which left the door with no single function to call.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from causalab.neural.shared.metrics import (
    compute_metric,
    compute_windowed_metric,
    gathered_metric,
    matched_metric,
)
from causalab.protocol.answers import metric_token_ids, names_answers
from causalab.protocol.results import Unavailable
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import AggregationSpec
from causalab.protocol.schema.types import METRIC_KINDS

pytestmark = pytest.mark.numerical_unit

ROWS: list[dict[str, Any]] = [
    {
        "a": " Monday",
        "b": " Friday",
        "forms": [" Monday", "Monday"],
        "set": [" Monday", " Friday"],
    },
    {
        "a": " Friday",
        "b": " Sunday",
        "forms": [" Friday"],
        "set": [" Sunday", " Friday"],
    },
    {"a": None, "b": " Sunday", "forms": [], "set": []},  # carries no answer
]


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def _id(tokenizer: Any, text: str) -> int:
    (token,) = tokenizer.encode(text, add_special_tokens=False)
    return token


@pytest.mark.parametrize(
    ("kind", "fields", "expected"),
    [
        (
            "logit_diff",
            {"a": "a", "b": "b"},
            {"a": [" Monday", " Friday"], "b": [" Friday", " Sunday"]},
        ),
        (
            "soft_accuracy",
            {"a": "a", "b": "b"},
            {"a": [" Monday", " Friday"], "b": [" Friday", " Sunday"]},
        ),
        ("token_logit", {"token": "a"}, {"token": [" Monday", " Friday"]}),
        ("cross_entropy", {"target": "a"}, {"target": [" Monday", " Friday"]}),
        (
            "class_probs",
            {"groups": {"end": [" Friday", " Sunday"]}},
            {"groups.end": [" Friday", " Sunday"]},
        ),
        (
            "token_logits",
            {"tokens": [" Monday", " Sunday"]},
            {"tokens": [" Monday", " Sunday"]},
        ),
    ],
)
def test_the_ids_are_the_answers_as_written(
    kind: str, fields: dict[str, Any], expected: dict[str, list[str]], tokenizer: Any
) -> None:
    """Per-row fields over the rows that carry an answer (the third row is
    excluded); literal fields once for the run."""
    got = metric_token_ids(AggregationSpec(kind=kind, fields=fields), ROWS, tokenizer)
    assert got == {
        field: [_id(tokenizer, text) for text in texts]
        for field, texts in expected.items()
    }


def test_match_credits_each_rows_forms(tokenizer: Any) -> None:
    got = metric_token_ids(
        AggregationSpec(kind="match", fields={"expected": "forms"}), ROWS, tokenizer
    )
    assert got == {
        "expected": [
            {_id(tokenizer, " Monday"), _id(tokenizer, "Monday")},
            {_id(tokenizer, " Friday")},
        ]
    }


def test_a_restricted_js_resolves_each_rows_set(tokenizer: Any) -> None:
    metric = AggregationSpec(kind="js", fields={"target": "unused", "restrict": "set"})
    got = metric_token_ids(metric, ROWS, tokenizer)
    assert got == {
        "restrict": [
            [_id(tokenizer, " Monday"), _id(tokenizer, " Friday")],
            [_id(tokenizer, " Sunday"), _id(tokenizer, " Friday")],
        ]
    }


@pytest.mark.parametrize(
    ("kind", "fields"),
    [
        ("kl", {"target": "unused"}),
        ("js", {"target": "unused"}),
        ("top_k", {"k": 2, "by": "value"}),
        ("decode", {}),
    ],
)
def test_a_kind_that_names_no_answer_resolves_nothing(
    kind: str, fields: dict[str, Any], tokenizer: Any
) -> None:
    metric = AggregationSpec(kind=kind, fields=fields)
    assert metric_token_ids(metric, ROWS, tokenizer) == {}


def test_the_score_uses_the_ids_the_door_resolved(tokenizer: Any) -> None:
    """One-hot logits at the resolved ids: ``match`` scores 1 on every row
    that carries an answer, and ``logit_diff`` reads exactly the two
    entries."""
    rows = ROWS[:2]
    ids = metric_token_ids(
        AggregationSpec(kind="logit_diff", fields={"a": "a", "b": "b"}), rows, tokenizer
    )
    logits = torch.zeros(len(rows), 1, len(tokenizer))
    for i, (a, b) in enumerate(zip(ids["a"], ids["b"])):
        logits[i, 0, a] = 3.0
        logits[i, 0, b] = 1.0
    margins = compute_metric(
        AggregationSpec(kind="logit_diff", fields={"a": "a", "b": "b"}),
        logits,
        rows,
        tokenizer,
    )
    assert margins == [2.0, 2.0]
    matched = compute_metric(
        AggregationSpec(kind="match", fields={"expected": "a"}), logits, rows, tokenizer
    )
    assert matched == [1.0, 1.0]


def test_the_refusal_reads_the_same_at_the_door_and_the_score(tokenizer: Any) -> None:
    """``'Tiffany'`` is three gpt2 tokens. The door and the score raise one
    message, because they run one function."""
    rows = [{"a": "Tiffany", "b": " Sean"}]
    metric = AggregationSpec(kind="logit_diff", fields={"a": "a", "b": "b"})
    with pytest.raises(ProtocolError) as door:
        metric_token_ids(metric, rows, tokenizer)
    with pytest.raises(ProtocolError) as score:
        compute_metric(metric, torch.zeros(1, 1, len(tokenizer)), rows, tokenizer)
    assert str(door.value) == str(score.value)
    assert "metric logit_diff.a" in str(door.value)


def test_an_excluded_row_is_not_resolved(tokenizer: Any) -> None:
    """The row with no answer is an excluded measurement at the score, and
    the door does not resolve it either."""
    metric = AggregationSpec(kind="token_logit", fields={"token": "a"})
    scored = compute_metric(metric, torch.zeros(3, 1, len(tokenizer)), ROWS, tokenizer)
    assert isinstance(scored[2], Unavailable)
    assert len(metric_token_ids(metric, ROWS, tokenizer)["token"]) == 2


#: one minimal spec per metric kind, and a second for ``js``, restricted.
#: `test_every_kind_has_a_minimal_spec` holds this to ``METRIC_KINDS``, so a
#: new kind needs an entry here before the census below can pass.
MINIMAL: dict[str, list[dict[str, Any]]] = {
    "logit_diff": [{"a": "a", "b": "b"}],
    "soft_accuracy": [{"a": "a", "b": "b"}],
    "token_logit": [{"token": "a"}],
    "cross_entropy": [{"target": "a"}],
    "kl": [{"target": "unused"}],
    "js": [{"target": "unused"}, {"target": "unused", "restrict": "set"}],
    "class_probs": [{"groups": {"end": [" Friday", " Sunday"]}}],
    "token_logits": [{"tokens": [" Monday"]}],
    "top_k": [{"k": 2, "by": "value"}],
    "match": [{"expected": "forms"}],
    "decode": [{}],
}


def test_every_kind_has_a_minimal_spec() -> None:
    assert set(MINIMAL) == set(METRIC_KINDS)


@pytest.mark.parametrize(
    ("kind", "fields"),
    [(kind, fields) for kind, specs in MINIMAL.items() for fields in specs],
)
def test_a_kind_names_answers_exactly_when_it_resolves_some(
    kind: str, fields: dict[str, Any], tokenizer: Any
) -> None:
    """``names_answers`` decides whether the run door loads a tokenizer for
    a metric, and ``metric_token_ids`` is what the door checks. A kind that
    says it names answers but resolves none would be checked by nothing."""
    metric = AggregationSpec(kind=kind, fields=fields)
    assert bool(metric_token_ids(metric, ROWS, tokenizer)) == names_answers(metric)


def test_every_failing_value_of_every_field_is_named(tokenizer: Any) -> None:
    """Two values of ``a`` and one of ``b`` split under gpt2. One refusal
    has a line per field; the ``a`` line counts both values and names the
    first row, so one fix-and-rerun loop clears the table."""
    rows = [
        {"a": " Tiffanyx", "b": " Kevin"},
        {"a": " Seanathan", "b": "Kevinoss"},
        {"a": " Jennifer", "b": " Kevin"},
    ]
    metric = AggregationSpec(kind="logit_diff", fields={"a": "a", "b": "b"})
    with pytest.raises(ProtocolError) as err:
        metric_token_ids(metric, rows, tokenizer, row_numbers=[4, 5, 6])
    a, b = err.value.message.splitlines()
    assert a.startswith("metric logit_diff.a: metric column value ' Tiffanyx' (row 4)")
    assert "2 of 3 values fail this way, among them ' Seanathan'" in a, a
    assert b.startswith("metric logit_diff.b: metric column value 'Kevinoss' (row 5)")
    assert "fail this way" not in b, b


@pytest.mark.parametrize("score", [compute_metric, gathered_metric, matched_metric])
def test_the_score_names_no_row_when_it_cannot_know_the_table_row(
    score: Any, tokenizer: Any
) -> None:
    """Row 0 carries no answer, so the score reduces row 1 alone. The score
    is handed a batch of base rows, whose table rows it does not know, so
    its refusal names the value and no row. On the base it named a position
    as the row: ``(row 0)``, the position among the rows it resolved, from
    ``compute_metric`` and ``matched_metric``, and ``(row 1)``, the position
    in the batch, from ``gathered_metric``."""
    kind, fields = (
        ("match", {"expected": "a"})
        if score is matched_metric
        else ("token_logit", {"token": "a"})
    )
    rows = [{"a": None}, {"a": "Tiffany"}]
    with pytest.raises(ProtocolError) as err:
        score(
            AggregationSpec(kind=kind, fields=fields),
            torch.zeros(2, 1, len(tokenizer)),
            rows,
            tokenizer,
        )
    text = err.value.message
    assert f"metric {kind}." in text and "'Tiffany'" in text, text
    assert "(row" not in text, text


def test_a_windowed_refusal_counts_rows_not_positions(tokenizer: Any) -> None:
    """Each of three rows addresses three positions, and two rows hold a split
    answer. The refusal counts 2 of 3 rows. On the base it counted every
    addressed position, 6 of 9, and named the first position as a row."""
    rows = [{"t": "Tiffany"}, {"t": "Seanathan"}, {"t": " Sean"}]
    windows = [torch.zeros(3, len(tokenizer)) for _ in rows]
    metric = AggregationSpec(kind="token_logit", fields={"token": "t"})
    with pytest.raises(ProtocolError) as err:
        compute_windowed_metric(metric, windows, rows, tokenizer)
    text = err.value.message
    assert "2 of 3 values fail this way, among them 'Seanathan'" in text, text
    assert "(row" not in text, text


def test_a_restricted_js_at_the_score_names_no_row(tokenizer: Any) -> None:
    """``restrict_token_ids`` flattens every row's answer set into one
    refusal. Row 0's set is empty, so the score reduces row 1 alone. At the
    score no table row is known, so it names none. On the base it named
    ``(row 0)`` for the member of the second row."""
    rows = [{"set": []}, {"set": [" Sunday", "Tiffany"]}]
    metric = AggregationSpec(kind="js", fields={"target": "unused", "restrict": "set"})
    logits = torch.zeros(2, 1, len(tokenizer))
    with pytest.raises(ProtocolError) as err:
        compute_metric(metric, logits, rows, tokenizer, target_value=logits)
    text = err.value.message
    assert "metric js.restrict" in text and "'Tiffany'" in text, text
    assert "(row" not in text, text
