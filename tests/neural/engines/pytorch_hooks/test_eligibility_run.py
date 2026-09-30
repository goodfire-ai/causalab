"""Eligibility-aware metrics, the engine half (spec §2.10 "Eligibility", §4.1)
— on the tiny Llama fixture, end to end through
``run_protocol``.

* **T1** — a cohort with *k* of *n* rows structurally unobservable produces a
  metric cell with ``n_eligible = n − k``, each excluded row carrying its
  reason code, and a value computed over the eligible rows only. Two sources:
  a ``variable`` row whose value occurs twice (``alignment_ambiguous``, the
  read's own cell staying unavailable) and a row whose answer column is
  ``null`` (``alignment_missing``, the read available). *Mutation:* a mean
  over *n* (the excluded row counted as 0) fails the value assertion.
* the **twin**: every row answered — every row ``eligible: true``, no
  ``reason_code`` column, the summary's ``eligibility`` exactly the two
  counts, nothing ``unavailable``.
* a metric whose **every** row is excluded is an ``unavailable`` cell under
  the rows' reason, counted in the denominator.

The pure halves are ``tests/protocol/test_eligibility.py`` and
``tests/neural/shared/test_metric_eligibility.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
from causalab.protocol import run_protocol
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns
from causalab.protocol.results import Available, Unavailable
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.io.tables import read_table

from .conftest import TINY_LLAMA
from tests.protocol._docs import aggregation, in_order, saved

pytestmark = pytest.mark.smoke


def _doc(*, pos: Any) -> dict[str, Any]:
    """A ``token_logit`` over the ``entity`` column at ``pos`` — a ``variable``
    anchor on that same column, or the last position."""
    doc: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main", "dtype": "fp32"},
        "data": {"base": {"dataset": "elig/rows", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": pos}},
            "save": [
                saved(
                    "logits",
                    "original",
                    "tl.json",
                    aggregation("token_logit", token="entity"),
                )
            ],
        },
    }
    if pos == "ent":
        doc["method"]["positions"] = {"ent": {"variable": "entity"}}
    return in_order(doc)


def _env_with_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> ResolutionEnv:
    table = tmp_path / "data" / "elig" / "rows.json"
    table.parent.mkdir(parents=True)
    table.write_text(json.dumps(rows))
    return ResolutionEnv(
        datasets=FileDatasets(root=tmp_path / "data"),
        artifacts=FileArtifacts(root=tmp_path / "artifacts"),
    )


def _run(tmp_path: Path, doc: dict[str, Any], rows: list[dict[str, Any]]):
    # callers take the `llama_bundle` fixture: loading the fixture model is
    # what registers its static config, which `load` needs for the digest
    env = _env_with_rows(tmp_path, rows)
    loaded = compile_protocol(doc, env=env)
    out = tmp_path / "out"
    result = run_protocol(loaded, env, PytorchHooksEngine(device="cpu"), out)
    return result, read_table(out / "tl.json")


#: one undivided pool (§2.2, V22); row 0's entity occurs twice in its text
AMBIGUOUS_ROWS = [
    {"input": "day after day", "entity": "day", "split": "all"},
    {"input": "one two three", "entity": "two", "split": "all"},
    {"input": "four five six", "entity": "five", "split": "all"},
]
#: every entity occurs once; row 1 carries no answer
NULL_ANSWER_ROWS = [
    {"input": "one two three", "entity": "two", "split": "all"},
    {"input": "four five six", "entity": None, "split": "all"},
    {"input": "seven eight nine", "entity": "eight", "split": "all"},
]
ANSWERED_ROWS = [
    {**row, "entity": "five"} if i == 1 else row
    for i, row in enumerate(NULL_ANSWER_ROWS)
]


def _assert_t1(result, table, *, excluded_row: int, reason: str) -> None:
    n = 3
    rows = sorted(table, key=lambda r: r["example_id"])
    assert [r["example_id"] for r in rows] == ["0", "1", "2"]
    # each excluded row carries its reason code and a null value
    assert rows[excluded_row]["eligible"] is False
    assert rows[excluded_row]["reason_code"] == reason
    assert rows[excluded_row]["value"] is None
    eligible = [r for i, r in enumerate(rows) if i != excluded_row]
    assert all(r["eligible"] is True for r in eligible)
    assert not any("reason_code" in r for r in eligible)
    scored = [r["value"] for r in eligible]
    assert all(isinstance(v, float) for v in scored)
    # the cell: n_eligible = n − k, n_considered = n, the excluded by reason
    (summary,) = result.summaries
    assert summary["eligibility"]["tl"] == {
        "n_eligible": n - 1,
        "n_considered": n,
        "excluded": {reason: 1},
    }
    # the value is computed over the eligible rows only — a mean over n
    # (the excluded row as 0) is a different number
    assert summary["metrics"]["tl"] == pytest.approx(sum(scored) / (n - 1))
    assert summary["metrics"]["tl"] != pytest.approx(sum(scored) / n)
    metric_cell = result.cells[-1]
    assert isinstance(metric_cell, Available)
    assert metric_cell.denominator_key == "tl"
    assert metric_cell.mapping["n_eligible"] == n - 1
    assert metric_cell.mapping["n_considered"] == n


def test_t1_an_ambiguous_row_is_excluded_and_the_other_rows_are_scored(
    tmp_path, llama_bundle
):
    """The read's cell stays unavailable (`alignment_ambiguous`, the saved gather
    width zero on that row); the metric over it inherits row by row."""
    result, table = _run(tmp_path, _doc(pos="ent"), AMBIGUOUS_ROWS)
    _assert_t1(result, table, excluded_row=0, reason="alignment_ambiguous")
    (metric_cell,) = result.cells
    assert isinstance(metric_cell, Available)
    assert result.denominator.render() == "1 / 1 eligible"
    (summary,) = result.summaries
    assert "unavailable" not in summary  # the metric's cell measured


def test_t1_a_row_without_an_answer_is_excluded_under_alignment_missing(
    tmp_path, llama_bundle
):
    result, table = _run(tmp_path, _doc(pos=-1), NULL_ANSWER_ROWS)
    _assert_t1(result, table, excluded_row=1, reason="alignment_missing")
    assert result.denominator.render() == "1 / 1 eligible"


def test_twin_every_row_answered_records_eligible_true_and_nothing_else(
    tmp_path, llama_bundle
):
    result, table = _run(tmp_path, _doc(pos=-1), ANSWERED_ROWS)
    assert len(table) == 3
    assert all(row["eligible"] is True for row in table)
    assert not any("reason_code" in row for row in table)
    assert all(isinstance(row["value"], float) for row in table)
    (summary,) = result.summaries
    assert summary["eligibility"] == {"tl": {"n_eligible": 3, "n_considered": 3}}
    assert "unavailable" not in summary
    assert summary["metrics"]["tl"] == pytest.approx(
        sum(row["value"] for row in table) / 3
    )
    (cell,) = result.cells
    assert isinstance(cell, Available)
    assert cell.mapping == {
        "file_path": "tl.json",
        "metric": "tl",
        "n_eligible": 3,
        "n_considered": 3,
    }


def test_a_metric_whose_every_row_is_excluded_is_an_unavailable_cell(
    tmp_path, llama_bundle
):
    rows = [{**row, "entity": None} for row in NULL_ANSWER_ROWS]
    result, table = _run(tmp_path, _doc(pos=-1), rows)
    assert [row["eligible"] for row in table] == [False, False, False]
    assert {row["reason_code"] for row in table} == {"alignment_missing"}
    (cell,) = result.cells
    assert isinstance(cell, Unavailable)
    assert cell.reason == "alignment_missing"
    assert cell.denominator_key == "tl"
    assert "all 3 rows are excluded" in cell.detail
    assert (
        result.denominator.render()
        == "0 / 1 eligible; 1 excluded: alignment_missing ×1"
    )
    (summary,) = result.summaries
    assert summary["eligibility"]["tl"] == {
        "n_eligible": 0,
        "n_considered": 3,
        "excluded": {"alignment_missing": 3},
    }
    assert set(summary["unavailable"]) == {"tl"}


#: the answered rows under the author's own labels (§2.2)
LABELLED_ROWS = [
    {**row, "example_id": label}
    for row, label in zip(ANSWERED_ROWS, ("w1", "w2", "w3"))
]


def test_an_authored_example_id_labels_the_metric_rows(tmp_path, llama_bundle):
    """The row label is the author's when the table carries one (§2.2): the
    same rows under the same document write the same values, keyed by
    ``example_id`` instead of the index."""
    _, table = _run(tmp_path, _doc(pos=-1), LABELLED_ROWS)
    assert [row["example_id"] for row in table] == ["w1", "w2", "w3"]
    _, positional = _run(tmp_path / "positional", _doc(pos=-1), ANSWERED_ROWS)
    assert [row["example_id"] for row in positional] == ["0", "1", "2"]
    assert [row["value"] for row in table] == [row["value"] for row in positional]


def test_a_repeated_example_id_is_refused_at_validate_and_before_the_first_forward(
    tmp_path, llama_bundle
):
    """One label, one row. ``validate --data`` refuses the table under rule 4
    naming the role; a run that skipped it is refused by ``resolve_roles``
    before any executor is built."""
    rows = [{**row, "example_id": "same"} for row in LABELLED_ROWS[:2]] + LABELLED_ROWS[
        2:
    ]
    env = _env_with_rows(tmp_path, rows)
    loaded = compile_protocol(_doc(pos=-1), env=env)
    with pytest.raises(ValidationError) as err:
        check_data_columns(loaded, env)
    assert err.value.rule == 4 and err.value.path == "data.base"
    assert "'same' labels rows 0 and 1" in str(err.value)
    with pytest.raises(ProtocolError, match="labels rows 0 and 1"):
        run_protocol(loaded, env, PytorchHooksEngine(device="cpu"), tmp_path / "out")
    assert not (tmp_path / "out" / "tl.json").exists()


#: row 0's entity occurs nowhere in its text, and the tiny Llama tokenizer
#: splits it (``['sevent', 'een']``); row 1's is one token and aligns
UNALIGNED_SPLIT_ROWS = [
    {"input": "one two three", "entity": "seventeen", "split": "all"},
    {"input": "one two three", "entity": "two", "split": "all"},
]


def test_a_split_answer_on_a_row_its_read_does_not_align_runs_and_scores(
    tmp_path, llama_bundle
):
    """The score never tokenizes the answer of a row whose read aligns on
    nothing (§4.1), so the run door does not either: the run scores 1 of 2
    rows and excludes row 0 under ``alignment_missing``, as it did before
    the door resolved answers."""
    result, table = _run(tmp_path, _doc(pos="ent"), UNALIGNED_SPLIT_ROWS)
    rows = sorted(table, key=lambda r: r["example_id"])
    assert [r["eligible"] for r in rows] == [False, True]
    assert rows[0]["reason_code"] == "alignment_missing"
    (summary,) = result.summaries
    assert summary["eligibility"]["tl"] == {
        "n_eligible": 1,
        "n_considered": 2,
        "excluded": {"alignment_missing": 1},
    }


def test_twin_a_split_answer_on_an_aligned_row_refuses_before_the_first_forward(
    tmp_path, llama_bundle
):
    """The same split answer on a row whose read aligns is scored, so the
    door refuses it before the weights, naming the row."""
    rows = [{**UNALIGNED_SPLIT_ROWS[0], "input": "one seventeen three"}] + (
        UNALIGNED_SPLIT_ROWS[1:]
    )
    with pytest.raises(ProtocolError) as err:
        _run(tmp_path, _doc(pos="ent"), rows)
    text = str(err.value)
    assert err.value.path == "save[0].aggregation", text
    assert "'seventeen' (row 0) is not a single token" in text, text
    assert not (tmp_path / "out" / "tl.json").exists()
