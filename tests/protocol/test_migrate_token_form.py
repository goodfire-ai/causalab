"""``causalab migrate`` retires ``token_form`` (§2.10). For a metric whose
answers are literal strings in the document, the retired value is written into
the strings it used to rewrite and the key goes. For a metric whose answers are
dataset columns, the migration refuses, because the strings are in a table it
does not read. One refusal names every metric that needs an author step, and a
Markdown page's fenced example is refused the way a JSON file is. The
retirement runs on the protocol-3 form, before the reads-first rewrite carries
the metric onto the save entry that consumes it as its ``aggregation``
(§2.12)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.migrate import (
    migrate_document,
    migrate_markdown,
    needs_migration,
)
from causalab.protocol.rules.errors import ParseError

pytestmark = pytest.mark.unit


def _v3_with_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """A protocol-3 interchange document whose one metric ``m`` reduces
    ``patched``'s logits and is saved as ``m.json``."""
    return {
        "header": {"protocol_version": "3"},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "weekdays/data#train", "field": "input"},
            "counterfactual": {
                "dataset": "weekdays/data#train",
                "field": "counterfactual_inputs[0]",
            },
        },
        "method": {
            "sites": {
                "tgt": {"component": "block_output", "layers": [3]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {
                    "site": "tgt",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                },
                "logits": {
                    "site": "lm_head",
                    "pos": -1,
                    "model": "patched",
                    "input": "base",
                },
                "cf_logits": {
                    "site": "lm_head",
                    "pos": -1,
                    "model": "original",
                    "input": "counterfactual",
                },
            },
            "writes": {"patch": {"site": "tgt", "pos": -1, "do": {"swap": "v_cf"}}},
            "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
            "metrics": {"m": {"of": "logits", **metric}},
            "save": [
                {
                    "value": "m",
                    "model": "patched",
                    "input": "base",
                    "file_path": "m.json",
                }
            ],
        },
    }


def _aggregation(migrated: dict[str, Any]) -> dict[str, Any]:
    """The reduction the migrated document's one save entry carries."""
    (entry,) = migrated["method"]["save"]
    assert entry["read"] == "logits" and entry["model"] == "patched"
    return entry["aggregation"]


@pytest.mark.parametrize("form", ["auto", "bare", "space_prefixed"])
def test_a_column_metric_under_a_retired_form_is_refused_naming_the_columns(
    form: str,
) -> None:
    """The retired form rewrote each column value's leading space when the
    run resolved it. The migration cannot read the table, so dropping the key
    could change the scored token without an error. It refuses instead, and
    names the metric, its fields and its columns."""
    raw = _v3_with_metric(
        {
            "kind": "logit_diff",
            "a": "cf_answer",
            "b": "base_answer",
            "token_form": form,
        }
    )
    before = copy.deepcopy(raw)
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert err.value.code == "P2"
    assert err.value.path == "method.metrics.m.token_form"
    message = str(err.value)
    assert f"{form!r}" in message
    assert "a: 'cf_answer'" in message and "b: 'base_answer'" in message
    assert raw == before  # nothing was rewritten on the way to the refusal


@pytest.mark.parametrize(
    ("form", "rewrite"),
    [
        ("bare", "as s.lstrip(' ')"),
        ("space_prefixed", "as ' ' + s.lstrip(' ')"),
        ("auto", "so the rewrite needs the tokenizer"),
    ],
)
def test_the_column_refusal_states_the_table_rewrite(form: str, rewrite: str) -> None:
    """``bare`` and ``space_prefixed`` fixed one rewrite per answer string, so
    the refusal can state it. ``auto`` chose per string with the tokenizer."""
    raw = _v3_with_metric(
        {"kind": "match", "expected": "cf_answer", "token_form": form}
    )
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert rewrite in str(err.value)


def test_a_column_of_answer_lists_is_refused_with_the_rewrite_per_member() -> None:
    """A ``match.expected`` column may hold a list of equivalent forms per
    row, such as ``[' Thursday', 'Thursday']``. The retired resolver
    rewrote each member, so the refusal's rewrite reaches each member."""
    raw = _v3_with_metric(
        {
            "kind": "match",
            "expected": "label_forms",
            "mode": "first_token",
            "token_form": "space_prefixed",
        }
    )
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    message = str(err.value)
    assert "expected: 'label_forms'" in message
    assert "each member of a list of forms included" in message


def _v3_with_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`_v3_with_metric` with several metrics over ``patched``'s logits, each
    saved as ``<name>.json``."""
    raw = _v3_with_metric({})
    method = raw["method"]
    method["metrics"] = {name: {"of": "logits", **m} for name, m in metrics.items()}
    method["save"] = [
        {
            "value": name,
            "model": "patched",
            "input": "base",
            "file_path": f"{name}.json",
        }
        for name in metrics
    ]
    return raw


def test_one_refusal_names_every_metric_that_needs_an_author_step() -> None:
    """A document with several column metrics under retired forms is refused
    once, at the ``metrics`` section, and the message names each metric with
    its columns and rewrite. A literal metric beside them is still respelled
    by the run that follows the author's edits."""
    raw = _v3_with_metrics(
        {
            "ld": {
                "kind": "logit_diff",
                "a": "cf_answer",
                "b": "base_answer",
                "token_form": "space_prefixed",
            },
            "iia": {"kind": "match", "expected": "cf_answer", "token_form": "bare"},
            "tl": {"kind": "token_logit", "token": "cf_answer", "token_form": "auto"},
            "lit": {
                "kind": "token_logits",
                "tokens": ["?", "."],
                "token_form": "space_prefixed",
            },
        }
    )
    before = copy.deepcopy(raw)
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert err.value.code == "P2"
    assert err.value.path == "method.metrics"
    message = str(err.value)
    assert message.count("reads its answers from dataset columns") == 3
    assert "(1) metric 'ld' reads" in message
    assert "(2) metric 'iia' reads" in message
    assert "(3) metric 'tl' reads" in message
    assert "'lit'" not in message
    assert "a: 'cf_answer'; b: 'base_answer'" in message
    assert "as ' ' + s.lstrip(' ')" in message and "as s.lstrip(' ')" in message
    assert raw == before
    # the author's step for each: the tables hold the forms, the keys go
    for name in ("ld", "iia", "tl"):
        del raw["method"]["metrics"][name]["token_form"]
    migrated = migrate_document(raw)
    tokens = migrated["method"]["save"][3]["aggregation"]["tokens"]
    assert tokens == [" ?", " ."]


def test_a_literal_refusal_and_a_column_refusal_come_in_one_message() -> None:
    raw = _v3_with_metrics(
        {
            "cp": {"kind": "class_probs", "groups": {"q": ["?"]}, "token_form": "auto"},
            "iia": {"kind": "match", "expected": "cf_answer", "token_form": "bare"},
        }
    )
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    message = str(err.value)
    assert err.value.path == "method.metrics"
    assert "(1) metric 'cp' lists literal answers under token_form 'auto'" in message
    assert "(2) metric 'iia' reads its answers from dataset columns" in message


def _page(*documents: dict[str, Any]) -> str:
    """A guide page with each document in its own fenced ``json`` block."""
    blocks = [f"```json\n{json.dumps(doc, indent=2)}\n```\n" for doc in documents]
    return "# Guide\n\nSome prose.\n\n" + "\nMore prose.\n\n".join(blocks)


def test_a_markdown_example_is_refused_like_a_json_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fenced example is a whole old document, so the column refusal
    reaches its author as it does for a ``.json`` file: named on stderr with
    the block's line, and counted in the exit code. ``migrate_markdown``
    leaves the block as it is."""
    refused = _v3_with_metric(
        {"kind": "logit_diff", "a": "io", "b": "s", "token_form": "space_prefixed"}
    )
    page = tmp_path / "guide.md"
    text = _page(refused)
    page.write_text(text)
    assert migrate_markdown(text) == text
    assert main(["migrate", "--check", str(page)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"refused: {page}: line 5: [P2] at method.metrics.m.token_form" in (
        captured.err
    )
    assert "a: 'io'; b: 's'" in captured.err
    assert main(["migrate", str(page)]) == 1
    assert page.read_text() == text


def test_the_other_blocks_of_a_page_still_migrate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refused block is left as it is; the page's other old examples are
    rewritten, as the files before a refused path stay migrated."""
    refused = _v3_with_metric(
        {"kind": "match", "expected": "cf_answer", "token_form": "bare"}
    )
    plain = _v3_with_metric({"kind": "match", "expected": "cf_answer"})
    page = tmp_path / "guide.md"
    page.write_text(_page(refused, plain))
    assert main(["migrate", "--check", str(page)]) == 1
    captured = capsys.readouterr()
    assert captured.out == f"would migrate {page}\n"
    assert "refused:" in captured.err
    assert main(["migrate", str(page)]) == 1
    out = page.read_text()
    assert out.count('"protocol_version": "3"') == 1
    assert out.count('"protocol_version": "4"') == 1
    assert "refused:" in capsys.readouterr().err


def test_a_swept_column_field_is_refused_like_a_plain_one() -> None:
    """A sweep over column names still reads the answers from the table."""
    raw = _v3_with_metric(
        {
            "kind": "token_logit",
            "token": {"sweep": ["cf_answer", "base_answer"]},
            "token_form": "bare",
        }
    )
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert err.value.path == "method.metrics.m.token_form"
    assert "token: 'cf_answer', 'base_answer'" in str(err.value)


def test_a_column_metric_without_the_key_migrates() -> None:
    """The refusal is about the retired key only. The same metric without it
    moves onto its save entry unchanged."""
    metric = {"kind": "logit_diff", "a": "cf_answer", "b": "base_answer"}
    migrated = migrate_document(_v3_with_metric(metric))
    assert _aggregation(migrated) == metric
    assert migrate_document(migrated) == migrated  # idempotent
    assert not needs_migration(migrated)


def test_space_prefixed_writes_the_space_into_a_literal_group() -> None:
    raw = _v3_with_metric(
        {
            "kind": "class_probs",
            "groups": {"city": ["Seattle", " Seattle"], "no": ["Portland"]},
            "token_form": "space_prefixed",
        }
    )
    groups = _aggregation(migrate_document(raw))["groups"]
    # the retired resolver folded both spellings onto the spaced row; the
    # migration writes exactly that row, twice — which the run then refuses
    # as a collision, the way it always did for this idiom
    assert groups == {"city": [" Seattle", " Seattle"], "no": [" Portland"]}


def test_bare_strips_the_space_from_a_literal_token_list() -> None:
    raw = _v3_with_metric(
        {"kind": "token_logits", "tokens": [" ?", "."], "token_form": "bare"}
    )
    assert _aggregation(migrate_document(raw)) == {
        "kind": "token_logits",
        "tokens": ["?", "."],
    }


def test_a_literal_restrict_is_respelled_and_a_column_restrict_is_refused() -> None:
    """A ``restrict`` list is literal strings, which the migration respells.
    A ``restrict`` string names a column, which it cannot read. The ``js``
    target is a read, so the refusal names only the ``restrict`` column."""
    literal = _v3_with_metric(
        {
            "kind": "js",
            "target": "cf_logits",
            "restrict": ["Yes", "No"],
            "token_form": "space_prefixed",
        }
    )
    assert _aggregation(migrate_document(literal))["restrict"] == [" Yes", " No"]
    column = _v3_with_metric(
        {
            "kind": "js",
            "target": "cf_logits",
            "restrict": "valid_answers",
            "token_form": "space_prefixed",
        }
    )
    with pytest.raises(ParseError) as err:
        migrate_document(column)
    assert err.value.path == "method.metrics.m.token_form"
    assert "restrict: 'valid_answers'" in str(err.value)
    assert "cf_logits" not in str(err.value)


def test_auto_over_a_literal_list_is_refused_naming_the_metric() -> None:
    """``auto`` picked each string's form with the tokenizer in hand; the
    migration has no tokenizer and does not guess."""
    raw = _v3_with_metric(
        {"kind": "class_probs", "groups": {"q": ["?"]}, "token_form": "auto"}
    )
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert err.value.code == "P2"
    assert err.value.path == "method.metrics.m.token_form"
    assert "'auto'" in str(err.value)


def test_an_artifact_held_literal_field_is_refused_not_respelled() -> None:
    """An artifact reference is legal anywhere a value is and resolves after
    the migration, so its strings are not here to rewrite — walking into the
    reference would corrupt its ``artifact`` and ``key``."""
    raw = _v3_with_metric(
        {
            "kind": "token_logits",
            "tokens": {"artifact": "fit", "key": "answer_strings"},
            "token_form": "space_prefixed",
        }
    )
    before = copy.deepcopy(raw)
    with pytest.raises(ParseError) as err:
        migrate_document(raw)
    assert err.value.code == "P2"
    assert "artifact" in str(err.value)
    assert raw == before  # nothing was rewritten on the way to the refusal


def test_a_migrated_document_is_returned_as_it_is() -> None:
    migrated = migrate_document(
        _v3_with_metric({"kind": "match", "expected": "cf_answer"})
    )
    assert not needs_migration(migrated)
    assert migrate_document(migrated) == migrated
