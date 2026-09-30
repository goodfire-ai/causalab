"""Eligibility-aware metrics, the pure half (spec §2.10 "Eligibility", §4.1,
§5 rule 4, §6).

* [`Eligibility`][causalab.protocol.results.Eligibility] — ``n_eligible`` of
  ``n_considered`` over a metric cell's rows, the excluded rows by reason: the
  row-level twin of ``Denominator``, and a third denominator named
  apart from ``save.reduce: "count"`` and a workflow reduction's ``unit``.
* ``minimum_count`` — the one authored field: parsed as a positive integer,
  never swept, in the canonical form only when authored, so every document
  that declares none keeps its digest (T3's half of this file).
* **T2** — a threshold above the resolved base table's **maximum eligible
  count** is refused by ``validate --data`` under rule 4, naming the maximum,
  the table's size and the empty column. *Mutation:* the check is held to the
  maximum, not to the row count — a table with one empty answer among three
  refuses a threshold of three. Every refusal has its valid-work twin: a
  threshold at the maximum passes, a metric with none makes no claim.
* **T3** — no shipped or corpus document authors a threshold, and none
  canonicalizes with one; the pinned digests (``test_corpus.py``) are the
  proof that nothing moved.

The model-backed half — T1's excluded rows, the record on disk and the mean
over the eligible rows — is ``tests/neural/engines/pytorch_hooks/
test_eligibility_run.py`` and ``tests/neural/shared/test_metric_eligibility.py``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.cli import main
from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.data import check_data_columns, maximum_eligible_count
from causalab.protocol.results import Eligibility, unavailable
from causalab.io.env import FileArtifacts, FileDatasets, ResolutionEnv
from causalab.protocol.schema import (
    MINIMUM_COUNT_FIELD,
    PROTOCOL_VERSION,
    parse_document,
)

from tests.protocol._docs import aggregation, by_label, saved
from tests.protocol._env import CORPUS_DIR, steps_of


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]

#: Every place a runnable document is committed (T3).
SHIPPED = (
    "tests/protocols/*.json",
    # the shipped method documents under demos/methods/ included
    "demos/*/protocols/*.json",
)

#: One undivided pool (§2.2, V22); one row carries no answer.
ROWS_ONE_EMPTY = [
    {"input": "the number is one", "entity": "one", "split": "all"},
    {"input": "the number is two", "entity": None, "split": "all"},
    {"input": "the number is three", "entity": "three", "split": "all"},
]
ROWS_ALL_ANSWERED = [
    {**row, "entity": answer}
    for row, answer in zip(ROWS_ONE_EMPTY, ["one", "two", "three"])
]


def _doc(ref: str, *, minimum_count: int | None = None) -> dict[str, Any]:
    spec = aggregation("token_logit", token="entity")
    if minimum_count is not None:
        spec[MINIMUM_COUNT_FIELD] = minimum_count
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {"base": {"dataset": ref, "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [saved("logits", "original", "tl.json", spec)],
        },
    }


def _env_with_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> ResolutionEnv:
    table = tmp_path / "data" / "elig" / "rows.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(rows))
    return ResolutionEnv(
        datasets=FileDatasets(root=tmp_path / "data"),
        artifacts=FileArtifacts(root=tmp_path / "artifacts"),
    )


REF = "elig/rows"


# --------------------------------------------------------------------------- #
# Eligibility — the row-level denominator
# --------------------------------------------------------------------------- #


class TestEligibility:
    def test_counts_eligible_of_considered_and_the_excluded_by_reason(self):
        values = [
            0.5,
            unavailable("alignment_missing", "row 1", "tl"),
            2.0,
            unavailable("alignment_ambiguous", "row 3", "tl"),
            unavailable("alignment_missing", "row 4", "tl"),
        ]
        e = Eligibility.of(values)
        assert (e.n_eligible, e.n_considered) == (2, 5)
        assert e.excluded == {"alignment_ambiguous": 1, "alignment_missing": 2}
        assert e.as_record() == {
            "n_eligible": 2,
            "n_considered": 5,
            "excluded": {"alignment_ambiguous": 1, "alignment_missing": 2},
        }

    def test_a_cell_with_every_row_eligible_records_exactly_the_two_counts(self):
        e = Eligibility.of([0.1, 0.2, {"indices": [1]}])
        assert (e.n_eligible, e.n_considered) == (3, 3)
        assert e.as_record() == {"n_eligible": 3, "n_considered": 3}

    def test_no_rows_is_zero_of_zero(self):
        assert Eligibility.of([]).as_record() == {"n_eligible": 0, "n_considered": 0}

    def test_the_record_is_frozen(self):
        e = Eligibility.of([1.0])
        with pytest.raises(Exception):
            e.n_eligible = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# minimum_count — the one authored field
# --------------------------------------------------------------------------- #


class TestMinimumCountField:
    def test_parses_as_a_positive_integer(self):
        doc = parse_document(_doc(REF, minimum_count=3))
        assert by_label(doc)["tl"].minimum_count == 3

    def test_absent_is_none(self):
        assert by_label(parse_document(_doc(REF)))["tl"].minimum_count is None

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_threshold_is_refused_at_parse(self, bad):
        with pytest.raises(ParseError) as err:
            parse_document(_doc(REF, minimum_count=bad))
        assert err.value.code == "P2"
        assert err.value.path == f"save[0].aggregation.{MINIMUM_COUNT_FIELD}"

    @pytest.mark.parametrize("bad", ["3", 2.5, True])
    def test_a_non_integer_threshold_is_refused_at_parse(self, bad):
        raw = _doc(REF)
        raw["method"]["save"][0]["aggregation"][MINIMUM_COUNT_FIELD] = bad
        with pytest.raises(ParseError) as err:
            parse_document(raw)
        assert err.value.code == "P2"

    def test_a_threshold_is_not_a_research_variable(self):
        """Never swept: a campaign forks on hypotheses, not on its bar."""
        raw = _doc(REF)
        raw["method"]["save"][0]["aggregation"][MINIMUM_COUNT_FIELD] = {"sweep": [2, 3]}
        with pytest.raises(ValidationError) as err:
            parse_document(raw)
        assert err.value.rule == 14

    def test_authored_it_enters_the_canonical_form_and_the_digest(self, tmp_path):
        env = _env_with_rows(tmp_path, ROWS_ALL_ANSWERED)
        plain = canonicalize(_doc(REF), env)
        declared = canonicalize(_doc(REF, minimum_count=2), env)
        assert MINIMUM_COUNT_FIELD not in plain["method"]["save"][0]["aggregation"]
        assert declared["method"]["save"][0]["aggregation"][MINIMUM_COUNT_FIELD] == 2
        assert digest(plain) != digest(declared)

    def test_unauthored_the_canonical_form_has_no_such_key(self, tmp_path):
        """No default is materialized: absent stays absent, so a document that
        declares nothing has the digest it had before the field existed (§7)."""
        env = _env_with_rows(tmp_path, ROWS_ALL_ANSWERED)
        canonical = canonicalize(_doc(REF), env)
        assert MINIMUM_COUNT_FIELD not in json.dumps(canonical)


# --------------------------------------------------------------------------- #
# T2 — a threshold above the maximum eligible count is refused by validate --data
# --------------------------------------------------------------------------- #


class TestMaximumEligibleCount:
    def test_a_row_with_no_answer_cannot_be_eligible(self):
        metric = by_label(parse_document(_doc(REF)))["tl"]
        assert maximum_eligible_count(metric, ROWS_ONE_EMPTY) == (2, {"entity": 1})
        assert maximum_eligible_count(metric, ROWS_ALL_ANSWERED) == (3, {})

    def test_an_absent_key_and_an_empty_form_group_count_as_no_answer(self):
        raw = _doc(REF)
        raw["method"]["save"][0]["aggregation"] = aggregation(
            "match", expected="entity"
        )
        metric = by_label(parse_document(raw))["tl"]
        rows = [
            {"input": "a", "entity": ["one", " one"], "split": "all"},
            {"input": "b", "entity": [], "split": "all"},
            {"input": "c", "split": "all"},
        ]
        assert maximum_eligible_count(metric, rows) == (1, {"entity": 2})

    def test_a_kind_naming_no_column_makes_every_row_eligible(self):
        raw = _doc(REF)
        raw["method"]["save"][0]["aggregation"] = aggregation("top_k", k=2, by="prob")
        metric = by_label(parse_document(raw))["tl"]
        assert maximum_eligible_count(metric, ROWS_ONE_EMPTY) == (3, {})


class TestT2Refusal:
    def test_t2_a_threshold_above_the_maximum_is_refused_naming_the_maximum(
        self, tmp_path
    ):
        """Three rows, one with no answer: the maximum eligible count is 2, and
        ``minimum_count: 3`` — the row count — is refused. The mutation the
        acceptance names (compare against ``n_considered``) would let this
        document through."""
        env = _env_with_rows(tmp_path, ROWS_ONE_EMPTY)
        with pytest.raises(ValidationError) as err:
            check_data_columns(
                compile_protocol(_doc(REF, minimum_count=3), env=env), env
            )
        assert err.value.rule == 4
        assert err.value.rule_id == "references_resolve"
        assert err.value.path == f"save[0].aggregation.{MINIMUM_COUNT_FIELD}"
        message = str(err.value)
        assert "[V4]" in message
        assert "minimum_count=3" in message
        assert "at most 2" in message and "3 rows" in message
        assert "'entity'" in message  # the empty column, named

    def test_t2_twin_a_threshold_at_the_maximum_passes(self, tmp_path):
        env = _env_with_rows(tmp_path, ROWS_ONE_EMPTY)
        assert check_data_columns(
            compile_protocol(_doc(REF, minimum_count=2), env=env), env
        )

    def test_t2_twin_a_metric_with_no_threshold_makes_no_claim(self, tmp_path):
        env = _env_with_rows(tmp_path, ROWS_ONE_EMPTY)
        assert check_data_columns(compile_protocol(_doc(REF), env=env), env)

    def test_t2_a_threshold_above_the_row_count_is_refused_on_a_full_table(
        self, tmp_path
    ):
        env = _env_with_rows(tmp_path, ROWS_ALL_ANSWERED)
        with pytest.raises(ValidationError) as err:
            check_data_columns(
                compile_protocol(_doc(REF, minimum_count=4), env=env), env
            )
        assert err.value.rule == 4
        assert "at most 3 of its 3 rows" in str(err.value)
        # and the twin, at the maximum
        assert check_data_columns(
            compile_protocol(_doc(REF, minimum_count=3), env=env), env
        )

    def test_t2_the_bare_load_does_not_read_the_table(self, tmp_path):
        """Like rule 20 and rule 25, the check needs the resolved rows, so it
        is ``validate --data``'s and not the bare load's."""
        env = _env_with_rows(tmp_path, ROWS_ONE_EMPTY)
        loaded = compile_protocol(_doc(REF, minimum_count=3), env=env)  # no raise
        assert by_label(loaded.document)["tl"].minimum_count == 3

    def test_t2_through_the_cli(self, tmp_path, capsys):
        env_root = tmp_path
        _env_with_rows(env_root, ROWS_ONE_EMPTY)
        refused = tmp_path / "refused.json"
        refused.write_text(json.dumps(_doc(REF, minimum_count=3)))
        passes = tmp_path / "passes.json"
        passes.write_text(json.dumps(_doc(REF, minimum_count=2)))
        argv = [
            "--data-root",
            str(env_root / "data"),
            "--artifacts-root",
            str(env_root / "artifacts"),
            "--data",
        ]
        assert main(["validate", "--engine", "auto", str(refused), *argv]) == 1
        err = capsys.readouterr().err
        assert "refused: [V4]" in err and "at most 2" in err
        assert main(["validate", "--engine", "auto", str(passes), *argv]) == 0
        assert "OK" in capsys.readouterr().out
        # `validate` runs the data rules by default, so the bare
        # verb refuses the same document; `--data` names the default
        assert main(["validate", "--engine", "auto", str(refused), *argv[:-1]]) == 1


# --------------------------------------------------------------------------- #
# T3 — the legitimate campaign: nothing shipped declares a threshold
# --------------------------------------------------------------------------- #


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _shipped() -> list[str]:
    return sorted(
        path for pattern in SHIPPED for path in glob.glob(str(REPO / pattern))
    )


def test_t3_the_shipped_documents_were_found():
    assert len(_shipped()) >= 20


@pytest.mark.parametrize("path", _shipped(), ids=lambda p: Path(p).name)
def test_t3_no_shipped_document_authors_a_threshold(path):
    """The field is optional and absent everywhere, so no digest moved —
    ``test_corpus.py`` holds the pins; this says why they could not have."""
    raw = json.loads(Path(path).read_text())
    assert not any(MINIMUM_COUNT_FIELD in mapping for mapping in _walk(raw)), path


@pytest.mark.parametrize(
    "name", sorted(p.name for p in CORPUS_DIR.glob("*_im.json")), ids=str
)
def test_t3_every_corpus_document_canonicalizes_without_the_field(name, env):
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    for entry in loaded.canonical["method"]["save"]:
        assert MINIMUM_COUNT_FIELD not in entry.get("aggregation", {})
    for point in steps_of(loaded, env).documents:
        assert all(m.minimum_count is None for m in by_label(point).values())
