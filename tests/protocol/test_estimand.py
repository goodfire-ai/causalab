"""Estimand identity, the protocol side (spec §2.10, §7; ``estimand.py``).

* **the censuses** — §2.10's unit table is exactly ``UNITS``, the kind table's
  ``unit`` column is exactly ``METRIC_UNITS``, and ``METRIC_UNITS`` covers
  exactly ``METRIC_KINDS``; the vocabulary is defined once and no second
  spelling of a member appears in the package or the specs. Each parse asserts
  it found something first (``test_vocabulary_census.py``'s guard);
* **the grammar** — ``<estimand>/v<n>`` accepts the load-bearing identifiers
  and refuses the malformed ones with the grammar in the message;
* **the metric fields** — ``unit`` / ``estimand_version`` parse on a metric
  and enter the canonical form **only when authored**; an authored value the
  kind does not compute is a ``P4`` naming what it does compute;
* **T7** — a ``fraction`` record against a ``percentage_points`` record is
  refused naming both units and both records. *Mutation:* compare the metric
  *names* instead of the units and the test fails: the two records share a
  name;
* **T8** — two arms with nothing declared compare cleanly, and the same unit
  under two declared estimands is a labelled version comparison, not a refusal;
* **T9** — every corpus, golden and shipped document canonicalizes to its
  pinned digest with neither field authored and no identity key in any metric
  entry;
* **T10** — a claim bound to a record is refused when the record's value
  changes, naming the file and the point digest; a claim bound to a point the
  file no longer holds is refused the same way. *Mutation:* bind by file path
  only and T10 fails, because the path is unchanged.

Every test here fails without the change: the module under test does not
exist on the base and ``unit`` is an unknown metric key (rule 1).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.estimand import (
    IDENTIFIER,
    METRIC_UNITS,
    REDUCTION_ESTIMANDS,
    UNITS,
    Claim,
    EstimandError,
    Record,
    check_claim,
    compare,
    metric_identity,
    metric_record_identity,
    parse_identifier,
    table_record,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.schema import METRIC_KINDS, PROTOCOL_VERSION, parse_document
from tests._helpers.tracked import tracked_files
from tests.golden import _env as golden_env
from tests.protocol._docs import aggregation, by_label, saved
from tests.protocol._env import CORPUS_DIR, steps_of
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "intervention_protocol.md"
WORKFLOW_SPEC = REPO / "docs" / "workflow_protocol.md"
PROTOCOLS = PROTOCOLS_DIR
ESTIMAND = REPO / "causalab" / "protocol" / "estimand.py"
CORPUS_PINS = json.loads((REPO / "tests/protocol/corpus_digests.json").read_text())
GOLDEN_PINS = json.loads((REPO / "tests/golden/golden_digests.json").read_text())

ROW = re.compile(r"^[ \t]*\|(.+)\|\s*$", re.M)
CODE = re.compile(r"`([^`]+)`")

# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #


def _section(heading: str) -> str:
    depth = len(heading) - len(heading.lstrip("#"))
    body = SPEC.read_text().split(heading, 1)
    assert len(body) == 2, f"{heading!r} is not in {SPEC.name}"
    stop = re.compile(rf"^#{{1,{depth}}} ", re.M)
    end = stop.search(body[1])
    return body[1][: end.start()] if end else body[1]


def _rows(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for match in ROW.finditer(text):
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        out.append(cells)
    return out


def _table(header: str) -> list[list[str]]:
    """§2.10's table whose header row starts with ``header``: its body rows,
    up to the first row that is not a backticked member."""
    rows = _rows(_section("### 2.10 `aggregation`: reductions over a read"))
    start = next(index for index, row in enumerate(rows) if row[0] == header)
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def test_the_unit_table_was_found() -> None:
    assert len(_table("unit")) >= len(UNITS)


def test_the_unit_table_is_exactly_the_vocabulary() -> None:
    """Fails without the change: no table, no tuple. Fails on a member added
    to one side only."""
    tabulated = [CODE.findall(row[0])[0] for row in _table("unit")]
    assert len(set(tabulated)) == len(tabulated), (
        f"§2.10 lists a unit twice: {tabulated}"
    )
    assert set(tabulated) == set(UNITS), (
        "§2.10's unit table and UNITS disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(UNITS))}; "
        f"only in the code: {sorted(set(UNITS) - set(tabulated))}"
    )


def test_every_unit_row_says_what_it_measures_and_who_produces_it() -> None:
    for row in _table("unit"):
        assert len(row) == 3 and len(row[1]) > 10 and len(row[2]) > 5, row


def test_metric_units_cover_exactly_the_metric_kinds() -> None:
    """``estimand.py`` keys its table by kind name rather than importing
    ``METRIC_KINDS`` (``schema.py`` imports it), so this holds the two equal."""
    assert set(METRIC_UNITS) == set(METRIC_KINDS)


def test_the_kind_table_unit_column_is_metric_units() -> None:
    """The kind table gained a ``unit`` column; each cell is the kind's
    ``METRIC_UNITS`` entry, ``none`` for a kind with no scalar. Fails on the
    base: the table has four columns."""
    rows = _table("kind")
    assert len(rows) == len(METRIC_KINDS)
    for row in rows:
        kind = CODE.findall(row[0])[0]
        assert len(row) == 6, f"{kind}: the kind table has six columns now"
        own = METRIC_UNITS[kind]
        if own is None:
            assert row[4].startswith("none"), f"{kind} has no scalar unit: {row[4]}"
        else:
            assert CODE.findall(row[4])[0] == own, f"{kind}: {row[4]} vs {own}"
        assert CODE.findall(row[5]) == [metric_identity(kind)], row[5]


def test_every_unit_a_kind_produces_is_in_the_vocabulary() -> None:
    assert set(METRIC_UNITS.values()) - {None} <= set(UNITS)


def test_the_admits_table_says_no_kind_admits_two_arithmetics() -> None:
    """The decision the spec records: identity is derived for every kind and
    required nowhere; the reduction's identifiers are the workflow spec's."""
    rows = _rows(_section("### 2.10 `aggregation`: reductions over a read"))
    start = next(i for i, row in enumerate(rows) if row[0] == "kind or estimator")
    body = rows[start + 1 : start + 3]
    assert body[0][1] == "one" and "derived" in body[0][2]
    assert "derived" in body[1][2] and "rule 13" in body[1][2]


#: Alternative spellings of the members, as a reader might write them. A
#: regression list: each is a real way the same unit has been named in prose.
SECOND_SPELLINGS: tuple[str, ...] = (
    '"nats"',
    '"bits"',
    '"percent"',
    '"pct"',
    '"percentage_point"',
    "`nats`",
    "`bits`",
    "`percent`",
    "`pct`",
    "`percentage_point`",
)


def test_no_second_spelling_of_a_unit_anywhere() -> None:
    """One vocabulary, one spelling: the tuple is defined exactly once in the
    package, and no alternate spelling appears as a literal or a backticked
    token in the package or the specs."""
    # the module under test is included by name: on the commit that adds it,
    # `git ls-files` does not list it until it is staged
    sources = sorted(
        set(tracked_files(REPO / "causalab", "*.py"))
        | set(tracked_files(REPO / "docs", "*.md"))
        | {ESTIMAND}
    )
    assert len(sources) > 50
    definitions = [
        path
        for path in sources
        if path.suffix == ".py" and re.search(r"^UNITS\s*[:=]", path.read_text(), re.M)
    ]
    assert [p.name for p in definitions] == ["estimand.py"], definitions
    offenders = [
        f"{path.relative_to(REPO)}: {spelling}"
        for path in sources
        for spelling in SECOND_SPELLINGS
        if spelling in path.read_text()
    ]
    assert not offenders, offenders


def test_the_identifier_grammar_is_defined_once() -> None:
    sources = sorted(set(tracked_files(REPO / "causalab", "*.py")) | {ESTIMAND})
    hits = [p for p in sources if re.search(r"^IDENTIFIER\s*=", p.read_text(), re.M)]
    assert [p.name for p in hits] == ["estimand.py"]
    # the workflow spec's identifier table is the same grammar
    for entry in REDUCTION_ESTIMANDS:
        assert IDENTIFIER.match(entry.identifier), entry.identifier


# --------------------------------------------------------------------------- #
# the grammar
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mean_of_eligible_row_ratios/v1", ("mean_of_eligible_row_ratios", 1)),
        ("ratio_of_sums/v1", ("ratio_of_sums", 1)),
        ("kl/v12", ("kl", 12)),
        ("top_k/v1", ("top_k", 1)),
    ],
)
def test_the_grammar_accepts_snake_case_and_a_positive_version(text, expected):
    assert parse_identifier(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "mean",  # no version
        "mean/1",  # no v
        "mean/v0",  # not positive
        "mean/v01",  # leading zero
        "Mean/v1",  # not snake_case
        "mean of ratios/v1",  # spaces
        "mean__of/v1",  # double underscore
        "_mean/v1",  # leading underscore
        "mean/v1/v2",
        "",
        None,
        3,
    ],
)
def test_the_grammar_refuses_the_rest_naming_the_grammar(text):
    with pytest.raises(EstimandError, match="<estimand>/v<n>"):
        parse_identifier(text)


def test_every_kind_has_a_derived_identity_in_the_grammar() -> None:
    for kind in METRIC_KINDS:
        assert parse_identifier(metric_identity(kind)) == (kind, 1)


def test_derived_identity_carries_the_kind_unit() -> None:
    assert metric_record_identity("kl") == {"unit": "nat", "estimand_version": "kl/v1"}
    assert metric_record_identity("decode") == {
        "unit": None,
        "estimand_version": "decode/v1",
    }
    assert metric_record_identity(
        "match", unit="fraction", estimand_version="match/v1"
    ) == {"unit": "fraction", "estimand_version": "match/v1"}


# --------------------------------------------------------------------------- #
# the metric fields: parse, canonical form, digest
# --------------------------------------------------------------------------- #


def _document(**metric: Any) -> dict[str, Any]:
    """A minimal document whose one aggregation is a ``match`` over an lm_head
    read, saved as ``acc`` — the fixture the loader tests use, reduced to what
    an aggregation needs."""
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "Qwen/Qwen3-8B", "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["logits"]}},
            "sites": {"lm_head": {"component": "lm_head"}},
            "reads": {"logits": {"site": "lm_head", "pos": -1}},
            "save": [
                saved(
                    "logits",
                    "original",
                    "acc.json",
                    aggregation(
                        "match",
                        expected="cf_answer",
                        **metric,
                    ),
                )
            ],
        },
    }


def _canonical_metric(env, raw: dict[str, Any]) -> dict[str, Any]:
    loaded = compile_protocol(raw, env=env)
    return loaded.canonical["method"]["save"][0]["aggregation"]


def test_the_fields_parse_and_land_on_the_spec() -> None:
    doc = parse_document(_document(unit="fraction", estimand_version="match/v1"))
    spec = by_label(doc)["acc"]
    assert spec.unit == "fraction" and spec.estimand_version == "match/v1"
    bare = by_label(parse_document(_document()))["acc"]
    assert bare.unit is None and bare.estimand_version is None


def test_authored_identity_is_canonical_and_moves_the_digest(env) -> None:
    """§7: digest-bearing when authored. Fails on the base: unknown key."""
    plain = compile_protocol(_document(), env=env)
    stated = compile_protocol(
        _document(unit="fraction", estimand_version="match/v1"), env=env
    )
    entry = stated.canonical["method"]["save"][0]["aggregation"]
    assert entry["unit"] == "fraction" and entry["estimand_version"] == "match/v1"
    assert stated.digests.document != plain.digests.document
    unit_only = compile_protocol(_document(unit="fraction"), env=env)
    assert (
        "estimand_version"
        not in unit_only.canonical["method"]["save"][0]["aggregation"]
    )
    assert unit_only.digests.document not in {
        plain.digests.document,
        stated.digests.document,
    }


def test_unauthored_identity_is_absent_from_the_canonical_form(env) -> None:
    """The twin of the test above and the mechanism behind T9: nothing is
    materialized. Fails under the mutation 'route the fields through
    OPTIONAL_METRIC_FIELDS' — every metric entry would gain both keys."""
    entry = _canonical_metric(env, _document())
    assert "unit" not in entry and "estimand_version" not in entry


@pytest.mark.parametrize(
    "metric, expect",
    [
        ({"unit": "percentage_points"}, "computes a value in 'fraction'"),
        ({"unit": "pct"}, "is not one of"),
        ({"estimand_version": "ratio_of_sums/v1"}, "computes 'match/v1'"),
        ({"estimand_version": "Match/V1"}, "<estimand>/v<n>"),
    ],
)
def test_an_identity_the_kind_does_not_compute_is_a_p4(metric, expect) -> None:
    """A document may state a kind's unit and arithmetic, not change them."""
    with pytest.raises(ParseError) as err:
        parse_document(_document(**metric))
    assert err.value.code == "P4" and expect in str(err.value)
    assert err.value.path is not None and err.value.path.startswith(
        "save[0].aggregation."
    )


def test_the_identity_fields_are_not_sweepable() -> None:
    """An identity is not a research variable: a sweep wrapper on either
    field is rule 14's refusal, as on `token_form`."""
    for field in ("unit", "estimand_version"):
        with pytest.raises(ValidationError, match="sweep wrapper is not allowed"):
            parse_document(_document(**{field: {"sweep": ["fraction", "count"]}}))


def test_a_kind_with_no_scalar_refuses_an_authored_unit() -> None:
    raw = _document()
    raw["method"]["save"][0]["aggregation"] = {"kind": "decode", "unit": "fraction"}
    with pytest.raises(ParseError, match="produces no scalar"):
        parse_document(raw)


def test_stating_the_derived_identity_is_legitimate_for_every_scalar_kind() -> None:
    """The twin: every kind accepts its own unit and identifier. Fails if the
    parser ever refuses the truth."""
    for kind, own in METRIC_UNITS.items():
        if own is None:
            continue
        raw = _document()
        fields = {
            "logit_diff": {"a": "cf_answer", "b": "base_answer"},
            "soft_accuracy": {"a": "cf_answer", "b": "base_answer"},
            "token_logit": {"token": "cf_answer"},
            "cross_entropy": {"target": "cf_answer"},
            "kl": {"target": {"read": "logits", "model": "original"}},
            "js": {"target": {"read": "logits", "model": "original"}},
            "class_probs": {"groups": {"x": ["a"]}},
            "token_logits": {"tokens": ["a"]},
            "match": {"expected": "cf_answer"},
        }[kind]
        raw["method"]["save"][0]["aggregation"] = {
            "kind": kind,
            "unit": own,
            "estimand_version": metric_identity(kind),
            **fields,
        }
        spec = by_label(parse_document(raw))["acc"]
        assert spec.unit == own and spec.estimand_version == f"{kind}/v1"


# --------------------------------------------------------------------------- #
# T7 / T8 — the mismatch refusal and its legitimate twins
# --------------------------------------------------------------------------- #


def test_t7_fraction_against_percentage_points_is_refused_naming_both() -> None:
    """*Mutation:* compare the metric names rather than the units — both
    records are named `iia.json`, so a name comparison passes and this test's
    `raises` fails."""
    left = Record("arm_a/iia.json", unit="fraction", estimand_version="mean/v1")
    right = Record(
        "arm_b/iia.json", unit="percentage_points", estimand_version="mean/v1"
    )
    with pytest.raises(EstimandError) as err:
        compare(left, right)
    message = str(err.value)
    for word in ("fraction", "percentage_points", "arm_a/iia.json", "arm_b/iia.json"):
        assert word in message
    assert "fraction cannot be compared to percentage points" in message


def test_t7_the_refusal_is_symmetric() -> None:
    with pytest.raises(EstimandError, match="unit mismatch"):
        compare(Record("b", unit="percentage_points"), Record("a", unit="fraction"))


def test_t8_two_arms_with_nothing_declared_compare_cleanly() -> None:
    """Valid work is not refused: a unit check that refuses an arm-vs-arm
    comparison is worse than no unit check."""
    result = compare(Record("arm_a/iia.json"), Record("arm_b/iia.json"))
    assert result.kind == "arm" and result.unit is None


def test_t8_same_unit_same_estimand_is_an_arm_comparison() -> None:
    result = compare(
        Record("arm_a", unit="fraction", estimand_version="mean/v1"),
        Record("arm_b", unit="fraction", estimand_version="mean/v1"),
    )
    assert result.kind == "arm" and result.unit == "fraction"


def test_t8_one_declared_side_is_not_a_mismatch() -> None:
    """An unknown unit is not a wrong unit: a table written before units
    existed compares with one written after."""
    result = compare(Record("old", unit=None), Record("new", unit="fraction"))
    assert result.kind == "arm" and result.unit == "fraction"


def test_t8_same_unit_two_estimands_is_a_labelled_version_comparison() -> None:
    result = compare(
        Record("a", unit="fraction", estimand_version="mean_of_eligible_row_ratios/v1"),
        Record("b", unit="fraction", estimand_version="ratio_of_sums/v1"),
    )
    assert result.kind == "version"


def test_a_table_in_two_units_is_one_refusal_naming_both() -> None:
    rows = [
        {"value": 0.5, "unit": "fraction"},
        {"value": 50.0, "unit": "percentage_points"},
    ]
    with pytest.raises(EstimandError) as err:
        table_record(rows, name="mixed.json")
    assert "fraction" in str(err.value) and "percentage_points" in str(err.value)
    assert "mixed.json" in str(err.value)


def test_a_table_record_reads_the_repeated_columns() -> None:
    rows = [
        {"value": 0.5, "unit": "nat", "estimand_version": "kl/v1"} for _ in range(3)
    ]
    record = table_record(rows, name="kl.json")
    assert record == Record("kl.json", "nat", "kl/v1")
    # rows of two metrics: one record, undeclared estimand, still one unit
    rows[0]["estimand_version"] = "cross_entropy/v1"
    assert table_record(rows, name="t").estimand_version is None
    # a table with no identity columns at all is an undeclared record
    assert table_record([{"value": 1.0}], name="t") == Record("t")


# --------------------------------------------------------------------------- #
# T10 — the stale claim
# --------------------------------------------------------------------------- #

#: a reduced row of a one-axis group-by: the coordinate is what a claim
#: selects the row by
WHERE = {"sites.target.layers": 3}
RECORD = [
    {
        **WHERE,
        "value": 0.5625,
        "n": 4,
        "unit": "fraction",
        "estimand_version": "mean_of_eligible_row_ratios/v1",
    }
]
CLAIM = Claim(
    file="reduced.json",
    where=WHERE,
    estimand_version="mean_of_eligible_row_ratios/v1",
    unit="fraction",
    value=0.5625,
)


def test_t10_a_claim_its_record_still_supports_passes() -> None:
    """The twin: a claim that is true is not refused."""
    check_claim(CLAIM, RECORD)


def test_t10_a_recomputed_value_refuses_naming_record_and_coordinates() -> None:
    """*Mutation:* bind by file path only — the path is `reduced.json` before
    and after, so a path check passes and this `raises` fails."""
    recomputed = copy.deepcopy(RECORD)
    recomputed[0]["value"] = 0.5454545454545454
    with pytest.raises(EstimandError) as err:
        check_claim(CLAIM, recomputed)
    message = str(err.value)
    assert "stale claim" in message and "reduced.json" in message
    assert '"sites.target.layers": 3' in message
    assert "0.5625" in message and "0.5454545454545454" in message


def test_t10_a_row_that_moved_off_the_claims_coordinates_binds_to_nothing() -> None:
    """The group-by changed and the row the claim named is gone from the file
    — the path did not change."""
    moved = copy.deepcopy(RECORD)
    moved[0]["sites.target.layers"] = 4
    with pytest.raises(EstimandError) as err:
        check_claim(CLAIM, moved)
    assert "binds to nothing" in str(err.value)
    assert '"sites.target.layers": 3' in str(err.value)


def test_t10_an_empty_where_binds_a_one_row_table() -> None:
    """A reduction with no group-by writes one row; the claim names no
    column and binds it."""
    check_claim(dataclasses.replace(CLAIM, where={}), RECORD)


def test_t10_a_where_column_no_row_carries_is_refused_naming_it() -> None:
    """A typo'd or dropped column is a different table, not "no row at those
    coordinates": the refusal names the column."""
    with pytest.raises(EstimandError) as err:
        check_claim(dataclasses.replace(CLAIM, where={"layer": 3}), RECORD)
    assert "['layer']" in str(err.value)
    assert "no row of the file carries" in str(err.value)


def test_t10_an_absent_column_does_not_match_none() -> None:
    """A row without the column is not a row where the column is `None`: a
    claim at `None` binds nothing on such a row (the reduced row that dropped
    its group-by coordinate is not "the same record")."""
    rows = copy.deepcopy(RECORD)
    rows.append({k: v for k, v in RECORD[0].items() if k not in WHERE})
    with pytest.raises(EstimandError) as err:
        check_claim(
            dataclasses.replace(CLAIM, where={"sites.target.layers": None}), rows
        )
    assert "binds to nothing" in str(err.value)


def test_t10_a_claim_in_the_wrong_unit_or_estimand_is_refused() -> None:
    with pytest.raises(EstimandError, match="'percentage_points'"):
        check_claim(
            Claim(
                CLAIM.file, WHERE, CLAIM.estimand_version, "percentage_points", 56.25
            ),
            RECORD,
        )
    with pytest.raises(EstimandError, match="ratio_of_sums/v1"):
        check_claim(
            Claim(CLAIM.file, WHERE, "ratio_of_sums/v1", "fraction", 0.5625), RECORD
        )


def test_t10_a_claim_binds_one_record() -> None:
    """Many rows at one set of coordinates is a table, not a record: reduce
    first, or name every column that singles the row out."""
    with pytest.raises(EstimandError, match="reduce the table first"):
        check_claim(CLAIM, RECORD + RECORD)


def test_t10_tolerance_is_opt_in_and_exact_by_default() -> None:
    drifted = copy.deepcopy(RECORD)
    drifted[0]["value"] = 0.5625 + 1e-9
    with pytest.raises(EstimandError):
        check_claim(CLAIM, drifted)
    check_claim(CLAIM, drifted, tolerance=1e-6)


# --------------------------------------------------------------------------- #
# T9 — the legitimate campaign: every document, nothing authored, pins still
# --------------------------------------------------------------------------- #


def _aggregations(method: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The aggregations a method's save entries carry, by file stem."""
    return {
        str(entry["file_path"]).rsplit("/", 1)[-1].rsplit(".", 1)[0]: entry[
            "aggregation"
        ]
        for entry in method.get("save", [])
        if isinstance(entry, dict) and "aggregation" in entry
    }


def _assert_no_identity_authored(canonical: dict[str, Any], name: str) -> None:
    for metric, entry in _aggregations(canonical).items():
        assert "unit" not in entry and "estimand_version" not in entry, (
            f"{name}: aggregation {metric!r} gained an identity key"
        )


CORPUS = sorted(p.name for p in CORPUS_DIR.glob("*_im.json"))
GOLDEN = sorted(p.name for p in golden_env.GOLDEN_PROTOCOLS.glob("*_im.json"))


def test_t9_the_census_found_documents() -> None:
    assert len(CORPUS) >= 10 and len(GOLDEN) >= 10


@pytest.mark.parametrize("name", CORPUS)
def test_t9_corpus_documents_pin_unchanged_with_nothing_authored(name, env) -> None:
    """Fails under a materialized default (every pin moves, every metric entry
    gains two keys); passes on the base and here alike — the twin."""
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    pin = CORPUS_PINS[name]
    assert loaded.digests.document == (
        pin["document"] if isinstance(pin, dict) else pin
    )
    _assert_no_identity_authored(loaded.canonical, name)
    for point in steps_of(loaded, env).canonical:
        _assert_no_identity_authored(point, name)


@pytest.mark.parametrize("name", GOLDEN)
def test_t9_golden_documents_pin_unchanged_with_nothing_authored(name) -> None:
    env = golden_env.build_env(Path(tempfile.mkdtemp()))
    loaded = compile_protocol(golden_env.GOLDEN_PROTOCOLS / name, env=env)
    pin = GOLDEN_PINS[name]
    assert loaded.digests.document == pin["document"]
    assert list(steps_of(loaded, env).digests) == pin["points"]
    _assert_no_identity_authored(loaded.canonical, name)


def test_t9_shipped_presets_author_no_identity() -> None:
    """The shipped presets have no digest pin; what is checked is that none
    authors either field and none gains one in its canonical form. The
    four run-tree-only presets do not load standalone
    (``test_protocol_presets.py``); their authored form is checked."""
    presets = sorted(PROTOCOLS.glob("*.json"))
    assert len(presets) >= 5
    for path in presets:
        raw = json.loads(path.read_text())
        for name, entry in _aggregations(raw.get("method", {})).items():
            assert "unit" not in entry and "estimand_version" not in entry, (
                f"{path.name}: {name} authors an identity field"
            )
        try:
            canonical = canonicalize(parse_document(raw))
        except Exception:  # noqa: BLE001 — a run-tree preset needs its step
            continue
        _assert_no_identity_authored(canonical, path.name)
        assert len(digest(canonical)) == 64


def test_t9_a_document_that_authors_nothing_canonicalizes_as_before(env) -> None:
    """Direct form of the twin: the canonical metric entry of an unauthored
    document has exactly the keys it had on the base."""
    entry = _canonical_metric(env, _document())
    assert set(entry) == {"kind", "expected", "mode"}
