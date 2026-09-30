"""Estimand identity, the workflow side (workflow spec §2.6, rule 13; IM spec
§2.10; ``causalab/protocol/estimand.py``).

* **T6** — the same table reduced as ``mean_of_eligible_row_ratios/v1`` and as
  ``ratio_of_sums/v1`` gives **different numbers** and **different
  identifiers**, and a document declaring one while computing the other fails
  at load (rule 13) naming both. *Mutation:* equalize the fixture's row
  denominators → the two arithmetics coincide, and the inequality this test
  asserts is exactly what fails; ``test_t6_the_fixture_is_not_degenerate``
  is the guard against the guard, and
  ``test_t6_equal_denominators_would_make_the_test_vacuous`` shows why;
* **T7** — a ``fraction`` record compared with a ``percentage_points`` record
  is refused naming both, at both shipped comparison sites: the reduction (rows
  in two units) and ``paired_ttest`` (two inputs). *Mutation:* compare the
  metric *names* rather than the units — both sides are ``iia.json`` — and the
  refusal never fires;
* **T8** — two arms with nothing declared compare cleanly through
  ``paired_ttest``, and the same unit under two declared estimands is labelled
  ``version``, never refused;
* **T9** — every shipped and demo workflow canonicalizes to its pin with no
  ``estimand_version`` anywhere; the derived identity appears on metric rows
  (``MetricTable``) and reduced rows;
* **T10** — a claim bound to a reduced record is refused after the table is
  recomputed to a different value, naming the record and the point digest.
  *Mutation:* bind by file path only and the test fails: the path is
  unchanged;
* **the censuses** — §5 has ``MAX_RULE`` items, rule 13 is the estimand
  rule, and §2.6's identifier table is exactly ``REDUCTION_ESTIMANDS``.

Every test here fails without the change: ``estimand_version`` is an unknown
reduction key (rule 12), ``MAX_RULE`` is 12, and the identity columns do not
exist.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.analysis import paired_ttest
from causalab.io.step_io import StepError
from causalab.neural.shared.results import MetricTable
from causalab.protocol.estimand import (
    REDUCTION_ESTIMANDS,
    Claim,
    EstimandError,
    admissible_reduction_identifiers,
    check_claim,
    metric_record_identity,
    reduction_identity,
    table_record,
)
from causalab.io.env import ResolutionEnv
from causalab.io.tables import read_table
from causalab.workflow.document import MAX_RULE, WorkflowError, load_workflow
from causalab.workflow.reduction import (
    IDENTITY_RULE,
    OUTPUT_COLUMNS,
    REDUCTION_INPUT,
    ReductionSpecError,
    parse_reduction,
)
from causalab.workflow.runner import run_workflow
from causalab.workflow.scripts import reduce
from tests.step_scripts import put_table, run_step
from tests._helpers.paths import WORKFLOWS_DIR

from tests._helpers.demos import demo_env, demo_workflows

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"
WORKFLOWS = WORKFLOWS_DIR
DEMOS = REPO / "demos"

# --------------------------------------------------------------------------- #
# the fixture — a per-row ratio with unequal denominators
# --------------------------------------------------------------------------- #

#: (numerator, denominator) per row. Denominators differ on purpose: a mean
#: of ratios and a ratio of sums coincide exactly when they do not.
RATIOS: tuple[tuple[int, int], ...] = ((1, 2), (3, 4), (1, 8), (7, 8))


def ratio_table(pairs=RATIOS, *, unit: str | None = "fraction") -> list[dict[str, Any]]:
    """One row per example: ``value`` is the row's ratio, ``denominator`` its
    denominator — the shape `addition`'s normalized recovery takes."""
    rows = []
    for example, (numerator, denominator) in enumerate(pairs):
        row: dict[str, Any] = {
            "example_id": str(example),
            "metric": "recovery",
            "value": numerator / denominator,
            "denominator": denominator,
            "estimand_version": "logit_diff/v1",
        }
        if unit is not None:
            row["unit"] = unit
        rows.append(row)
    return rows


MEAN_OF_RATIOS: dict[str, Any] = {
    "estimator": {"kind": "mean"},
    "unit": {"kind": "row"},
    "group_by": [],
    "weight": None,
    "missing": "exclude",
    "uncertainty": {"kind": "none"},
    "estimand_version": "mean_of_eligible_row_ratios/v1",
}
RATIO_OF_SUMS: dict[str, Any] = {
    **MEAN_OF_RATIOS,
    "estimator": {"kind": "weighted_mean"},
    "weight": "denominator",
    "estimand_version": "ratio_of_sums/v1",
}


def _reduce(tmp_path: Path, rows: list[dict], block: dict, tag: str) -> dict[str, Any]:
    table = put_table(tmp_path / tag / "recovery.json", rows)
    out = tmp_path / tag / "reduced.json"
    run_step(reduce, {"table": table, REDUCTION_INPUT: block}, {"table": out})
    (result,) = read_table(out)
    return result


def reduce_workflow(table: Path, reduction: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "facts": {
                "type": "script",
                "script": {"module": "causalab.workflow.scripts.reduce"},
                "inputs": {"table": {"path": str(table)}},
                "outputs": {
                    "table": {
                        "file": "reduced.json",
                        "columns": {"value": "float64", "n": "int64"},
                    }
                },
                "reduction": reduction,
            }
        },
    }


# --------------------------------------------------------------------------- #
# T6 — two identifiers, two numbers, and the load refusal
# --------------------------------------------------------------------------- #


def test_t6_the_fixture_is_not_degenerate() -> None:
    """The guard against the guard: with equal denominators the two
    arithmetics coincide and every T6 assertion below passes vacuously."""
    assert len({d for _, d in RATIOS}) > 1


def test_t6_equal_denominators_would_make_the_test_vacuous(tmp_path) -> None:
    """Shown rather than asserted away: on an equalized fixture the two
    reductions give the *same* number, which is why the fixture's
    denominators must differ for the test after this one to mean anything."""
    equal = ratio_table(((1, 4), (3, 4), (1, 4), (3, 4)))
    a = _reduce(tmp_path, equal, MEAN_OF_RATIOS, "a")
    b = _reduce(tmp_path, equal, RATIO_OF_SUMS, "b")
    assert a["value"] == pytest.approx(b["value"])


def test_t6_two_identifiers_give_two_numbers(tmp_path) -> None:
    """Fails without the change: `estimand_version` is an unknown key. Fails
    on an equalized fixture: the two values are equal."""
    rows = ratio_table()
    a = _reduce(tmp_path, rows, MEAN_OF_RATIOS, "a")
    b = _reduce(tmp_path, rows, RATIO_OF_SUMS, "b")
    assert a["value"] == pytest.approx((1 / 2 + 3 / 4 + 1 / 8 + 7 / 8) / 4)  # 0.5625
    assert b["value"] == pytest.approx((1 + 3 + 1 + 7) / (2 + 4 + 8 + 8))  # 12/22
    assert a["value"] != b["value"]
    assert a["estimand_version"] == "mean_of_eligible_row_ratios/v1"
    assert b["estimand_version"] == "ratio_of_sums/v1"
    assert a["estimand_version"] != b["estimand_version"]
    # same table, same unit — only the arithmetic differs
    assert a["unit"] == b["unit"] == "fraction"
    assert a["n"] == b["n"] == 4 and a["n_excluded"] == b["n_excluded"] == 0


def test_t6_eligible_means_the_rows_exclude_kept(tmp_path) -> None:
    """The name's 'eligible' is `missing: exclude`'s count, nothing new (8·4)."""
    rows = ratio_table()
    rows[1]["value"] = None
    result = _reduce(tmp_path, rows, MEAN_OF_RATIOS, "a")
    assert result["n"] == 3 and result["n_excluded"] == 1
    assert result["value"] == pytest.approx((1 / 2 + 1 / 8 + 7 / 8) / 3)


def test_t6_declaring_one_while_computing_the_other_fails_at_load(env, tmp_path):
    """Rule 13, naming the declared identifier, the computed one and the
    admissible set. Fails without the change: rule 12 refuses the key
    instead, with no identifier in the message."""
    table = put_table(tmp_path / "recovery.json", ratio_table())
    lying = {**MEAN_OF_RATIOS, "estimand_version": "ratio_of_sums/v1"}
    with pytest.raises(WorkflowError) as err:
        load_workflow(reduce_workflow(table, lying), env, workflow_dir=tmp_path)
    assert err.value.rule == IDENTITY_RULE == 13
    message = str(err.value)
    assert "ratio_of_sums/v1" in message and "mean/v1" in message
    assert "mean_of_eligible_row_ratios/v1" in message
    assert "steps.facts.reduction.estimand_version" in message


def test_t6_the_other_direction_is_refused_too(env, tmp_path) -> None:
    table = put_table(tmp_path / "recovery.json", ratio_table())
    lying = {**RATIO_OF_SUMS, "estimand_version": "mean_of_eligible_row_ratios/v1"}
    with pytest.raises(WorkflowError, match="weighted_mean/v1") as err:
        load_workflow(reduce_workflow(table, lying), env, workflow_dir=tmp_path)
    assert err.value.rule == 13


def test_t6_the_honest_declarations_load_run_and_digest_apart(env, tmp_path):
    """The twin of the refusal, and the digest half of 'different
    identifiers': two documents that differ only in the arithmetic they name
    are two documents."""
    table = put_table(tmp_path / "recovery.json", ratio_table())
    a = load_workflow(
        reduce_workflow(table, MEAN_OF_RATIOS), env, workflow_dir=tmp_path
    )
    b = load_workflow(reduce_workflow(table, RATIO_OF_SUMS), env, workflow_dir=tmp_path)
    assert a.digest != b.digest
    assert a.canonical["steps"]["facts"]["reduction"]["estimand_version"] == (
        "mean_of_eligible_row_ratios/v1"
    )
    result = run_workflow(a, env, tmp_path / "run_a", engine=None)
    (row,) = read_table(result.run_root / "facts" / "reduced.json")
    assert row["estimand_version"] == "mean_of_eligible_row_ratios/v1"
    assert row["value"] == pytest.approx(0.5625)


def test_t6_the_identifier_is_canonical_only_when_authored(env, tmp_path) -> None:
    """A block that names nothing keeps the digest it had before the
    identifier existed, and its rows carry the estimator's own identifier."""
    table = put_table(tmp_path / "recovery.json", ratio_table())
    bare = {k: v for k, v in MEAN_OF_RATIOS.items() if k != "estimand_version"}
    loaded = load_workflow(reduce_workflow(table, bare), env, workflow_dir=tmp_path)
    assert "estimand_version" not in loaded.canonical["steps"]["facts"]["reduction"]
    named = load_workflow(
        reduce_workflow(table, MEAN_OF_RATIOS), env, workflow_dir=tmp_path
    )
    assert loaded.digest != named.digest
    result = run_workflow(loaded, env, tmp_path / "run", engine=None)
    (row,) = read_table(result.run_root / "facts" / "reduced.json")
    assert row["estimand_version"] == "mean/v1"


def test_t6_a_malformed_identifier_is_refused_naming_the_grammar(env, tmp_path):
    table = put_table(tmp_path / "recovery.json", ratio_table())
    bad = {**MEAN_OF_RATIOS, "estimand_version": "Mean Of Ratios"}
    with pytest.raises(WorkflowError, match="<estimand>/v<n>"):
        load_workflow(reduce_workflow(table, bad), env, workflow_dir=tmp_path)


def test_t6_the_script_revalidates_the_identifier(tmp_path) -> None:
    """A direct call is refused as a document would be."""
    lying = {**MEAN_OF_RATIOS, "estimand_version": "ratio_of_sums/v1"}
    with pytest.raises(StepError, match="ratio_of_sums/v1"):
        _reduce(tmp_path, ratio_table(), lying, "a")


def test_admissible_identifiers_follow_the_block() -> None:
    """Which estimators admit which identifiers — the spec table as code."""
    mean_row = parse_reduction(
        {k: v for k, v in MEAN_OF_RATIOS.items() if k != "estimand_version"}
    )
    assert admissible_reduction_identifiers(mean_row.canonical()) == [
        "mean/v1",
        "mean_of_eligible_row_ratios/v1",
    ]
    # `mean` over examples, not rows: the row-ratio name is not admissible
    by_example = parse_reduction(
        {**mean_row.canonical(), "unit": {"kind": "example", "columns": ["example_id"]}}
    )
    assert admissible_reduction_identifiers(by_example.canonical()) == ["mean/v1"]
    # `missing: error` keeps every row, so no row was "eligible" rather than kept
    strict = parse_reduction({**mean_row.canonical(), "missing": "error"})
    assert admissible_reduction_identifiers(strict.canonical()) == ["mean/v1"]
    ratio = parse_reduction(
        {k: v for k, v in RATIO_OF_SUMS.items() if k != "estimand_version"}
    )
    assert admissible_reduction_identifiers(ratio.canonical()) == [
        "weighted_mean/v1",
        "ratio_of_sums/v1",
    ]
    for kind in ("sum", "count", "median"):
        block = {**mean_row.canonical(), "estimator": {"kind": kind}}
        assert admissible_reduction_identifiers(block) == [f"{kind}/v1"]
    assert reduction_identity(mean_row.canonical()) == "mean/v1"
    with pytest.raises(EstimandError, match="admissible here"):
        reduction_identity(mean_row.canonical(), "ratio_of_sums/v1")


def test_parse_reduction_names_the_field_and_the_rule() -> None:
    with pytest.raises(ReductionSpecError) as err:
        parse_reduction({**MEAN_OF_RATIOS, "estimand_version": "ratio_of_sums/v1"})
    assert err.value.field == "estimand_version" and err.value.rule == 13
    with pytest.raises(ReductionSpecError) as err:
        parse_reduction({**MEAN_OF_RATIOS, "estimand_version": 7})
    assert err.value.rule == 13 and "<estimand>/v<n>" in str(err.value)


# --------------------------------------------------------------------------- #
# T7 — the mismatch refusal at both comparison sites
# --------------------------------------------------------------------------- #


def test_t7_a_reduction_over_rows_in_two_units_is_refused_naming_both(tmp_path):
    """*Mutation:* compare metric names — every row is `recovery` — and the
    refusal never fires."""
    rows = ratio_table()
    rows[2]["unit"] = "percentage_points"
    rows[2]["value"] *= 100
    with pytest.raises(StepError) as err:
        _reduce(tmp_path, rows, MEAN_OF_RATIOS, "a")
    message = str(err.value)
    assert "fraction" in message and "percentage_points" in message
    assert "recovery.json" in message


def _ttest(tmp_path: Path, a: list[dict], b: list[dict]) -> dict[str, Any]:
    left = put_table(tmp_path / "arm_a" / "iia.json", a)
    right = put_table(tmp_path / "arm_b" / "iia.json", b)
    out = tmp_path / "stats.json"
    run_step(paired_ttest, {"a": left, "b": right}, {"stats": out})
    (row,) = read_table(out)
    return row


def _arm(values, *, unit="fraction", estimand="match/v1") -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(i),
            "metric": "iia",
            "value": v,
            "unit": unit,
            "estimand_version": estimand,
        }
        for i, v in enumerate(values)
    ]


def test_t7_paired_ttest_refuses_a_fraction_against_percentage_points(tmp_path):
    """Both inputs are `iia.json`, so a check on names could never fire; the
    refusal names the units and the slots."""
    with pytest.raises(StepError) as err:
        _ttest(
            tmp_path,
            _arm([0.5, 0.75, 0.25, 1.0]),
            _arm([50.0, 75.0, 25.0, 100.0], unit="percentage_points"),
        )
    message = str(err.value)
    assert "fraction" in message and "percentage_points" in message
    assert "input 'a'" in message and "input 'b'" in message
    assert "fraction cannot be compared to percentage points" in message


# --------------------------------------------------------------------------- #
# T8 — the legitimate comparisons
# --------------------------------------------------------------------------- #


def test_t8_two_arms_with_nothing_declared_compare_cleanly(tmp_path) -> None:
    """Valid work is not refused: two tables written before units existed
    still compare."""
    a = [{"example_id": str(i), "value": v} for i, v in enumerate([1.0, 2.0, 3.0, 4.0])]
    b = [{"example_id": str(i), "value": v} for i, v in enumerate([0.5, 1.0, 2.5, 3.0])]
    row = _ttest(tmp_path, a, b)
    assert row["comparison"] == "arm" and row["unit"] is None
    assert row["mean_difference"] == pytest.approx(0.75)


def test_t8_two_arms_in_the_same_unit_and_estimand_are_an_arm_comparison(tmp_path):
    row = _ttest(tmp_path, _arm([0.5, 0.75, 0.25, 1.0]), _arm([0.25, 0.5, 0.0, 0.75]))
    assert row["comparison"] == "arm" and row["unit"] == "fraction"
    assert row["mean_difference"] == pytest.approx(0.25)


def test_t8_a_version_comparison_is_labelled_not_refused(tmp_path) -> None:
    row = _ttest(
        tmp_path,
        _arm([0.5, 0.75, 0.25, 1.0], estimand="mean_of_eligible_row_ratios/v1"),
        _arm([0.4, 0.7, 0.3, 0.9], estimand="ratio_of_sums/v1"),
    )
    assert row["comparison"] == "version" and row["unit"] == "fraction"


def test_t8_one_declared_side_compares_cleanly(tmp_path) -> None:
    """A table with a unit against one without: unknown is not wrong."""
    old = [
        {"example_id": str(i), "value": v} for i, v in enumerate([0.5, 0.75, 0.25, 1.0])
    ]
    row = _ttest(tmp_path, old, _arm([0.25, 0.5, 0.0, 0.75]))
    assert row["comparison"] == "arm" and row["unit"] == "fraction"


def test_t8_a_reduction_over_an_unlabelled_table_runs_with_an_unknown_unit(tmp_path):
    """A pre-identity table reduces; its unit is `null`, its estimand the
    estimator's own."""
    rows = [
        {"example_id": str(i), "value": v} for i, v in enumerate([0.5, 0.75, 0.25, 1.0])
    ]
    block = {k: v for k, v in MEAN_OF_RATIOS.items() if k != "estimand_version"}
    result = _reduce(tmp_path, rows, block, "a")
    assert result["unit"] is None and result["estimand_version"] == "mean/v1"
    assert result["value"] == pytest.approx(0.625)


# --------------------------------------------------------------------------- #
# T9 — nothing authored, nothing moved; the derived identity on every row
# --------------------------------------------------------------------------- #


def _demo_env(document: Path) -> ResolutionEnv:
    """A demo carries its own tables (``tests/_helpers/demos.py``)."""
    return demo_env(document)


SHIPPED = sorted(WORKFLOWS.glob("*.json"))
DEMO_WORKFLOWS = demo_workflows()


def _assert_no_identity(canonical: dict[str, Any], name: str) -> None:
    for step, entry in canonical["steps"].items():
        assert "estimand_version" not in json.dumps(entry), f"{name}: {step}"


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_t9_shipped_workflows_author_no_identity(env, path):
    loaded = load_workflow(path, env)
    _assert_no_identity(loaded.canonical, path.name)


@pytest.mark.parametrize(
    "path",
    DEMO_WORKFLOWS,
    ids=[f"{p.parents[1].name}/{p.name}" for p in DEMO_WORKFLOWS],
)
def test_t9_demo_workflows_author_no_identity(path) -> None:
    _assert_no_identity(load_workflow(path, _demo_env(path)).canonical, str(path))


def test_t9_the_workflow_census_found_something() -> None:
    assert len(SHIPPED) >= 2 and len(DEMO_WORKFLOWS) >= 1


def test_t9_metric_rows_carry_the_derived_identity() -> None:
    """`MetricTable._row` (spec §2.10, §6): every row repeats `unit` and
    `estimand_version` beside its coordinate columns, derived from the kind
    when the document says nothing. Fails without the change: the columns do
    not exist."""
    table = MetricTable()
    table.add(
        "ce",
        [0.5, 1.5],
        {"sites.target.layers": 3},
        identity=metric_record_identity("cross_entropy"),
    )
    table.add_windowed(
        "said",
        [[1.0], []],
        {},
        identity=metric_record_identity(
            "match", unit="fraction", estimand_version="match/v1"
        ),
        steps=[[0], []],
        matched=[True, False],
    )
    ce = [row for row in table.rows if row["metric"] == "ce"]
    # `eligible` is the eligibility record (§2.10 "Eligibility"), after
    # the identity; an eligible row carries no reason_code
    assert [list(row) for row in ce] == [
        [
            "example_id",
            "metric",
            "value",
            "sites.target.layers",
            "unit",
            "estimand_version",
            "eligible",
        ]
    ] * 2
    assert {(row["unit"], row["estimand_version"]) for row in ce} == {
        ("nat", "cross_entropy/v1")
    }
    said = [row for row in table.rows if row["metric"] == "said"]
    assert all(
        row["unit"] == "fraction" and row["estimand_version"] == "match/v1"
        for row in said
    )
    assert said[1]["value"] is None and said[1]["matched"] is False
    assert table_record(ce, name="ce.json").unit == "nat"


def test_t9_a_structure_kind_carries_a_null_unit() -> None:
    table = MetricTable()
    table.add("top", [{"indices": [1]}], {}, identity=metric_record_identity("top_k"))
    (row,) = table.rows
    assert row["unit"] is None and row["estimand_version"] == "top_k/v1"


def test_t9_reduced_rows_carry_the_identity_columns_in_order(tmp_path) -> None:
    result = _reduce(tmp_path, ratio_table(), MEAN_OF_RATIOS, "a")
    assert list(result) == list(OUTPUT_COLUMNS)
    assert OUTPUT_COLUMNS[-2:] == ("unit", "estimand_version")


def test_t9_count_is_in_count_whatever_the_table_holds(tmp_path) -> None:
    block = {**MEAN_OF_RATIOS, "estimator": {"kind": "count"}}
    del block["estimand_version"]
    result = _reduce(tmp_path, ratio_table(), block, "a")
    assert result["unit"] == "count" and result["estimand_version"] == "count/v1"
    assert result["value"] == 4


# --------------------------------------------------------------------------- #
# T10 — the stale claim, end to end through the built-in
# --------------------------------------------------------------------------- #


def test_t10_a_recomputed_record_refuses_the_claim_naming_record_and_row(tmp_path):
    """*Mutation:* bind by file path only — `reduced.json` is unchanged as a
    path, so a path check passes and the `raises` below fails. The reduction
    has no group-by, so the file is one row and the claim's `where` is
    empty."""
    result = _reduce(tmp_path, ratio_table(), MEAN_OF_RATIOS, "first")
    claim = Claim(
        file="first/reduced.json",
        where={},
        estimand_version=result["estimand_version"],
        unit=result["unit"],
        value=result["value"],
    )
    check_claim(claim, read_table(tmp_path / "first" / "reduced.json"))  # the twin
    # recompute: one row of the source moved, the output path did not
    changed = ratio_table()
    changed[0]["value"] = 1.0
    put_table(tmp_path / "first" / "recovery.json", changed)
    run_step(
        reduce,
        {
            "table": tmp_path / "first" / "recovery.json",
            REDUCTION_INPUT: MEAN_OF_RATIOS,
        },
        {"table": tmp_path / "first" / "reduced.json"},
    )
    with pytest.raises(EstimandError) as err:
        check_claim(claim, read_table(tmp_path / "first" / "reduced.json"))
    message = str(err.value)
    assert "first/reduced.json" in message
    assert "0.5625" in message


# --------------------------------------------------------------------------- #
# the censuses
# --------------------------------------------------------------------------- #

ROW = re.compile(r"^[ \t]*\|(.+)\|\s*$", re.M)
CODE = re.compile(r"`([^`]+)`")
ITEM = re.compile(r"^(\d+)\. ", re.M)


def _section(heading: str) -> str:
    depth = len(heading) - len(heading.lstrip("#"))
    body = SPEC.read_text().split(heading, 1)
    assert len(body) == 2, f"{heading!r} is not in {SPEC.name}"
    stop = re.compile(rf"^#{{1,{depth}}} ", re.M)
    end = stop.search(body[1])
    return body[1][: end.start()] if end else body[1]


def _rows(table: str) -> list[list[str]]:
    out: list[list[str]] = []
    for match in ROW.finditer(table):
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        out.append(cells)
    return out


def _identifier_table() -> list[list[str]]:
    rows = _rows(_section("### 2.6 `reduction`"))
    start = next(index for index, row in enumerate(rows) if row[0] == "identifier")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def test_rule_13_is_the_estimand_rule() -> None:
    """Rule 14 (controls, `tests/workflow/test_controls.py`) came after; the
    estimand rule keeps its number."""
    assert IDENTITY_RULE == 13 and MAX_RULE >= 13


def test_the_checklist_ends_at_max_rule_and_13_is_the_estimand_rule() -> None:
    numbers = [int(n) for n in ITEM.findall(_section("## 5. Validation"))]
    assert numbers == list(range(1, MAX_RULE + 1)), numbers
    item = re.search(
        r"^13\. (.+?)(?=^\d+\. |\Z)", _section("## 5. Validation"), re.M | re.S
    )
    assert item is not None
    text = item.group(1)
    assert "`reduction.estimand_version`" in text and "<estimand>/v<n>" in text
    assert "ratio_of_sums/v1" in text


def test_the_identifier_table_was_found() -> None:
    assert len(_identifier_table()) >= len(REDUCTION_ESTIMANDS)


def test_the_identifier_table_is_exactly_reduction_estimands() -> None:
    tabulated = [CODE.findall(row[0])[0] for row in _identifier_table()]
    assert len(set(tabulated)) == len(tabulated), tabulated
    coded = [entry.identifier for entry in REDUCTION_ESTIMANDS]
    assert tabulated == coded, (
        "§2.6's identifier table and REDUCTION_ESTIMANDS disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(coded))}; "
        f"only in the code: {sorted(set(coded) - set(tabulated))}"
    )
    for row, entry in zip(_identifier_table(), REDUCTION_ESTIMANDS):
        assert CODE.findall(row[1])[0] == entry.estimator, row
        assert len(row[2]) > 20, row


def test_every_estimator_has_exactly_one_own_identifier() -> None:
    """The unauthored identity is unambiguous: one `<estimator>/v1` each."""
    from causalab.workflow.reduction import ESTIMATORS

    own = [e.identifier for e in REDUCTION_ESTIMANDS if e.unit_kind is None]
    assert own == [f"{kind}/v1" for kind in ESTIMATORS]
    campaign = [e for e in REDUCTION_ESTIMANDS if e.unit_kind is not None]
    assert {e.identifier for e in campaign} == {
        "mean_of_eligible_row_ratios/v1",
        "ratio_of_sums/v1",
    }
    assert {e.estimator for e in campaign} == {"mean", "weighted_mean"}


def test_a_copy_of_the_block_with_no_identifier_is_the_original_form() -> None:
    """Nothing pinned before the identifier moves: the canonical block of an
    unauthored declaration has exactly the six keys it had."""
    block = copy.deepcopy(MEAN_OF_RATIOS)
    del block["estimand_version"]
    assert set(parse_reduction(block).canonical()) == {
        "estimator",
        "unit",
        "group_by",
        "weight",
        "missing",
        "uncertainty",
    }
