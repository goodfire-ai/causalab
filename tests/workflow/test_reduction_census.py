"""Censuses for the reduction contract (workflow spec §2.6, §5) — T4 and T5.

Two kinds of guard, the same discipline as ``tests/protocol/test_vocabulary_census.py``
(whose table parser this copies rather than imports: that module's ``SPEC`` is
the intervention protocol's, and a guard should not depend on another test
module's globals).

* **The checklist census.** §5's numbered list has exactly ``MAX_RULE`` items,
  numbered 1 … ``MAX_RULE`` in order. Until now the workflow checklist had a
  code constant and no census, so a rule could be added to the spec and never
  to the code, or the reverse. Each parse asserts it found something first: a
  census over a list the parser failed to find passes for the wrong reason.
* **The vocabulary censuses.** §2.6's five tables — unit kinds, missing
  policies, uncertainty procedures, estimators, curve scales — are exactly
  the code's five tuples. A table behind the code is worse than no table: it reads as
  complete.
* **T4, the legitimate campaign.** Every shipped workflow
  (``demos/methods/workflows/*.json``) and every demo workflow
  (``demos/*/workflows/*.json``) loads, digests to its pin, and carries **no**
  ``reduction`` key in any canonical step entry. This is what proves the block
  is modelled on ``runtime`` (absent when unauthored) and not on
  ``is_deterministic`` (materialized): a materialized default fails this test
  on every one of them. ``tests/golden/`` holds no workflow documents, so that
  clause of T4 is vacuous on this base and is not faked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from causalab.io.env import ResolutionEnv
from causalab.workflow.document import MAX_RULE, load_workflow
from causalab.workflow.reduction import (
    ESTIMATORS,
    MISSING_POLICIES,
    UNCERTAINTY_KINDS,
    UNIT_KINDS,
    X_SCALES,
)
from tests._helpers.paths import WORKFLOWS_DIR

from tests._helpers.demos import demo_env, demo_workflows

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "workflow_protocol.md"
WORKFLOWS = WORKFLOWS_DIR
DEMOS = REPO / "demos"

ROW = re.compile(r"^[ \t]*\|(.+)\|\s*$", re.M)
CODE = re.compile(r"`([^`]+)`")
#: A numbered list item at the start of a line: ``12. …``
ITEM = re.compile(r"^(\d+)\. ", re.M)


def _section(heading: str) -> str:
    """The text under ``heading``, up to the next heading of equal or lesser
    depth — the same rule as the protocol census, for the same reason."""
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


def _table(header: str) -> list[list[str]]:
    """§2.6's table whose header row starts with ``header``: its body rows,
    up to the first row that is not a backticked member."""
    rows = _rows(_section("### 2.6 `reduction`"))
    start = next(index for index, row in enumerate(rows) if row[0] == header)
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def _members(header: str) -> list[str]:
    return [CODE.findall(row[0])[0] for row in _table(header)]


# --------------------------------------------------------------------------- #
# T5 — the checklist census
# --------------------------------------------------------------------------- #


def _checklist_numbers() -> list[int]:
    return [int(n) for n in ITEM.findall(_section("## 5. Validation"))]


def test_the_checklist_parse_found_something() -> None:
    assert len(_checklist_numbers()) >= 11


def test_the_checklist_and_max_rule_agree() -> None:
    """Fails without the change: the spec has 12 items and `MAX_RULE` is 11
    (or the reverse, on a spec edit with no code change)."""
    numbers = _checklist_numbers()
    assert numbers == list(range(1, MAX_RULE + 1)), (
        f"§5 lists rules {numbers}; MAX_RULE is {MAX_RULE}"
    )


def test_rule_12_is_the_reduction_rule() -> None:
    """Numbered by rule, so a renumbering is a deliberate edit here."""
    section = _section("## 5. Validation")
    item = re.search(r"^12\. (.+?)(?=^\d+\. |\Z)", section, re.M | re.S)
    assert item is not None
    text = " ".join(item.group(1).split())
    assert "`reduction`" in text and "names the field" in text


# --------------------------------------------------------------------------- #
# T5 — the vocabulary censuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "header, vocabulary",
    [
        ("unit", UNIT_KINDS),
        ("policy", MISSING_POLICIES),
        ("procedure", UNCERTAINTY_KINDS),
        ("estimator", ESTIMATORS),
        ("scale", X_SCALES),
    ],
    ids=["unit", "missing", "uncertainty", "estimator", "x_scale"],
)
def test_the_vocabulary_tables_are_not_empty(header, vocabulary) -> None:
    assert len(_table(header)) >= len(vocabulary)


@pytest.mark.parametrize(
    "header, vocabulary, name",
    [
        ("unit", UNIT_KINDS, "UNIT_KINDS"),
        ("policy", MISSING_POLICIES, "MISSING_POLICIES"),
        ("procedure", UNCERTAINTY_KINDS, "UNCERTAINTY_KINDS"),
        ("estimator", ESTIMATORS, "ESTIMATORS"),
        ("scale", X_SCALES, "X_SCALES"),
    ],
    ids=["unit", "missing", "uncertainty", "estimator", "x_scale"],
)
def test_each_vocabulary_matches_its_spec_table(header, vocabulary, name) -> None:
    tabulated = _members(header)
    assert len(set(tabulated)) == len(tabulated), f"§2.6 lists a {header} twice"
    assert set(tabulated) == set(vocabulary), (
        f"§2.6's {header} table and {name} disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(vocabulary))}; "
        f"only in the code: {sorted(set(vocabulary) - set(tabulated))}"
    )


def test_every_vocabulary_row_says_what_it_means() -> None:
    """A row that names the member and stops documents nothing a reader could
    not guess from the name."""
    for header in ("unit", "policy", "procedure", "estimator", "scale"):
        for row in _table(header):
            assert len(row) >= 2 and len(" ".join(row[1:])) > 20, (
                f"§2.6 {header} row {row[0]} is bare"
            )


def _dimension_rows() -> list[list[str]]:
    """The dimension table: its first cells are prose, so its body is every
    three-cell row after the header up to the next (two-cell) table."""
    rows = _rows(_section("### 2.6 `reduction`"))
    start = next(index for index, row in enumerate(rows) if row[0] == "dimension")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if len(row) != 3:
            break
        body.append(row)
    return body


def test_the_eight_dimensions_are_tabulated_once() -> None:
    """The dimension table names exactly the eight, by their fields."""
    fields = [CODE.findall(row[1])[0] for row in _dimension_rows()]
    assert fields == [
        "unit",
        "group_by",
        "weight",
        "missing",
        "uncertainty.kind",
        "uncertainty.resample_unit",
        "uncertainty.repetitions",
        "uncertainty.seed",
    ]


# --------------------------------------------------------------------------- #
# T4 — the legitimate campaign: nothing authored, nothing moved
# --------------------------------------------------------------------------- #


def _demo_env(document: Path) -> ResolutionEnv:
    """A demo carries its own tables (``tests/_helpers/demos.py``)."""
    return demo_env(document)


SHIPPED = sorted(WORKFLOWS.glob("*.json"))
DEMO_WORKFLOWS = demo_workflows()


def test_the_workflow_census_found_something() -> None:
    assert len(SHIPPED) >= 2 and len(DEMO_WORKFLOWS) >= 1


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_t4_shipped_workflows_author_no_reduction(env, path) -> None:
    """Fails on a materialized default (every entry gains a key); passes on
    the base and on this change alike — it is the twin."""
    loaded = load_workflow(path, env)
    for name, entry in loaded.canonical["steps"].items():
        assert "reduction" not in entry, f"{path.name}: {name} gained a reduction key"


@pytest.mark.parametrize(
    "path",
    DEMO_WORKFLOWS,
    ids=[f"{p.parents[1].name}/{p.name}" for p in DEMO_WORKFLOWS],
)
def test_t4_demo_workflows_author_no_reduction(path) -> None:
    """The demo pins live in the demos' markdown and are held by
    ``tests/demos/test_demos.py``; this holds the canonical entries."""
    loaded = load_workflow(path, _demo_env(path))
    for name, entry in loaded.canonical["steps"].items():
        assert "reduction" not in entry, f"{path}: {name} gained a reduction key"


def test_t4_no_golden_workflows_exist_to_check() -> None:
    """Said rather than faked: `tests/golden/` holds intervention specifications only.
    If a workflow lands there, add it to this census."""
    assert not list((REPO / "tests" / "golden").rglob("workflows/*.json"))
