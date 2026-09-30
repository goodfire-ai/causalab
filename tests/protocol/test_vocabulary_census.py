"""The closed vocabularies and the spec tables that document them agree.

**Four** vocabularies are closed and grow by PR — site components (§2.4),
metric kinds (§2.10), ``save.reduce`` verbs (§2.12), and engine capabilities
(§8) — and each is compared here to the table that documents it. The §5
checklist has had that guard since it existed
(``test_the_checklist_and_the_spec_agree_on_every_rule`` censuses ``RULES``
against §5's items, slug by slug); the other four did not, so a name could be
added to the code with no row, or a row written for a name that does not exist,
and both readings would look authoritative.

That is the failure this file exists for, and it is not hypothetical for a
closed vocabulary: the *only* thing making "closed" a usable design is that
the table says what is in it. A table that is behind the code is worse than no
table, because it reads as a complete list.

What is checked:

* §2.4's component list is exactly ``COMPONENTS``, compared as a *set*: §2.4
  lists them in a reading order that walks a block, which is not the literal's
  order and nothing depends on either. The count is deliberately not restated
  here — that is the thing this test exists to derive;
* the kinds in §2.10's table are exactly ``METRIC_KINDS``;
* each kind's mandatory and optional value fields, as the table's ``fields``
  cell states them, are exactly ``METRIC_FIELDS`` / ``OPTIONAL_METRIC_FIELDS``;
* the verbs in §2.12's ``reduce`` table are exactly ``SAVE_REDUCTIONS``;
* every ``reduce`` verb has an implementation, and every implementation
  collapses the rows — which is the property ``reduce`` exists for;
* §8's two capability tables are exactly ``CAPABILITIES`` — both the
  ``required when`` table and the per-engine one — and every ✓/✗ cell in the
  latter equals that engine's ``capabilities`` frozenset, with its two
  ``N of M`` counts equal to ``len(engine.components)`` and ``len(COMPONENTS)``. That table replaced
  an aspirational matrix claiming ``grad`` for an engine that never declared
  it, which is the drift this half exists to stop;
* the mechanisms in §2.8's ``do`` table are exactly ``MECHANISMS`` (the
  table stays hand-written because its other two columns are prose);
* the groups in §2.5's ``group`` table are exactly ``GATE_GROUPS``, and each
  row's *axis* cell is the axis kind ``GATE_GROUP_AXES`` ties it to; every
  group has a site selector the legality check reads. This vocabulary has a
  second way to drift that the other two do not: a group with no axis behind
  it does not fail loudly, it resolves to "no grouping" — a coordinate-wise
  gate wearing a grouped gate's name.
* §11.1's five normative nouns are defined once each, are each actually *used*
  somewhere outside their own table, and the overloaded word they replaced does
  not come back — in **every tracked `.md` and `.py` file**, minus four named
  carve-outs. That is wider than the scope §11.1 states ("this spec,
  `docs/workflow_protocol.md`, or a module docstring"), on purpose: an
  inclusion list made each new directory a blind spot, and `README.md` regressed
  within a day of the vocabulary landing. This last vocabulary has no code-side
  enum to drift from — the drift it guards against is prose regressing, one
  sentence at a time, to the word that named five things;
* the hashed-script exemption list is exactly the set of modules a workflow
  `script` step names, so the one place the old wording is deliberately left
  standing cannot quietly grow.

Each parse asserts it found something before comparing: a census against an
empty table passes for the wrong reason, which is the one way a guard like
this fails silently.
"""

from __future__ import annotations

import bisect
import json
import re
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import get_args

import pytest

from causalab.protocol.rules.errors import RULES, ParseError

from causalab.causal.pair_validation import GRADE_VALUES, GRADES
from causalab.causal.scoring import PROTOCOL_MODES, SCORING_FIELDS, STRING_MODES
from causalab.protocol.lowering import AXIS_KINDS, RULE_KINDS
from causalab.protocol.engine import CAPABILITIES
from causalab.protocol.registry import GROUP_SITE_SELECTORS
from causalab.protocol.schema import (
    COMPONENTS,
    GATE_GROUP_AXES,
    FEATURIZER_FAMILIES,
    FEATURIZER_FIELD_CONDITIONS,
    FEATURIZER_FIELDS,
    FEATURIZER_KINDS,
    GATE_DEFAULT_MAP,
    GATE_GROUPS,
    GATE_MAPS,
    GATE_PARAMETRIZATIONS,
    MATCH_MODES,
    METRIC_FIELDS,
    METRIC_KINDS,
    OPTIONAL_METRIC_FIELDS,
    PROTOCOL_VERSION,
    DRAW_KINDS,
    FORWARD_MASKS,
    GATE_AXES,
    REGULARIZER_COSTS,
    REGULARIZER_KINDS,
    TRAINABLE_KINDS,
    parse_document,
    SAVE_REDUCTIONS,
)
from causalab.protocol.registry.shapes import AxisKind
from causalab.workflow.document import is_workflow
from tests._helpers import tracked
from tests._helpers.tracked import tracked_files
from tests.protocol._docs import saved


pytestmark = pytest.mark.unit

SPEC = Path(__file__).resolve().parents[2] / "docs" / "intervention_protocol.md"
#: The spec's §6–8 and §9.1, moved beside it.
INTERNALS = SPEC.with_name("intervention_protocol_internals.md")

#: A markdown table row: the cells between the outer pipes. Leading
#: whitespace is allowed because §2.12's table sits inside a list item.
ROW = re.compile(r"^[ \t]*\|(.+)\|\s*$", re.M)
#: A backtick-quoted run.
CODE = re.compile(r"`([^`]+)`")


def _section(heading: str, spec: Path = SPEC) -> str:
    """The text under ``heading``, up to the next heading of equal or lesser depth.

    "Or lesser" matters: §2.12 is the last `###` of §2, so stopping only at
    the next `###` runs on into §3 and picks up the CLI's option table. That
    is exactly how this guard would have passed against the wrong table.
    """
    depth = len(heading) - len(heading.lstrip("#"))
    body = spec.read_text().split(heading, 1)
    assert len(body) == 2, f"{heading!r} is not in {spec.name}"
    stop = re.compile(rf"^#{{1,{depth}}} ", re.M)
    end = stop.search(body[1])
    return body[1][: end.start()] if end else body[1]


def _rows(table: str) -> list[list[str]]:
    """Body rows of the first markdown table in ``table``, cells stripped."""
    out: list[list[str]] = []
    for match in ROW.finditer(table):
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # the ---|--- separator
        out.append(cells)
    return out


def _metric_table() -> list[list[str]]:
    """§2.10's kind table — the one whose header starts with ``kind``."""
    section = _section("### 2.10 `aggregation`: reductions over a read")
    rows = _rows(section)
    start = next(index for index, row in enumerate(rows) if row[0] == "kind")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break  # the next table (domains) has a different first column
        body.append(row)
    return body


def _reduce_table() -> list[list[str]]:
    """§2.12's ``reduce`` table — the one whose header starts with ``verb``."""
    rows = _rows(_section("### 2.12 `save`"))
    start = next(index for index, row in enumerate(rows) if row[0] == "verb")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def _fields(cell: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(mandatory, optional)`` from a ``fields`` cell.

    The first backticked run is the mandatory list (``of, a, b``); any later
    run is optional (``(+ optional `mode`)``). ``of`` is dropped: every kind
    has it, so ``METRIC_FIELDS`` does not carry it.
    """
    if cell.strip() == "none":  # a kind with no value fields (`decode`)
        return (), ()
    runs = CODE.findall(cell)
    assert runs, f"no backticked field list in {cell!r}"
    mandatory = tuple(
        name.strip()
        for name in runs[0].split(",")
        if name.strip() and name.strip() != "of"
    )
    optional = tuple(
        name.strip() for run in runs[1:] for name in run.split(",") if name.strip()
    )
    return mandatory, optional


def test_the_metric_table_is_not_empty() -> None:
    """A census over a table the parser failed to find passes vacuously."""
    assert len(_metric_table()) >= len(METRIC_KINDS)


def test_metric_kinds_match_spec() -> None:
    tabulated = [CODE.findall(row[0])[0] for row in _metric_table()]
    assert len(set(tabulated)) == len(tabulated), (
        f"§2.10 lists a kind twice: {tabulated}"
    )
    assert set(tabulated) == set(METRIC_KINDS), (
        "§2.10's kind table and MetricKind disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(METRIC_KINDS))}; "
        f"only in the code: {sorted(set(METRIC_KINDS) - set(tabulated))}"
    )


def test_metric_fields_match_spec() -> None:
    """The `fields` column, not just the kind names.

    A kind in both lists with the wrong fields is the drift that actually
    bites: the table is what an author writes a document from.
    """
    for row in _metric_table():
        kind = CODE.findall(row[0])[0]
        mandatory, optional = _fields(row[1])
        assert set(mandatory) == set(METRIC_FIELDS[kind]), (
            f"§2.10 says {kind} takes {mandatory}, METRIC_FIELDS says "
            f"{METRIC_FIELDS[kind]}"
        )
        assert set(optional) == set(OPTIONAL_METRIC_FIELDS.get(kind, ())), (
            f"§2.10 says {kind}'s optional fields are {optional}, "
            f"OPTIONAL_METRIC_FIELDS says {OPTIONAL_METRIC_FIELDS.get(kind, ())}"
        )


def test_every_tabulated_kind_has_a_domain() -> None:
    """Step 3 of §2.10's recipe. A kind with no domain makes the planner
    guess whether to materialize a vocabulary projection."""
    from causalab.protocol.schema import METRIC_DOMAINS

    assert set(METRIC_DOMAINS) == set(METRIC_KINDS)


def test_the_reduce_table_is_not_empty() -> None:
    assert len(_reduce_table()) >= len(SAVE_REDUCTIONS)


def test_save_reductions_match_spec() -> None:
    tabulated = [CODE.findall(row[0])[0] for row in _reduce_table()]
    assert len(set(tabulated)) == len(tabulated), (
        f"§2.12 lists a verb twice: {tabulated}"
    )
    assert set(tabulated) == set(SAVE_REDUCTIONS), (
        "§2.12's reduce table and SAVE_REDUCTIONS disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(SAVE_REDUCTIONS))}; "
        f"only in the code: {sorted(set(SAVE_REDUCTIONS) - set(tabulated))}"
    )


def test_every_reduce_row_says_why_the_verb_exists() -> None:
    """Step 3 of §2.12's recipe: a row that names the verb and stops is a row
    that documents nothing a reader could not guess from the name."""
    for row in _reduce_table():
        verb = CODE.findall(row[0])[0]
        assert len(row) >= 3 and len(row[2]) > 20, (
            f"§2.12's row for {verb} has no 'why it is here' text"
        )


# -- §2.10 the scoring translation table and the ScoringSpec fields -------- #

TASKS_README = SPEC.parents[1] / "causalab" / "tasks" / "README.md"


def _translation_table() -> list[list[str]]:
    """§2.10's translation table — the one whose header starts with ``task
    `string_mode```: a task's string mode → the ``mode`` a ``match`` metric
    declares over its table."""
    rows = _rows(_section("### 2.10 `aggregation`: reductions over a read"))
    start = next(
        index
        for index, row in enumerate(rows)
        if row[0].startswith("task `string_mode`")
    )
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def _scoring_fields_table() -> list[list[str]]:
    """``causalab/tasks/README.md``'s ``ScoringSpec`` fields table — the one
    whose header starts with ``field``."""
    rows = _rows(TASKS_README.read_text())
    start = next(index for index, row in enumerate(rows) if row[0] == "field")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def test_the_translation_table_is_not_empty() -> None:
    assert len(_translation_table()) >= len(STRING_MODES)


def test_the_two_match_mode_vocabularies_are_one_table() -> None:
    """The bridge between a task's ``string_mode`` and the protocol's ``match``
    ``mode`` used to live in a parse-error string. It is a
    type now — ``PROTOCOL_MODES`` — and §2.10's table is that type's census:
    the left column is exactly ``STRING_MODES``, the right column exactly
    ``MATCH_MODES``, and the map is total in both directions."""
    tabulated = {
        CODE.findall(row[0])[0]: CODE.findall(row[1])[0] for row in _translation_table()
    }
    assert tabulated == dict(PROTOCOL_MODES), (
        "§2.10's translation table and PROTOCOL_MODES disagree — "
        f"spec: {tabulated}; code: {dict(PROTOCOL_MODES)}"
    )
    assert set(tabulated) == set(STRING_MODES)
    assert set(tabulated.values()) == set(MATCH_MODES)
    assert len(set(tabulated.values())) == len(tabulated)  # a bijection


def _grades_table() -> list[list[str]]:
    """§2.10's ``grade`` table — the one whose header is ``per-example
    `grade```: the per-example status vocabulary against what
    ``ScoringSpec.grade`` returns. The header does not start with
    a backtick on purpose: the scanners above read a table's body as "rows
    until one whose first cell is not code", so a code-first header would be
    swallowed into the translation table that precedes this one."""
    rows = _rows(_section("### 2.10 `aggregation`: reductions over a read"))
    start = next(
        index for index, row in enumerate(rows) if row[0] == "per-example `grade`"
    )
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


def test_the_grades_table_is_not_empty() -> None:
    assert len(_grades_table()) >= len(GRADES)


def test_grades_are_the_spec_table_and_map_onto_scoring_spec_grade() -> None:
    """The per-example status is one closed vocabulary, ``GRADES``, and §2.10's
    table is its census: the left column is exactly ``GRADES`` in order, the
    right column the three values ``ScoringSpec.grade`` returns, and the map is
    ``GRADE_VALUES`` — total and injective (a 1:1 relabelling, never a
    rounding)."""
    tabulated = [
        (CODE.findall(row[0])[0], CODE.findall(row[1])[0]) for row in _grades_table()
    ]
    assert [name for name, _ in tabulated] == list(GRADES), tabulated
    spelled = {"1.0": 1.0, "0.0": 0.0, "null": None}
    assert {name: spelled[value] for name, value in tabulated} == dict(GRADE_VALUES)
    assert len({value for _, value in tabulated}) == len(tabulated)  # injective
    for row in _grades_table():
        assert len(row) >= 3 and len(row[2]) > 15, f"the row for {row[0]} says nothing"


def test_grade_and_matched_have_definitions() -> None:
    """Document the meaning of each output field beside its vocabulary."""
    section = _section("### 2.10 `aggregation`: reductions over a read")
    assert "`grade` describes" in section
    assert "`matched` records whether the addressed token position existed" in section


def test_the_scoring_fields_table_is_not_empty() -> None:
    assert len(_scoring_fields_table()) >= len(SCORING_FIELDS)


def test_scoring_spec_fields_match_the_tasks_readme() -> None:
    """The fields table in ``causalab/tasks/README.md`` is what a task author
    writes a spec from, so it is exactly ``SCORING_FIELDS`` — in order, since
    the table is also the field order the spec's identity lists."""
    tabulated = [CODE.findall(row[0])[0] for row in _scoring_fields_table()]
    assert tabulated == list(SCORING_FIELDS), (
        "the README's ScoringSpec fields table and SCORING_FIELDS disagree — "
        f"only in the README: {sorted(set(tabulated) - set(SCORING_FIELDS))}; "
        f"only in the code: {sorted(set(SCORING_FIELDS) - set(tabulated))}"
    )
    for row in _scoring_fields_table():
        assert len(row) >= 3 and len(row[2]) > 15, (
            f"the README's row for {row[0]} does not say what the field retires"
        )


#: The retired spellings of a task's scoring declaration. They may
#: survive in exactly two places in code — the derived read-only view on
#: ``CausalModel`` (kept under the name its readers grew up on) and the
#: recipe sidecar's human-readable copy, written by the serializer as an
#: advisory field — and in the two documents that describe those two places.
RETIRED_SCORING_SPELLINGS: tuple[str, ...] = ("match_modes", "declared_match_mode")
RETIRED_SPELLING_HOMES: frozenset[str] = frozenset(
    {
        "causalab/causal/model.py",  # the derived views
        "causalab/tasks/serialize.py",  # the manifest copy
        "causalab/tasks/README.md",  # documents both, and what they retired
        "docs/CODEBASE.md",  # the module map names the views
    }
)


def test_no_second_spelling_of_the_scoring_declaration() -> None:
    """``git grep match_modes|declared_match_mode`` over ``causalab/``,
    ``scripts/`` and ``docs/`` returns only the derived view and the manifest
    copy. A third spelling is a third place the task's definition of correct
    can drift from."""
    pattern = re.compile(
        "|".join(re.escape(word) for word in RETIRED_SCORING_SPELLINGS)
    )
    offences = [
        f"{rel}:{number}: {line.strip()!r}"
        for path in tracked_files(ROOT, "*.md", "*.py", "*.json")
        if (rel := _relative(path)).startswith(("causalab/", "scripts/", "docs/"))
        and rel not in RETIRED_SPELLING_HOMES
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offences, "\n".join(offences)
    for home in RETIRED_SPELLING_HOMES:
        assert (ROOT / home).is_file(), home


# -- §2.5 `group` ---------------------------------------------------------- #


def _group_table() -> list[list[str]]:
    """§2.5's ``group`` table — the one whose header starts with ``group``."""
    rows = _rows(_section("### 2.5 `featurizers`"))
    start = next(index for index, row in enumerate(rows) if row[0] == "group")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    return body


# -- §11.1 one word per object -------------------------------------------- #

ROOT = SPEC.parents[1]
PACKAGE = ROOT / "causalab"

#: The five nouns §11.1 makes normative, and what each names. Duplicated here
#: deliberately: a census whose expectation is read out of the file it checks
#: passes whatever the file says.
FIVE_NAMES: dict[str, str] = {
    "research pipeline": "the methodology",
    "runtime implementation": "the installed package and its engine",
    "intervention specification": "the authored, serializable document",
    "compiled intervention": "the resolved, expanded, digested form",
    "run receipt": "the metadata a completed run leaves",
}

#: Phrases that name one of the five objects with the overloaded word instead
#: of its normative name — plus ``run record``, the predecessor of the fifth
#: name, which survived the first sweep *unguarded* and so left `run receipt`
#: normative in name only. Each entry is a real phrasing that was in the tree
#: before this vocabulary landed, so the list is a regression list, not a
#: guess.
#:
#: Matching is by substring, so a singular entry also catches its plural
#: (``protocol documents``, ``point protocols``); listing both would report
#: one offending line twice. ``intervention protocols`` is listed in the
#: plural on purpose — the *singular* lowercase form is a legitimate reference
#: to the format ("the intervention protocol is usable on its own",
#: `causalab/cli.py`), while a format cannot be plural, so only the plural is
#: unambiguously one of the five objects. The singular followed by
#: ``document`` is *not* legitimate, and needs no entry of its own: it contains
#: ``protocol document``.
BANNED_PHRASES: tuple[str, ...] = (
    "protocol document",
    "point protocol",
    "intervention protocols",
    "run record",
    # the per-example status is `grade`; `criterion` / `qualification` are
    # the words for its parts, and nothing is declared under the field's
    # earlier name, which the census renamed.
    "correctness qualification",
)

#: Where the word legitimately survives, and why. A ban with no stated
#: exemptions is a ban that gets deleted the first time it is inconvenient.
#:
#: * ``Intervention Protocol`` / ``Workflow Protocol`` — the *formats*, which is
#:   what the two specs are named after;
#: * ``intervention_protocol`` — a serialized step-type value in the workflow
#:   format, and ``type: protocol`` a serialized value in the intervention
#:   format. Renaming either breaks every existing document;
#: * ``causalab/protocol/`` and ``tests/protocol/`` — module paths;
#: * ``protocols/`` — a directory name.
#:
#: **Masking is case-sensitive while the bans are not**, and that asymmetry is
#: the mechanism, not an oversight: it is the only thing separating
#: ``Intervention Protocol`` (the format, legal) from ``intervention
#: protocols`` (the objects, banned). The cost is that Title Case is a general
#: bypass — ``Intervention Protocol documents`` masks to ``documents`` and no
#: ban fires. Two lines in the tree are exactly that shape
#: (`causalab/workflow/document.py:1`, `docs/CODEBASE.md:85`) and both are
#: right, because they genuinely name the format; nothing here can tell them
#: apart from capitalizing one's way out of a failure, so this is a rule about
#: intent that a reviewer enforces and a test cannot.
#:
#: ``RUN_RECORD_NAME`` and ``write_run_record`` need no entry: an underscore is
#: not a space, so ``run record`` cannot match either. Renaming them would
#: break the public API for a word, which is the same trade §11.1 refuses for
#: ``type: protocol`` — ``RUN_RECORD_NAME`` is re-exported from
#: `causalab.protocol` (`__init__.py:28,41`), while ``write_run_record`` is
#: exported from `causalab.protocol.run` only.
ALLOWED_CONTEXTS: tuple[str, ...] = (
    "Intervention Protocol",
    "Workflow Protocol",
    "intervention_protocol",
    "causalab/protocol",
    "causalab.protocol",
    "tests/protocol",
    "protocols/",
)

#: Directory prefixes the vocabulary does not reach, with the reason. Empty:
#: no tracked directory is on its own vocabulary (§11.1). Kept as the seam a
#: future carve-out would use, so the existence check below keeps guarding it.
EXEMPT_PREFIXES: tuple[str, ...] = ()

#: Files the vocabulary cannot reach, with the reason.
#:
#: * this file — it holds the banned phrases as *data*, so it necessarily
#:   contains every one of them. The exemption is real and worth naming: a
#:   genuine offence in this file's own prose would not be caught.
EXEMPT_FILES: frozenset[str] = frozenset({"tests/protocol/test_vocabulary_census.py"})

#: The modules a workflow ``script`` step names. `check_script` hashes the
#: module's *bytes* (`causalab/workflow/document.py`, `docs/workflow_protocol.md`
#: §7), so their prose is part of an experiment's identity down to the
#: docstring and is **frozen** (§11.1): renaming a noun in one of them moves a
#: recorded run's digest. Listed rather than described, so that adding a sixth
#: script step fails ``test_the_frozen_scripts_are_the_hashed_ones`` loudly
#: instead of silently widening the exemption.
#:
#: A ``path``-form script step (a template outside the package, such as a
#: user's own ``summarize.py``) is hashed
#: the same way but lives outside the package, so it is out of this census's
#: scope rather than exempt from it.
#:
#: The modules these five *import* are **not** digest-bearing — a script's
#: identity is its own bytes, and the package is runtime identity
#: (`docs/workflow_protocol_internals.md` §4.2, `docs/workflow_protocol.md` §7;
#: `tests/workflow/test_closure_census.py`
#: freezes what they reach as a layering census) — so they are deliberately not
#: exempt: a vocabulary fix in `causalab/protocol/schema.py` moves no digest
#: and is ordinary prose this census may touch.
HASHED_SCRIPTS: frozenset[str] = frozenset(
    {
        "causalab/analysis/fit_pca.py",
        "causalab/analysis/harvest_difference.py",
        "causalab/analysis/random_mask.py",
        "causalab/io/plots/workflow_figures.py",
        "causalab/workflow/scripts/reduce.py",
        "causalab/workflow/scripts/select.py",
    }
)

#: What separates two words of a banned phrase in prose: **whitespace and
#: presentation markup**, stated once. A reader sees ``**protocol** document``,
#: ``protocol\n> document`` and ``protocol document`` as the same two words;
#: the census has to as well. It was widened three times, by one case each
#: — indentation, then a line-lead marker, then inline emphasis
#: (`demos/README.md:69` was live and green with two asterisks between the
#: words) — which is the sign that the separator was being enumerated rather
#: than defined. So:
#:
#: * ``MARKUP`` is the inline emphasis and code markers that can close one word
#:   and open the next (``**``, ``_``, `````);
#: * the whitespace between is an ordinary run of spaces, or a soft wrap onto a
#:   line that is very likely indented and may carry a **line-lead marker** —
#:   a wrapped blockquote repeats its ``>``, a wrapped Python comment its ``#``,
#:   a wrapped list item is indented under its ``*``.
#:
#: Exactly one newline is allowed. A blank line contributes two, so a paragraph
#: boundary still cannot be read as a phrase — and a marker-only line leaves
#: two as well, so allowing the marker does not weaken that. Markup contributes
#: none, so admitting it does not weaken it either.
MARKUP = r"[*_`]*"
GAP = rf"{MARKUP}(?:[ \t]*\n[ \t]*(?:[>#*]+[ \t]*)?|[ \t]+){MARKUP}"

#: §11.1's heading, as `_section` matches it.
VOCABULARY_SECTION = "### 11.1 Terms"


def _tree_files() -> list[Path]:
    """Every **tracked** markdown and python file in the tree.

    The whole tree, deliberately. An *inclusion* list grew one glob at a time,
    and each new glob turned up one more hole — `causalab/**/*.py`,
    then `causalab/**/*.md`, then the repo `README.md`, then `demos/` and
    `scripts/`. An inclusion list makes the next new directory a blind spot;
    scanning everything and naming the carve-outs makes it a failure. That is
    also this file's own stated standard for `ALLOWED_CONTEXTS`, and the shape
    `HASHED_SCRIPTS` already has.

    *Tracked*, not *present*: every carve-out below is a path from the repo
    root, so the files it is checked against have to be named the same way.
    A filesystem walk is not — a worktree under ``worktrees/`` or a setuptools
    ``build/lib/`` is a second copy of the tree at which all four carve-outs
    miss at once, green in CI and red on a developer's checkout. `git
    ls-files` is the enumeration the carve-outs were written against
    (`tests/_helpers/tracked.py` says why, and what the fallback costs).
    """
    out = tracked_files(ROOT, "*.md", "*.py")
    assert len(out) > 100, f"the walk found only {len(out)} files"
    return out


def _relative(path: Path) -> str:
    """``path`` as the carve-outs name it — or as given, for a fixture that is
    not in the tree at all (the `GAP` regression test writes to ``tmp_path``)."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _corpus() -> list[Path]:
    """Every file §11.1 binds: the tree, minus the named carve-outs."""
    skip = EXEMPT_FILES | HASHED_SCRIPTS
    out = [
        path
        for path in _tree_files()
        if (rel := _relative(path)) not in skip and not rel.startswith(EXEMPT_PREFIXES)
    ]
    assert len(out) > 100, "the carve-outs swallowed the corpus"
    return out


def _offences(path: Path) -> list[str]:
    """The banned phrases in ``path``, as ``path:line: phrase`` messages.

    Allowed contexts are **masked out of** each line rather than causing the
    line to be skipped. Skipping was the first version and it was wrong: a
    markdown paragraph here is frequently one long line, so any line that
    happened to cite a module path was exempt from *every* ban — which is how
    ``docs/TESTS.md:3`` said "serializable intervention protocols" next to
    ``docs/intervention_protocol.md`` and stayed green.

    Lines are then scanned as one joined text, with the offsets mapped back to
    line numbers, because prose soft-wraps: §1.1's "written into the run
    record" straddles a line break and is invisible to a per-line scan while
    being one phrase to every reader. Masking pads with ``" " * len(context)``
    rather than collapsing, and the join adds exactly one character per line,
    so those offsets stay true.

    The gap between a phrase's words is therefore **whitespace**, not a
    literal space: a continuation line is usually *indented*, so a literal
    space caught soft-wraps in flush-left markdown and none at all inside a
    docstring or a list item (the retired `causalab/protocol/method.py:106` was live and
    green on it). At most one newline, so the two spaces a paragraph break
    contributes cannot join the end of one paragraph to the start of the next.

    Read with ``errors="replace"``, like every other read over this corpus: the
    corpus is every tracked text file, and a fixture written for an encoding
    test must not take the module down with a `UnicodeDecodeError` that names
    no file.
    """
    lines = path.read_text(errors="replace").splitlines()
    starts: list[int] = []
    masked: list[str] = []
    cursor = 0
    for line in lines:
        scanned = line
        for context in ALLOWED_CONTEXTS:
            scanned = scanned.replace(context, " " * len(context))
        masked.append(scanned)
        starts.append(cursor)
        cursor += len(scanned) + 1
    corpus = "\n".join(masked).lower()
    found: list[tuple[int, str]] = []
    for phrase in BANNED_PHRASES:
        pattern = GAP.join(re.escape(word) for word in phrase.split())
        for match in re.finditer(pattern, corpus):
            number = bisect.bisect_right(starts, match.start())
            found.append((number, phrase))
    return [
        f"{_relative(path)}:{number}: {phrase!r} in {lines[number - 1].strip()!r}"
        for number, phrase in sorted(found)
    ]


def _script_modules() -> set[str]:
    """The module of every ``script`` step in every tracked workflow document.

    Every tracked JSON file rather than a fixed list of directories, so a
    workflow document added somewhere new is still seen — and *tracked* rather
    than *present*, for the same reason as `_tree_files`: a run's output tree
    and a virtualenv each hold more JSON than the repository does, and neither
    is a recorded experiment's step.
    """
    out: set[str] = set()
    for path in tracked_files(ROOT, "*.json"):
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(raw, dict) or not is_workflow(raw):
            continue
        steps = raw["steps"]
        entries = steps.values() if isinstance(steps, dict) else steps
        for step in entries:
            script = step.get("script") if isinstance(step, dict) else None
            module = script.get("module") if isinstance(script, dict) else None
            if isinstance(module, str):
                out.add(module.replace(".", "/") + ".py")
    return out


def _vocabulary_table() -> list[list[str]]:
    """§11.1's table — the one whose header starts with ``term``."""
    rows = _rows(_section(VOCABULARY_SECTION))
    start = next(index for index, row in enumerate(rows) if row[0] == "term")
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("**"):
            break
        body.append(row)
    return body


def test_the_group_table_is_not_empty() -> None:
    assert len(_group_table()) >= len(GATE_GROUPS)


def test_gate_groups_match_spec() -> None:
    tabulated = [CODE.findall(row[0])[0] for row in _group_table()]
    assert len(set(tabulated)) == len(tabulated), (
        f"§2.5 lists a group twice: {tabulated}"
    )
    assert set(tabulated) == set(GATE_GROUPS), (
        "§2.5's group table and GATE_GROUPS disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(GATE_GROUPS))}; "
        f"only in the code: {sorted(set(GATE_GROUPS) - set(tabulated))}"
    )


def test_group_is_a_closed_vocabulary_of_three_values() -> None:
    """The vocabulary is exactly the three derivations the registry has. There
    is no literal for the default: an absent ``group`` is the per-coordinate
    gate, and the literal spelling of it is refused as an unknown value; a
    rank-slot group — a parameter indexed by routing rank, not stable across
    routing — is not a value either."""
    assert set(GATE_GROUPS) == {"head", "expert_neuron", "site"}


def test_every_group_row_names_the_axis_the_code_resolves_it_to() -> None:
    """The cell that makes the table load-bearing.

    A group whose row says one axis while ``GATE_GROUP_AXES`` says another is
    the drift that resolves silently: the gate is built, it just groups by
    something else than the spec claims.
    """
    assert set(GATE_GROUP_AXES) == set(GATE_GROUPS)
    for row in _group_table():
        group = CODE.findall(row[0])[0]
        axes = CODE.findall(row[2])
        assert axes, f"§2.5's row for {group} names no axis"
        assert axes[0] == GATE_GROUP_AXES[group], (
            f"§2.5 says {group} groups over the {axes[0]!r} axis, "
            f"GATE_GROUP_AXES says {GATE_GROUP_AXES[group]!r}"
        )


def test_every_group_axis_is_an_axis_kind() -> None:
    """``schema`` spells the axes as plain strings (it sits below ``shapes`` in
    the import order); they have to be ``AxisKind`` values or the table ties a
    group to an axis no shape can declare."""
    assert set(GATE_GROUP_AXES.values()) <= set(get_args(AxisKind))


def test_every_group_has_a_site_selector() -> None:
    """`site_group_map` refuses a group whose axis the *site* already selects a
    single member of, from a table keyed by group. A group added to the
    vocabulary with no row there would raise a bare `KeyError` instead of
    deciding rule 23, so the table is asserted complete, and its values are
    site fields — or ``None`` for a group no site field can collapse (``site``
    over one head's slice is still one unit)."""
    assert set(GROUP_SITE_SELECTORS) == set(GATE_GROUPS)
    assert set(GROUP_SITE_SELECTORS.values()) <= {"head", "expert", None}


# -- §3.2 named axes: the axis kinds and the rule kinds ------------------- #

AXES_SECTION = "### 3.2 `axes`: correlated rows and dependent axes"


def _axes_table(first_header: str) -> list[str]:
    """The first column of §3.2's table whose header starts with
    ``first_header`` — a plain word, never a backtick (the scanner would fold a
    code-first header into the preceding table)."""
    rows = _rows(_section(AXES_SECTION))
    start = next(index for index, row in enumerate(rows) if row[0] == first_header)
    body: list[str] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row[0])
    return body


def test_the_axis_kinds_are_the_spec_table() -> None:
    """§3.2's ``key`` table names every kind of named axis, in the code's
    order: a kind added to `AXIS_KINDS` without a row, or a row without a
    kind, fails here."""
    kinds = [CODE.match(cell).group(1) for cell in _axes_table("key")]  # type: ignore[union-attr]
    assert tuple(kinds) == AXIS_KINDS
    assert len(set(AXIS_KINDS)) == len(AXIS_KINDS)


def test_the_rule_kinds_are_the_spec_table() -> None:
    """§3.2's ``kind`` table is the closed vocabulary of dependent-axis rules
    — one row per kind, and nothing the code does not know (T3's
    `routed_experts` is refused, and is not here)."""
    kinds = [CODE.match(cell).group(1) for cell in _axes_table("kind")]  # type: ignore[union-attr]
    assert tuple(kinds) == RULE_KINDS
    assert "routed_experts" not in kinds


# -- §2.5 `axis` — the position gate ------------------------------------------ #


def test_every_gate_axis_is_spelled_in_the_featurizer_section() -> None:
    """§2.5's ``axis`` bullet spells each member as a backticked word, and the
    gate row names the field; a member added to the code without its spelling
    fails here."""
    section = _section("### 2.5 `featurizers`")
    assert GATE_AXES == ("position",)
    assert "`axis`" in section, "§2.5 does not describe `axis`"
    for axis in GATE_AXES:
        assert f"`{axis}`" in section, f"§2.5 does not spell axis {axis!r}"


# -- §2.5 the mapping form of `parametrization` — `forward` --------------------- #


def test_every_forward_mask_is_spelled_in_the_featurizer_section() -> None:
    """§2.5's mapping-form paragraph spells each `forward` value as a
    backticked word; a value added to the code without its spelling fails
    here — the census that holds §7's one-spelling argument to the code."""
    section = _section("### 2.5 `featurizers`")
    assert FORWARD_MASKS == ("hard",)
    assert '{"forward": "hard", "backward":' in section, "§2.5 has no mapping form"
    for value in FORWARD_MASKS:
        assert f"`{value}`" in section, f"§2.5 does not spell forward {value!r}"


# -- §2.2 `draw` — the counterfactual-set verb -------------------------------- #


def test_every_draw_kind_is_spelled_in_the_data_section() -> None:
    """§2.2's ``draw`` bullet spells each kind as a backticked word beside
    ``draw``; a kind added to the code without its spelling fails here."""
    section = _section("### 2.2 `data`")
    assert DRAW_KINDS == ("uniform",)
    assert "`draw:" in section or "`draw`" in section, "§2.2 does not describe `draw`"
    for kind in DRAW_KINDS:
        assert f"`{kind}`" in section, f"§2.2 does not spell draw kind {kind!r}"


# -- §2.11 `train.objective` regularizers ---------------------------------- #


def _objective_row() -> str:
    """§2.11's table row for ``objective`` — the whole line, because its cell
    quotes ``\\|`` between the two regularizer spellings and a cell split on
    pipes would cut it in half."""
    section = _section("### 2.11 `train`")
    rows = [line for line in section.splitlines() if line.startswith("| `objective` |")]
    assert len(rows) == 1, "§2.11 has no single `objective` row"
    return rows[0]


def test_regularizer_kinds_are_a_closed_vocabulary_of_three_values() -> None:
    """A regularizer is ``l1``, ``l2`` or ``l0`` and nothing else: the parser's
    suggestion for a misspelling, the canonical form's list sorting and the
    train loop's ``|p|`` / ``p²`` / expected-L0 branches all key on the same
    tuple; ``l0`` was added last so no earlier suggestion ordering moved."""
    assert REGULARIZER_KINDS == ("l1", "l2", "l0")


def test_every_regularizer_cost_word_is_in_the_objective_row() -> None:
    """§2.11 ``costs`` takes a table or a closed word; each word is spelled
    in the objective row as a JSON string, so a rule added to the code
    without its spelling fails here."""
    row = _objective_row()
    assert REGULARIZER_COSTS == ("parameter_count",)
    assert '"costs"' in row, "§2.11's objective row does not name the 'costs' key"
    for word in REGULARIZER_COSTS:
        assert f'"{word}"' in row, f"§2.11's objective row does not spell {word!r}"


def test_every_regularizer_kind_is_in_the_objective_row() -> None:
    """The §2.11 row spells each kind as a JSON key, ``{"l1": names}``, so a
    kind added to the code without a spelling in the spec fails here."""
    row = _objective_row()
    for kind in REGULARIZER_KINDS:
        assert f'{{"{kind}": names}}' in row, (
            f"§2.11's objective row does not spell {kind!r}"
        )
    # an aggregation term is spelled by its bound read and its reduction (§2.10)
    for key in ("weight", "read", "model", "aggregation"):
        assert f'"{key}"' in row, f"§2.11's objective row does not name the {key!r} key"


def test_the_vocabulary_table_is_not_empty() -> None:
    assert len(_vocabulary_table()) >= len(FIVE_NAMES)


def test_the_five_names_are_defined_once_each() -> None:
    tabulated = [row[0].strip("* ") for row in _vocabulary_table()]
    assert len(set(tabulated)) == len(tabulated), (
        f"§11.1 lists a term twice: {tabulated}"
    )
    assert set(tabulated) == set(FIVE_NAMES), (
        "§11.1's table and the census disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(FIVE_NAMES))}; "
        f"only in the census: {sorted(set(FIVE_NAMES) - set(tabulated))}"
    )


def test_every_name_has_a_definition() -> None:
    """Each term has a definition in the glossary."""
    for row in _vocabulary_table():
        term = row[0].strip("* ")
        assert len(row) >= 2, f"§11.1's row for {term} has no definition"
        assert len(row[1]) > 20, f"§11.1 does not say what {term} means"


def test_every_normative_name_is_actually_used() -> None:
    """Each glossary term is used elsewhere in the documentation."""
    corpus = "\n".join(path.read_text(errors="replace") for path in _corpus())
    table = _section(VOCABULARY_SECTION)
    assert table in corpus, "§11.1's table is not in the corpus it was cut from"
    corpus = corpus.replace(table, "").lower()
    for term in FIVE_NAMES:
        assert corpus.count(term) >= 1, (
            f"§11.1 makes {term!r} normative but nothing outside its own table uses it"
        )


def test_the_overloaded_word_does_not_come_back() -> None:
    """The regression this vocabulary exists to prevent, over the whole tree.

    One test rather than one per corpus, because the corpus is no longer
    assembled: it is everything minus four named carve-outs, and a failure
    that says which file it is in says everything a split would have.

    §11.1 binds "this spec, `docs/workflow_protocol.md`, or a module
    docstring". The census is deliberately *wider* than that — `demos/`,
    `scripts/`, `tests/` and the repo `README.md` are all prose a reader meets
    and none of them is named by the rule — because the alternative is a scope
    sentence that has to be re-litigated every time a directory is added.
    """
    offences = [line for path in _corpus() for line in _offences(path)]
    assert not offences, (
        "'protocol' is being used for one of §11.1's five objects — say which "
        "one. If the file is genuinely on another vocabulary (a hashed "
        "script's frozen prose), it "
        "belongs in a named carve-out, not in a reword:\n" + "\n".join(offences)
    )


def test_the_carve_outs_all_still_exist() -> None:
    """A carve-out for a file that is gone is a widened blind spot, not a free
    line — the same argument that retired an earlier directory exemption."""
    for name in HASHED_SCRIPTS | EXEMPT_FILES:
        assert (ROOT / name).is_file(), f"{name} is exempt but does not exist"
    for prefix in EXEMPT_PREFIXES:
        assert (ROOT / prefix).is_dir(), f"{prefix} is exempt but does not exist"


def test_the_frozen_scripts_are_the_hashed_ones() -> None:
    """`HASHED_SCRIPTS` is the exemption list, so it has to *be* the hashed set.

    A sixth script step would otherwise inherit no exemption and start failing
    the package census for prose it is not allowed to change; a script step
    that went away would leave a live module silently exempt.

    The walk covers *every* workflow document in the tree, test fixtures
    included, so a throwaway fixture with a `script` step fails this too. That
    is the right default — the test cannot know which runs are recorded — but
    the message has to say so, because "add it to `HASHED_SCRIPTS`" is the
    wrong fix for a fixture and the right one for a real step.
    """
    named = _script_modules()
    assert named, "no workflow script steps found — the walk or the filter is wrong"
    assert named == HASHED_SCRIPTS, (
        "the hashed-script set moved. Freezing a module's prose is only right "
        "for a module a *recorded* run's digest depends on — if the step that "
        "moved this set is a test fixture, point it at a module already in the "
        "list instead of growing the list.\n"
        f"named by a step but not exempt: {sorted(named - HASHED_SCRIPTS)}\n"
        f"exempt but named by no step: {sorted(HASHED_SCRIPTS - named)}"
    )


@pytest.fixture
def nested_copy() -> Iterator[Path]:
    """An untracked second copy of an offending file, where a checkout puts one.

    ``worktrees/`` is gitignored and is where a developer's tooling may park
    a git worktree — a full copy of the tree, at which every root-anchored
    carve-out misses. The probe is one file, not a tree, because one file is
    enough to show whether the enumeration is *tracked* or *present*. Removed
    afterwards, along with ``worktrees/`` itself if the fixture created it.
    """
    worktrees = ROOT / "worktrees"
    existed = worktrees.is_dir()
    probe = worktrees / f"census-probe-{uuid.uuid4().hex}"
    (probe / "docs").mkdir(parents=True)
    path = probe / "docs" / "offence.md"
    path.write_text("A protocol document, in a copy no carve-out names.\n")
    try:
        yield path
    finally:
        shutil.rmtree(probe)
        if not existed and not any(worktrees.iterdir()):
            worktrees.rmdir()


@pytest.mark.parametrize("git_available", [True, False], ids=["git", "fallback"])
def test_a_nested_untracked_copy_of_the_tree_is_not_scanned(
    nested_copy: Path, monkeypatch: pytest.MonkeyPatch, git_available: bool
) -> None:
    """The corpus is the tracked tree, so a copy of it that git does not track
    is not an offence — in either enumeration.

    The fallback walk is exercised by making ``git`` unavailable, because it is
    the one path a developer never takes and so the one that rots unseen; it
    has to prune the same nested copy by name.
    """
    if not git_available:

        def unavailable(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git")

        monkeypatch.setattr(tracked.subprocess, "run", unavailable)
    assert _offences(nested_copy), "the probe is not an offence, so this proves nothing"
    files = _tree_files()
    assert nested_copy not in files
    assert not any(_relative(path).startswith("worktrees/") for path in files)


@pytest.mark.parametrize(
    ("text", "fires"),
    [
        pytest.param("a **protocol** document", True, id="inline-emphasis"),
        pytest.param("a *protocol* _document_", True, id="mixed-emphasis"),
        pytest.param("the `protocol` document", True, id="inline-code"),
        pytest.param("a protocol\n    document", True, id="indented-wrap"),
        pytest.param("> a protocol\n> document", True, id="blockquote-wrap"),
        pytest.param("    # a protocol\n    # document", True, id="comment-wrap"),
        pytest.param(
            "**protocol**\n**document**", True, id="emphasis-both-sides-of-a-wrap"
        ),
        pytest.param("a protocol\n\ndocument", False, id="paragraph-boundary"),
        pytest.param("> a protocol\n>\n> document", False, id="marker-only-line"),
    ],
)
def test_the_gap_is_whitespace_and_presentation_markup(
    tmp_path: Path, text: str, fires: bool
) -> None:
    """Every way prose separates two words of one phrase reads as the phrase;
    a paragraph boundary never does. `demos/README.md:69` was the inline-
    emphasis case, live and green."""
    path = tmp_path / "fixture.md"
    path.write_text(text + "\n")
    assert bool(_offences(path)) is fires, _offences(path)


# --------------------------------------------------------------------------- #
# §2.4's component vocabulary
# --------------------------------------------------------------------------- #
#
# The third closed vocabulary in §2, and by far the largest: 62 names. It is
# the one a document addresses most often and the one most likely to grow, and
# until now it had no guard — the metric kinds and the `reduce` verbs did.
# The spec's sentence "the census checks the set, all 62 of them" once
# described a test that did not exist. This is that test; the sentence is now
# true.
#
# Unlike §2.10 and §2.12 the list is prose, not a table — a single
# `·`-separated run of backticked names — so it gets its own reader.


def _component_list() -> list[str]:
    """§2.4's component vocabulary, in the order the spec lists it."""
    section = _section("### 2.4 `sites`")
    paragraphs = [
        block
        for block in section.split("\n\n")
        if " · " in block and block.lstrip().startswith("`")
    ]
    assert len(paragraphs) == 1, (
        f"expected exactly one `·`-separated component list in §2.4, "
        f"found {len(paragraphs)} — the reader or the section changed"
    )
    return re.findall(r"`([a-z0-9_]+)`", paragraphs[0])


def test_the_component_list_is_not_empty() -> None:
    """A parametrization-free guard still needs its vacuity floor: a reader
    that silently matched nothing would make every assertion below trivial.

    `>=`, matching `METRIC_KINDS`/`SAVE_REDUCTIONS`'s floors in this file:
    equality here can only fail in company with the set comparison below, and
    it would fail *first*, with a bare `assert 62 == 61` instead of the diff.
    """
    assert len(_component_list()) >= len(COMPONENTS)


def test_the_component_vocabulary_and_the_spec_agree() -> None:
    """§2.4 lists exactly the components the code accepts.

    A table behind the code is worse than no table, because it reads as a
    complete list — and this is the list a document author works from. Order
    is deliberately not asserted: §2.4 walks a block for a reader, which is a
    different and equally valid order from `COMPONENTS`' tuple.
    """
    tabulated = _component_list()
    assert len(set(tabulated)) == len(tabulated), (
        "§2.4 lists a component twice: "
        f"{sorted({c for c in tabulated if tabulated.count(c) > 1})}"
    )
    assert set(tabulated) == set(COMPONENTS), (
        "§2.4's component list and the code disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(COMPONENTS))}; "
        f"only in the code: {sorted(set(COMPONENTS) - set(tabulated))}"
    )


# --------------------------------------------------------------------------- #
# §8's capability tables
# --------------------------------------------------------------------------- #
#
# The fourth closed vocabulary, and the one most tightly coupled to code that
# the spec cites. §8 draws two tables over it: the coarse verbs and when each
# is required, and — since the audit replaced §8's aspirational "reference
# matrix" — what the two shipped engines actually declare. That replacement is
# 7 rows that must equal `CAPABILITIES` and 14 ✓/✗ cells that must equal two
# frozensets, none of it checked.
#
# The old matrix is the argument: it claimed `grad ✓` for an engine whose class
# has never declared it, and read as authoritative for as long as it took an
# audit to notice. A census would have failed on the day `PytorchHooksEngine`
# landed.


def _capability_rows(second_header: str) -> list[list[str]]:
    """Body rows of the §8 capability table whose *second* header cell matches.

    Both tables head their first column `capability`, so the first column
    cannot pick one: `required when` is the requirement table and
    `` `pytorch_hooks` `` the per-engine one. Selecting on the wrong cell is
    how this guard would silently check the same table twice.
    """
    rows = _rows(_section("## 8. Engine contract", INTERNALS))
    start = next(
        (
            index
            for index, row in enumerate(rows)
            if row[0] == "capability" and row[1].startswith(second_header)
        ),
        None,
    )
    assert start is not None, (
        f"§8 has no capability table whose second column is {second_header!r}"
    )
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break  # the next table, or prose
        body.append(row)
    # The floor this module's docstring promises for *each* parse, and the one
    # parse that lacked it. The loop above stops at the first row whose first
    # cell is not backticked, so a reformat that unbackticks the first row
    # yields an empty body — and every §8 test would then compare an empty set,
    # failing for the wrong reason instead of naming the parse.
    assert body, (
        f"§8's table headed {second_header!r} parsed to zero rows — the reader "
        "or the table changed, and the comparisons below would be vacuous"
    )
    return body


def _row_name(row: list[str]) -> str:
    """The capability a §8 row's first cell names.

    One spelling for all three tests. They must agree: `_verbs` and the ✓/✗
    test drop the component row while the counts test selects *only* it, so if
    the predicates ever diverge one row is checked twice or not at all — and
    both readings would pass.
    """
    return row[0].strip("`").split("`")[0].strip()


def _is_component_row(row: list[str]) -> bool:
    """Whether a §8 row is the *generated* `component:<name>` entry."""
    return _row_name(row).startswith("component:")


def _verbs(rows: list[list[str]]) -> list[str]:
    """The coarse capability named by each row's first cell.

    Rows for the *generated* component entry are dropped: `component:<name>`
    is not in `CAPABILITIES` by design (§8 says so — component entries are
    generated per site, never listed), so it is not part of this comparison.
    """
    return [_row_name(row) for row in rows if not _is_component_row(row)]


def test_the_required_when_table_lists_every_capability() -> None:
    """§8's `capability | required when` table is exactly `CAPABILITIES`."""
    tabulated = _verbs(_capability_rows("required when"))
    assert len(set(tabulated)) == len(tabulated), (
        f"§8 lists a capability twice: {tabulated}"
    )
    assert set(tabulated) == set(CAPABILITIES), (
        "§8's requirement table and the code disagree — "
        f"only in the spec: {sorted(set(tabulated) - set(CAPABILITIES))}; "
        f"only in the code: {sorted(set(CAPABILITIES) - set(tabulated))}"
    )


def _engine_declaration_rows() -> list[list[str]]:
    """§8's per-engine table — the one headed by the two engine names."""
    return _capability_rows("`pytorch_hooks`")


def test_the_engine_table_lists_every_capability() -> None:
    """§8's per-engine table covers exactly `CAPABILITIES`, once each."""
    tabulated = _verbs(_engine_declaration_rows())
    assert len(set(tabulated)) == len(tabulated), (
        f"§8's per-engine table lists a capability twice: {tabulated}"
    )
    assert set(tabulated) == set(CAPABILITIES), (
        "§8's per-engine table and the code disagree on which capabilities "
        f"exist — only in the spec: {sorted(set(tabulated) - set(CAPABILITIES))}; "
        f"only in the code: {sorted(set(CAPABILITIES) - set(tabulated))}"
    )


def test_the_engine_table_matches_what_each_engine_declares() -> None:
    """Every ✓/✗ cell equals that engine's `capabilities` frozenset.

    This is the check the retired "reference matrix" needed: it claimed `grad`
    and `quantized_weights` for the nnsight engine, which declares neither.

    Engines are imported inside the test — they pull torch, and this module is
    a `unit` test about markdown.
    """
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine

    engines = [PytorchHooksEngine, NnsightEngine]
    wrong: list[str] = []
    for row in _engine_declaration_rows():
        if _is_component_row(row):
            continue
        name = _row_name(row)
        # strict: deleting the `nnsight` column would leave 2-cell rows and a
        # lenient zip would truncate — all three §8 tests passing while
        # checking one engine. A third engine column must be red here too.
        for engine, cell in zip(engines, row[1:], strict=True):
            claimed = cell.startswith("✓")
            assert claimed or cell.startswith("✗"), (
                f"§8's cell for {engine.name}/{name} starts with neither ✓ nor "
                f"✗: {cell!r}"
            )
            if claimed is not (name in engine.capabilities):
                wrong.append(
                    f"{engine.name}/{name}: spec says "
                    f"{'✓' if claimed else '✗'}, class says "
                    f"{'✓' if name in engine.capabilities else '✗'}"
                )
    assert not wrong, "§8's per-engine table and the classes disagree — " + "; ".join(
        wrong
    )


def test_the_component_counts_match_each_engine() -> None:
    """§8's `N of 62` cells equal what each engine's `components` set holds.

    This row is skipped by the capability comparison above — `component:<name>`
    is generated per site and is not in `CAPABILITIES` — which left the only
    two *numbers* in the table unguarded. They are also the only cells nobody
    has to touch to make wrong: both engines compute `components` as
    `frozenset(COMPONENTS) - <interior set>`, so adding one name to
    `Component` moves 50/62 and 49/62 by itself.

    The sharp part, and the reason this is worth its own test: that same edit
    *does* fail §2.4's census, which tells its author to update §2.4's list and
    says nothing about §8. One edit, a red pointer in the wrong file and a
    green §8.

    `writable_components` is asserted equal to `components` because the row's
    header carries `[:write]`, so the counts are claimed for writes too.
    """
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine

    rows = [row for row in _engine_declaration_rows() if _is_component_row(row)]
    assert len(rows) == 1, f"expected one component row in §8, found {len(rows)}"

    wrong: list[str] = []
    for engine, cell in zip(
        [PytorchHooksEngine, NnsightEngine], rows[0][1:], strict=True
    ):
        match = re.match(r"(\d+) of (\d+)", cell)
        assert match, f"§8's component cell for {engine.name} is not 'N of M': {cell!r}"
        served, total = int(match.group(1)), int(match.group(2))
        if total != len(COMPONENTS):
            wrong.append(
                f"{engine.name}: spec says {total} components exist, "
                f"the code has {len(COMPONENTS)}"
            )
        if served != len(engine.components):
            wrong.append(
                f"{engine.name}: spec says it serves {served}, "
                f"the class serves {len(engine.components)}"
            )
        if set(engine.writable_components) != set(engine.components):
            wrong.append(
                f"{engine.name}: the row claims [:write] over the same set, but "
                "writable_components differs from components"
            )
    assert not wrong, "§8's component counts and the engines disagree — " + "; ".join(
        wrong
    )


# --------------------------------------------------------------------------- #
# the capability registry — the third instance
# --------------------------------------------------------------------------- #
#
# One row per component in `registry.CAPABILITIES` is the truth every other
# statement of component / mechanism / availability is generated from: the two
# engines' `components` sets, the write policy the executor and `validate`
# apply, the stream table both halves of the stream check read, the docs
# tables, and the A3B sweep's buckets. Each of those used to be its own table
# (the census counted thirteen facts stated in two to nine places each), and
# two of them disagreed with the spec. These guards hold every derived
# statement to the rows, and the rows to the vocabulary.

RUNNING_EXPERIMENTS = SPEC.parent / "running_experiments.md"
#: The model page that carries the generated component table.
QWEN36_PAGE = SPEC.parent / "qwen36_35b_a3b.md"


def test_every_component_has_exactly_one_capability_row() -> None:
    """`set(rows) == set(COMPONENTS)`, and the vacuity floor: a registry that
    lost half its rows must fail here by count, not only by diff."""
    from causalab.protocol.registry import CAPABILITIES

    assert len(CAPABILITIES) >= 54  # 62 before eight spellings became aliases
    assert set(CAPABILITIES) == set(COMPONENTS), (
        f"rows without a component: {sorted(set(CAPABILITIES) - set(COMPONENTS))}; "
        f"components without a row: {sorted(set(COMPONENTS) - set(CAPABILITIES))}"
    )
    for component, row in CAPABILITIES.items():
        assert row.component == component


#: The two engines' `components` sets as the engines declared them before the
#: registry generated them (hand-written in `pytorch_hooks/engine.py` and
#: `nnsight_tracing/engine.py`): 50 and 49 names — then one decided change
#: applied: the
#: eight `deltanet_*` spellings that named the reference engine's `delta_*`
#: tensors are aliases (schema.DEPRECATED_COMPONENTS), so they leave the
#: nnsight set and the eight `delta_*` names enter it (50 and 49 members
#: still, 54 names in all). Listed, not derived, so that generating the sets
#: from the rows is proven to reproduce the routing the suite was green on.
PRE_REGISTRY_PYTORCH_HOOKS_COMPONENTS: frozenset[str] = frozenset(
    {
        "input_ids", "embeddings", "block_input", "attention_input_norm",
        "delta_qkv", "delta_gate", "delta_conv", "delta_query", "delta_key",
        "delta_value", "delta_beta", "delta_decay", "delta_kv_mem",
        "delta_state_update", "delta_state", "delta_kernel_output",
        "attention_query_pre_rope", "attention_key_pre_rope",
        "attention_value_states", "attention_gate", "attention_query",
        "attention_key", "attention_scores", "attention_z", "attention_result",
        "delta_premix", "attention_output", "attention_premix", "attention_probs",
        "block_mid", "mlp_input_norm", "mlp_input", "mlp_output",
        "mlp_activation", "router_logits", "router_scores", "expert_idx",
        "expert_gate_proj", "expert_up_proj", "expert_activation",
        "expert_output", "routed_output", "shared_expert_gate_proj",
        "shared_expert_up_proj", "shared_expert_activation",
        "shared_expert_output", "shared_expert_gate", "block_output", "ln_final",
        "lm_head",
    }
)  # fmt: skip
PRE_REGISTRY_NNSIGHT_COMPONENTS: frozenset[str] = frozenset(
    {
        "input_ids", "embeddings", "block_input", "attention_input_norm",
        "attention_query_pre_rope", "attention_key_pre_rope",
        "attention_value_states", "attention_gate", "attention_query",
        "attention_key", "attention_scores", "attention_z", "delta_qkv",
        "delta_gate", "delta_conv", "deltanet_query", "deltanet_key",
        "delta_value", "delta_beta", "delta_decay", "deltanet_state",
        "delta_kernel_output", "delta_premix", "attention_result",
        "attention_output", "attention_premix", "attention_probs", "block_mid",
        "mlp_input_norm", "mlp_input", "mlp_output", "mlp_activation",
        "router_logits", "router_scores", "expert_idx", "expert_gate_proj",
        "expert_up_proj", "expert_activation", "expert_permutation",
        "expert_output", "routed_output", "shared_expert_gate_proj",
        "shared_expert_up_proj", "shared_expert_activation",
        "shared_expert_output", "shared_expert_gate", "block_output", "ln_final",
        "lm_head",
    }
)  # fmt: skip


def test_the_pre_pr_literals_have_their_recorded_sizes() -> None:
    assert len(PRE_REGISTRY_PYTORCH_HOOKS_COMPONENTS) == 50
    assert len(PRE_REGISTRY_NNSIGHT_COMPONENTS) == 49


def test_engine_component_sets_are_generated_from_the_rows() -> None:
    """Each engine's `components` and `writable_components` are exactly the
    rows whose `reads` name it — nothing declared by hand survives in the
    classes. Engines are imported inside the test (they pull torch)."""
    from causalab.neural.engines.nnsight_tracing.engine import NnsightEngine
    from causalab.neural.engines.pytorch_hooks.engine import PytorchHooksEngine
    from causalab.protocol.registry import CAPABILITIES, components_served_by

    for engine in (PytorchHooksEngine, NnsightEngine):
        from_rows = frozenset(
            c for c, row in CAPABILITIES.items() if engine.name in row.reads
        )
        assert engine.components == from_rows == components_served_by(engine.name)
        assert engine.writable_components == engine.components


def test_the_generated_sets_equal_the_pre_pr_declarations() -> None:
    """Both engines add complete neuron outputs to the recorded component set."""
    from causalab.protocol.registry import components_served_by

    assert components_served_by(
        "pytorch_hooks"
    ) == PRE_REGISTRY_PYTORCH_HOOKS_COMPONENTS | {
        "mlp_neuron_output",
        "expert_neuron_output",
    }
    assert components_served_by("nnsight") == PRE_REGISTRY_NNSIGHT_COMPONENTS | {
        "mlp_neuron_output",
        "expert_neuron_output",
    }


def test_the_capability_verbs_are_byte_identical_to_the_base() -> None:
    """`requires()`'s coarse verbs — `writable_attention_probs` included, now
    generated from the rows' `write_capability` cell — are the base's tuple,
    in the base's order, plus the three training verbs appended since (§2.11,
    rule 30: what a fit authors that an engine's loop may not implement) and
    `generation_writes` beside `generate` (§2.9 `writes_during_generation`:
    a write hook kept installed across the decode steps)."""
    assert CAPABILITIES == (
        "grad",
        "paired_forward",
        "full_logits",
        "writable_attention_probs",
        "pytorch_fn_local",
        "generate",
        "generation_writes",
        "quantized_weights",
        "train_free_params",
        "train_loss_precision",
        "train_eval_updates",
    )


def test_component_streams_equal_the_rows() -> None:
    """`COMPONENT_STREAMS` is a view of the rows' `stream` cell, and the
    schema module no longer carries a second copy."""
    import causalab.protocol.schema as schema
    from causalab.protocol.registry import CAPABILITIES, COMPONENT_STREAMS

    assert dict(COMPONENT_STREAMS) == {
        c: row.stream for c, row in CAPABILITIES.items() if row.stream is not None
    }
    assert not hasattr(schema, "COMPONENT_STREAMS")


def _component_table_rows() -> list[str]:
    """The body rows of the component tables on the Qwen3.6-35B-A3B page,
    as raw lines, so the comparison is row for row and cell for cell."""
    text = QWEN36_PAGE.read_text()
    start = text.index("**Model boundary (no `layer`)**")
    end = text.index("<!-- generated: end component-table -->")
    section = text[start:end]
    return [
        line
        for line in section.splitlines()
        if line.startswith("| `")  # a body row; headers and separators are not
    ]


def test_the_component_table_on_the_model_page_is_the_rendering() -> None:
    """`docs/qwen36_35b_a3b.md`'s component table equals
    `registry.render_component_tables()` row for row — the docs table is
    generated, and a later change can swap the committed text for the
    rendering call."""
    from causalab.protocol.registry import render_component_tables

    committed = _component_table_rows()
    rendered = [
        line
        for line in render_component_tables().splitlines()
        if line.startswith("| `")
    ]
    assert len(committed) >= 54  # 62 before eight spellings became aliases
    assert committed == rendered, (
        "docs/qwen36_35b_a3b.md's component table is not the rendering "
        "of registry.CAPABILITIES — regenerate it with "
        "registry.render_component_tables()"
    )


def test_the_component_table_headings_are_the_rendering() -> None:
    """The group headings (with their layer counts) are rendered too."""
    from causalab.protocol.registry import render_component_tables

    text = QWEN36_PAGE.read_text()
    for line in render_component_tables().splitlines():
        if line.startswith("**"):
            assert line in text, f"heading missing from qwen36_35b_a3b.md: {line}"


def test_the_engine_component_counts_are_the_rendering() -> None:
    """Spec §8's generated `component:<name>` row and running_experiments §6's
    `components` row both open with `registry.engine_component_summary`."""
    from causalab.protocol.registry import ENGINES, engine_component_summary

    spec_row = [row for row in _engine_declaration_rows() if _is_component_row(row)]
    assert len(spec_row) == 1
    guide_rows = [
        row for row in _rows(RUNNING_EXPERIMENTS.read_text()) if row[0] == "components"
    ]
    assert len(guide_rows) == 1
    for row in (spec_row[0], guide_rows[0]):
        for engine, cell in zip(ENGINES, row[1:], strict=True):
            assert cell.startswith(engine_component_summary(engine)), (
                f"{engine}: {cell!r} does not open with "
                f"{engine_component_summary(engine)!r}"
            )


def test_the_a3b_sweep_buckets_are_the_registry() -> None:
    """The sweep helper's partition is a query over the rows: every component
    in exactly one bucket, and each bucket equal to its row predicate."""
    from causalab.protocol.registry import CAPABILITIES

    from tests._helpers import a3b_sweep as sweep  # imports torch

    assert not sweep.unclaimed_components()
    assert not sweep.double_claimed_components()
    assert sweep.READ_ONLY == {c for c, r in CAPABILITIES.items() if r.writes is None}
    assert sweep.SWAP_ONLY_WRITES == {
        c for c, r in CAPABILITIES.items() if r.writes == frozenset({"swap"})
    }
    assert set(sweep.HOOKS_ONLY) == {
        c for c, r in CAPABILITIES.items() if r.reads == {"pytorch_hooks"}
    }
    assert set(sweep.NNSIGHT_ONLY) == {
        c for c, r in CAPABILITIES.items() if r.reads == {"nnsight"}
    }
    assert set(sweep.SHARED_FULL_ONLY) == {
        c
        for c, r in CAPABILITIES.items()
        if r.stream == "full_attention" and len(r.reads) == 2
    }
    assert set(sweep.SHARED_LINEAR_ONLY) == {
        c
        for c, r in CAPABILITIES.items()
        if r.stream == "linear_attention" and len(r.reads) == 2
    }
    assert len(sweep.SHARED_LINEAR_ONLY) == 8  # the eight aliased pairs' names
    assert set(sweep.ABSENT_ON_A3B) == {"mlp_activation", "mlp_neuron_output"}


def _capability_table(first_header: str) -> list[list[str]]:
    """Body rows of the §2.4 table whose first header cell is ``first_header``."""
    rows = _rows(_section("### 2.4 `sites`"))
    start = next((i for i, row in enumerate(rows) if row[0] == first_header), None)
    assert start is not None, f"§2.4 has no table headed {first_header!r}"
    body: list[list[str]] = []
    for row in rows[start + 1 :]:
        if not row[0].startswith("`"):
            break
        body.append(row)
    assert body, f"§2.4's {first_header!r} table parsed to zero rows"
    return body


def test_reason_codes_match_spec() -> None:
    """§2.4's reason-code table is exactly `errors.REASON_CODES`, and every row
    says which code path (or which PR) emits it."""
    from causalab.protocol.rules.errors import REASON_CODES

    rows = _capability_table("reason")
    tabulated = [row[0].strip("`") for row in rows]
    assert len(set(tabulated)) == len(tabulated), (
        f"a reason is listed twice: {tabulated}"
    )
    assert set(tabulated) == set(REASON_CODES), (
        f"only in the spec: {sorted(set(tabulated) - set(REASON_CODES))}; "
        f"only in the code: {sorted(set(REASON_CODES) - set(tabulated))}"
    )
    for row in rows:
        assert row[1].strip(), f"reason {row[0]} names no emitter"


#: A reason code leaving the code: as a refusal's `reason=` keyword, or as the
#: first argument of the `unavailable(...)` value (spec §4.1).
_EMITS_REASON = re.compile(r"""(?:\breason=|\bunavailable\(\s*)["'](\w+)["']""")


def test_reason_code_emitters_match_the_code() -> None:
    """The table marks unused reason codes as reserved.

    Compare that marker with the codes emitted in refusals and unavailable
    results. Adding an emitter requires an update to its documentation row.
    """
    from causalab.protocol.rules.errors import REASON_CODES

    emitted: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        emitted.update(_EMITS_REASON.findall(path.read_text()))
    assert emitted <= set(REASON_CODES), f"emitted outside the vocabulary: {emitted}"
    declared_only = {
        row[0].strip("`")
        for row in _capability_table("reason")
        if row[1].strip().startswith("Reserved")
    }
    assert emitted == set(REASON_CODES) - declared_only, (
        f"emitted in code but marked as reserved: {sorted(emitted & declared_only)}; "
        f"marked as emitted but no code emits it: "
        f"{sorted(set(REASON_CODES) - declared_only - emitted)}"
    )


def test_predicates_match_spec_and_have_a_probe() -> None:
    """§2.4's predicate table is exactly `registry.PREDICATES`, and the site
    resolver has one module-tree probe per predicate — a predicate added to a
    row without a probe fails here, not a document."""
    from causalab.protocol.registry import CAPABILITIES, PREDICATES

    rows = _capability_table("predicate")
    tabulated = [row[0].strip("`") for row in rows]
    assert set(tabulated) == set(PREDICATES), (
        f"only in the spec: {sorted(set(tabulated) - set(PREDICATES))}; "
        f"only in the code: {sorted(set(PREDICATES) - set(tabulated))}"
    )
    from causalab.neural.shared.model_tree import _PREDICATE_PROBES  # imports torch

    assert set(_PREDICATE_PROBES) == set(PREDICATES)
    used = {p for row in CAPABILITIES.values() for p in row.requires}
    assert used == set(PREDICATES), (
        f"predicates no row requires: {set(PREDICATES) - used}"
    )


def test_tap_kinds_are_the_closed_vocabulary() -> None:
    from causalab.protocol.registry import CAPABILITIES, TAP_KINDS

    assert {row.tap for row in CAPABILITIES.values()} == set(TAP_KINDS)


def test_override_keys_and_packings_match_spec() -> None:
    """§2.4's two per-family tap table vocabularies are exactly the code's
    (`registry.OVERRIDE_KEYS`, `registry.PACKINGS`) — the tap table's census
    guard."""
    from causalab.protocol.registry import OVERRIDE_KEYS, PACKINGS

    keys = [row[0].strip("`") for row in _capability_table("override key")]
    assert keys == list(OVERRIDE_KEYS), keys
    packings = [row[0].strip("`") for row in _capability_table("packing")]
    assert packings == list(PACKINGS), packings


def test_the_interior_rows_are_the_resolvers_interior_set() -> None:
    """The rows that carry per-family addresses are the rows the site
    resolver routes to `_attention_interior_site` — one set, two readers."""
    from causalab.protocol.registry import INTERIOR_ROWS
    from causalab.neural.shared.sites import _ATTENTION_INTERIOR  # imports torch

    assert set(INTERIOR_ROWS) == set(_ATTENTION_INTERIOR)


# --------------------------------------------------------------------------- #
# §2.8's `do` table (censused, not generated — its `write` and `class`
# columns are prose the reader needs)
# --------------------------------------------------------------------------- #

#: A `do` row's first cell: the mechanism is the one key of the object.
_DO_CELL = re.compile(r'^`\{"(\w+)"')


def _do_table() -> list[str]:
    """The mechanism named by each row of §2.8's `do` table — the rows whose
    first cell is a one-key object literal — in the table's order."""
    rows = _rows(_section("### 2.8 `writes` and the `do` algebra"))
    names = [m.group(1) for row in rows if (m := _DO_CELL.match(row[0]))]
    assert names, (
        "§2.8's `do` table parsed to zero rows — the reader or the table changed"
    )
    return names


def test_the_do_table_is_exactly_MECHANISMS() -> None:
    """§2.8's `do` table lists exactly `schema.MECHANISMS`, once each. The
    write-policy cells of the generated tables (`writes`, per component) name
    mechanisms out of this same set, so a mechanism added to the code with no
    row here — or a row for a mechanism the code lacks — fails by name."""
    from causalab.protocol.schema import MECHANISMS

    tabulated = _do_table()
    assert len(set(tabulated)) == len(tabulated), (
        f"a mechanism is listed twice: {tabulated}"
    )
    assert set(tabulated) == set(MECHANISMS), (
        f"only in the spec: {sorted(set(tabulated) - set(MECHANISMS))}; "
        f"only in the code: {sorted(set(MECHANISMS) - set(tabulated))}"
    )


def _running_experiments_family_table_rows() -> list[str]:
    text = RUNNING_EXPERIMENTS.read_text()
    start = text.index("### The attention interior, per family")
    end = text.index("### `delta_*` and `deltanet_*`")
    return [line for line in text[start:end].splitlines() if line.startswith("| `")]


def test_the_family_table_in_running_experiments_is_the_rendering() -> None:
    """`docs/running_experiments.md` §5's per-family attention-interior table
    equals `registry.render_family_table()` row for row, header included —
    the measured three-family table lives in the rows and nowhere else."""
    from causalab.protocol.registry import INTERIOR_ROWS, render_family_table

    rendered = render_family_table().splitlines()
    committed = _running_experiments_family_table_rows()
    body = [line for line in rendered if line.startswith("| `")]
    assert len(body) == len(INTERIOR_ROWS) >= 4
    assert committed == body, (
        "docs/running_experiments.md §5's per-family table is not the rendering "
        "of the rows' overrides — regenerate it with registry.render_family_table()"
    )
    assert rendered[0] in RUNNING_EXPERIMENTS.read_text()  # the family header


def test_the_reason_code_carried_by_a_refusal_is_in_the_vocabulary() -> None:
    """No code path can invent a reason: the error classes check it."""
    from causalab.protocol.rules.errors import ProtocolError, ValidationError

    assert ProtocolError("P4", "x", reason="component_unavailable").reason
    with pytest.raises(AssertionError, match="unknown reason code"):
        ProtocolError("P4", "x", reason="made_up")  # type: ignore[arg-type]
    err = ValidationError(4, "x", path="sites.a", reason="unsupported_mechanism")
    assert err.reason == "unsupported_mechanism" and err.code == "V4"


# --------------------------------------------------------------------------- #
# §2.5 — the featurizer families, the gate maps and the field legality table
# (docs/DOCUMENTATION.md): the tables the method pages render
# are the tables the parser and the validator read, and they cover the model.
# --------------------------------------------------------------------------- #


def test_every_featurizer_kind_has_a_family() -> None:
    """One family record per kind and no record for a kind that is not one —
    the family paragraph a method page opens with, and the ``featurize`` cell
    of the kinds table, exist for exactly the closed set."""
    assert set(FEATURIZER_FAMILIES) == set(FEATURIZER_KINDS)
    for kind, family in FEATURIZER_FAMILIES.items():
        assert family.sentence.endswith("."), kind
        assert family.featurize.startswith("`"), kind


def test_the_gate_maps_are_the_parametrization_vocabulary() -> None:
    """``GATE_PARAMETRIZATIONS`` is derived from ``GATE_MAPS`` and the default
    map is one of them; every map admits at most one mask penalty, and that
    penalty is a regularizer kind."""
    assert GATE_PARAMETRIZATIONS == tuple(GATE_MAPS)
    assert GATE_DEFAULT_MAP in GATE_MAPS
    for name, gate_map in GATE_MAPS.items():
        assert gate_map.name == name
        assert gate_map.penalty is None or gate_map.penalty in REGULARIZER_KINDS


def _gate_map_table() -> list[list[str]]:
    section = _section("### 2.5 `featurizers`")
    start = section.index("| parametrization | soft mask (train) |")
    end = section.index("\n\n", start)
    return _rows(section[start:end])


def test_the_gate_map_table_is_exactly_GATE_MAPS() -> None:
    """§2.5's per-parametrization table has one row per map, in the
    vocabulary's order (the block is generated; this is the census's own
    reading of it, independent of the generator's byte check)."""
    rows = _gate_map_table()
    assert rows, "§2.5 has no parametrization table"
    names = [m.group(1) for row in rows if (m := CODE.search(row[0]))]  # body rows
    assert names == list(GATE_PARAMETRIZATIONS)


def test_every_authorable_field_has_a_legality_row() -> None:
    """Every field a kind may author is in the legality table, and the table
    names no field no kind may author — the two lists are one list."""
    authorable = set().union(*FEATURIZER_FIELDS.values())
    assert set(FEATURIZER_FIELD_CONDITIONS) == authorable


def test_every_featurizer_spec_field_has_an_attribute_doc() -> None:
    """Every field of the protocol object model's featurizer record
    carries a ``#:`` attribute doc — the method pages pull them through the
    support-tables tool's ``attrs`` reader, so a missing one is a blank page
    entry."""
    import dataclasses
    import sys

    from causalab.protocol.schema import FeaturizerSpec

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from generate_support_tables import attribute_docs
    finally:
        sys.path.pop(0)
    docs = attribute_docs(FeaturizerSpec)
    missing = [f.name for f in dataclasses.fields(FeaturizerSpec) if f.name not in docs]
    assert not missing, f"FeaturizerSpec fields without a #: doc: {missing}"


def test_every_rule_kind_is_a_featurizer_kind() -> None:
    """``Rule.kinds`` names featurizer kinds and nothing else (``errors`` sits
    below ``schema`` and cannot import the vocabulary, so this is the check);
    and the rules whose text is about a gate say so."""
    for rule in RULES.values():
        assert rule.kinds <= set(FEATURIZER_KINDS), (rule.slug, rule.kinds)
    assert RULES["group_legality"].kinds == {"gate"}
    assert RULES["scores_init"].kinds == {"gate"}
    assert TRAINABLE_KINDS <= set(FEATURIZER_KINDS)


#: A parseable value for each gate field, for the legality census below.
_GATE_FIELD_VALUES: dict[str, object] = {
    "parametrization": "sigmoid",
    "group": "head",
    "axis": "position",
    "init": {"fill": 0.5},
    "temperature": 0.5,
    "stretch": [-0.1, 1.1],
    "dead": {"leak": 0.1},
    "top_k": 3,
    "k_schedule": {"kind": "fixed", "k": 2},
    "stop_grad_shift": True,
    "pool": "shared",
}


def _gate_document(gate: dict[str, object]) -> dict[str, object]:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "gpt2", "revision": "main"},
        "data": {
            "base": {"dataset": "d#train", "field": "input"},
            "counterfactual": {"dataset": "d#train", "field": "cf"},
        },
        "method": {
            "intervened_models": {
                "original_counterfactual": {"input": "counterfactual", "reads": ["v"]}
            },
            "sites": {"t": {"component": "attention_premix", "layers": [1]}},
            "featurizers": {"g": {"kind": "gate", **gate}},
            "reads": {"v": {"site": "t", "pos": -1, "featurizer": "g"}},
            "save": [saved("v", "original_counterfactual", "v.json")],
        },
    }


@pytest.mark.parametrize("field", sorted(FEATURIZER_FIELDS["gate"]))
@pytest.mark.parametrize("parametrization", GATE_PARAMETRIZATIONS)
@pytest.mark.parametrize("loaded", [False, True], ids=["fit", "loaded"])
def test_the_parser_refuses_exactly_what_the_legality_table_says(
    field: str, parametrization: str, loaded: bool
) -> None:
    """The table and the parser agree cell by cell. For every gate field,
    map and fit state, a document authoring the field parses if and only if
    ``FEATURIZER_FIELD_CONDITIONS`` lists the map for that state. The base
    document carries whatever else the state needs — a ``k_schedule`` on a
    budget fit, a ``top_k`` beside a loaded ``pool`` — so the one field under
    test is the only thing that can be refused."""
    if field == "parametrization":
        pytest.skip("the field under test is the map itself")
    gate: dict[str, object] = {"parametrization": parametrization}
    if loaded:
        gate["file_path"] = "fit/g.safetensors"
    if parametrization == "budget" and not loaded and field != "k_schedule":
        gate["k_schedule"] = _GATE_FIELD_VALUES["k_schedule"]
    if field == "pool" and loaded:
        # a pooled readout needs a top_k — where the table lets this map have
        # one; where it does not, the pool row itself is the refusal under test
        top_k_legal = FEATURIZER_FIELD_CONDITIONS["top_k"].legal(loaded=True)
        if top_k_legal is None or parametrization in top_k_legal:
            gate["top_k"] = _GATE_FIELD_VALUES["top_k"]
    gate[field] = _GATE_FIELD_VALUES[field]
    legal = FEATURIZER_FIELD_CONDITIONS[field].legal(loaded=loaded)
    expected = legal is None or parametrization in legal
    try:
        parse_document(_gate_document(gate))
    except ParseError as err:
        assert not expected, (
            f"{field} under {parametrization} ({'loaded' if loaded else 'fit'}) is legal per the table but the parser refused it: {err}"
        )
        assert err.path == f"featurizers.g.{field}", err.path
    else:
        assert expected, (
            f"{field} under {parametrization} ({'loaded' if loaded else 'fit'}) parsed but the table refuses it"
        )
