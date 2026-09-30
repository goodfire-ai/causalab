"""Validate counterfactual pairs and their edit groups.

The row records which spans change together. Validation checks answer changes,
answer correctness, intended edits, token stability, and location coverage.
Atomic groups require an intervention to cover the whole group. These checks
use serialized values; callers supply the tokenizer when token checks are needed.

``GRADES`` names the outcomes returned by the task's scoring rule. This module
uses the standard library so document validation can read it before a model loads."""

from __future__ import annotations

import dataclasses
import difflib
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BASE_TEXT_COLUMN",
    "COUNTERFACTUAL_TEXT_COLUMN",
    "EDIT_GROUPS_COLUMN",
    "GRADES",
    "GRADE_VALUES",
    "SIDES",
    "EditGroup",
    "EditGroupError",
    "TokenEdit",
    "check_answer_change",
    "check_component_wise",
    "check_correctness",
    "check_intended_token_change",
    "check_location_coverage",
    "check_no_unintended_edits",
    "check_tokenizer_stability",
    "grade_name",
    "grade_value",
    "parse_edit_groups",
    "row_texts",
    "token_edits",
    "token_runs",
]

#: The optional per-row column (spec §2.2) — a reserved column name
#: (``causalab/tasks/serialize.py``), written only when the example declares
#: groups.
EDIT_GROUPS_COLUMN = "edit_groups"

#: The two sides of a pair, in the order a group's ``spans`` lists them —
#: the ``data`` roles of a paired document (§2.2).
SIDES: tuple[str, ...] = ("base", "counterfactual")

#: Where each side's text lives in the row vocabulary: ``input`` and
#: ``counterfactual_inputs[0]`` (``serialize.py``; the fixture tables' shape).
BASE_TEXT_COLUMN = "input"
COUNTERFACTUAL_TEXT_COLUMN = "counterfactual_inputs"

#: The per-example status vocabulary (§2.10), in [`ScoringSpec.grade`][causalab.causal.scoring.ScoringSpec.grade]'s
#: order of values: ``1.0``, ``0.0``, ``None``. ``grade`` is not ``matched``
#: (the module docstring).
GRADES: tuple[str, ...] = ("correct", "incorrect", "unscored")

#: The 1:1 map onto [`ScoringSpec.grade`][causalab.causal.scoring.ScoringSpec.grade]'s return values.
GRADE_VALUES: Mapping[str, float | None] = MappingProxyType(
    {"correct": 1.0, "incorrect": 0.0, "unscored": None}
)

#: The keys a group object carries — exactly these (rule 1's spirit: no
#: authored extras).
_GROUP_KEYS: frozenset[str] = frozenset({"name", "atomic", "spans"})


class EditGroupError(ValueError):
    """A pair-validity check failed, or an ``edit_groups`` value is malformed.
    The callers that sit on the checklist (the loader, the executor) re-raise
    it as rule 27; this module spells no rule number itself."""


# --------------------------------------------------------------------------- #
# grade
# --------------------------------------------------------------------------- #


def grade_name(value: float | None) -> str:
    """The [`GRADES`][] member for one [`ScoringSpec.grade`][causalab.causal.scoring.ScoringSpec.grade] value —
    ``1.0 → correct``, ``0.0 → incorrect``, ``None → unscored``. Anything
    else is not a grade and is refused, not rounded."""
    for name, expected in GRADE_VALUES.items():
        if value is expected or (value is not None and value == expected):
            return name
    raise EditGroupError(
        f"{value!r} is not a grade — ScoringSpec.grade returns 1.0, 0.0 or None "
        f"({', '.join(GRADES)})"
    )


def grade_value(name: str) -> float | None:
    """The inverse of [`grade_name`][]."""
    if name not in GRADE_VALUES:
        raise EditGroupError(f"{name!r} is not one of {GRADES}")
    return GRADE_VALUES[name]


# --------------------------------------------------------------------------- #
# the column
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class EditGroup:
    """One group of the ``edit_groups`` column (the module docstring):
    ``spans[side][k]`` is constituent ``k``'s char span on ``side``."""

    name: str
    atomic: bool
    spans: Mapping[str, tuple[tuple[int, int], ...]]

    @property
    def constituents(self) -> int:
        """How many constituents the group has (the same on both sides)."""
        return len(self.spans[SIDES[0]])

    def as_row_value(self) -> dict[str, Any]:
        """The JSON shape the column carries."""
        return {
            "name": self.name,
            "atomic": self.atomic,
            "spans": {
                side: [[start, end] for start, end in self.spans[side]]
                for side in SIDES
            },
        }


def row_texts(row: Mapping[str, Any]) -> dict[str, str]:
    """The two prompts of a row, by side — ``input`` and
    ``counterfactual_inputs[0]``."""
    base = row.get(BASE_TEXT_COLUMN)
    counterfactuals = row.get(COUNTERFACTUAL_TEXT_COLUMN)
    if not isinstance(base, str):
        raise EditGroupError(
            f"a pair's base text is the {BASE_TEXT_COLUMN!r} column, a string; "
            f"got {type(base).__name__}"
        )
    if (
        not isinstance(counterfactuals, list)
        or len(counterfactuals) != 1
        or not isinstance(counterfactuals[0], str)
    ):
        raise EditGroupError(
            f"a pair's counterfactual text is {COUNTERFACTUAL_TEXT_COLUMN}[0], a "
            f"one-element list of one string; got {counterfactuals!r}"
        )
    return {SIDES[0]: base, SIDES[1]: counterfactuals[0]}


def parse_edit_groups(row: Mapping[str, Any]) -> tuple[EditGroup, ...]:
    """The row's ``edit_groups`` as objects, checked against the row's own
    texts — the load-time half of rule 27 (the module docstring).

    A missing column, or a ``null``, is no declaration: ``()``. Otherwise a
    list of group objects with exactly ``name`` / ``atomic`` / ``spans``;
    names non-empty and distinct; ``spans`` keyed by exactly the two sides,
    each a list of ``[start, end]`` integer pairs with ``0 <= start <= end <=
    len(text)``, sorted and non-overlapping within a side, the same count on
    both sides, at least one; an ``atomic`` group has two or more.
    """
    raw = row.get(EDIT_GROUPS_COLUMN)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise EditGroupError(
            f"{EDIT_GROUPS_COLUMN} must be a list of groups, got {type(raw).__name__}"
        )
    texts = row_texts(row)
    groups: list[EditGroup] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        where = f"{EDIT_GROUPS_COLUMN}[{index}]"
        if not isinstance(item, Mapping):
            raise EditGroupError(
                f"{where} must be an object, got {type(item).__name__}"
            )
        keys = set(item)
        if keys != _GROUP_KEYS:
            raise EditGroupError(
                f"{where} carries {sorted(keys)}; a group has exactly "
                f"{sorted(_GROUP_KEYS)}"
            )
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise EditGroupError(
                f"{where}.name must be a non-empty string, got {name!r}"
            )
        if name in names:
            raise EditGroupError(f"{where}.name {name!r} is declared twice in the row")
        names.add(name)
        atomic = item["atomic"]
        if not isinstance(atomic, bool):
            raise EditGroupError(
                f"{where}.atomic must be true or false, got {atomic!r}"
            )
        spans = item["spans"]
        if not isinstance(spans, Mapping) or set(spans) != set(SIDES):
            raise EditGroupError(
                f"{where}.spans must be keyed by exactly {list(SIDES)}, got "
                f"{sorted(spans) if isinstance(spans, Mapping) else spans!r}"
            )
        parsed = {
            side: _parse_side_spans(spans[side], texts[side], f"{where}.spans.{side}")
            for side in SIDES
        }
        counts = {side: len(parsed[side]) for side in SIDES}
        if len(set(counts.values())) != 1:
            raise EditGroupError(
                f"{where} ({name!r}) declares {counts[SIDES[0]]} constituent span(s) "
                f"on {SIDES[0]!r} and {counts[SIDES[1]]} on {SIDES[1]!r}; constituent k "
                "is spans.base[k] paired with spans.counterfactual[k], so the two "
                "sides must declare the same number"
            )
        if counts[SIDES[0]] == 0:
            raise EditGroupError(f"{where} ({name!r}) declares no constituent span")
        if atomic and counts[SIDES[0]] < 2:
            raise EditGroupError(
                f"{where} ({name!r}) is atomic with one constituent — an atomic "
                "group is a coordinated edit of two or more spans; a single span "
                "is declared with atomic: false"
            )
        groups.append(
            EditGroup(name=name, atomic=atomic, spans=MappingProxyType(parsed))
        )
    return tuple(groups)


def _parse_side_spans(value: Any, text: str, where: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise EditGroupError(
            f"{where} must be a list of [start, end] pairs, got {value!r}"
        )
    out: list[tuple[int, int]] = []
    for k, pair in enumerate(value):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in pair)
        ):
            raise EditGroupError(
                f"{where}[{k}] must be an integer [start, end] pair, got {pair!r}"
            )
        start, end = int(pair[0]), int(pair[1])
        if not 0 <= start <= end <= len(text):
            raise EditGroupError(
                f"{where}[{k}] = [{start}, {end}] is not inside the side's text "
                f"(length {len(text)}: {text!r})"
            )
        if out and start < out[-1][1]:
            raise EditGroupError(
                f"{where}[{k}] = [{start}, {end}] overlaps or precedes "
                f"{where}[{k - 1}] = {list(out[-1])}; constituent spans are sorted "
                "and disjoint within a side"
            )
        out.append((start, end))
    return tuple(out)


# --------------------------------------------------------------------------- #
# chars ↔ tokens
# --------------------------------------------------------------------------- #


def token_runs(
    group: EditGroup, side: str, offsets: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Each constituent's token run on ``side``: the indices of the tokens
    whose char ``offsets`` overlap the constituent's span. ``(0, 0)`` entries
    are padding / specials the text does not spell and never match — the
    same reading ``variable`` positions take of an offset mapping
    (``protocol/positions/encoding.py``), so a declared span and a ``variable``
    anchor over the same characters name the same tokens.
    """
    return tuple(
        tuple(
            index
            for index, (a, b) in enumerate(offsets)
            if not (a == 0 and b == 0) and a < end and b > start
        )
        for start, end in group.spans[side]
    )


def _offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and char offsets of ``text`` under a fast tokenizer."""
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    try:
        mapping = encoded["offset_mapping"]
        ids = encoded["input_ids"]
    except (KeyError, TypeError) as err:
        raise EditGroupError(
            "the pair-validity checks need a tokenizer that reports char offsets "
            "(a `transformers` fast tokenizer: tokenizer(text, "
            "return_offsets_mapping=True)); this one does not"
        ) from err
    return [int(i) for i in ids], [(int(a), int(b)) for a, b in mapping]


@dataclasses.dataclass(frozen=True)
class TokenEdit:
    """One non-``equal`` opcode of a token-level diff between the two sides:
    the char ranges (in each side's text) of the tokens it replaced, deleted
    or inserted."""

    op: str
    base: tuple[tuple[int, int], ...]
    counterfactual: tuple[tuple[int, int], ...]

    def ranges(self, side: str) -> tuple[tuple[int, int], ...]:
        return self.base if side == SIDES[0] else self.counterfactual


def token_edits(
    tokenizer: Any, base: str, counterfactual: str
) -> tuple[TokenEdit, ...]:
    """The token-level differences of two texts, located: each text is
    tokenized as one string (a tokenizer may merge across an edit's boundary,
    which is exactly what the located form makes visible) and diffed by token
    id; every changed token comes back as its char range on its side.

    The same three-way reading ``protocol/positions/alignment.py``'s ``pair_differences``
    takes (prompt / prefix / full context) is a matter of what the caller
    passes as ``base`` and ``counterfactual``; this function adds the offsets
    the checks below need to say *where* a change fell.
    """
    ids_a, offsets_a = _offsets(tokenizer, base)
    ids_b, offsets_b = _offsets(tokenizer, counterfactual)
    matcher = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    return tuple(
        TokenEdit(
            op=op,
            base=tuple(offsets_a[i1:i2]),
            counterfactual=tuple(offsets_b[j1:j2]),
        )
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op != "equal"
    )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _stripped(text: str, span: tuple[int, int]) -> tuple[int, int]:
    """``span`` without its leading whitespace — a byte-level BPE token owns
    the space before its word, which no declared span needs to spell."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    return start, end


def _snippet(text: str, span: tuple[int, int]) -> str:
    return repr(text[span[0] : span[1]])


# --------------------------------------------------------------------------- #
# the six checks
# --------------------------------------------------------------------------- #


def check_answer_change(row: Mapping[str, Any]) -> None:
    """Requirement 1 — the pair's two answers differ (``base_answer`` vs
    ``cf_answer``, the row's expected target change; ``label`` is the answer
    *after* the interchange and is a different fact). A noise-floor table
    (``generate_resample_dataset``) legitimately fails this on purpose, which
    is why it is a check a builder opts into and never a load refusal."""
    base, counterfactual = row.get("base_answer"), row.get("cf_answer")
    if base is None or counterfactual is None:
        raise EditGroupError(
            "answer change: the row carries no base_answer / cf_answer to compare"
        )
    if str(base) == str(counterfactual):
        raise EditGroupError(
            f"answer change: both sides answer {base!r} — the counterfactual "
            "changes nothing the task grades"
        )


def check_correctness(
    spec: Any,
    generated: Mapping[str, str],
    row: Mapping[str, Any],
    *,
    variable: str | None = None,
) -> dict[str, str]:
    """Requirement 2 — the model is *correct* on both sides of the pair: each
    side's generated string, graded by the task's ``ScoringSpec`` against that
    side's own answer, is ``correct``. Returns the two grades by side (the
    [`GRADES`][] names); raises naming the first side that is not.

    ``spec`` is a [`ScoringSpec`][causalab.causal.scoring.ScoringSpec] (anything with
    its ``grade(generated, expected, *, variable)``); ``generated`` is the
    decoded text by side — what a ``decode`` read over a no-intervention
    document returns (§2.10).
    """
    expected = {SIDES[0]: row.get("base_answer"), SIDES[1]: row.get("cf_answer")}
    grades: dict[str, str] = {}
    for side in SIDES:
        if side not in generated:
            raise EditGroupError(f"correctness: no generated string for side {side!r}")
        grades[side] = grade_name(
            spec.grade(generated[side], expected[side], variable=variable)
        )
    for side in SIDES:
        if grades[side] != GRADES[0]:
            raise EditGroupError(
                f"correctness: the {side} side is graded {grades[side]!r} — "
                f"generated {generated[side]!r} against the expected "
                f"{expected[side]!r}"
            )
    return grades


def check_intended_token_change(
    tokenizer: Any,
    row: Mapping[str, Any],
    groups: Sequence[EditGroup] | None = None,
) -> tuple[TokenEdit, ...]:
    """Requirement 3 — the intended edit was made, in tokens: the two prompts
    tokenize differently, and every declared constituent overlaps a changed
    token on at least one side (a declared edit the texts do not carry is a
    claim about the pair that the pair does not make). Returns the edits.
    ``groups`` defaults to the row's own declaration."""
    texts = row_texts(row)
    groups = parse_edit_groups(row) if groups is None else groups
    edits = token_edits(tokenizer, texts[SIDES[0]], texts[SIDES[1]])
    if not edits:
        raise EditGroupError(
            "intended token change: the two prompts tokenize identically — "
            "there is no edit for a counterfactual pair to validate"
        )
    changed = {
        side: [span for edit in edits for span in edit.ranges(side)] for side in SIDES
    }
    for group in groups:
        for k in range(group.constituents):
            if not any(
                _overlaps(span, group.spans[side][k])
                for side in SIDES
                for span in changed[side]
            ):
                raise EditGroupError(
                    f"intended token change: group {group.name!r} constituent {k} "
                    f"({_snippet(texts[SIDES[0]], group.spans[SIDES[0]][k])} → "
                    f"{_snippet(texts[SIDES[1]], group.spans[SIDES[1]][k])}) "
                    "overlaps no changed token on either side — the declared "
                    "edit is not in the texts"
                )
    return edits


def check_no_unintended_edits(
    tokenizer: Any,
    row: Mapping[str, Any],
    groups: Sequence[EditGroup] | None = None,
    *,
    base_prefix: str = "",
    counterfactual_prefix: str = "",
) -> tuple[TokenEdit, ...]:
    """Requirement 4 — nothing changed but what was declared: over the full
    context (prompt + teacher-forced answer prefix, tokenized as one string,
    the reading ``pair_differences`` calls ``full_context``), every changed
    token lies — its leading whitespace aside — inside a declared span on its
    side, or inside the prefix. A one-character edit that re-merges its
    neighbour (``tomorrow → tomorrew``: ``row`` becomes ``re w``) is a changed
    token the one-character span does not contain, and is what this check
    exists to catch; widening the span to the word it re-tokenizes is the
    honest declaration. The default (no prefix) is every v1 corpus row, which
    ends at the prompt."""
    texts = row_texts(row)
    groups = parse_edit_groups(row) if groups is None else groups
    prefixes = {SIDES[0]: base_prefix, SIDES[1]: counterfactual_prefix}
    full = {side: texts[side] + prefixes[side] for side in SIDES}
    edits = token_edits(tokenizer, full[SIDES[0]], full[SIDES[1]])
    allowed = {
        side: [span for group in groups for span in group.spans[side]]
        + [(len(texts[side]), len(full[side]))]
        for side in SIDES
    }
    for edit in edits:
        for side in SIDES:
            for token_span in edit.ranges(side):
                start, end = _stripped(full[side], token_span)
                if start >= end:
                    continue  # a token that is only whitespace
                if not any(s <= start and end <= e for s, e in allowed[side]):
                    raise EditGroupError(
                        f"unintended edit: the {side} token {_snippet(full[side], token_span)} "
                        f"at chars [{token_span[0]}, {token_span[1]}) changed but lies "
                        f"outside every declared span "
                        f"({[list(s) for s in allowed[side][:-1]]}) — either the "
                        "pair carries an edit the groups do not declare, or the "
                        "declared edit re-tokenized its neighbour and the span "
                        "must widen to the token it moved"
                    )
    return edits


def check_tokenizer_stability(
    recorded: Mapping[str, Any] | None, running: Mapping[str, Any]
) -> tuple[str, ...]:
    """Requirement 5 — the run holds the tokenizer the pair was validated
    under: a recorded ``{"key", "revision"}`` equals the loaded bundle's.
    Returns the fields compared. A pure function no run-time check calls — a
    table is held to nothing beside it (§2.2), and where an author keeps the
    record is theirs — stated here so the six requirements are six
    functions."""
    if recorded is None:
        raise EditGroupError(
            "tokenizer stability: the table records no tokenizer — the pair was "
            "never validated under one (build it with --validate-pairs)"
        )
    compared: list[str] = []
    for field in ("key", "revision"):
        want, got = recorded.get(field), running.get(field)
        if want is None or got is None or str(want) != str(got):
            raise EditGroupError(
                f"tokenizer stability: the table was validated under tokenizer."
                f"{field} = {want!r} but the run holds {got!r}"
            )
        compared.append(field)
    return tuple(compared)


def check_location_coverage(
    groups: Iterable[EditGroup],
    offsets: Sequence[tuple[int, int]],
    addressed: Iterable[int],
    *,
    side: str,
    where: str,
    text: str | None = None,
) -> int:
    """Requirement 6 — full-set location coverage: every constituent of every
    group (atomic or not) resolves to at least one token on ``side`` and every
    one of those tokens is in ``addressed`` (the token indices a run's
    positions resolved to, in the same frame as ``offsets``). Returns the
    number of constituents covered."""
    hit = set(addressed)
    covered = 0
    for group in groups:
        for k, run in enumerate(token_runs(group, side, offsets)):
            missing = [i for i in run if i not in hit]
            if not run or missing:
                raise EditGroupError(
                    f"{where}: group {group.name!r} constituent {k}"
                    + (f" ({_snippet(text, group.spans[side][k])})" if text else "")
                    + f" on {side!r} is not fully addressed — "
                    + (
                        "it resolves to no token"
                        if not run
                        else f"tokens {missing} of its run {list(run)} are not"
                    )
                )
            covered += 1
    return covered


def check_component_wise(
    groups: Iterable[EditGroup],
    offsets: Sequence[tuple[int, int]],
    addressed: Iterable[int],
    *,
    side: str,
    where: str,
    text: str | None = None,
) -> None:
    """The coordinated-edit-group refusal — all or none, per intervened model
    (the module docstring): for every ``atomic`` group, the ``addressed``
    tokens either include every token of every constituent on ``side`` or
    touch none of them. Touching some is refused, naming the group, the
    constituents addressed and the siblings that were not. A non-atomic group
    is not checked: its constituents were declared, not coordinated.

    ``offsets`` are the side's token char offsets in the frame ``addressed``
    indexes (an encoded batch's ``offset_mapping[row]``); ``where`` names the
    intervened model and row for the message.
    """
    hit = set(addressed)
    for group in groups:
        if not group.atomic:
            continue
        runs = token_runs(group, side, offsets)
        touched = [k for k, run in enumerate(runs) if hit & set(run)]
        if not touched:
            continue
        whole = [k for k, run in enumerate(runs) if run and set(run) <= hit]
        if len(whole) == len(runs):
            continue  # the group is addressed as one edit
        missing = [k for k in range(len(runs)) if k not in whole]

        def name(k: int) -> str:
            span = group.spans[side][k]
            return f"{k}" + (f" ({_snippet(text, span)})" if text else "")

        raise EditGroupError(
            f"{where}: group {group.name!r} is atomic — a coordinated edit the pair "
            f"was validated as one — but the addressed positions on {side!r} touch "
            f"constituent(s) {', '.join(name(k) for k in touched)} without "
            f"{'whole ' if any(k in touched for k in missing) else ''}"
            f"sibling(s) {', '.join(name(k) for k in missing)}. Address every "
            "constituent (one write per span, or one atomic span over them), or "
            "declare the group atomic: false to apply its parts independently"
        )
