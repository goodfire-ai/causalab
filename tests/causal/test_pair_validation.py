"""``causalab/causal/pair_validation.py`` — the counterfactual pair as a validation
layer over the row, torch-free. ``docs/intervention_protocol.md`` states the
contract: sec. 2.2 (coordinated edits), sec. 2.10 (the ``grade`` vocabulary)
and rule 27 in sec. 5.

**T7** — three counterfactual kinds as fixtures: the
single-numeral edit validates standalone; the relation-word-plus-total edit
and the two-mapping-entry swap are refused when one constituent is addressed
alone, naming the group and the missing sibling. *Mutation:* ``atomic: false``
and both refusals disappear — the flag is what bites. The token frame here is
synthetic (one token per word, offsets by hand), because the refusal is
arithmetic over char spans and token indices; the same refusal on the real
tokenizer and a real engine is ``tests/neural/engines/pytorch_hooks/
test_edit_groups_run.py``.

**T8** — the six pair-validity requirements as six independent assertions,
each with its own negative fixture. "Absence of unintended edits" is the one
that needs the real tokenizer (the byte-level BPE fixture, no model, no
torch): ``tomorrow → tomorrew`` re-merges ``row`` into ``re w``, so the span
that declared the one changed character does not contain the changed token,
and the same edit with the span widened to the word passes.

Plus the column's shape refusals (the part of rule 27 checked at ``validate``)
and ``GRADES`` as the 1:1 relabelling of ``ScoringSpec.grade``.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalab.causal.pair_validation import (
    EDIT_GROUPS_COLUMN,
    GRADE_VALUES,
    GRADES,
    SIDES,
    EditGroup,
    EditGroupError,
    check_answer_change,
    check_component_wise,
    check_correctness,
    check_intended_token_change,
    check_location_coverage,
    check_no_unintended_edits,
    check_tokenizer_stability,
    grade_name,
    grade_value,
    parse_edit_groups,
    row_texts,
    token_edits,
    token_runs,
)
from causalab.causal.scoring import ScoringSpec
from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# fixtures — three counterfactual kinds, as rows
# --------------------------------------------------------------------------- #


def _span(text: str, needle: str, *, occurrence: int = 0) -> list[int]:
    """The char span of the ``occurrence``-th ``needle`` in ``text``."""
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    return [start, start + len(needle)]


def _row(
    base: str,
    counterfactual: str,
    *,
    base_answer: str,
    cf_answer: str,
    groups: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "input": base,
        "counterfactual_inputs": [counterfactual],
        "base_answer": base_answer,
        "cf_answer": cf_answer,
        "label": cf_answer,
        "split": "all",
    }
    if groups is not None:
        row[EDIT_GROUPS_COLUMN] = groups
    return row


def _group(
    name: str,
    atomic: bool,
    base: str,
    counterfactual: str,
    constituents: list[tuple[str, str]],
) -> dict[str, Any]:
    """A group over ``constituents`` — ``(base needle, counterfactual needle)``
    pairs, each located once in its text."""
    return {
        "name": name,
        "atomic": atomic,
        "spans": {
            "base": [_span(base, b) for b, _ in constituents],
            "counterfactual": [_span(counterfactual, c) for _, c in constituents],
        },
    }


#: (i) the single-numeral edit — one constituent, nothing coordinated
NUMERAL_BASE = "Q: what is 12 plus 30? A:"
NUMERAL_CF = "Q: what is 13 plus 30? A:"
NUMERAL = _row(
    NUMERAL_BASE,
    NUMERAL_CF,
    base_answer=" 42",
    cf_answer=" 43",
    groups=[_group("numeral", False, NUMERAL_BASE, NUMERAL_CF, [("12", "13")])],
)

#: (ii) the relation word and the total it changes — one coordinated edit
RELATION_BASE = "Q: 2 plus 3 = 5. True or false? A:"
RELATION_CF = "Q: 2 minus 3 = -1. True or false? A:"


def relation_row(atomic: bool = True) -> dict[str, Any]:
    return _row(
        RELATION_BASE,
        RELATION_CF,
        base_answer=" True",
        cf_answer=" True",
        groups=[
            _group(
                "relation_total",
                atomic,
                RELATION_BASE,
                RELATION_CF,
                [("plus", "minus"), ("5.", "-1.")],
            )
        ],
    )


#: (iii) the two-mapping-entry swap — two constituents that are one edit
SWAP_BASE = "a maps to 1, b maps to 2. a maps to"
SWAP_CF = "a maps to 2, b maps to 1. a maps to"


def swap_row(atomic: bool = True) -> dict[str, Any]:
    return _row(
        SWAP_BASE,
        SWAP_CF,
        base_answer=" 1",
        cf_answer=" 2",
        groups=[
            {
                "name": "mapping_swap",
                "atomic": atomic,
                "spans": {
                    "base": [_span(SWAP_BASE, "1"), _span(SWAP_BASE, "2")],
                    "counterfactual": [_span(SWAP_CF, "2"), _span(SWAP_CF, "1")],
                },
            }
        ],
    )


def word_offsets(text: str) -> list[tuple[int, int]]:
    """A synthetic token frame: one token per whitespace-separated word, the
    space owned by the word after it (the byte-level BPE convention), plus a
    leading ``(0, 0)`` special that must never match a span."""
    offsets: list[tuple[int, int]] = [(0, 0)]
    start = 0
    for index, char in enumerate(text):
        if char == " " and index > start:
            offsets.append((start, index))
            start = index
    if start < len(text):
        offsets.append((start, len(text)))
    return offsets


def _tokens_of(text: str, needle: str) -> set[int]:
    """The synthetic-frame token indices a ``variable`` anchor over ``needle``
    would address."""
    start, end = _span(text, needle)
    return {
        i
        for i, (a, b) in enumerate(word_offsets(text))
        if not (a == 0 and b == 0) and a < end and b > start
    }


# --------------------------------------------------------------------------- #
# T7 — the coordinated-edit-group refusal, and its mutation
# --------------------------------------------------------------------------- #


def test_t7_the_single_numeral_edit_validates_standalone() -> None:
    (group,) = parse_edit_groups(NUMERAL)
    assert group.name == "numeral" and not group.atomic and group.constituents == 1
    offsets = word_offsets(NUMERAL_BASE)
    # addressing the one constituent alone is the whole group
    check_component_wise(
        [group], offsets, _tokens_of(NUMERAL_BASE, "12"), side="base", where="im"
    )
    assert (
        check_location_coverage(
            [group], offsets, _tokens_of(NUMERAL_BASE, "12"), side="base", where="im"
        )
        == 1
    )


@pytest.mark.parametrize(
    ("row", "text", "alone", "sibling"),
    [
        (relation_row(), RELATION_BASE, "plus", "5."),
        (relation_row(), RELATION_BASE, "5.", "plus"),
        (swap_row(), SWAP_BASE, "1", "2"),
        (swap_row(), SWAP_BASE, "2", "1"),
    ],
    ids=["relation-alone", "total-alone", "first-entry-alone", "second-entry-alone"],
)
def test_t7_a_constituent_addressed_alone_is_refused_naming_the_group_and_sibling(
    row: dict[str, Any], text: str, alone: str, sibling: str
) -> None:
    """The relation-word-plus-total edit and the two-mapping-entry swap: an
    intervention that addresses one constituent of the atomic group and not
    the other is refused, and the message names the group, the constituent it
    touched and the sibling it left out."""
    (group,) = parse_edit_groups(row)
    assert group.atomic and group.constituents == 2
    with pytest.raises(EditGroupError) as err:
        check_component_wise(
            [group],
            word_offsets(text),
            _tokens_of(text, alone),
            side="base",
            where="intervened_models.patched, row 0",
            text=text,
        )
    message = str(err.value)
    assert group.name in message
    assert "atomic" in message and "sibling" in message
    assert repr(alone) in message and repr(sibling) in message
    assert "intervened_models.patched, row 0" in message


@pytest.mark.parametrize(
    ("row", "text", "constituents"),
    [
        (relation_row(), RELATION_BASE, ["plus", "5."]),
        (swap_row(), SWAP_BASE, ["1", "2"]),
    ],
    ids=["relation-total", "mapping-swap"],
)
def test_t7_twin_the_whole_group_addressed_runs(
    row: dict[str, Any], text: str, constituents: list[str]
) -> None:
    (group,) = parse_edit_groups(row)
    addressed = set().union(*(_tokens_of(text, c) for c in constituents))
    check_component_wise(
        [group], word_offsets(text), addressed, side="base", where="im"
    )
    # and so does an intervention that touches none of it
    check_component_wise([group], word_offsets(text), {1}, side="base", where="im")


@pytest.mark.parametrize(
    ("make_row", "text", "alone"),
    [(relation_row, RELATION_BASE, "plus"), (swap_row, SWAP_BASE, "1")],
    ids=["relation-total", "mapping-swap"],
)
def test_t7_mutation_atomic_false_makes_the_refusals_disappear(
    make_row: Any, text: str, alone: str
) -> None:
    """The flag is what bites: the same spans, the same lone address, declared
    ``atomic: false`` — no refusal. A declared-but-not-coordinated group is a
    declaration only."""
    (group,) = parse_edit_groups(make_row(atomic=False))
    assert not group.atomic
    check_component_wise(
        [group], word_offsets(text), _tokens_of(text, alone), side="base", where="im"
    )


def test_t7_the_counterfactual_side_is_checked_by_its_own_spans() -> None:
    """Constituent ``k`` is ``spans.base[k]`` paired with
    ``spans.counterfactual[k]``: on the swap, constituent 0 is ``1`` on base
    and ``2`` on the counterfactual — addressing the counterfactual's ``2``
    alone is the same lone constituent, refused the same way."""
    (group,) = parse_edit_groups(swap_row())
    with pytest.raises(EditGroupError, match="mapping_swap"):
        check_component_wise(
            [group],
            word_offsets(SWAP_CF),
            _tokens_of(SWAP_CF, "2"),
            side="counterfactual",
            where="im",
        )


def test_t7_a_partially_addressed_constituent_is_not_whole() -> None:
    """A constituent whose run is two tokens, addressed at one of them, is
    touched but not whole — refused, and the message says which."""
    base = "the total is forty two dollars"
    cf = "the total is fifty three dollars"
    row = _row(
        base,
        cf,
        base_answer="42",
        cf_answer="53",
        groups=[
            _group(
                "amount",
                True,
                base,
                cf,
                [("forty two", "fifty three"), ("dollars", "dollars")],
            )
        ],
    )
    (group,) = parse_edit_groups(row)
    offsets = word_offsets(base)
    with pytest.raises(EditGroupError) as err:
        check_component_wise(
            [group],
            offsets,
            _tokens_of(base, "forty") | _tokens_of(base, "dollars"),
            side="base",
            where="im",
            text=base,
        )
    assert "whole sibling" in str(err.value)


# --------------------------------------------------------------------------- #
# T8 — six requirements, six assertions, six negative fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bpe_tokenizer():
    """The byte-level BPE fixture's tokenizer alone — no model, no torch
    (the shape ``tests/protocol/test_alignment.py`` uses)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)


def test_t8_1_answer_change() -> None:
    check_answer_change(NUMERAL)
    same = {**NUMERAL, "cf_answer": NUMERAL["base_answer"]}
    with pytest.raises(EditGroupError, match="answer change"):
        check_answer_change(same)  # a noise-floor pair: nothing the task grades moved
    with pytest.raises(EditGroupError, match="no base_answer"):
        check_answer_change({"input": "x", "counterfactual_inputs": ["y"]})


TRUE_FALSE = ScoringSpec(
    forms={"answer": {"True": [" True", "True"], "False": [" False", "False"]}},
    invalid_output="unscored",
)


def test_t8_2_correctness_is_the_grade_on_both_sides() -> None:
    row = _row(
        RELATION_BASE, RELATION_CF, base_answer=" True", cf_answer=" False", groups=None
    )
    grades = check_correctness(
        TRUE_FALSE, {"base": " True", "counterfactual": "False"}, row
    )
    assert grades == {"base": "correct", "counterfactual": "correct"}
    # a wrong answer on one side is `incorrect`, named with its side
    with pytest.raises(EditGroupError) as err:
        check_correctness(TRUE_FALSE, {"base": " True", "counterfactual": " True"}, row)
    assert "counterfactual side is graded 'incorrect'" in str(err.value)
    # an off-space string under `invalid_output: unscored` is `unscored`, not 0
    with pytest.raises(EditGroupError) as err:
        check_correctness(
            TRUE_FALSE, {"base": "maybe", "counterfactual": " False"}, row
        )
    assert "base side is graded 'unscored'" in str(err.value)
    with pytest.raises(EditGroupError, match="no generated string for side"):
        check_correctness(TRUE_FALSE, {"base": " True"}, row)


def test_t8_3_intended_token_change(bpe_tokenizer) -> None:
    edits = check_intended_token_change(bpe_tokenizer, relation_row())
    assert edits and all(edit.op != "equal" for edit in edits)
    # negative (a): the prompts tokenize identically — there is no edit
    same = _row(
        RELATION_BASE, RELATION_BASE, base_answer="a", cf_answer="b", groups=None
    )
    with pytest.raises(EditGroupError, match="tokenize identically"):
        check_intended_token_change(bpe_tokenizer, same)
    # negative (b): a declared constituent the texts do not change — the
    # declaration claims an edit the pair does not carry
    undeclared_edit = _row(
        RELATION_BASE,
        RELATION_CF,
        base_answer=" True",
        cf_answer=" True",
        groups=[
            _group(
                "wrong_span",
                True,
                RELATION_BASE,
                RELATION_CF,
                [("plus", "minus"), ("True", "True")],
            )
        ],
    )
    with pytest.raises(EditGroupError) as err:
        check_intended_token_change(bpe_tokenizer, undeclared_edit)
    assert "constituent 1" in str(err.value) and "'True'" in str(err.value)


TOMORROW = "If today is Thursday, tomorrow is"
TOMORREW = "If today is Thursday, tomorrew is"


def test_t8_4_absence_of_unintended_edits_on_the_real_tokenizer(bpe_tokenizer) -> None:
    """The one requirement that needs the tokenizer: a one-character edit
    inside ``tomorrow`` re-merges its neighbour — ``row`` becomes ``re`` +
    ``w`` — so the changed token reaches outside the one-character span that
    declared the edit. That is an unintended edit; the same edit with the span
    widened to the word is not."""
    # the premise, pinned: the changed token on base is wider than the edit
    (edit,) = token_edits(bpe_tokenizer, TOMORROW, TOMORREW)
    changed_char = _span(
        TOMORROW, "o", occurrence=TOMORROW.count("o") - 1
    )  # the 'o' of 'row'
    assert TOMORROW[changed_char[0]] == "o" and TOMORREW[changed_char[0]] == "e"
    assert any(a < changed_char[0] or b > changed_char[1] for a, b in edit.base)
    narrow = _row(
        TOMORROW,
        TOMORREW,
        base_answer=" Friday",
        cf_answer=" Friday",
        groups=[
            {
                "name": "typo",
                "atomic": False,
                "spans": {"base": [changed_char], "counterfactual": [changed_char]},
            }
        ],
    )
    with pytest.raises(EditGroupError) as err:
        check_no_unintended_edits(bpe_tokenizer, narrow)
    message = str(err.value)
    assert "unintended edit" in message and "'row'" in message
    widened = _row(
        TOMORROW,
        TOMORREW,
        base_answer=" Friday",
        cf_answer=" Friday",
        groups=[_group("word", False, TOMORROW, TOMORREW, [("tomorrow", "tomorrew")])],
    )
    edits = check_no_unintended_edits(bpe_tokenizer, widened)
    assert edits == (edit,)
    # independent of requirement 3: the widened declaration passes both
    check_intended_token_change(bpe_tokenizer, widened)


def test_t8_4_an_undeclared_second_edit_is_an_unintended_edit(bpe_tokenizer) -> None:
    """The plain case: the pair changes the total too, the group declares the
    relation word only."""
    row = _row(
        RELATION_BASE,
        RELATION_CF,
        base_answer=" True",
        cf_answer=" True",
        groups=[
            _group("relation", False, RELATION_BASE, RELATION_CF, [("plus", "minus")])
        ],
    )
    with pytest.raises(EditGroupError, match="unintended edit"):
        check_no_unintended_edits(bpe_tokenizer, row)
    check_no_unintended_edits(bpe_tokenizer, relation_row())  # both declared: clean


def test_t8_4_a_recomputed_prefix_is_not_an_unintended_edit(bpe_tokenizer) -> None:
    """A second class of edit error, read the right way: the teacher-forced
    answer prefix the task recomputes per input changes the full context, and
    is not an edit of the prompt the groups describe."""
    row = _row(
        RELATION_BASE,
        RELATION_CF,
        base_answer=" True",
        cf_answer=" True",
        groups=relation_row()[EDIT_GROUPS_COLUMN],
    )
    check_no_unintended_edits(
        bpe_tokenizer, row, base_prefix=" True", counterfactual_prefix=" False"
    )


def test_t8_5_tokenizer_stability() -> None:
    recorded = {"key": "hf-internal-testing/tiny-random-gpt2", "revision": "main"}
    assert check_tokenizer_stability(recorded, dict(recorded)) == ("key", "revision")
    with pytest.raises(EditGroupError, match="tokenizer.revision = 'v2'"):
        check_tokenizer_stability({**recorded, "revision": "v2"}, recorded)
    with pytest.raises(EditGroupError, match="tokenizer.key"):
        check_tokenizer_stability({**recorded, "key": "gpt2"}, recorded)
    with pytest.raises(EditGroupError, match="records no tokenizer"):
        check_tokenizer_stability(None, recorded)


def test_t8_5_compares_key_and_revision_and_nothing_else() -> None:
    """The two fields a tokenizer is named by (a document's ``model.key`` /
    ``model.revision``): the pure function compares exactly those, so a
    record kept anywhere — a workflow's description, a README — is checked
    the same way."""
    recorded = {"key": "k", "revision": "main", "extra": "ignored"}
    assert check_tokenizer_stability(recorded, recorded) == ("key", "revision")


def test_t8_6_full_set_location_coverage() -> None:
    (group,) = parse_edit_groups(relation_row())
    offsets = word_offsets(RELATION_BASE)
    both = _tokens_of(RELATION_BASE, "plus") | _tokens_of(RELATION_BASE, "5.")
    assert check_location_coverage([group], offsets, both, side="base", where="im") == 2
    # negative: one constituent of one example is not addressed
    with pytest.raises(EditGroupError) as err:
        check_location_coverage(
            [group],
            offsets,
            _tokens_of(RELATION_BASE, "plus"),
            side="base",
            where="im",
            text=RELATION_BASE,
        )
    assert "constituent 1 ('5.')" in str(err.value) and "not fully addressed" in str(
        err.value
    )
    # a constituent that resolves to no token cannot be covered
    empty = EditGroup("empty", False, {"base": ((0, 0),), "counterfactual": ((0, 0),)})
    with pytest.raises(EditGroupError, match="resolves to no token"):
        check_location_coverage([empty], offsets, both, side="base", where="im")


def test_t8_the_six_are_independent() -> None:
    """Each requirement fails on its own fixture and passes on the others':
    the six functions share no verdict."""
    equal_answers = {**relation_row(), "cf_answer": relation_row()["base_answer"]}
    with pytest.raises(EditGroupError):
        check_answer_change(equal_answers)
    (group,) = parse_edit_groups(equal_answers)  # the declaration is still fine
    check_component_wise(
        [group],
        word_offsets(RELATION_BASE),
        _tokens_of(RELATION_BASE, "plus") | _tokens_of(RELATION_BASE, "5."),
        side="base",
        where="im",
    )


# --------------------------------------------------------------------------- #
# the column's shape — the part of rule 27 checked at validate
# --------------------------------------------------------------------------- #


def test_no_column_or_null_declares_nothing() -> None:
    row = _row("a b", "a c", base_answer="1", cf_answer="2", groups=None)
    assert parse_edit_groups(row) == ()
    assert parse_edit_groups({**row, EDIT_GROUPS_COLUMN: None}) == ()
    assert parse_edit_groups({**row, EDIT_GROUPS_COLUMN: []}) == ()


def test_row_texts_are_input_and_the_one_counterfactual() -> None:
    assert row_texts(NUMERAL) == {"base": NUMERAL_BASE, "counterfactual": NUMERAL_CF}
    with pytest.raises(EditGroupError, match="one-element list"):
        row_texts({"input": "a", "counterfactual_inputs": ["b", "c"]})
    with pytest.raises(EditGroupError, match="'input' column"):
        row_texts({"counterfactual_inputs": ["b"]})


def test_the_row_value_round_trips() -> None:
    groups = parse_edit_groups(relation_row())
    assert [g.as_row_value() for g in groups] == relation_row()[EDIT_GROUPS_COLUMN]
    assert set(groups[0].spans) == set(SIDES)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda g: {**g, "spans": {**g["spans"], "base": [[0, 99], [14, 16]]}},
            "not inside",
        ),
        (lambda g: {**g, "spans": {**g["spans"], "base": [[5, 9]]}}, "same number"),
        (lambda g: {**g, "spans": {"base": g["spans"]["base"]}}, "keyed by exactly"),
        (lambda g: {**g, "atomic": "yes"}, "true or false"),
        (lambda g: {**g, "name": ""}, "non-empty string"),
        (lambda g: {**g, "extra": 1}, "exactly"),
        (
            lambda g: {**g, "spans": {**g["spans"], "base": [[9, 5], [14, 16]]}},
            "not inside",
        ),
        (
            lambda g: {**g, "spans": {**g["spans"], "base": [[5, 9], [7, 16]]}},
            "overlaps",
        ),
        (
            lambda g: {**g, "spans": {**g["spans"], "base": [[5, "9"], [14, 16]]}},
            "integer",
        ),
        (
            lambda g: {**g, "spans": {"base": [], "counterfactual": []}},
            "no constituent",
        ),
    ],
    ids=[
        "span-outside-text",
        "lopsided-sides",
        "missing-side",
        "atomic-not-bool",
        "empty-name",
        "extra-key",
        "start-after-end",
        "overlapping-spans",
        "non-integer",
        "no-constituent",
    ],
)
def test_a_malformed_group_is_refused(mutate: Any, match: str) -> None:
    row = relation_row()
    row[EDIT_GROUPS_COLUMN] = [mutate(row[EDIT_GROUPS_COLUMN][0])]
    with pytest.raises(EditGroupError, match=match):
        parse_edit_groups(row)


def test_an_atomic_group_of_one_is_refused_and_its_non_atomic_twin_loads() -> None:
    row = relation_row()
    row[EDIT_GROUPS_COLUMN][0]["spans"] = {
        "base": [[5, 9]],
        "counterfactual": [[5, 10]],
    }
    with pytest.raises(EditGroupError, match="atomic with one constituent"):
        parse_edit_groups(row)
    row[EDIT_GROUPS_COLUMN][0]["atomic"] = False
    (group,) = parse_edit_groups(row)
    assert group.constituents == 1


def test_two_groups_may_not_share_a_name() -> None:
    row = relation_row()
    row[EDIT_GROUPS_COLUMN] = row[EDIT_GROUPS_COLUMN] * 2
    with pytest.raises(EditGroupError, match="declared twice"):
        parse_edit_groups(row)
    with pytest.raises(EditGroupError, match="must be a list"):
        parse_edit_groups({**row, EDIT_GROUPS_COLUMN: {"name": "x"}})


def test_token_runs_read_offsets_the_way_variable_positions_do() -> None:
    """``(0, 0)`` entries never match; a token overlapping the span by one
    char is in the run — the ``_chars_to_tokens`` reading, so a declared span
    and a ``variable`` anchor over the same characters name the same tokens."""
    (group,) = parse_edit_groups(relation_row())
    offsets = [
        (0, 0),
        (0, 2),
        (2, 4),
        (4, 7),
        (7, 9),
        (9, 11),
        (11, 13),
        (13, 15),
        (15, 16),
    ]
    runs = token_runs(group, "base", offsets)
    assert runs == ((3, 4), (7, 8))  # 'plus' = ' pl' + 'us'; '5.' = ' 5' + '.'


# --------------------------------------------------------------------------- #
# grade — the per-example status (sec. 2.10), 1:1 onto ScoringSpec.grade
# --------------------------------------------------------------------------- #


def test_grades_are_a_bijection_onto_scoring_spec_grade_values() -> None:
    assert GRADES == ("correct", "incorrect", "unscored")
    assert list(GRADE_VALUES) == list(GRADES)
    assert [grade_name(GRADE_VALUES[name]) for name in GRADES] == list(GRADES)
    assert [grade_value(name) for name in GRADES] == [1.0, 0.0, None]
    spec = ScoringSpec(
        forms={"total": {"85": [" 85", "85"]}}, invalid_output="unscored"
    )
    assert grade_name(spec.grade(" 85", "85")) == "correct"
    assert grade_name(spec.grade(" 87", "85")) == "unscored"  # names no declared value
    strict = ScoringSpec(forms={"total": {"85": [" 85", "85"]}})
    assert grade_name(strict.grade(" 87", "85")) == "incorrect"


def test_a_value_that_is_not_a_grade_is_refused_not_rounded() -> None:
    with pytest.raises(EditGroupError, match="not a grade"):
        grade_name(0.5)
    with pytest.raises(EditGroupError, match="not one of"):
        grade_value("matched")  # grade is not matched
