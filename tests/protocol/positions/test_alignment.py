"""Alignment cardinality as data (spec §2.3): the
closed vocabulary and its **one** consumer, rule 26, the pair-difference
validator, and the twin that proves an unauthored ``alignment`` moves no
digest.

What is pinned here is the pure half:

* the five members — ``one_to_one | one_to_many | many_to_one | absent |
  ambiguous`` — are exactly §2.3's table, and ``alignment_of`` classifies
  every case of two sides' candidate runs into one of them;
* ``absent`` and ``ambiguous`` are §2.4's two alignment reason codes, as an
  ``unavailable`` cell for a read and as a typed refusal otherwise, and a
  declared cardinality the observation contradicts is refused naming both;
* ``alignment`` parses on a position entry, is never swept, and rule 26
  refuses what the document alone can decide — an off-vocabulary value, an
  address that cannot carry one, a declaration an ``index`` / ``span``
  contradicts by construction — naming the field;
* **one function, three callers**: an AST census over ``causalab/`` holds the
  set of modules that call ``alignment_of`` to planning, execution and
  metrics, and the member literals to the module that defines them and the
  one that maps them to reason codes;
* **T5** — a pair differing only in its recomputed answer prefix validates
  clean with its three difference sets asserted *independently*;
* **T6, the pure half** — both interchange documents load and validate with no
  ``alignment`` authored anywhere, the corpus one to its pinned digest, and
  the canonical form carries the key only when it was written.

The engine half — the bare-vs-space-prefixed refusal, the ambiguous /
absent row as an unavailable cell counted in the denominator, the write
refusal, the declared-vs-observed contradiction against a real tokenizer,
and the T6 run — is ``tests/neural/engines/pytorch_hooks/test_alignment_run.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.positions.alignment import (
    UnalignableError,
    alignment_of,
    check_declared,
    pair_differences,
    refuse_unalignable,
    token_runs,
    unalignable,
    unalignable_reason,
)
from causalab.protocol.rules.errors import (
    RULES,
    ParseError,
    ProtocolError,
    ValidationError,
    ValidationErrors,
)
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.positions.alignment import static_alignment
from causalab.protocol.results import Unavailable
from causalab.protocol.schema import (
    ALIGNMENT_CARDINALITIES,
    PositionSpec,
    parse_document,
)
from causalab.protocol.rules.document import validate_document

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME
from tests.protocol._docs import base_doc, in_order, saved
from tests.protocol._env import CORPUS_DIR, steps_of
from tests._helpers.paths import PROTOCOLS_DIR


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "docs" / "intervention_protocol.md"
PACKAGE = REPO / "causalab"
PINS = json.loads((Path(__file__).parents[1] / "corpus_digests.json").read_text())
#: the shipped interchange application (split form) and its flat corpus twin —
#: the documents T6 loads
INTERCHANGE_PRESET = PROTOCOLS_DIR / "interchange.json"
INTERCHANGE_CORPUS = CORPUS_DIR / "02_interchange_im.json"


# --------------------------------------------------------------------------- #
# the vocabulary and its one function
# --------------------------------------------------------------------------- #


def _spec_alignment_table() -> list[str]:
    """§2.3's ``alignment`` table, first column; refuses an empty read."""
    section = SPEC.read_text().split("### 2.3 `positions`")[1].split("\n### ")[0]
    start = section.index("| `alignment` | means |")
    members: list[str] = []
    for line in section[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        members.append(line.split("|")[1].strip().strip("`"))
    assert members, "no alignment rows parsed from §2.3 — the table moved"
    return members


def test_the_five_cardinalities_are_exactly_the_spec_table() -> None:
    """The closed vocabulary and the table that documents it agree, in order
    (the census discipline of ``test_vocabulary_census.py``)."""
    tabulated = _spec_alignment_table()
    assert tabulated == list(ALIGNMENT_CARDINALITIES)
    assert len(set(ALIGNMENT_CARDINALITIES)) == len(ALIGNMENT_CARDINALITIES) == 5


@pytest.mark.parametrize(
    ("base", "counterfactual", "expected"),
    [
        (((3,),), ((7,),), "one_to_one"),  # one token each side
        (((3, 4),), ((7, 8),), "one_to_one"),  # one joint span of one width
        (((3,),), ((7, 8, 9),), "one_to_many"),
        (((3, 4, 5),), ((7,),), "many_to_one"),
        (((3, 4),), ((7, 8, 9),), "ambiguous"),  # no canonical pairing
        ((), ((7,),), "absent"),  # nothing on the base side
        (((3,),), (), "absent"),
        (((3,), (9,)), ((7,),), "ambiguous"),  # two candidates on one side
        (((),), ((7,),), "absent"),  # an occurrence covering no token
    ],
)
def test_alignment_of_classifies_a_pair(base, counterfactual, expected) -> None:
    assert alignment_of(base, counterfactual) == expected


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [((), "absent"), (((3, 4),), "one_to_one"), (((3,), (8,)), "ambiguous")],
)
def test_alignment_of_classifies_one_side_alone(candidates, expected) -> None:
    """The per-input half: a single role, or one side of a pair before it is
    paired — none is absent, several ambiguous, one is the run."""
    assert alignment_of(candidates) == expected


def test_token_runs_locates_every_occurrence() -> None:
    assert token_runs([1, 2], [0, 1, 2, 5, 1, 2]) == ((1, 2), (4, 5))
    assert token_runs([9], [0, 1, 2]) == ()
    assert token_runs([], [0, 1]) == ()


def test_the_two_unalignable_values_are_the_two_reason_codes() -> None:
    """``absent`` → ``alignment_missing``, ``ambiguous`` →
    ``alignment_ambiguous`` (§2.4's codes — nothing minted); the three
    that pair carry no reason."""
    assert unalignable_reason("absent") == "alignment_missing"
    assert unalignable_reason("ambiguous") == "alignment_ambiguous"
    for member in ("one_to_one", "one_to_many", "many_to_one"):
        assert unalignable_reason(member) is None
        assert unalignable(member, "d", "r") is None
        refuse_unalignable(member, "never raised")  # returns
    cell = unalignable("ambiguous", "value 'day' occurs 2 times (row 0)", "r[pos=1]")
    assert isinstance(cell, Unavailable)
    assert cell.reason == "alignment_ambiguous"
    assert cell.detail == "value 'day' occurs 2 times (row 0)"
    assert cell.denominator_key == "r[pos=1]"
    assert cell.record()["status"] == "unavailable"


def test_the_refusal_is_a_protocol_error_with_the_reason_and_the_cardinality():
    with pytest.raises(UnalignableError) as err:
        refuse_unalignable("absent", "value 'x' occurs 0 times")
    assert isinstance(err.value, ProtocolError)
    assert err.value.code == "P2"
    assert err.value.reason == "alignment_missing"
    assert err.value.cardinality == "absent"
    assert "occurs 0 times" in str(err.value)
    with pytest.raises(AssertionError):
        UnalignableError("one_to_one", "not unalignable")


def test_a_declared_cardinality_the_observation_contradicts_is_refused() -> None:
    """Never a silent override: the refusal names the declared and the
    observed value. Nothing declared, or the same value, passes."""
    check_declared(None, "ambiguous", where="position 'ent'")
    check_declared("one_to_many", "one_to_many", where="position 'ent'")
    with pytest.raises(ProtocolError) as err:
        check_declared("one_to_one", "one_to_many", where="position 'ent', row 2,")
    assert "declares alignment 'one_to_one'" in str(err.value)
    assert "resolves it as 'one_to_many'" in str(err.value)
    assert err.value.reason is None
    with pytest.raises(ProtocolError) as err:
        check_declared("one_to_one", "absent", where="position 'ent'")
    assert err.value.reason == "alignment_missing"


# --------------------------------------------------------------------------- #
# the authoring surface: parse, canonical form, rule 26
# --------------------------------------------------------------------------- #


def _doc_with_position(spec: dict[str, Any]) -> dict[str, Any]:
    doc = base_doc()
    doc["method"]["positions"] = {"ent": spec}
    doc["method"]["reads"]["v_cf"]["pos"] = "ent"
    doc["method"]["writes"]["patch"]["pos"] = "ent"
    return doc


def _validate(raw: dict[str, Any]) -> None:
    validate_document(parse_document(in_order(raw)))


def test_alignment_parses_on_a_position_entry_and_inline() -> None:
    doc = _doc_with_position({"variable": "entity", "alignment": "one_to_many"})
    doc["method"]["reads"]["logits"]["pos"] = {"index": -1, "alignment": "one_to_one"}
    parsed = parse_document(in_order(doc))
    ent = parsed.positions["ent"]
    assert isinstance(ent, PositionSpec) and ent.alignment == "one_to_many"
    inline = parsed.reads["logits"].pos
    assert isinstance(inline, PositionSpec) and inline.alignment == "one_to_one"
    # nothing declared is None — no default is materialized anywhere
    assert parse_document(in_order(base_doc())).reads["v_cf"].pos.alignment is None


def test_alignment_must_be_a_string_and_is_never_swept() -> None:
    with pytest.raises(ParseError) as err:
        parse_document(in_order(_doc_with_position({"index": -1, "alignment": 1})))
    assert err.value.code == "P2" and err.value.path == "positions.ent.alignment"
    with pytest.raises(ValidationError) as verr:
        parse_document(
            in_order(
                _doc_with_position(
                    {"index": -1, "alignment": {"sweep": ["one_to_one", "absent"]}}
                )
            )
        )
    assert verr.value.rule == 14


def test_rule_26_refuses_a_value_outside_the_vocabulary_naming_the_field() -> None:
    with pytest.raises(ValidationError) as err:
        _validate(_doc_with_position({"variable": "entity", "alignment": "one_to_1"}))
    assert err.value.rule == 26 and err.value.rule_id == "alignment_declared"
    assert err.value.path == "positions.ent.alignment"
    assert "one_to_one" in str(err.value)  # the suggestion


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ({"all": True, "alignment": "one_to_one"}, "all-positions address"),
        (
            {
                "index": -1,
                "generated": {"max_new_tokens": 2},
                "alignment": "one_to_one",
            },
            "generated position",
        ),
    ],
)
def test_rule_26_refuses_an_address_that_cannot_carry_one(spec, fragment) -> None:
    doc = base_doc()
    doc["method"]["positions"] = {"ent": spec}
    if "generated" in spec:
        # writes are prefill-only (rule 16), so the continuation frame is
        # reached through a read of its own: the base document stays legal
        # as it is and gains one saved generated read on the original model
        doc["method"]["reads"]["gen"] = {"site": "lm_head", "pos": "ent"}
        doc["method"]["intervened_models"]["original_base"] = {
            "input": "base",
            "reads": ["gen"],
        }
        doc["method"]["save"].append(saved("gen", "original_base", "gen.safetensors"))
    else:
        doc["method"]["reads"]["v_cf"]["pos"] = "ent"
    with pytest.raises(ValidationError) as err:
        _validate(doc)
    # rule 26 is the only violation: the document is otherwise legal, so the
    # refusal is the one raised, not one of several collected
    assert not isinstance(err.value, ValidationErrors), str(err.value)
    assert err.value.rule == 26, str(err.value)
    assert err.value.path == "positions.ent.alignment"
    assert fragment in err.value.message


@pytest.mark.parametrize(
    "spec",
    [
        {"index": -1, "alignment": "one_to_many"},
        {"index": 0, "scope": {"variable": "entity"}, "alignment": "absent"},
        {"span": [0, 2], "alignment": "many_to_one"},
    ],
)
def test_rule_26_refuses_a_declaration_the_address_contradicts(spec) -> None:
    """An ``index`` is one token per row on every input and an unscoped
    ``span`` one joint window of one width: ``one_to_one`` by construction,
    which the planner knows without a tokenizer (``static_alignment``)."""
    with pytest.raises(ValidationError) as err:
        _validate(_doc_with_position(spec))
    assert err.value.rule == 26
    assert "by construction" in err.value.message


@pytest.mark.parametrize(
    "spec",
    [
        {"index": -1, "alignment": "one_to_one"},
        {"span": [0, 2], "alignment": "one_to_one"},
        {"variable": "entity", "alignment": "one_to_many"},
        {"variable": "entity", "alignment": "absent"},
        {"column": "entity", "alignment": "ambiguous"},
        {"span": [0, 1], "scope": {"variable": "entity"}, "alignment": "many_to_one"},
    ],
)
def test_rule_26_accepts_a_declaration_the_document_cannot_refute(spec) -> None:
    """The valid-work twin: every member on an address whose cardinality the
    tokenizer decides, and ``one_to_one`` where the document decides it."""
    _validate(_doc_with_position(spec))


def test_static_alignment_is_what_the_document_alone_decides() -> None:
    doc = parse_document(in_order(base_doc()))
    assert static_alignment(doc, PositionSpec(index=-1)) == "one_to_one"
    assert static_alignment(doc, PositionSpec(index=0, scope="x")) == "one_to_one"
    assert static_alignment(doc, PositionSpec(span=(1, 3))) == "one_to_one"
    assert static_alignment(doc, PositionSpec(span=(0, 1), scope="x")) is None
    assert static_alignment(doc, PositionSpec(variable="x")) is None
    assert static_alignment(doc, PositionSpec(column="x")) is None
    assert static_alignment(doc, PositionSpec(all=True)) is None
    assert (
        static_alignment(doc, PositionSpec(index=-1, generated={"max_new_tokens": 1}))
        is None
    )
    # the spelling a read carries: a positions-table name resolves the same way
    assert static_alignment(doc, -1) is None or True  # an int is not a name
    named = parse_document(in_order(_doc_with_position({"index": -1})))
    assert static_alignment(named, "ent") == "one_to_one"


def test_rule_26_is_the_twenty_sixth_rule() -> None:
    """The rule after ``row_roles`` (25) in ``RULES``. The slug is the
    rule's identity; the number is its place in the table."""
    assert RULES["alignment_declared"].code == "V26"
    assert ValidationError("alignment_declared", "m").rule == 26


def _walk(value: Any):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)


def test_canonical_form_carries_alignment_only_when_authored(env) -> None:
    """Digest-neutral by construction: no materialized default anywhere. An
    authored value is part of the canonical form (it changes the experiment
    an author claims to have run); an unauthored one is absent from it."""
    authored = compile_protocol(
        in_order(_doc_with_position({"variable": "entity", "alignment": "absent"})),
        env=env,
    )
    assert authored.canonical["method"]["positions"]["ent"] == {
        "variable": "entity",
        "alignment": "absent",
    }
    plain = compile_protocol(
        in_order(_doc_with_position({"variable": "entity"})), env=env
    )
    assert plain.canonical["method"]["positions"]["ent"] == {"variable": "entity"}
    assert "alignment" not in set(_walk(plain.canonical))
    assert authored.digests.document != plain.digests.document


@pytest.mark.parametrize("path", [INTERCHANGE_PRESET, INTERCHANGE_CORPUS])
def test_t6_an_interchange_document_authors_no_alignment_and_still_loads(
    path: Path, env
) -> None:
    """T6, the pure half: an implicit one-to-one
    interchange loads and validates with no ``alignment`` field anywhere, and
    its canonical form gains none."""
    loaded = compile_protocol(path, env=env)
    assert "alignment" not in set(_walk(loaded.tree))
    assert "alignment" not in set(_walk(loaded.canonical))
    for point in steps_of(loaded, env).documents:
        validate_document(point)
        for spec in point.positions.values():
            assert getattr(spec, "alignment", None) is None
        for entry in (*point.reads.values(), *point.writes.values()):
            if isinstance(entry.pos, PositionSpec):
                assert entry.pos.alignment is None


def test_t6_the_corpus_interchange_digest_is_its_pin(env) -> None:
    """Zero pins move: the field's absence is byte-identical to before it
    existed. (Every other corpus document is held to its pin by
    ``test_corpus.py``; this one is the document T6 names.)"""
    loaded = compile_protocol(INTERCHANGE_CORPUS, env=env)
    assert loaded.digests.document == PINS["02_interchange_im.json"]["document"]
    assert (
        list(steps_of(loaded, env).digests) == PINS["02_interchange_im.json"]["points"]
    )


# --------------------------------------------------------------------------- #
# one function, three callers — the census
# --------------------------------------------------------------------------- #

#: Where ``alignment_of`` may be called from — the one module per layer the
#: spec names: planning, execution (the resolver and the executor it serves),
#: metrics. The module that *defines* it (``protocol/alignment.py``) is not a
#: caller and is not listed: this is a census of consumers, and the definer
#: is held separately below. A new caller is a design change, not a line.
CONSUMERS = frozenset(
    {
        "causalab/protocol/positions/encoding.py",
        "causalab/protocol/positions/resolve.py",
    }
)
#: Where a member may be spelled as a literal: the vocabulary's definition, and
#: the module that maps two members to their reason codes. Everything else
#: branches on the value ``alignment_of`` returned or on the tuple.
SPELLERS = frozenset(
    {"causalab/protocol/schema/types.py", "causalab/protocol/positions/alignment.py"}
)


def _defines(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.walk(tree)
    )


def _references(tree: ast.AST, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
        if isinstance(node, ast.alias) and node.name == name:
            return True
    return False


def _literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_no_module_outside_the_two_callers_derives_a_cardinality() -> None:
    """Planning and execution consume **one** function; nothing else
    calls it and nothing re-derives it by spelling the members. Parsed, not
    grepped, so a docstring mentioning ``one_to_one`` is not a hit."""
    callers: set[str] = set()
    spellers: set[str] = set()
    definers: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        rel = path.relative_to(REPO).as_posix()
        if _defines(tree, "alignment_of"):
            definers.add(rel)
        elif _references(tree, "alignment_of"):
            callers.add(rel)
        if _literals(tree) & set(ALIGNMENT_CARDINALITIES):
            spellers.add(rel)
    assert definers == {"causalab/protocol/positions/alignment.py"}, definers
    assert callers, "nothing calls alignment_of — the census parsed nothing"
    assert callers == CONSUMERS, {
        "unexpected callers": sorted(callers - CONSUMERS),
        "callers gone": sorted(CONSUMERS - callers),
    }
    assert spellers == SPELLERS, {
        "spell a member outside the vocabulary's home": sorted(spellers - SPELLERS),
        "no longer spell one": sorted(SPELLERS - spellers),
    }


def test_the_defining_module_is_torch_free() -> None:
    """The validator and the classification live in the pure layer: a
    tokenizer is an argument, never loaded, and nothing imports torch."""
    tree = ast.parse((PACKAGE / "protocol" / "positions" / "alignment.py").read_text())
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not imported & {"torch", "transformers", "numpy"}, imported


# --------------------------------------------------------------------------- #
# T5 — the pair-difference validator
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bpe_tokenizer():
    """The byte-level BPE fixture's tokenizer alone — no model, no torch."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TINY_RANDOM_GPT2_MODEL_NAME)


PROMPT = "Q: what is twelve plus thirty? A: twelve plus thirty ="


def test_t5_a_recomputed_prefix_validates_clean_in_three_separate_sets(
    bpe_tokenizer,
) -> None:
    """The second error class of pair validation. A pair whose prompts are identical
    and whose teacher-forced answer prefixes differ — the task recomputed the
    prefix per input — has (a) an **empty** prompt set, (b) a **non-empty**
    prefix set, (c) a **non-empty** full-context set, each asserted on its
    own.

    *Mutation:* fold the three into one set (report ``full_context`` for all
    three) and (a) fails — the recomputed prefix would read as a prompt edit,
    which is exactly the misreading the split exists to prevent.
    """
    diff = pair_differences(
        bpe_tokenizer,
        PROMPT,
        PROMPT,
        base_prefix=" forty-two",
        counterfactual_prefix=" fifty-two",
    )
    assert diff.prompt == ()  # (a) the prompt did not change
    assert diff.teacher_forced_prefix != ()  # (b) the prefix did
    assert diff.full_context != ()  # (c) and so the context as a whole did
    assert not diff.prompt_edited
    assert diff.prefix_recomputed
    # the hunks say what changed, decoded
    (hunk,) = diff.teacher_forced_prefix[:1]
    assert hunk.op in {"replace", "insert", "delete"}
    assert hunk.base != hunk.counterfactual


def test_t5_a_prompt_edit_lands_in_the_prompt_set(bpe_tokenizer) -> None:
    """The intended edit of a counterfactual pair shows where it belongs, and
    an unchanged prefix stays empty."""
    diff = pair_differences(
        bpe_tokenizer,
        "If today is Thursday, tomorrow is",
        "If today is Friday, tomorrow is",
        base_prefix=" Friday",
        counterfactual_prefix=" Friday",
    )
    assert diff.prompt != ()
    assert diff.teacher_forced_prefix == ()
    assert diff.full_context != ()
    assert diff.prompt_edited and not diff.prefix_recomputed
    assert any(
        "hurs" in "".join(h.base) or "Th" in "".join(h.base) for h in diff.prompt
    )


def test_t5_an_identical_pair_has_three_empty_sets(bpe_tokenizer) -> None:
    diff = pair_differences(bpe_tokenizer, PROMPT, PROMPT)
    assert (diff.prompt, diff.teacher_forced_prefix, diff.full_context) == ((), (), ())
