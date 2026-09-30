"""Semantic spans (spec §2.3) — the pure half.

* every span form parses where a position is accepted (``positions.<name>``
  and inline), is a ``PositionSpec``, and is refused on the shapes it cannot
  take (two selectors, ``atomic: false``, sugar inside a span, a member with
  its own ``alignment``, ``generated``);
* the span-key vocabulary is exactly §2.3's span table (the census);
* the algebra — ``indices``, ``union``, ``intersection``, ``before`` /
  ``after`` / ``between``, ``atomic`` — resolves as documented over a fake
  frame, and an empty selection is refused with reason ``empty_selector``;
* an ``atomic`` span is **one address**: rule 8 refuses a second absolute
  write overlapping it and accepts a provably disjoint one; rule 26 classifies
  a static atomic set ``one_to_one`` through ``plan.static_alignment`` (the
  planning caller of ``alignment_of`` — no second classifier);
* rule 27's span half: an atomic set the document fixes at one member is
  refused, its two-member twin accepted;
* a span canonicalizes verbatim, with no default materialized.

Every refusal has its valid-work twin beside it.
"""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.positions.alignment import alignment_of
from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.positions.alignment import static_alignment
from causalab.protocol.schema import PositionSpec, parse_document
from causalab.protocol.positions.spans import (
    COMPOSITE_KEYS,
    PREDICATE_KEYS,
    SPAN_KEYS,
    SpanSpec,
    constituents,
    resolve_span,
    segment_anchors,
    selector,
    static_indices,
    walk,
)
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import UNWRITTEN, base_doc, in_order


pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "intervention_protocol.md"


def _doc_with(position: Any, *, write_too: bool = True) -> dict[str, Any]:
    """The base document with ``positions.p`` = ``position``, read (and
    written) at ``p``."""
    doc = base_doc()
    doc["method"]["positions"] = {"p": position}
    doc["method"]["reads"]["v_cf"]["pos"] = "p"
    if write_too:
        doc["method"]["writes"]["patch"]["pos"] = "p"
    return in_order(doc)


def _parse(position: Any) -> PositionSpec:
    return parse_document(_doc_with(position)).positions["p"]


def _validate(raw: dict[str, Any]) -> None:
    validate_document(parse_document(in_order(raw)))


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

FORMS: list[tuple[dict[str, Any], str]] = [
    ({"indices": [0, 2]}, "indices"),
    ({"indices": [0, 1], "atomic": True}, "indices"),
    ({"indices": [0, -1], "scope": {"variable": "entity"}}, "indices"),
    ({"union": [{"index": 0}, {"variable": "entity"}]}, "union"),
    ({"intersection": [{"span": [0, 4]}, {"variable": "entity"}]}, "intersection"),
    ({"before": {"variable": "entity"}}, "before"),
    ({"after": {"column": "entity"}}, "after"),
    ({"between": [{"index": 0}, {"variable": "entity"}]}, "between"),
    ({"span": [0, 2], "atomic": True}, "span"),
    ({"variable": "entity", "atomic": True}, "variable"),
    ({"column": "entity", "atomic": True}, "column"),
    (
        {"union": [{"indices": [0, 1], "atomic": True}, {"after": {"index": 3}}]},
        "union",
    ),
]


@pytest.mark.parametrize(("raw", "which"), FORMS, ids=[f[1] for f in FORMS])
def test_every_span_form_parses_as_a_position(raw, which) -> None:
    spec = _parse(raw)
    assert isinstance(spec, SpanSpec) and isinstance(spec, PositionSpec)
    assert selector(spec) == which
    assert spec.atomic is bool(raw.get("atomic", False))
    assert spec.generated is None


def test_a_span_is_accepted_inline_on_a_read_and_a_write() -> None:
    doc = base_doc()
    doc["method"]["reads"]["v_cf"]["pos"] = {"indices": [0, 1], "atomic": True}
    doc["method"]["writes"]["patch"]["pos"] = {"indices": [0, 1], "atomic": True}
    parsed = parse_document(doc)
    assert isinstance(parsed.reads["v_cf"].pos, SpanSpec)
    assert isinstance(parsed.writes["patch"].pos, SpanSpec)
    validate_document(parsed)


REFUSED: list[tuple[dict[str, Any], str, str]] = [
    ({"indices": [0], "union": [{"index": 0}, {"index": 1}]}, "P2", "exactly one"),
    ({"indices": [0, 1], "atomic": False}, "P2", "no literal spelling"),
    ({"variable": "entity", "atomic": True, "atomicity": 1}, "P3", "unknown key"),
    ({"index": 0, "indices": [1, 2]}, "P2", "does not combine"),
    ({"all": True, "atomic": True}, "P2", "does not combine"),
    ({"indices": [0, 1], "generated": {"max_new_tokens": 2}}, "P2", "prompt frame"),
    ({"union": [-1, {"index": 0}]}, "P2", 'no int or "all" sugar'),
    ({"union": [{"index": 0}]}, "P2", "two or more"),
    ({"between": [{"index": 0}]}, "P2", "exactly 2"),
    (
        {"union": [{"index": 0, "alignment": "one_to_one"}, {"index": 1}]},
        "P2",
        "member carries no alignment",
    ),
    (
        {"before": {"index": 0, "generated": {"max_new_tokens": 1}}},
        "P2",
        "prompt frame",
    ),
    ({"indices": []}, "P2", "non-empty"),
    ({"indices": [1, 1]}, "P2", "repeats"),
    (
        {"union": [{"index": 0}, {"index": 1}], "scope": {"variable": "entity"}},
        "P2",
        "scope/relative_to modify",
    ),
    (
        {
            "indices": [0, 1],
            "scope": {"variable": "a"},
            "relative_to": {"variable": "b"},
        },
        "P2",
        "mutually exclusive",
    ),
    ({"segment": ""}, "P2", "declared segment"),
    ({"span": [0, 1, 2], "atomic": True}, "P2", "exactly two"),
]


@pytest.mark.parametrize(
    ("raw", "code", "fragment"), REFUSED, ids=[r[2][:24] for r in REFUSED]
)
def test_a_malformed_span_is_refused_at_parse(raw, code, fragment) -> None:
    with pytest.raises(ParseError) as err:
        _parse(raw)
    assert err.value.code == code
    assert fragment in str(err.value)


def test_a_bare_anchor_with_a_false_atomic_is_refused_not_promoted() -> None:
    """``{"variable": x, "atomic": false}`` is neither a plain position nor a
    span: the default has no spelling, so nothing is silently accepted."""
    with pytest.raises(ParseError, match="no literal spelling"):
        _parse({"variable": "entity", "atomic": False})


# --------------------------------------------------------------------------- #
# the census: §2.3's span table is exactly the code's vocabulary
# --------------------------------------------------------------------------- #


def _section(heading: str) -> str:
    depth = len(heading) - len(heading.lstrip("#"))
    body = SPEC.read_text().split(heading, 1)
    assert len(body) == 2, f"{heading!r} is not in the spec"
    end = re.compile(rf"^#{{1,{depth}}} ", re.M).search(body[1])
    return body[1][: end.start()] if end else body[1]


def _table(section: str, first_header: str) -> list[list[str]]:
    """Body rows of the one table in ``section`` whose first header cell is
    ``first_header`` — the table ends at the first non-table line, so a later
    table in the same section is never read as more rows."""
    lines = section.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("|")
            and line.strip().strip("|").split("|")[0].strip() == first_header
        ),
        None,
    )
    assert start is not None, f"no table headed {first_header!r}"
    body: list[list[str]] = []
    for line in lines[start + 1 :]:
        if not line.strip().startswith("|"):
            break
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if set(row[0]) <= set("-: "):
            continue
        body.append(row)
    assert body, f"the {first_header!r} table parsed to zero rows"
    return body


def test_the_span_table_is_exactly_the_span_keys() -> None:
    """Words-per-object: every span key has a row and every row is a key."""
    rows = _table(_section("### 2.3 `positions`"), "span key")
    tabulated = [row[0].strip("`") for row in rows]
    assert len(set(tabulated)) == len(tabulated), tabulated
    assert set(tabulated) == set(SPAN_KEYS), {
        "spec only": sorted(set(tabulated) - set(SPAN_KEYS)),
        "code only": sorted(set(SPAN_KEYS) - set(tabulated)),
    }
    assert set(COMPOSITE_KEYS) | set(PREDICATE_KEYS) < set(SPAN_KEYS)
    for row in rows:
        assert row[1].strip(), f"span key {row[0]} has no shape cell"


def test_the_span_module_is_torch_free_and_spells_no_cardinality() -> None:
    """The algebra is pure over a frame; it imports no torch and derives no
    cardinality of its own (the one-classifier rule: ``alignment_of`` alone
    classifies)."""
    source = (REPO / "causalab/protocol/positions/spans.py").read_text()
    tree = ast.parse(source)
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
    assert "torch" not in imported
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert (
        "alignment_of" not in names
    )  # the docstring may cite it; the code never calls it
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not literals & {
        "one_to_one",
        "one_to_many",
        "many_to_one",
        "absent",
        "ambiguous",
    }


# --------------------------------------------------------------------------- #
# what the document alone decides
# --------------------------------------------------------------------------- #


def test_static_indices_for_the_fixed_forms() -> None:
    assert static_indices(_parse({"indices": [3, 0, 2]})) == (0, 2, 3)
    assert static_indices(_parse({"span": [1, 4], "atomic": True})) == (1, 2, 3)
    assert static_indices(PositionSpec(index=-1)) == (-1,)
    assert static_indices(PositionSpec(span=(0, 2))) == (0, 1)
    union = _parse({"union": [{"indices": [0, 1]}, {"span": [1, 3], "atomic": True}]})
    assert static_indices(union) == (0, 1, 2)
    meet = _parse({"intersection": [{"indices": [0, 1, 2]}, {"indices": [1, 2, 5]}]})
    assert static_indices(meet) == (1, 2)


@pytest.mark.parametrize(
    "raw",
    [
        {"variable": "entity", "atomic": True},
        {"before": {"index": 0}},
        {"indices": [0, 1], "scope": {"variable": "entity"}},
        {"union": [{"indices": [0, 1]}, {"variable": "entity"}]},
        {"segment": "user"},
    ],
)
def test_static_indices_is_none_where_only_the_tokenizer_can_say(raw) -> None:
    assert static_indices(_parse(raw)) is None


def test_constituents_are_one_for_atomic_and_each_member_otherwise() -> None:
    atomic = _parse({"indices": [0, 1], "atomic": True})
    assert constituents(atomic) == (atomic,)
    loose = _parse({"indices": [0, 1], "scope": {"variable": "entity"}})
    parts = constituents(loose)
    assert [p.index for p in parts] == [0, 1]
    assert all(p.scope == "entity" and p.anchor_source == "variable" for p in parts)
    union = _parse({"union": [{"index": 0}, {"variable": "entity"}]})
    assert constituents(union) == tuple(union.union or ())
    assert constituents(PositionSpec(index=-1)) == (PositionSpec(index=-1),)


def test_walk_and_segment_anchors_reach_every_nested_spec() -> None:
    nested = parse_document(
        in_order(
            {
                **_doc_with(
                    {
                        "union": [
                            {"before": {"segment": "user"}},
                            {"index": -1, "scope": {"segment": "assistant_prefix"}},
                            {"between": [{"variable": "entity"}, {"segment": "ent"}]},
                        ]
                    }
                ),
                "segments": {"frame": "chat", "declare": {"ent": {"column": "entity"}}},
            }
        )
    ).positions["p"]
    assert len(list(walk(nested))) == 7  # union, before+anchor, scoped index, between+2
    assert [name for _, name in segment_anchors(nested)] == [
        "user",
        "assistant_prefix",
        "ent",
    ]


# --------------------------------------------------------------------------- #
# the algebra over a fake frame
# --------------------------------------------------------------------------- #

#: (first real token, content start, padded length): two prefix tokens at 2–3,
#: content at 4–9
FRAME = (2, 4, 10)
RUNS = {"x": [5, 6], "y": [8], "z": [4, 5]}


def _resolve(member: PositionSpec) -> list[int]:
    """A stand-in for the engine's resolver: the frame math for `index`,
    the table above for variables, recursion for nested spans."""
    if isinstance(member, SpanSpec):
        return resolve_span(member, frame=FRAME, resolve=_resolve, segment_run=_segment)
    if member.index is not None:
        n = int(member.index)
        if member.scope is not None:
            return [RUNS[str(member.scope)][n]]
        return [FRAME[2] + n if n < 0 else FRAME[1] + n]
    if member.variable is not None:
        return list(RUNS[str(member.variable)])
    if member.span is not None:
        lo, hi = member.span
        return list(range(FRAME[1] + lo, FRAME[1] + hi))
    raise AssertionError(member)


def _segment(name: str) -> list[int]:
    return {"user": [4, 5, 6, 7], "assistant_prefix": [8, 9]}[name]


def _algebra(raw: dict[str, Any]) -> list[int]:
    spec = _parse(raw)
    assert isinstance(spec, SpanSpec)
    return resolve_span(spec, frame=FRAME, resolve=_resolve, segment_run=_segment)


ALGEBRA: list[tuple[dict[str, Any], list[int]]] = [
    ({"indices": [0, 2]}, [4, 6]),
    ({"indices": [-1, 0]}, [4, 9]),
    ({"indices": [0, 1], "atomic": True}, [4, 5]),
    ({"indices": [0, -1], "scope": {"variable": "x"}}, [5, 6]),
    ({"union": [{"index": 0}, {"variable": "x"}]}, [4, 5, 6]),
    ({"union": [{"variable": "x"}, {"variable": "z"}]}, [4, 5, 6]),
    ({"intersection": [{"variable": "x"}, {"variable": "z"}]}, [5]),
    ({"before": {"variable": "x"}}, [2, 3, 4]),
    ({"after": {"variable": "x"}}, [7, 8, 9]),
    ({"between": [{"variable": "x"}, {"variable": "y"}]}, [7]),
    ({"between": [{"variable": "y"}, {"variable": "x"}]}, [7]),
    ({"variable": "x", "atomic": True}, [5, 6]),
    ({"span": [1, 3], "atomic": True}, [5, 6]),
    ({"segment": "user"}, [4, 5, 6, 7]),
    ({"before": {"segment": "user"}}, [2, 3]),
    (
        {"union": [{"segment": "assistant_prefix"}, {"before": {"segment": "user"}}]},
        [2, 3, 8, 9],
    ),
    (
        {
            "intersection": [
                {"after": {"index": 0}},
                {"before": {"segment": "assistant_prefix"}},
            ]
        },
        [5, 6, 7],
    ),
]


@pytest.mark.parametrize(("raw", "expected"), ALGEBRA, ids=[str(a[0]) for a in ALGEBRA])
def test_the_algebra_resolves_as_documented(raw, expected) -> None:
    """`before` ranges over the row's real tokens (the prefix included), so
    the mutation that swaps `before` and `after`, or that ranges `before` from
    the content start, fails the third and eighth rows respectively."""
    assert _algebra(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        {"intersection": [{"variable": "x"}, {"variable": "y"}]},
        {"between": [{"variable": "z"}, {"variable": "x"}]},  # runs touch
        {"before": {"index": -2}},  # nothing before the first real token? no:
    ],
)
def test_an_empty_selection_is_refused_with_empty_selector(raw) -> None:
    if raw == {"before": {"index": -2}}:
        # `index -2` is token 8; `before` it is non-empty — make the case empty
        raw = {"before": {"segment": "user"}}
        FRAME_EMPTY = (4, 4, 10)  # no prefix: nothing before the user turn
        spec = _parse(raw)
        assert isinstance(spec, SpanSpec)
        with pytest.raises(ProtocolError) as err:
            resolve_span(
                spec, frame=FRAME_EMPTY, resolve=_resolve, segment_run=_segment
            )
        assert err.value.reason == "empty_selector"
        return
    with pytest.raises(ProtocolError) as err:
        _algebra(raw)
    assert err.value.reason == "empty_selector"
    assert "no token" in str(err.value)


# --------------------------------------------------------------------------- #
# atomic = one address: rule 8, rule 26 (through the one classifier), rule 27
# --------------------------------------------------------------------------- #


def _two_absolute_writes(first: Any, second: Any) -> dict[str, Any]:
    doc = base_doc()
    doc["method"]["positions"] = {"a": first, "b": second}
    doc["method"]["reads"]["v_cf"]["pos"] = "a"
    doc["method"]["reads"]["v_cf2"] = {**doc["method"]["reads"]["v_cf"], "pos": "b"}
    doc["method"]["intervened_models"][UNWRITTEN]["reads"] = ["v_cf", "v_cf2"]
    doc["method"]["writes"]["patch"]["pos"] = "a"
    doc["method"]["writes"]["patch2"] = {
        "site": "tgt",
        "pos": "b",
        "do": {"swap": "v_cf2"},
    }
    doc["method"]["intervened_models"]["patched"]["writes"] = ["patch", "patch2"]
    return in_order(doc)


def test_rule_8_refuses_a_second_absolute_write_inside_an_atomic_span() -> None:
    with pytest.raises(ValidationError) as err:
        _validate(
            _two_absolute_writes({"indices": [0, 1], "atomic": True}, {"index": 1})
        )
    assert err.value.rule == 8


def test_rule_8_refuses_a_text_located_span_against_anything_at_the_site() -> None:
    """Nothing about a `variable` span is provable at load, so it is assumed
    to overlap — the conservative reading the module docstring states."""
    with pytest.raises(ValidationError) as err:
        _validate(
            _two_absolute_writes({"variable": "entity", "atomic": True}, {"index": 7})
        )
    assert err.value.rule == 8


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"indices": [0, 1], "atomic": True}, {"index": 5}),
        ({"indices": [0, 1], "atomic": True}, {"indices": [2, 3], "atomic": True}),
        ({"span": [0, 2], "atomic": True}, {"indices": [4, 6]}),
        ({"union": [{"indices": [0, 1]}, {"index": 2}]}, {"index": 3}),
    ],
)
def test_rule_8_twin_two_provably_disjoint_static_spans_are_accepted(first, second):
    _validate(_two_absolute_writes(first, second))


def test_rule_8_mixed_sign_regimes_are_unknowable_and_refused() -> None:
    with pytest.raises(ValidationError) as err:
        _validate(
            _two_absolute_writes({"indices": [0, 1], "atomic": True}, {"index": -1})
        )
    assert err.value.rule == 8


def test_a_static_atomic_span_is_one_to_one_through_the_one_classifier() -> None:
    doc = parse_document(_doc_with({"indices": [0, 1], "atomic": True}))
    assert static_alignment(doc, "p") == "one_to_one"
    assert static_alignment(doc, "p") == alignment_of(((0, 1),), ((0, 1),))
    loose = parse_document(_doc_with({"union": [{"variable": "a"}, {"variable": "b"}]}))
    assert static_alignment(loose, "p") is None  # the tokenizer decides


def test_rule_26_refuses_a_declaration_a_static_atomic_span_contradicts() -> None:
    with pytest.raises(ValidationError) as err:
        _validate(
            _doc_with({"indices": [0, 1], "atomic": True, "alignment": "one_to_many"})
        )
    assert err.value.rule == 26
    assert err.value.path == "positions.p.alignment"


@pytest.mark.parametrize(
    "raw",
    [
        {"indices": [0, 1], "atomic": True, "alignment": "one_to_one"},
        {"union": [{"variable": "entity"}, {"index": 0}], "alignment": "one_to_many"},
        {"variable": "entity", "atomic": True, "alignment": "many_to_one"},
    ],
)
def test_rule_26_twin_a_declaration_the_document_cannot_refute_loads(raw) -> None:
    _validate(_doc_with(raw))


@pytest.mark.parametrize(
    "raw",
    [
        {"indices": [3], "atomic": True},
        {"span": [2, 3], "atomic": True},
        {"union": [{"index": 1}, {"span": [1, 2], "atomic": True}], "atomic": True},
    ],
    ids=["one index", "one-wide span", "union collapsing to one"],
)
def test_rule_27_refuses_an_atomic_span_of_one_member(raw) -> None:
    with pytest.raises(ValidationError) as err:
        _validate(_doc_with(raw))
    assert err.value.rule == 27
    assert err.value.rule_id == "segment_declared"
    assert "two or more" in str(err.value)


@pytest.mark.parametrize(
    "raw",
    [
        {"indices": [3, 4], "atomic": True},
        {"span": [2, 4], "atomic": True},
        {"variable": "entity", "atomic": True},  # width is the tokenizer's
    ],
)
def test_rule_27_twin_a_joint_address_of_two_or_unknown_width_loads(raw) -> None:
    _validate(_doc_with(raw))


# --------------------------------------------------------------------------- #
# canonical form
# --------------------------------------------------------------------------- #


def test_a_span_canonicalizes_verbatim_with_no_default_materialized(env) -> None:
    raw = _doc_with({"union": [{"indices": [0, 1]}, {"variable": "entity"}]})
    canonical = canonicalize(raw, env)
    assert canonical["method"]["positions"]["p"] == raw["method"]["positions"]["p"]
    assert "atomic" not in canonical["method"]["positions"]["p"]
    atomic = copy.deepcopy(raw)
    atomic["method"]["positions"]["p"]["atomic"] = True
    assert digest(canonicalize(atomic, env)) != digest(canonical)  # it is address
