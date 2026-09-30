"""The ``segments`` section, rule 27, the ``location_ledger`` save kind and the
ledger's rows (spec §2.2.1, §2.12, §6, §7) — the pure half.

* ``segments`` is an optional top-level section right after ``data``; its
  parse, its closed vocabularies (``frame``, the chat segment names, the
  source keys) against the spec's tables, and the section's declared names;
* rule 27: an undeclared ``segment:`` anchor, an off-enum ``frame``, a
  ``system`` turn outside the chat frame, a reserved name in ``declare``, a
  ``continuation`` anchor without ``generated`` — each refused naming its
  field, each with the twin that loads; the rule is the 27th;
* **T3** (the legitimate campaign): every ``tests/protocols/`` document and
  the shipped ``weekdays_locate_scan.json`` load, author no ``segments``, and
  canonicalize to their pinned digests — the proof the section is optional and
  digest-neutral;
* the ``location_ledger`` save kind parses, is opt-in, and rule 10 holds it
  to one JSON entry; the ledger records an address once and refuses a
  contradiction at the same address.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import (
    REASON_CODES,
    RULES,
    ParseError,
    ValidationError,
)
from causalab.protocol.positions.ledger import LEDGER_COLUMNS, LedgerRow, LocationLedger
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.schema import SAVE_KINDS, SECTION_ORDER, parse_document
from causalab.protocol.segments import (
    CHAT_SEGMENTS,
    SEGMENT_FRAMES,
    SEGMENT_SOURCE_KEYS,
    SegmentsSpec,
    parse_segments,
)
from causalab.protocol.rules.document import validate_document

from tests.protocol._docs import base_doc, in_order, saved
from tests.protocol._env import CORPUS_DIR, steps_of
from tests._helpers.paths import PROTOCOLS_DIR

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "docs" / "intervention_protocol.md"
INTERNALS = REPO / "docs" / "intervention_protocol_internals.md"
SHIPPED_LOCATE_SCAN = PROTOCOLS_DIR / "weekdays_locate_scan.json"
PINS = json.loads((Path(__file__).parent / "corpus_digests.json").read_text())


def _validate(raw: dict[str, Any]) -> None:
    validate_document(parse_document(in_order(raw)))


def _chat_doc(position: Any, **segments: Any) -> dict[str, Any]:
    doc = base_doc()
    doc["method"]["segments"] = segments or {"frame": "chat"}
    doc["method"]["positions"] = {"p": position}
    doc["method"]["reads"]["v_cf"]["pos"] = "p"
    return doc


# --------------------------------------------------------------------------- #
# the section
# --------------------------------------------------------------------------- #


def test_segments_is_the_first_section_after_the_models() -> None:
    """`intervened_models` opens the method (§1); `segments` follows it,
    ahead of every section that can name a segment."""
    assert SECTION_ORDER.index("intervened_models") == SECTION_ORDER.index("data") + 1
    assert (
        SECTION_ORDER.index("segments") == SECTION_ORDER.index("intervened_models") + 1
    )
    assert SECTION_ORDER.index("segments") < SECTION_ORDER.index("positions")


def test_a_document_without_the_section_has_none() -> None:
    assert parse_document(base_doc()).segments is None


def test_the_section_parses_and_declares_its_names() -> None:
    chat = parse_segments({"frame": "chat"})
    assert chat == SegmentsSpec(frame="chat")
    assert chat.declared() == ("user", "assistant_prefix", "continuation")
    with_system = parse_segments({"frame": "chat", "system": {"column": "sys"}})
    assert with_system.declared() == CHAT_SEGMENTS
    plain = parse_segments({"declare": {"q": {"column": "question"}}})
    assert plain.frame is None and plain.declared() == ("continuation", "q")
    both = parse_segments({"frame": "chat", "declare": {"q": {"column": "question"}}})
    assert both.declared() == ("user", "assistant_prefix", "continuation", "q")


@pytest.mark.parametrize(
    ("raw", "code", "fragment"),
    [
        ({}, "P2", "declares nothing"),
        ({"frame": "chat", "mode": 1}, "P3", "unknown key"),
        ({"frame": 1}, "P2", "frame is a string"),
        ({"declare": {"q": {"col": "question"}}}, "P3", "unknown key"),
        ({"declare": {"q": {"column": 3}}}, "P2", "segment source"),
        ({"declare": {"": {"column": "q"}}}, "P2", "non-empty"),
        ({"system": "sys"}, "P2", "expected an object"),
    ],
)
def test_a_malformed_section_is_refused_at_parse(raw, code, fragment) -> None:
    with pytest.raises(ParseError) as err:
        parse_document(in_order({**base_doc(), "segments": raw}))
    assert err.value.code == code and fragment in str(err.value)


def test_the_section_is_parsed_onto_the_document() -> None:
    doc = parse_document(in_order({**base_doc(), "segments": {"frame": "chat"}}))
    assert doc.segments == SegmentsSpec(frame="chat")


# --------------------------------------------------------------------------- #
# the census: the spec's tables are exactly the code's vocabularies
# --------------------------------------------------------------------------- #


def _section(heading: str, spec: Path = SPEC) -> str:
    depth = len(heading) - len(heading.lstrip("#"))
    body = spec.read_text().split(heading, 1)
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


def _names(section: str, header: str) -> list[str]:
    return [row[0].strip("`") for row in _table(section, header)]


def test_the_frame_segment_and_source_tables_match_the_code() -> None:
    section = _section("### 2.2.1 `segments`")
    assert _names(section, "frame") == list(SEGMENT_FRAMES)
    assert _names(section, "segment") == list(CHAT_SEGMENTS)
    assert _names(section, "source") == list(SEGMENT_SOURCE_KEYS)


def test_the_save_kind_table_matches_the_code() -> None:
    assert _names(_section("### 2.12 `save`"), "kind") == list(SAVE_KINDS)


def test_the_chat_template_reason_code_is_in_the_vocabulary() -> None:
    """The one new reason code; the §2.4 table census is
    test_vocabulary_census's."""
    assert "chat_template_missing" in REASON_CODES


def test_the_ledger_columns_are_the_spec_row() -> None:
    section = _section("## 6. Derived: never authored", INTERNALS)
    assert (
        "(example, edit group, constituent, side, token index, token id, decoded token)"
        in section
    )
    assert LEDGER_COLUMNS == (
        "example",
        "edit_group",
        "constituent",
        "side",
        "token_index",
        "token_id",
        "decoded_token",
    )


def test_the_new_protocol_modules_are_torch_free() -> None:
    for name in ("positions/spans.py", "segments.py", "positions/ledger.py"):
        tree = ast.parse((REPO / "causalab/protocol" / name).read_text())
        modules = {
            (node.module or "").split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "torch" not in modules and "transformers" not in modules, name


# --------------------------------------------------------------------------- #
# rule 27
# --------------------------------------------------------------------------- #


def test_rule_27_is_the_twenty_seventh_rule() -> None:
    # rules 28 (`family_wrappers`, sec. 3.1), 29–31 (`kl` operands,
    # the engine-supported fit, one position per metric) and 32 (a gate's
    # `init.from_scores`) follow it
    assert len(RULES) == 32
    rule = RULES["segment_declared"]
    assert rule.number == 27 and rule.code == "V27"
    codes = json.loads((Path(__file__).parent / "rule_codes.json").read_text())
    assert codes["segment_declared"] == "V27"


def _expect_27(raw: dict[str, Any], path: str, fragment: str) -> None:
    with pytest.raises(ValidationError) as err:
        _validate(raw)
    assert err.value.rule == 27, err.value
    assert err.value.rule_id == "segment_declared"
    assert err.value.path == path, err.value
    assert fragment in str(err.value)


def test_rule_27_refuses_a_frame_outside_the_vocabulary_naming_the_field() -> None:
    _expect_27(
        {**base_doc(), "segments": {"frame": "chta"}},
        "segments.frame",
        "did you mean 'chat'",
    )


def test_rule_27_refuses_a_system_turn_outside_the_chat_frame() -> None:
    _expect_27(
        {**base_doc(), "segments": {"system": {"column": "sys"}}},
        "segments.system",
        "only in the chat frame",
    )


def test_rule_27_refuses_a_declared_name_the_chat_frame_owns() -> None:
    _expect_27(
        {**base_doc(), "segments": {"declare": {"user": {"column": "input"}}}},
        "segments.declare.user",
        "locates itself",
    )


def test_rule_27_refuses_a_segment_anchor_with_no_section() -> None:
    doc = base_doc()
    doc["method"]["positions"] = {"p": {"index": -1, "scope": {"segment": "user"}}}
    doc["method"]["reads"]["v_cf"]["pos"] = "p"
    _expect_27(doc, "positions.p", "no segments section")


def test_rule_27_refuses_an_undeclared_segment_with_a_suggestion() -> None:
    _expect_27(
        _chat_doc({"index": -1, "scope": {"segment": "assistant_prefx"}}),
        "positions.p",
        "did you mean 'assistant_prefix'",
    )


def test_rule_27_refuses_system_without_a_source() -> None:
    _expect_27(_chat_doc({"segment": "system"}), "positions.p", "not declared")


def test_rule_27_refuses_an_inline_anchor_too() -> None:
    doc = base_doc()
    doc["method"]["segments"] = {"frame": "chat"}
    doc["method"]["reads"]["v_cf"]["pos"] = {"before": {"segment": "nope"}}
    _expect_27(doc, "reads.v_cf.pos", "not declared")


def test_rule_27_refuses_a_continuation_anchor_without_generated() -> None:
    _expect_27(
        _chat_doc({"index": 0, "scope": {"segment": "continuation"}}),
        "positions.p",
        '"generated"',
    )


def test_rule_27_refuses_a_whole_continuation_span() -> None:
    _expect_27(_chat_doc({"segment": "continuation"}), "positions.p", '"all": true')


def _generated_doc(position: Any) -> dict[str, Any]:
    """A read of the continuation at lm_head on the un-intervened model — the
    shape rule 16 allows — saved as a tensor."""
    doc = base_doc()
    doc["method"]["segments"] = {"frame": "chat"}
    doc["method"]["positions"] = {"g": position}
    doc["method"]["reads"]["gen"] = {"site": "lm_head", "pos": "g"}
    doc["method"]["intervened_models"]["original_base"] = {
        "input": "base",
        "reads": ["gen"],
    }
    doc["method"]["save"].append(saved("gen", "original_base", "gen.safetensors"))
    return doc


def test_twin_a_continuation_anchor_with_generated_loads() -> None:
    doc = _generated_doc(
        {
            "generated": {"max_new_tokens": 2},
            "index": -1,
            "scope": {"segment": "continuation"},
        }
    )
    parsed = parse_document(in_order(doc))
    validate_document(parsed)
    spec = parsed.positions["g"]
    assert spec.anchor_source == "segment" and spec.scope == "continuation"


def test_a_generated_position_with_another_segment_scope_is_still_refused() -> None:
    with pytest.raises(ParseError, match="cannot combine with 'generated'"):
        parse_document(
            in_order(
                _generated_doc(
                    {
                        "generated": {"max_new_tokens": 2},
                        "index": -1,
                        "scope": {"segment": "user"},
                    }
                )
            )
        )


@pytest.mark.parametrize(
    "position",
    [
        {"index": -1, "scope": {"segment": "assistant_prefix"}},
        {"index": 0, "scope": {"segment": "user"}},
        {"segment": "user"},
        {"before": {"segment": "user"}},
        {"index": 1, "relative_to": {"segment": "assistant_prefix"}},
        {"between": [{"segment": "user"}, {"segment": "assistant_prefix"}]},
    ],
)
def test_twin_a_declared_chat_segment_anchor_loads(position) -> None:
    _validate(_chat_doc(position))


def test_twin_a_system_anchor_loads_with_a_source() -> None:
    _validate(_chat_doc({"segment": "system"}, frame="chat", system={"column": "sys"}))


def test_twin_a_plain_frame_declares_column_segments() -> None:
    _validate(
        _chat_doc(
            {"index": -1, "scope": {"segment": "ent"}},
            declare={"ent": {"column": "entity"}},
        )
    )


# --------------------------------------------------------------------------- #
# T3 — the legitimate campaign: optional, digest-neutral
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(PINS))
def test_t3_every_corpus_document_authors_no_segments_and_keeps_its_pin(env, name):
    loaded = compile_protocol(CORPUS_DIR / name, env=env)
    assert loaded.document.segments is None
    assert "segments" not in loaded.canonical
    assert loaded.digests.document == PINS[name]["document"]
    assert list(steps_of(loaded, env).digests) == PINS[name]["points"]


def test_t3_the_shipped_locate_scan_loads_with_no_segments(env) -> None:
    loaded = compile_protocol(SHIPPED_LOCATE_SCAN, env=env)
    assert loaded.document.segments is None
    assert "segments" not in loaded.canonical
    assert len(steps_of(loaded, env).digests) == 56  # 28 layers x 2 positions


def test_the_section_is_copied_through_canonicalization_only_when_authored(env):
    plain = canonicalize(base_doc(), env)
    assert "segments" not in plain
    framed = canonicalize(in_order({**base_doc(), "segments": {"frame": "chat"}}), env)
    assert framed["method"]["segments"] == {"frame": "chat"}
    assert digest(framed) != digest(plain)  # the frame is part of the address


# --------------------------------------------------------------------------- #
# the location_ledger save kind
# --------------------------------------------------------------------------- #

LEDGER_ENTRY = {"kind": "location_ledger", "file_path": "ledger.json"}


def test_the_ledger_save_kind_parses_and_validates() -> None:
    doc = base_doc()
    doc["method"]["save"].append(LEDGER_ENTRY)
    parsed = parse_document(doc)
    validate_document(parsed)
    entry = parsed.save[-1]
    assert entry.kind == "location_ledger" and entry.value == "location_ledger"
    assert entry.read is None and entry.site is None


def test_a_document_without_the_entry_asks_for_no_ledger() -> None:
    assert all(entry.kind is None for entry in parse_document(base_doc()).save)


@pytest.mark.parametrize(
    ("entry", "exc", "fragment"),
    [
        ({"kind": "ledger", "file_path": "l.json"}, ParseError, "not one of"),
        (
            {"kind": "location_ledger", "file_path": "l.json", "model": "patched"},
            ParseError,
            "unknown key",
        ),
        ({"kind": "location_ledger"}, ParseError, "file_path"),
        (
            {"kind": "location_ledger", "file_path": "l.safetensors"},
            ValidationError,
            "'.json'",
        ),
    ],
)
def test_a_malformed_ledger_entry_is_refused(entry, exc, fragment) -> None:
    doc = base_doc()
    doc["method"]["save"].append(entry)
    with pytest.raises(exc) as err:
        _validate(doc)
    assert fragment in str(err.value)
    if exc is ValidationError:
        assert err.value.rule == 10


def test_rule_10_holds_the_ledger_to_one_entry() -> None:
    doc = base_doc()
    doc["method"]["save"].extend(
        [LEDGER_ENTRY, {"kind": "location_ledger", "file_path": "l2.json"}]
    )
    with pytest.raises(ValidationError) as err:
        _validate(doc)
    assert err.value.rule == 10 and "saved twice" in str(err.value)


# --------------------------------------------------------------------------- #
# the ledger rows
# --------------------------------------------------------------------------- #


def _row(**over: Any) -> LedgerRow:
    base = dict(
        example=0,
        edit_group="original on base",
        constituent="last",
        side="base",
        token_index=6,
        token_id=318,
        decoded_token="Ġis",
    )
    base.update(over)
    return LedgerRow(**base)


def test_the_ledger_records_an_address_once_and_refuses_a_contradiction() -> None:
    ledger = LocationLedger()
    ledger.add(_row())
    ledger.add(_row())  # the width pre-flight, then the write
    assert len(ledger) == 1
    with pytest.raises(AssertionError, match="recorded twice"):
        ledger.add(_row(token_id=1))
