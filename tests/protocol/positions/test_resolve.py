"""One step's positions as a protocol-layer product
(``protocol/positions/resolve.py``): every address of every read and write
resolved on every row, the write refusals, the read records, the declared
alignment held to the pair, the positions key, and the ledger — all on a
real tokenizer and no model."""

from __future__ import annotations

from typing import Any

import pytest

from causalab.protocol.positions.alignment import UnalignableError
from causalab.protocol.positions.ledger import LEDGER_COLUMNS
from causalab.protocol.positions.resolve import (
    StepPositions,
    address_key,
    addressed,
    build_ledger,
    check_positions,
    encode_roles,
    positions_key,
    resolve_positions,
    spec_of,
)
from causalab.protocol.registry import component_shape, get_model_info
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PROTOCOL_VERSION, PositionSpec, parse_document

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME
from tests.protocol._docs import in_order, saved


pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from causalab.io.tokenizer import load_tokenizer

    return load_tokenizer(TINY_RANDOM_GPT2_MODEL_NAME)


def _doc(
    positions: dict[str, Any],
    reads: dict[str, Any],
    writes: dict[str, Any] | None = None,
    *,
    ledger: bool = False,
    model: str = "gpt2",
) -> Any:
    # each read is taken on the un-intervened model of its role (§2.9), named
    # as `causalab migrate` names it: `original` on base alone, else by role
    roles = list(dict.fromkeys(role for _pos, role in reads.values()))
    model_of = {
        role: "original" if roles == ["base"] else f"original_{role}" for role in roles
    }
    method: dict[str, Any] = {
        "intervened_models": {
            model_of[role]: {
                "input": role,
                "reads": [name for name, (_pos, r) in reads.items() if r == role],
            }
            for role in roles
        },
        "positions": positions,
        "sites": {"tap": {"component": "block_output", "layers": [0]}},
        "reads": {
            name: {"site": "tap", "pos": pos} for name, (pos, _role) in reads.items()
        },
        "save": [
            saved(name, model_of[role], f"{name}.safetensors")
            for name, (_pos, role) in reads.items()
        ],
    }
    data: dict[str, Any] = {"base": {"dataset": "inline", "field": "input"}}
    if writes:
        data["counterfactual"] = {
            "dataset": "inline",
            "field": "counterfactual_inputs[0]",
        }
        method["writes"] = {
            name: {"site": "tap", "pos": pos, "do": {"swap": source}}
            for name, (pos, source) in writes.items()
        }
        # a model nobody reads is refused at parse (§2.9): the written model
        # carries one probe read, saved like the others
        method["reads"]["probe"] = {"site": "tap", "pos": next(iter(positions))}
        method["save"].append(saved("probe", "patched", "probe.safetensors"))
        method["intervened_models"]["patched"] = {
            "input": "base",
            "reads": ["probe"],
            "writes": list(writes),
        }
    if ledger:
        method["save"].append({"kind": "location_ledger", "file_path": "ledger.json"})
    return parse_document(
        in_order(
            {
                "header": {"protocol_version": PROTOCOL_VERSION},
                "model": {"key": model, "revision": "main"},
                "data": data,
                "method": method,
            }
        )
    )


ROWS = [
    {
        "input": "If today is Friday, tomorrow is",
        "counterfactual_inputs": ["If today is Monday, tomorrow is"],
        "entity": "Friday",
        "input_variables": {"day": "Friday"},
        "counterfactual_inputs_variables": [{"day": "Monday"}],
    },
    {
        "input": "If today is Sunday, tomorrow is",
        "counterfactual_inputs": ["If today is Tuesday, tomorrow is"],
        "entity": "Sunday",
        "input_variables": {"day": "Sunday"},
        "counterfactual_inputs_variables": [{"day": "Tuesday"}],
    },
]
ROLE_ROWS = {"base": ROWS, "counterfactual": ROWS}
ROLE_FIELDS = {"base": "input", "counterfactual": "counterfactual_inputs[0]"}


def test_every_address_of_every_read_and_write_is_resolved_on_its_role(tokenizer):
    doc = _doc(
        {"last": {"index": -1}, "day": {"variable": "day"}},
        {"r_last": ("last", "base"), "r_day": ("day", "counterfactual")},
        {"w": ("day", "r_day")},
    )
    frames = encode_roles(tokenizer, doc, ROLE_ROWS, ROLE_FIELDS)
    assert set(frames) == {"base", "counterfactual"}
    positions = resolve_positions(doc, frames, ROLE_ROWS, ROLE_FIELDS)
    assert addressed(doc) == [
        ("last", "base", "r_last"),
        ("day", "counterfactual", "r_day"),
        ("last", "base", "probe"),  # the written model's own read (§2.9)
        ("day", "base", None),
    ]
    assert set(positions.addresses) == {
        ("last", "base"),
        ("day", "counterfactual"),
        ("day", "base"),
    }
    last = positions.addresses[("last", "base")]
    assert last.indices == ((frames["base"].padded_len - 1,),) * 2 and not last.problems
    day_cf = positions.addresses[("day", "counterfactual")]
    assert all(len(run) >= 1 for run in day_cf.indices) and not day_cf.problems
    # the token the address names, decoded, is the variable's value
    for row, run in enumerate(day_cf.indices):
        ids = [frames["counterfactual"].row_ids(row)[i] for i in run]
        assert (
            tokenizer.decode(ids).strip()
            == ROWS[row]["counterfactual_inputs_variables"][0]["day"]
        )
    check_positions(doc, positions)  # nothing to refuse


def test_a_write_on_an_unalignable_row_is_refused_and_a_read_records_it(tokenizer):
    rows = [dict(ROWS[0]), dict(ROWS[1])]
    rows[1]["input_variables"] = {"day": "Thursday"}  # absent from the text
    role_rows = {"base": rows, "counterfactual": rows}
    doc_read = _doc({"day": {"variable": "day"}}, {"r": ("day", "base")})
    frames = encode_roles(tokenizer, doc_read, role_rows, ROLE_FIELDS)
    positions = resolve_positions(doc_read, frames, role_rows, ROLE_FIELDS)
    resolved = positions.addresses[("day", "base")]
    assert resolved.indices[1] == () and list(resolved.problems) == [1]
    assert resolved.problems[1].cardinality == "absent"
    assert "occurs 0 times" in resolved.problems[1].message
    check_positions(doc_read, positions)  # a read's rows are cells, not refusals
    assert positions.rows("day", spec_of(doc_read, "day"), "base", cell="r") == [
        list(resolved.indices[0]),
        [],
    ]
    with pytest.raises(UnalignableError) as err:
        positions.rows("day", spec_of(doc_read, "day"), "base", cell=None)
    assert err.value.reason == "alignment_missing"

    doc_write = _doc(
        {"day": {"variable": "day"}},
        {"r": ("day", "counterfactual")},
        {"w": ("day", "r")},
    )
    # the write refuses as the addresses are resolved — before any declared
    # alignment is checked, as the executor always ordered it
    with pytest.raises(UnalignableError, match="occurs 0 times"):
        resolve_positions(doc_write, frames, role_rows, ROLE_FIELDS)
    # and again from the standalone check over a resolution built by hand
    positions = StepPositions(
        frames=frames, role_rows=role_rows, role_fields=ROLE_FIELDS
    )
    with pytest.raises(UnalignableError, match="occurs 0 times"):
        check_positions(doc_write, positions)


def test_a_declared_alignment_is_held_to_the_pair(tokenizer):
    """The pair below carries the same value on both sides, so the address is
    ``one_to_one`` whatever the tokenizer's pieces are (the tiny GPT-2 vocabulary
    splits a weekday into several); ``one_to_many`` contradicts the pair."""
    rows = [
        {
            **row,
            "counterfactual_inputs": [row["input"]],
            "counterfactual_inputs_variables": [row["input_variables"]],
        }
        for row in ROWS
    ]
    role_rows = {"base": rows, "counterfactual": rows}
    doc = _doc(
        {"day": {"variable": "day", "alignment": "one_to_many"}},
        {"r": ("day", "counterfactual")},
        {"w": ("day", "r")},
    )
    frames = encode_roles(tokenizer, doc, role_rows, ROLE_FIELDS)
    with pytest.raises(ProtocolError, match="declares alignment 'one_to_many'"):
        resolve_positions(doc, frames, role_rows, ROLE_FIELDS)
    fine = _doc(
        {"day": {"variable": "day", "alignment": "one_to_one"}},
        {"r": ("day", "counterfactual")},
        {"w": ("day", "r")},
    )
    resolve_positions(fine, frames, role_rows, ROLE_FIELDS)


def test_an_address_the_protocol_did_not_pre_resolve_is_resolved_on_demand(tokenizer):
    doc = _doc({"last": {"index": -1}}, {"r": ("last", "base")})
    frames = encode_roles(tokenizer, doc, {"base": ROWS}, {"base": "input"})
    positions = StepPositions(
        frames=frames, role_rows={"base": ROWS}, role_fields={"base": "input"}
    )
    assert positions.addresses == {}
    inline = PositionSpec(variable="day")
    resolved = positions.address(inline, inline, "base")
    assert positions.addresses[(address_key(inline, inline), "base")] is resolved
    assert positions.address(inline, inline, "base") is resolved  # cached


def test_the_positions_key_is_what_a_steps_positions_depend_on(tokenizer):
    a = _doc({"last": {"index": -1}}, {"r": ("last", "base")})
    b = _doc({"last": {"index": -1}}, {"r": ("last", "base")})
    assert positions_key(a) == positions_key(b)
    assert positions_key(a) != positions_key(
        _doc({"last": {"index": -2}}, {"r": ("last", "base")})
    )
    assert positions_key(a) != positions_key(
        _doc({"last": {"index": -1}}, {"r": ("last", "base")}, model="test")
    )
    with_save = _doc({"last": {"index": -1}}, {"r": ("last", "base")}, ledger=True)
    assert positions_key(a) == positions_key(
        with_save
    )  # a save entry moves no position


def test_the_ledger_is_built_from_the_resolved_positions(tokenizer):
    doc = _doc(
        {"last": {"index": -1}, "both": {"indices": [0, -1]}},
        {"r": ("last", "base"), "s": ("both", "base")},
        ledger=True,
    )
    frames = encode_roles(tokenizer, doc, {"base": ROWS}, {"base": "input"})
    positions = resolve_positions(doc, frames, {"base": ROWS}, {"base": "input"})
    info = get_model_info("gpt2")
    ledger = build_ledger(
        doc,
        positions,
        tokenizer,
        has_positions=lambda site: component_shape(
            info, str(doc.sites[site].component)
        ).has_contract_form,
    )
    records = ledger.records()
    assert records and all(tuple(r) == LEDGER_COLUMNS for r in records)
    constituents = {r["constituent"] for r in records}
    assert constituents == {"last", "both[0]", "both[1]"}
    for record in records:
        frame = frames["base"]
        ids = frame.row_ids(record["example"])
        first = frame.first_real(record["example"])
        assert ids[first + record["token_index"]] == record["token_id"]
        assert record["decoded_token"] == tokenizer.convert_ids_to_tokens(
            record["token_id"]
        )
        assert record["edit_group"] == "original on base" and record["side"] == "base"
