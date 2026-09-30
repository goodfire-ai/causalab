"""Semantic spans, the chat frame and the location ledger against real
tokenizers (spec §2.2.1, §2.3, §6) — the engine half, on the two tiny
fixtures.

* **T1** — one prompt bare and with a terminal separator, on the byte-level
  BPE fixture: the derived indices differ and the ledger rows differ, so the
  two tables tell the two runs apart.
* **T2** — `listsort`'s nine coordinate systems, each one authored position
  or span in a table, resolved with no Python between document and indices;
  `output rank` is marked as not a position (a metric-side notion). The
  mutation that deletes the `segment` anchor fails the full-sequence,
  assistant-prefix and continuation rows at parse.
* the chat frame: `frame: chat` on a tokenizer without a template is refused
  with reason `chat_template_missing`; the same document on a templated
  tokenizer runs with a real `prefix_lengths`; a plain document encodes
  byte-identically to `encode` with prefix 0 (the fail-closed twin).
* an `atomic` span is one address under rule 19 (ragged → refused before any
  forward; a fixed-width twin runs) and its constituents are classified
  separately under a declared `alignment`.
* the ledger is opt-in end to end: a run with the save entry writes the
  table; a run without it writes none; a run that loads a parameter fitted
  on other rows gets a ledger listing the tokens selected on its own rows.

`tiny-random-gpt2` ships no chat template, which is the refusal twin; the chat
tests set a minimal Jinja template **on the tokenizer object** — the template
is data on the tokenizer, and this file says so where it does it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.cli import main
from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.encoding import (
    Continuation,
    EncodedBatch,
    encode,
    resolve_position,
)
from causalab.protocol.rules.errors import ParseError, ProtocolError, ValidationError
from causalab.protocol.positions.framing import encode_framed
from causalab.protocol.positions.ledger import LEDGER_COLUMNS, wants_ledger
from causalab.protocol.schema import PositionSpec, parse_document
from causalab.protocol.segments import parse_segments

from ._drive import base_data_section, bundle_loader, executor_for
from .conftest import TINY_GPT2, TINY_LLAMA
from tests.protocol._docs import UNWRITTEN, in_order, saved
from tests.protocol._env import FIXTURES

pytestmark = pytest.mark.smoke

#: A minimal chat template — role markers, an end marker, a generation prompt.
#: Set on the fixture tokenizer object, which ships none (see the docstring).
MINIMAL_TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def gpt2_bundle():
    return load_model(TINY_GPT2)


@pytest.fixture()
def templated_tokenizer(gpt2_bundle):
    """The BPE fixture's tokenizer with the minimal template set for one
    test, then removed — other tests need the template-less twin."""
    tokenizer = gpt2_bundle.tokenizer
    assert not getattr(tokenizer, "chat_template", None)
    tokenizer.chat_template = MINIMAL_TEMPLATE
    try:
        yield tokenizer
    finally:
        tokenizer.chat_template = None


def _doc(
    positions: dict[str, Any],
    reads: dict[str, str],
    *,
    ledger: bool = False,
    segments: dict[str, Any] | None = None,
    counterfactual: bool = False,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "header": {"protocol_version": "4"},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=counterfactual),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": list(reads)}},
            "positions": positions,
            "sites": {"tap": {"component": "block_output", "layers": [0]}},
            "reads": {name: {"site": "tap", "pos": pos} for name, pos in reads.items()},
            "save": [saved(name, "original", f"{name}.safetensors") for name in reads],
        },
    }
    if segments is not None:
        doc["method"]["segments"] = segments
    if ledger:
        doc["method"]["save"].append(
            {"kind": "location_ledger", "file_path": "ledger.json"}
        )
    return in_order(doc)


# --------------------------------------------------------------------------- #
# T1 — a terminal separator is a changed experiment
# --------------------------------------------------------------------------- #

PROMPT = "If today is Friday, tomorrow is"


def test_t1_a_terminal_separator_moves_the_indices(gpt2_bundle):
    doc = _doc(
        {"last": {"index": -1}, "ent": {"variable": "entity"}},
        {"r_last": "last", "r_ent": "ent"},
        ledger=True,
    )
    bare = executor_for(
        doc, gpt2_bundle, base_texts=[PROMPT], extra_columns={"entity": ["Friday"]}
    )
    dotted = executor_for(
        doc,
        gpt2_bundle,
        base_texts=[PROMPT + "."],
        extra_columns={"entity": ["Friday"]},
    )
    first, second = bare.location_ledger(), dotted.location_ledger()
    # derived indices differ: the last token moved by one, the entity did not
    by_key = {(r["constituent"], r["token_index"]): r for r in first.records()}
    moved = {(r["constituent"], r["token_index"]): r for r in second.records()}
    assert {k for k in by_key if k[0] == "last"} != {k for k in moved if k[0] == "last"}
    assert {k for k in by_key if k[0] == "ent"} == {k for k in moved if k[0] == "ent"}
    assert first.records() != second.records()
    # and the twin: resolving the same batch again yields the same rows
    assert bare.location_ledger().records() == first.records()


def test_the_ledger_row_is_the_seven_columns_with_the_row_local_index(gpt2_bundle):
    doc = _doc({"last": {"index": -1}}, {"r": "last"}, ledger=True)
    executor = executor_for(doc, gpt2_bundle, base_texts=["one two", PROMPT])
    rows = executor.location_ledger().records()
    assert all(tuple(row) == LEDGER_COLUMNS for row in rows)
    batch = executor._batch("base")  # pyright: ignore[reportPrivateUsage]
    for row in rows:
        example = row["example"]
        padded = batch.padded_len - 1
        assert row["token_index"] == padded - batch.first_real(example)
        assert row["token_id"] == int(batch.input_ids[example, padded])
        assert row["decoded_token"] == gpt2_bundle.tokenizer.convert_ids_to_tokens(
            row["token_id"]
        )
        assert row["edit_group"] == "original on base" and row["side"] == "base"
        assert row["constituent"] == "last"
    # the shorter row is left-padded, so its index is not the padded index
    assert rows[0]["token_index"] < rows[1]["token_index"]


# --------------------------------------------------------------------------- #
# T2 — listsort's nine coordinate systems, each one authored address
# --------------------------------------------------------------------------- #

#: Candidate first items: the test takes the first one this tokenizer makes
#: **two or more** pieces of *in context* (a standalone piece count says
#: nothing — " 17" may be one piece where "17" is two).
TWO_TOKEN_CANDIDATES = ("17", "1742", "98765", "314159")


def _list_row(item_1: str) -> dict[str, str]:
    list_text = f"{item_1}, 3, 42"
    return {
        "input": f"Sort the list: {list_text}. Answer:",
        "item_1": item_1,
        "item_2": "3",
        "item_3": "42",
        "list_text": list_text,
    }


#: (coordinate system, the authored position, what the decoded tokens must be
#: — a string the decoded pieces join to, or None for a non-address row)
NINE: list[tuple[str, dict[str, Any] | None, str | None]] = [
    ("semantic item", {"variable": "item_2"}, "3"),
    ("char span (a serialized substring)", {"column": "list_text"}, "<list_text>"),
    ("prompt-local position", {"index": 0}, "S"),
    (
        "full-sequence position (the chat prefix)",
        {"before": {"segment": "user"}},
        "<|user|>",
    ),
    ("assistant prefix", {"index": -1, "scope": {"segment": "assistant_prefix"}}, ">"),
    (
        "continuation",
        {
            "generated": {"max_new_tokens": 3},
            "index": 0,
            "scope": {"segment": "continuation"},
        },
        None,  # a decode step, resolved against the continuation frame below
    ),
    (
        "first token of a two-token number",
        {"index": 0, "scope": {"variable": "item_1"}},
        None,
    ),
    (
        "whole donor span (one atomic address)",
        {"variable": "item_1", "atomic": True},
        "<item_1>",
    ),
    ("output rank — not a position (a metric-side notion)", None, None),
]


def _decoded(tokenizer: Any, batch: Any, indices: list[int]) -> str:
    return "".join(
        tokenizer.convert_ids_to_tokens(batch.input_ids[0, indices].tolist())
    )


def test_t2_the_nine_coordinate_systems_are_each_one_authored_address(
    gpt2_bundle, templated_tokenizer
):
    """No Python between document and indices: each row is parsed as a
    document position and handed to the resolver; the assertion reads the
    tokens the indices address. Rows 4–6 need the `segment` anchor — delete
    it and they fail at parse (`test_t2_mutation_…`)."""
    tokenizer = templated_tokenizer
    frame = parse_segments({"frame": "chat"})
    for item_1 in TWO_TOKEN_CANDIDATES:
        row = _list_row(item_1)
        batch = EncodedBatch.from_frame(encode_framed(tokenizer, [row], "input", frame))
        width = len(
            resolve_position(
                PositionSpec(variable="item_1"),
                batch,
                0,
                dataset_row=row,
                field="input",
            )
        )
        if width >= 2:
            break
    else:
        pytest.skip("this tokenizer makes every candidate number one piece in context")
    positions = {
        f"c{i}": spec for i, (_, spec, _) in enumerate(NINE) if spec is not None
    }
    positions["c6b"] = {"index": 1, "scope": {"variable": "item_1"}}  # second token
    doc = parse_document(
        _doc(
            positions,
            {f"r_{name}": name for name in positions},
            segments={"frame": "chat"},
        )
    )
    assert batch.prefix_lengths[0] > 0
    continuation = Continuation(
        token_ids=torch.tensor([[5, 6, 7]]),
        widths=(3,),
        texts=("abc",),
        offsets=(((0, 1), (1, 2), (2, 3)),),
    )
    seen = 0
    for i, (system, spec, expected) in enumerate(NINE):
        if spec is None:
            assert "not a position" in system  # output rank: no address invented
            continue
        indices = resolve_position(
            doc.positions[f"c{i}"],
            batch,
            0,
            dataset_row=row,
            field="input",
            continuation=continuation,
        )
        seen += 1
        if "continuation" in system:
            assert indices == [0]  # decode step 0
            continue
        decoded = _decoded(tokenizer, batch, indices).replace("Ġ", " ").strip()
        if expected is not None:
            want = expected.replace("<list_text>", row["list_text"]).replace(
                "<item_1>", item_1
            )
            assert decoded == want, (system, decoded)
        if "first token" in system:
            second = resolve_position(
                doc.positions["c6b"], batch, 0, dataset_row=row, field="input"
            )
            assert len(indices) == len(second) == 1 and second[0] == indices[0] + 1
            both = _decoded(tokenizer, batch, indices + second).replace("Ġ", "").strip()
            assert item_1.startswith(both) and len(both) >= 2  # its first two pieces
    assert seen == 8
    # the whole donor span is one address; its two tokens are two constituents
    whole = doc.positions["c7"]
    assert getattr(whole, "atomic", False) is True


@pytest.mark.parametrize(
    "row", [3, 4, 5], ids=["full-sequence", "assistant-prefix", "continuation"]
)
def test_t2_mutation_deleting_the_segment_anchor_fails_the_segment_rows(row):
    """The mutation clause: with `segment` removed from the anchor vocabulary,
    these three rows do not parse — which is what the rows depend on."""
    _, spec, _ = NINE[row]
    assert spec is not None
    stripped = json.loads(json.dumps(spec).replace('"segment"', '"segmen"'))
    with pytest.raises(ParseError):
        parse_document(_doc({"c": stripped}, {"r": "c"}, segments={"frame": "chat"}))


# --------------------------------------------------------------------------- #
# the chat frame: refusal, twin, and byte-identical plain text
# --------------------------------------------------------------------------- #


def test_frame_chat_without_a_template_is_refused_with_its_reason(gpt2_bundle):
    doc = _doc(
        {"p": {"index": -1, "scope": {"segment": "assistant_prefix"}}},
        {"r": "p"},
        segments={"frame": "chat"},
    )
    executor = executor_for(doc, gpt2_bundle, base_texts=[PROMPT])
    with pytest.raises(ProtocolError) as err:
        executor.read_value("r")
    assert err.value.reason == "chat_template_missing"
    assert err.value.path == "segments.frame"
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_twin_frame_chat_on_a_templated_tokenizer_runs_with_a_real_prefix(
    gpt2_bundle, templated_tokenizer
):
    doc = _doc(
        {
            "p": {"index": -1, "scope": {"segment": "assistant_prefix"}},
            "first": {"index": 0},
        },
        {"r": "p", "f": "first"},
        segments={"frame": "chat"},
    )
    executor = executor_for(doc, gpt2_bundle, base_texts=[PROMPT, PROMPT + "."])
    value = executor.read_value("r")
    assert isinstance(value, torch.Tensor) and value.shape[:2] == (2, 1)
    batch = executor._batch("base")  # pyright: ignore[reportPrivateUsage]
    rendered = templated_tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert batch.texts[0] == rendered
    prefix = templated_tokenizer.encode("<|user|>", add_special_tokens=False)
    assert batch.prefix_lengths == (len(prefix), len(prefix))
    first = executor._positions("first", batch, "base")  # pyright: ignore[reportPrivateUsage]
    assert (
        templated_tokenizer.convert_ids_to_tokens(int(batch.input_ids[0, first[0][0]]))
        == "I"
    )


def test_twin_the_llama_fixture_ships_a_template_and_frames_a_user_turn(llama_bundle):
    """The sentencepiece fixture's own (Llama-2) template: BOS inside the
    rendering, no generation prompt — so `user` locates, `assistant_prefix`
    is `absent`, and nothing is double-BOS."""
    doc = _doc({"u": {"segment": "user"}}, {"r": "u"}, segments={"frame": "chat"})
    executor = executor_for(doc, llama_bundle, base_texts=[PROMPT])
    value = executor.read_value("r")
    assert isinstance(value, torch.Tensor)
    batch = executor._batch("base")  # pyright: ignore[reportPrivateUsage]
    assert batch.texts[0].startswith(llama_bundle.tokenizer.bos_token)
    assert batch.prefix_lengths[0] >= 1
    assert batch.segments[0]["assistant_prefix"] == ()
    absent = _doc(
        {"a": {"segment": "assistant_prefix"}}, {"r": "a"}, segments={"frame": "chat"}
    )
    executor = executor_for(absent, llama_bundle, base_texts=[PROMPT])
    executor.read_value("r")
    assert executor.resolution("r").reason == "alignment_missing"  # pyright: ignore[reportAttributeAccessIssue]


def test_a_plain_document_encodes_byte_identically_with_prefix_zero(bundle):
    doc = _doc({"last": {"index": -1}}, {"r": "last"})
    executor = executor_for(doc, bundle, base_texts=[PROMPT, "one two"])
    batch = executor._batch("base")  # pyright: ignore[reportPrivateUsage]
    plain = encode(bundle.tokenizer, [PROMPT, "one two"])
    assert torch.equal(batch.input_ids, plain.input_ids)
    assert torch.equal(batch.attention_mask, plain.attention_mask)
    assert batch.offset_mapping == plain.offset_mapping
    assert batch.prefix_lengths == plain.prefix_lengths == (0, 0)
    assert batch.segments == ()


def test_a_plain_frame_locates_declared_column_segments(bundle):
    doc = _doc(
        {"e": {"index": -1, "scope": {"segment": "ent"}}},
        {"r": "e"},
        segments={"declare": {"ent": {"column": "entity"}}},
    )
    executor = executor_for(
        doc, bundle, base_texts=[PROMPT], extra_columns={"entity": ["Friday"]}
    )
    executor.read_value("r")
    batch = executor._batch("base")  # pyright: ignore[reportPrivateUsage]
    assert batch.prefix_lengths == (0,)
    (run,) = executor._positions("e", batch, "base")  # pyright: ignore[reportPrivateUsage]
    decoded = bundle.tokenizer.decode(batch.input_ids[0, run])
    assert "Friday".endswith(decoded.strip()) and decoded.strip()


# --------------------------------------------------------------------------- #
# atomic under rule 19 and a declared alignment
# --------------------------------------------------------------------------- #


def _write_doc(pos: Any) -> dict[str, Any]:
    return in_order(
        {
            "header": {"protocol_version": "4"},
            "model": {"key": "test", "revision": "main"},
            "data": base_data_section(with_counterfactual=True),
            "method": {
                "intervened_models": {
                    UNWRITTEN: {"input": "counterfactual", "reads": ["v_cf"]},
                    "patched": {"input": "base", "reads": ["out"], "writes": ["patch"]},
                },
                "positions": {"w": pos},
                "sites": {"tap": {"component": "block_output", "layers": [0]}},
                "reads": {
                    "v_cf": {"site": "tap", "pos": "w"},
                    "out": {"site": "tap", "pos": -1},
                },
                "writes": {
                    "patch": {"site": "tap", "pos": "w", "do": {"swap": "v_cf"}}
                },
                "save": [saved("out", "patched", "out.safetensors")],
            },
        }
    )


RAGGED_BASE = ["the cat sat", "the caterpillar sat"]
RAGGED_CF = ["the dog sat", "the doghouse sat"]


def test_a_ragged_atomic_span_write_is_refused_before_any_forward(llama_bundle):
    executor = executor_for(
        _write_doc({"variable": "entity", "atomic": True}),
        llama_bundle,
        base_texts=RAGGED_BASE,
        counterfactual_texts=RAGGED_CF,
        extra_columns={
            "entity": ["cat", "caterpillar"],
            "counterfactual_inputs_variables": [
                {"entity": "dog"},
                {"entity": "doghouse"},
            ],
        },
    )
    tok = llama_bundle.tokenizer
    widths = {
        len(tok.encode(e, add_special_tokens=False)) for e in ["cat", "caterpillar"]
    }
    if len(widths) == 1:
        pytest.skip("this tokenizer gives 'cat' and 'caterpillar' equal widths")
    with pytest.raises(ValidationError) as err:
        executor.read_value("out")
    assert err.value.rule == 19
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_twin_a_fixed_width_atomic_span_write_runs(bundle):
    executor = executor_for(
        _write_doc({"indices": [0, 1], "atomic": True}),
        bundle,
        base_texts=RAGGED_BASE,
        counterfactual_texts=RAGGED_CF,
    )
    assert isinstance(executor.read_value("out"), torch.Tensor)


def test_constituents_of_a_non_atomic_set_are_classified_one_by_one(bundle):
    """A non-atomic `indices` set declared `one_to_one` checks each single
    token address; the same set declared `one_to_many` contradicts every
    constituent and names which."""
    ok = executor_for(
        _write_doc({"indices": [0, 1], "alignment": "one_to_one"}),
        bundle,
        base_texts=RAGGED_BASE,
        counterfactual_texts=RAGGED_CF,
    )
    assert isinstance(ok.read_value("out"), torch.Tensor)
    with pytest.raises(ValidationError) as err:
        executor_for(
            _write_doc({"indices": [0, 1], "alignment": "one_to_many"}),
            bundle,
            base_texts=RAGGED_BASE,
            counterfactual_texts=RAGGED_CF,
        )
    assert err.value.rule == 26  # refused at load: the set is static


# --------------------------------------------------------------------------- #
# the ledger end to end: opt-in, stamped, certified
# --------------------------------------------------------------------------- #


def _run_doc(tmp_path: Path, *, ledger: bool) -> dict[str, Any]:
    doc = {
        "header": {"protocol_version": "4"},
        "model": {"key": TINY_LLAMA, "revision": "main"},
        "data": {"base": {"dataset": "weekdays/data#train", "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "positions": {"ent": {"index": -1, "scope": {"variable": "entity"}}},
            "sites": {"tap": {"component": "block_output", "layers": [0]}},
            "reads": {"r": {"site": "tap", "pos": "ent"}},
            "save": [saved("r", "original", "r.safetensors")],
        },
    }
    if ledger:
        doc["method"]["save"].append(
            {"kind": "location_ledger", "file_path": "ledger.json"}
        )
    return doc


def _run(tmp_path: Path, doc: dict[str, Any], out: Path) -> int:
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(doc))
    return main(
        [
            "run",
            "--engine",
            "auto",
            str(path),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(tmp_path),
            "--out",
            str(out),
            "--record",
        ]
    )


def test_a_run_with_the_entry_writes_the_ledger(tmp_path):
    out = tmp_path / "out"
    assert _run(tmp_path, _run_doc(tmp_path, ledger=True), out) == 0
    rows = json.loads((out / "ledger.json").read_text())
    assert rows and all(set(LEDGER_COLUMNS) <= set(row) for row in rows)
    assert {row["constituent"] for row in rows} == {"ent"}
    assert {row["point"] for row in rows} and all("coords" in row for row in rows)
    receipt = json.loads((out / "protocol.json").read_text())
    kinds = [e.get("kind") for e in receipt["canonical"]["method"]["save"]]
    assert "location_ledger" in kinds


def test_a_run_without_the_entry_writes_no_ledger(tmp_path):
    out = tmp_path / "out"
    assert _run(tmp_path, _run_doc(tmp_path, ledger=False), out) == 0
    assert not (out / "ledger.json").exists()
    assert (out / "r.safetensors").exists()


def _params_doc() -> dict[str, Any]:
    return in_order(
        {
            "header": {"protocol_version": "4"},
            "model": {"key": "test", "revision": "main"},
            "data": base_data_section(with_counterfactual=False),
            "method": {
                "intervened_models": {
                    "steered": {"input": "base", "reads": ["out"], "writes": ["steer"]}
                },
                "positions": {"last": {"index": -1}},
                "sites": {"tap": {"component": "block_output", "layers": [0]}},
                "params": {"c": {"file_path": "c.safetensors"}},
                "reads": {"out": {"site": "tap", "pos": "last"}},
                "writes": {
                    "steer": {
                        "site": "tap",
                        "pos": "last",
                        "do": {"add_scaled": {"op": "c", "alpha": 1.0}},
                    }
                },
                "save": [
                    saved("out", "steered", "out.safetensors"),
                    {"kind": "location_ledger", "file_path": "ledger.json"},
                ],
            },
        }
    )


def test_a_loaded_artifact_gets_the_ledger_of_its_own_rows(gpt2_bundle):
    """A run that loads a parameter fitted on other rows gets the ledger of
    the tokens it selected on its own rows, built before any forward."""
    doc = _params_doc()
    executor = executor_for(
        doc,
        gpt2_bundle,
        base_texts=[PROMPT],
        load_tensors=bundle_loader({"c.safetensors": {"value": torch.zeros(16)}}),
    )
    parsed = parse_document(doc)
    assert wants_ledger(parsed)
    mine = executor.location_ledger()
    other = executor_for(doc, gpt2_bundle, base_texts=[PROMPT + "."]).location_ledger()
    assert mine.records() != other.records()
    assert executor.location_ledger() is mine  # built once, before any forward
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]


def test_a_document_that_does_not_opt_in_gets_no_ledger(gpt2_bundle):
    doc = _doc({"last": {"index": -1}}, {"r": "last"})
    executor = executor_for(doc, gpt2_bundle, base_texts=[PROMPT])
    parsed = parse_document(doc)
    assert not wants_ledger(parsed)
    assert executor._ledger is None  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- #
# the protocol layer's ledger is the executor's, row for row
# --------------------------------------------------------------------------- #


def test_the_protocol_layer_builds_the_same_ledger_as_the_executor(gpt2_bundle):
    """``pipeline.resolve_positions`` builds the ledger before any weights load
    with the registry's answer to "does this tap have a position axis"; the
    executor builds it from the resolved tap. Same functions, same rows."""
    from causalab.protocol.positions.resolve import (
        build_ledger,
        encode_roles,
        resolve_positions,
    )
    from causalab.protocol.registry import component_shape

    doc = _doc(
        {"last": {"index": -1}, "both": {"indices": [0, -1]}},
        {"r": "last", "s": "both"},
        ledger=True,
    )
    executor = executor_for(doc, gpt2_bundle, base_texts=[PROMPT, PROMPT + " maybe"])
    parsed = parse_document(doc)
    mine = executor.location_ledger()
    frames = encode_roles(
        gpt2_bundle.tokenizer, parsed, executor.role_rows, executor.role_fields
    )
    positions = resolve_positions(
        parsed, frames, executor.role_rows, executor.role_fields
    )
    theirs = build_ledger(
        parsed,
        positions,
        gpt2_bundle.tokenizer,
        has_positions=lambda site: component_shape(
            gpt2_bundle.info, str(parsed.sites[site].component)
        ).has_contract_form,
    )
    assert theirs.records() == mine.records()
    assert len(mine) == 2 * 3  # two rows × (last, both[0], both[1])


def test_an_executor_handed_the_protocols_positions_resolves_nothing_itself(
    gpt2_bundle,
):
    """The engine hands ``StepPositions`` in; the executor's frames are the
    protocol's frames wrapped, its positions are read off the handed object,
    and the values it reads equal an executor that resolved its own."""
    from causalab.neural.shared.executor import RaggedValue
    from causalab.protocol.positions.resolve import encode_roles, resolve_positions

    doc = _doc(
        {"last": {"index": -1}, "ent": {"column": "entity"}}, {"r": "last", "s": "ent"}
    )
    texts = [PROMPT, "Alice gave Bob a book."]
    entities = ["Friday", "Bob"]
    own = executor_for(
        doc, gpt2_bundle, base_texts=texts, extra_columns={"entity": entities}
    )
    parsed = parse_document(doc)
    frames = encode_roles(gpt2_bundle.tokenizer, parsed, own.role_rows, own.role_fields)
    positions = resolve_positions(parsed, frames, own.role_rows, own.role_fields)
    handed = executor_for(
        doc, gpt2_bundle, base_texts=texts, extra_columns={"entity": entities}
    )
    handed._resolved = positions  # pyright: ignore[reportPrivateUsage]
    assert handed._batch("base").texts == frames["base"].texts  # pyright: ignore[reportPrivateUsage]
    assert handed._batch("base").input_ids.tolist() == [  # pyright: ignore[reportPrivateUsage]
        list(row) for row in frames["base"].token_ids
    ]
    for name in ("r", "s"):
        mine, theirs = handed.read_value(name), own.read_value(name)
        if isinstance(mine, RaggedValue):  # `ent` is as wide as each row's value
            assert isinstance(theirs, RaggedValue) and mine.widths == theirs.widths
            assert torch.equal(mine.flat, theirs.flat)
        else:
            assert torch.equal(mine, theirs)
    assert handed._step_positions() is positions  # pyright: ignore[reportPrivateUsage]
    # every address the executor asked for was one the protocol had resolved
    assert set(positions.addresses) >= {("last", "base"), ("ent", "base")}


def test_a_frame_the_bundles_tokenizer_does_not_reproduce_is_refused(
    gpt2_bundle, llama_bundle
):
    """The positions handed in are indices into a frame; a frame another
    tokenizer produced would make every one of them address the wrong token
    without a word — so the executor re-encodes (a memo hit for the same
    tokenizer) and refuses a mismatch by name before any forward."""
    from causalab.protocol.positions.resolve import encode_roles, resolve_positions

    doc = _doc({"last": {"index": -1}}, {"r": "last"})
    texts = [PROMPT, "Alice gave Bob a book."]
    executor = executor_for(doc, gpt2_bundle, base_texts=texts)
    parsed = parse_document(doc)
    frames = encode_roles(
        llama_bundle.tokenizer, parsed, executor.role_rows, executor.role_fields
    )
    executor._resolved = resolve_positions(  # pyright: ignore[reportPrivateUsage]
        parsed, frames, executor.role_rows, executor.role_fields
    )
    with pytest.raises(ProtocolError, match="does not reproduce"):
        executor.read_value("r")
    assert not executor._groups_run  # pyright: ignore[reportPrivateUsage]
