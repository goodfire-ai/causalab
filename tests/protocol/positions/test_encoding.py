"""The position frame is the protocol layer's: what
``protocol/positions/encoding.py`` resolves on a frame of Python ints is what
the engine resolves on the same frame with device tensors beside it — the
two frames agree field for field, every resolver gives the same indices on
both, and the protocol package imports without torch.

Real tokenizers, both fixtures: the byte-level BPE (tiny GPT-2, the real
GPT-2 vocabulary) and the sentencepiece one (tiny Llama, with a BOS) — the
surface forms a stand-in tokenizer cannot show.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.shared import encoding as engine
from causalab.protocol.positions import encoding as protocol
from causalab.protocol.positions.encoding import PositionFrame
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PositionSpec, SpanSpec

from tests._helpers.tiny import TINY_RANDOM_GPT2_MODEL_NAME, TINY_RANDOM_MODEL_NAME

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]

TEXTS = (
    "If today is Friday, tomorrow is",
    "Alice gave Bob a book yesterday.",
    "The capital of France is",
)
ROWS = (
    {"input": TEXTS[0], "entity": "Friday", "input_variables": {"day": "Friday"}},
    {"input": TEXTS[1], "entity": "Bob", "input_variables": {"day": "Alice"}},
    {"input": TEXTS[2], "entity": "France", "input_variables": {"day": "capital"}},
)
SPECS: tuple[Any, ...] = (
    PositionSpec(index=-1),
    PositionSpec(index=0),
    PositionSpec(index=2),
    PositionSpec(all=True),
    PositionSpec(span=(1, 3)),
    PositionSpec(variable="day"),
    PositionSpec(column="entity"),
    PositionSpec(index=0, scope="day"),
    PositionSpec(index=1, relative_to="day"),
    PositionSpec(span=(0, 1), scope="entity", anchor_source="column"),
    SpanSpec(after=PositionSpec(variable="day")),
    SpanSpec(union=(PositionSpec(index=-1), PositionSpec(variable="day"))),
    SpanSpec(indices=(0, -1), atomic=True),
)


@pytest.fixture(scope="module", params=["gpt2", "llama"])
def tokenizer(request: pytest.FixtureRequest) -> Any:
    from causalab.io.tokenizer import load_tokenizer

    name = (
        TINY_RANDOM_GPT2_MODEL_NAME
        if request.param == "gpt2"
        else TINY_RANDOM_MODEL_NAME
    )
    return load_tokenizer(name)


def test_the_positions_package_imports_without_torch() -> None:
    """The protocol layer stays torch-free (the protocol layer in
    ``docs/intervention_protocol_internals.md``): the whole package —
    encoding, framing, spans, alignment, roles, ledger, the resolver — in a
    fresh interpreter, then ``torch`` is not in ``sys.modules``."""
    probe = (
        "import sys; import causalab.protocol.positions.resolve, "
        "causalab.protocol.positions.roles, causalab.protocol.positions.framing; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", out.stdout


def test_the_protocol_frame_and_the_engine_frame_agree_field_for_field(
    tokenizer: Any,
) -> None:
    frame = protocol.encode(tokenizer, TEXTS)
    batch = engine.encode(tokenizer, TEXTS)
    assert isinstance(frame, PositionFrame)
    assert batch.input_ids.tolist() == [list(row) for row in frame.token_ids]
    assert batch.attention_mask.tolist() == [list(row) for row in frame.attention_mask]
    assert batch.offset_mapping == frame.offset_mapping
    assert batch.prefix_lengths == frame.prefix_lengths == (0, 0, 0)
    assert batch.first_reals == frame.first_reals
    assert batch.padded_len == frame.padded_len
    assert batch.texts == frame.texts == TEXTS
    for row in range(len(TEXTS)):
        assert batch.row_ids(row) == frame.row_ids(row)
        assert batch.content_start(row) == frame.content_start(row)
    # the engine frame is the protocol frame wrapped: same object graph
    assert engine.EncodedBatch.from_frame(frame).offset_mapping is frame.offset_mapping


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: repr(s)[:60])
def test_every_resolver_gives_the_same_indices_on_both_frames(
    tokenizer: Any, spec: Any
) -> None:
    frame = protocol.encode(tokenizer, TEXTS)
    batch = engine.encode(tokenizer, TEXTS)
    for row, dataset_row in enumerate(ROWS):
        kwargs = {"dataset_row": dataset_row, "field": "input"}
        mine = protocol.resolve_position(spec, frame, row, **kwargs)
        theirs = engine.resolve_position(spec, batch, row, **kwargs)
        assert mine == theirs, (spec, row)
        assert protocol.candidate_runs(
            spec, frame, row, **kwargs
        ) == protocol.candidate_runs(spec, batch, row, **kwargs)
        assert protocol.constituent_candidate_runs(
            spec, frame, row, **kwargs
        ) == protocol.constituent_candidate_runs(spec, batch, row, **kwargs)
        # every index addresses a real token of the row, never padding
        for index in mine:
            assert frame.attention_mask[row][index] == 1


def test_a_selection_of_the_frame_keeps_its_indices(tokenizer: Any) -> None:
    frame = protocol.encode(tokenizer, TEXTS)
    batch = engine.encode(tokenizer, TEXTS)
    picked = frame.select([2, 0])
    selected = batch.select([2, 0])
    assert picked.token_ids == tuple(tuple(r) for r in selected.input_ids.tolist())
    assert (
        picked.first_reals
        == selected.first_reals
        == (frame.first_reals[2], frame.first_reals[0])
    )
    assert picked.padded_len == frame.padded_len
    spec = PositionSpec(variable="day")
    assert protocol.resolve_position(
        spec, picked, 0, dataset_row=ROWS[2], field="input"
    ) == protocol.resolve_position(spec, frame, 2, dataset_row=ROWS[2], field="input")


def test_a_generated_spec_is_the_engines_and_not_this_frames(tokenizer: Any) -> None:
    frame = protocol.encode(tokenizer, TEXTS)
    with pytest.raises(ProtocolError, match="continuation"):
        protocol.resolve_position(
            PositionSpec(index=-1, generated={"max_new_tokens": 2}), frame, 0
        )


def test_the_refusals_read_lists_and_host_tensors_alike(tokenizer: Any) -> None:
    """``refuse_empty_rows`` / ``refuse_double_bos`` take the mask and ids as
    rows of ints or as the host tensors a caller already holds."""
    frame = protocol.encode(tokenizer, TEXTS)
    protocol.refuse_empty_rows(TEXTS, frame.attention_mask)
    protocol.refuse_empty_rows(TEXTS, torch.tensor(frame.attention_mask))
    protocol.refuse_double_bos(tokenizer, frame.token_ids, frame.attention_mask)
    protocol.refuse_double_bos(
        tokenizer, torch.tensor(frame.token_ids), torch.tensor(frame.attention_mask)
    )
    empty_mask = tuple(tuple(0 for _ in row) for row in frame.attention_mask)
    with pytest.raises(ProtocolError, match="encodes to no token"):
        protocol.refuse_empty_rows(TEXTS, empty_mask)
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        doubled = tuple(
            tuple(
                bos if j in (frame.first_reals[i], frame.first_reals[i] + 1) else t
                for j, t in enumerate(row)
            )
            for i, row in enumerate(frame.token_ids)
        )
        with pytest.raises(ProtocolError, match="twice"):
            protocol.refuse_double_bos(tokenizer, doubled, frame.attention_mask)


def test_first_real_indices_agree_between_the_two_layers(tokenizer: Any) -> None:
    frame = protocol.encode(tokenizer, TEXTS)
    assert protocol.first_real_indices(
        frame.attention_mask
    ) == engine.first_real_indices(torch.tensor(frame.attention_mask))
    assert protocol.first_real_indices(((0, 0, 0),)) == (0,)
