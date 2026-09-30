"""PositionFrame resolution (spec §2.3, §8): index/variable/span, scope and
relative_to, left-pad frame math, and the out-of-bounds refusal contract
(a stale position must fail legibly, never address the wrong token)."""

from __future__ import annotations

import pytest

import torch

from causalab.neural.shared.encoding import (
    Continuation,
    encode,
    resolve_position,
    resolve_steps,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import PositionSpec

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def tokenizer():
    from causalab.neural.engines.pytorch_hooks.loading import load_model

    return load_model(TINY_LLAMA).tokenizer


def test_negative_index_is_the_last_real_token(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        (pos,) = resolve_position(PositionSpec(index=-1), batch, row)
        assert pos == batch.padded_len - 1  # left padding right-aligns content


def test_nonnegative_index_counts_from_content_start(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    # row 0 is padded on the left; its index 0 is its first real token
    (pos,) = resolve_position(PositionSpec(index=0), batch, 0)
    assert pos == batch.content_start(0)
    assert batch.attention_mask[0, pos] == 1
    if pos > 0:
        assert batch.attention_mask[0, pos - 1] == 0


def test_variable_window_covers_the_value(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    run = resolve_position(
        PositionSpec(variable="subject"),
        batch,
        0,
        dataset_row={"input": text, "subject": "Thursday"},
        field="input",
    )
    decoded = tokenizer.decode(batch.input_ids[0, run])
    assert "Thursday" in decoded


def test_ambiguous_variable_occurrence_refuses(tokenizer):
    """Several occurrences is the ``ambiguous`` cardinality (spec §2.3): the
    refusal carries its reason code rather than being a bare P2."""
    text = "day after day after day"
    batch = encode(tokenizer, [text])
    with pytest.raises(ProtocolError) as err:
        resolve_position(
            PositionSpec(variable="word"),
            batch,
            0,
            dataset_row={"input": text, "word": "day"},
            field="input",
        )
    assert "occurs" in str(err.value)
    assert err.value.reason == "alignment_ambiguous"


def test_absent_variable_occurrence_refuses_as_missing(tokenizer):
    """Zero occurrences is the ``absent`` cardinality — ``alignment_missing``."""
    text = "one two three"
    batch = encode(tokenizer, [text])
    with pytest.raises(ProtocolError) as err:
        resolve_position(
            PositionSpec(variable="word"),
            batch,
            0,
            dataset_row={"input": text, "word": "four"},
            field="input",
        )
    assert "occurs 0 times" in str(err.value)
    assert err.value.reason == "alignment_missing"


def test_scope_indexes_inside_the_variable_window(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    row = {"input": text, "subject": "Thursday"}
    window = resolve_position(
        PositionSpec(variable="subject"), batch, 0, dataset_row=row, field="input"
    )
    (first,) = resolve_position(
        PositionSpec(index=0, scope="subject"), batch, 0, dataset_row=row, field="input"
    )
    (last,) = resolve_position(
        PositionSpec(index=-1, scope="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    assert first == window[0] and last == window[-1]


def test_relative_to_offsets_from_the_window(tokenizer):
    text = "If today is Thursday, tomorrow is"
    batch = encode(tokenizer, [text])
    row = {"input": text, "subject": "Thursday"}
    window = resolve_position(
        PositionSpec(variable="subject"), batch, 0, dataset_row=row, field="input"
    )
    (after,) = resolve_position(
        PositionSpec(index=1, relative_to="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    (before,) = resolve_position(
        PositionSpec(index=-1, relative_to="subject"),
        batch,
        0,
        dataset_row=row,
        field="input",
    )
    assert after == window[-1] + 1 and before == window[0] - 1


def test_span_is_a_content_frame_window(tokenizer):
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    window = resolve_position(PositionSpec(span=(0, 2)), batch, 0)
    start = batch.content_start(0)
    assert window == [start, start + 1]


def test_all_is_every_content_token(tokenizer):
    """``{"all": true}`` is the row's real tokens: contiguous from the row's
    content start to the padded end, nothing under the left pad."""
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        positions = resolve_position(PositionSpec(all=True), batch, row)
        start = batch.content_start(row)
        assert positions == list(range(start, batch.padded_len))
        assert all(batch.attention_mask[row, p] == 1 for p in positions)
    # the short row is genuinely shorter — the ragged case reads must carry
    assert len(resolve_position(PositionSpec(all=True), batch, 0)) < len(
        resolve_position(PositionSpec(all=True), batch, 1)
    )


def test_all_needs_no_dataset_row(tokenizer):
    """Unlike a variable window, an all spec is frame-only — it resolves
    without the dataset row, which is what makes it usable in any document."""
    batch = encode(tokenizer, ["one two three"])
    positions = resolve_position(PositionSpec(all=True), batch, 0)
    assert positions == list(range(batch.content_start(0), batch.padded_len))


def test_all_covers_the_index_forms(tokenizer):
    """The endpoints agree with the index spellings they generalize."""
    batch = encode(tokenizer, ["one two three", "a much longer sentence right here"])
    for row in range(2):
        positions = resolve_position(PositionSpec(all=True), batch, row)
        (first,) = resolve_position(PositionSpec(index=0), batch, row)
        (last,) = resolve_position(PositionSpec(index=-1), batch, row)
        assert positions[0] == first and positions[-1] == last


def test_out_of_bounds_refuses(tokenizer):
    batch = encode(tokenizer, ["one two three"])
    with pytest.raises(ProtocolError) as err:
        resolve_position(PositionSpec(index=500), batch, 0)
    assert "out of bounds" in str(err.value)


# the continuation frame (§2.3) ---------------------------------------------- #


def _continuation(widths: tuple[int, ...], steps: int = 4) -> Continuation:
    """A decode of ``steps`` steps whose rows stopped at ``widths``."""
    ids = torch.arange(len(widths) * steps).reshape(len(widths), steps)
    return Continuation(token_ids=ids, widths=widths)


def test_all_steps_is_the_rows_real_width():
    cont = _continuation((4, 2))
    spec = PositionSpec(generated={"max_new_tokens": 4}, all=True)
    assert resolve_steps(spec, cont, 0) == [0, 1, 2, 3]
    assert resolve_steps(spec, cont, 1) == [0, 1]


def test_last_step_is_the_last_real_token():
    """``index: -1`` on a row that stopped early means *its* last generated
    token, not the batch's last step — that is what "the final generated
    token" has to mean when rows end at different places."""
    cont = _continuation((4, 2))
    spec = PositionSpec(generated={"max_new_tokens": 4}, index=-1)
    assert resolve_steps(spec, cont, 0) == [3]
    assert resolve_steps(spec, cont, 1) == [1]


def test_a_row_that_generated_nothing_contributes_no_positions():
    """An immediate EOS is a result, not an authoring error: the row drops
    out of the read instead of failing the run."""
    cont = _continuation((3, 0))
    for spec in (
        PositionSpec(generated={"max_new_tokens": 3}, all=True),
        PositionSpec(generated={"max_new_tokens": 3}, index=-1),
        PositionSpec(generated={"max_new_tokens": 3}, index=0),
    ):
        assert resolve_steps(spec, cont, 1) == []


def test_a_window_past_a_rows_end_clips():
    cont = _continuation((4, 2))
    spec = PositionSpec(generated={"max_new_tokens": 4}, span=(0, 3))
    assert resolve_steps(spec, cont, 0) == [0, 1, 2]
    assert resolve_steps(spec, cont, 1) == [0, 1]


def test_an_index_past_a_rows_end_drops_it():
    cont = _continuation((4, 2))
    spec = PositionSpec(generated={"max_new_tokens": 4}, index=3)
    assert resolve_steps(spec, cont, 0) == [3]
    assert resolve_steps(spec, cont, 1) == []


def test_real_ids_stop_at_the_rows_width():
    cont = _continuation((4, 2))
    assert cont.real_ids(1) == [4, 5]
    assert cont.steps == 4


def test_a_generated_position_without_a_decode_refuses():
    """The frame does not exist until the model has run, so asking for it
    outside a decode is an engine misuse, not a document error."""
    batch = None
    spec = PositionSpec(generated={"max_new_tokens": 4}, index=-1)
    with pytest.raises(ProtocolError, match="does not exist until"):
        resolve_position(spec, batch, 0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# the `variable` anchor inside the continuation (§2.3)
# --------------------------------------------------------------------------- #


def _said(text: str, pieces: list[str]) -> Continuation:
    """A one-row decode whose tokens are ``pieces`` in order, with the char
    spans incremental detokenization would have produced."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    assert "".join(pieces) == text
    return Continuation(
        token_ids=torch.arange(len(pieces)).reshape(1, len(pieces)),
        widths=(len(pieces),),
        texts=(text,),
        offsets=(tuple(offsets),),
    )


def _variable_spec() -> PositionSpec:
    return PositionSpec(generated={"max_new_tokens": 8}, variable="answer")


def test_variable_covers_the_tokens_that_said_it():
    cont = _said(" the answer is Thursday", [" the", " answer", " is", " Thursday"])
    steps = resolve_steps(
        _variable_spec(),
        cont,
        0,
        dataset_row={"answer": "Thursday"},
        field="input",
    )
    assert steps == [3]


def test_variable_the_model_never_said_yields_no_positions():
    """Zero occurrences is the experiment's answer, not an error: the row
    contributes nothing and the run continues."""
    cont = _said(" the answer is Friday", [" the", " answer", " is", " Friday"])
    steps = resolve_steps(
        _variable_spec(),
        cont,
        0,
        dataset_row={"answer": "Thursday"},
        field="input",
    )
    assert steps == []


def test_variable_takes_the_first_occurrence():
    """The prompt side demands exactly one occurrence; a generation may
    repeat itself as a matter of course, so repetition must not crash."""
    cont = _said(" Thursday, Thursday", [" Thursday", ",", " Thursday"])
    steps = resolve_steps(
        _variable_spec(),
        cont,
        0,
        dataset_row={"answer": "Thursday"},
        field="input",
    )
    assert steps == [0]


def test_variable_spanning_a_merge_covers_every_token_it_touches():
    """A sentencepiece-style split is why the spans are built as the tokens
    arrive: the match starts inside one piece and ends inside another, and
    both pieces produced it."""
    cont = _said(" Thursday", [" Th", "urs", "day"])
    steps = resolve_steps(
        _variable_spec(),
        cont,
        0,
        dataset_row={"answer": "hursda"},
        field="input",
    )
    assert steps == [0, 1, 2]


def test_variable_does_not_reach_past_a_rows_width():
    """A row that stopped early cannot have said anything in the steps after
    its EOS, even though the batch decoded them."""
    cont = _said(" a Thursday", [" a", " Thursday"])
    narrowed = Continuation(
        token_ids=cont.token_ids,
        widths=(1,),
        texts=cont.texts,
        offsets=cont.offsets,
    )
    steps = resolve_steps(
        _variable_spec(),
        narrowed,
        0,
        dataset_row={"answer": "Thursday"},
        field="input",
    )
    assert steps == []


def test_variable_without_its_row_refuses():
    cont = _said(" Thursday", [" Thursday"])
    with pytest.raises(ProtocolError, match="needs its dataset row"):
        resolve_steps(_variable_spec(), cont, 0)


# --------------------------------------------------------------------------- #
#  the chat-template guard (§2.3): v1 has no chat field, so the rendered        #
#  template is the data — and the encoder must not silently prefix it twice     #
# --------------------------------------------------------------------------- #


def test_a_text_that_already_carries_bos_is_refused(tokenizer):
    """The double-BOS trap, which is a wrong number rather than a crash.

    `encode` adds special tokens as the tokenizer defines them, and the
    workaround for having no chat-template path is to bake the *rendered*
    template into the dataset's `input` column. A rendered template opens with
    BOS, so on a BOS-adding tokenizer every position in the row shifts by one
    and nothing raises. Refused rather than stripped: the text is the
    document's data, and its content digest is part of the canonical form.
    """
    assert tokenizer.bos_token_id is not None  # the premise
    with pytest.raises(ProtocolError) as err:
        encode(tokenizer, [tokenizer.bos_token + "already prefixed"])
    assert "twice" in str(err.value)
    assert "no chat field" in str(err.value)


def test_a_plain_text_still_encodes(tokenizer):
    """The guard is exactly two BOS at the content start — one is the
    tokenizer doing its job."""
    batch = encode(tokenizer, ["a plain sentence", "another one"])
    start = batch.content_start(0) - batch.prefix_lengths[0]
    assert int(batch.input_ids[0, start]) == tokenizer.bos_token_id
    assert int(batch.input_ids[0, start + 1]) != tokenizer.bos_token_id


def test_prefix_lengths_are_zero_in_the_plain_frame(tokenizer):
    """The spec's `n >= 0 is rebased past any chat prefix` rule is an identity
    for a plain document — one with no `segments` section — and §2.3 says so.
    Pinned here so the claim and the code cannot drift apart silently: the one
    thing that makes it non-zero is `segments.frame: chat` (§2.2.1,
    `framing.encode_framed`, `test_location_ledger.py`), never `encode` itself."""
    batch = encode(tokenizer, ["one two three", "four five"])
    assert batch.prefix_lengths == (0, 0)
    assert batch.segments == ()


def test_a_segment_anchor_resolves_inside_the_located_span(tokenizer):
    """A declared segment is located by the frame as char spans on the batch
    (spec §2.2.1) and resolved here through the same offset mapping a
    variable uses — `index` inside it, `relative_to` beside it."""
    text = "one two three four"
    span = (text.index("two"), text.index("two") + len("two three"))
    batch = encode(tokenizer, [text], segments=[{"middle": (span,)}])
    run = resolve_position(
        PositionSpec(index=-1, scope="middle", anchor_source="segment"), batch, 0
    )
    assert len(run) == 1
    assert "three" in tokenizer.decode(batch.input_ids[0, run])
    after = resolve_position(
        PositionSpec(index=1, relative_to="middle", anchor_source="segment"), batch, 0
    )
    assert "four" in tokenizer.decode(batch.input_ids[0, after])


def test_an_absent_segment_refuses_as_missing(tokenizer):
    batch = encode(tokenizer, ["one two"], segments=[{"gone": ()}])
    with pytest.raises(ProtocolError) as err:
        resolve_position(
            PositionSpec(index=0, scope="gone", anchor_source="segment"), batch, 0
        )
    assert err.value.reason == "alignment_missing"
