"""Text-framed recorded sequences must retain their exact predictive prefixes."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st
from tokenizers import Tokenizer, decoders, models, normalizers, processors
from transformers import PreTrainedTokenizerFast

from causalab.analysis.sequences import add_readouts, prepare_sequence
from causalab.protocol.schema import PROTOCOL_VERSION
from causalab.protocol.positions.encoding import encode, resolve_position
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.schema import parse_document

pytestmark = pytest.mark.unit


def _tokenizer(*, bos: bool = False, eos: bool = False) -> PreTrainedTokenizerFast:
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["[PAD]", "[UNK]", "[EOS]", "x", "b", "c", "bc", "y", "z", "[BOS]", "B"]
        )
    }
    backend = Tokenizer(models.BPE(vocabulary, merges=[("b", "c")], unk_token="[UNK]"))
    backend.decoder = decoders.Fuse()
    backend.normalizer = normalizers.Lowercase()
    if bos or eos:
        template = " ".join(
            [*(["[BOS]"] if bos else []), "$A", *(["[EOS]"] if eos else [])]
        )
        backend.post_processor = processors.TemplateProcessing(
            single=template, special_tokens=[("[BOS]", 9), ("[EOS]", 2)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        padding_side="left",
    )


@pytest.mark.parametrize("completion", [[4, 5], [10]])
def test_recorded_ids_that_change_when_encoded_are_refused(
    completion: list[int],
) -> None:
    """Both a BPE length change and same-length normalization change the experiment."""
    with pytest.raises(ProtocolError, match="recorded token IDs") as error:
        prepare_sequence(
            _tokenizer(),
            "xyz",
            completion,
            example_id="recorded",
            split="eval",
            prefix_condition="baseline_generated",
        )
    assert error.value.code == "P2"
    assert error.value.path == "completion"


def test_added_trailing_special_cannot_move_the_recorded_prefix() -> None:
    with pytest.raises(ProtocolError, match="recorded token IDs"):
        prepare_sequence(
            _tokenizer(eos=True), "xyz", [4], example_id="eos", split="eval"
        )


@pytest.mark.parametrize("bos", [False, True])
def test_exact_recorded_ids_keep_specials_and_prediction_positions(bos: bool) -> None:
    tokenizer = _tokenizer(bos=bos)
    completion = [6, 4, 2]
    row = prepare_sequence(
        tokenizer, "xyz", completion, example_id="exact", split="eval"
    )
    ids = tokenizer("xyz")["input_ids"] + completion
    assert tokenizer(row["input"])["input_ids"] == ids
    raw = add_readouts(
        {
            "header": {"protocol_version": PROTOCOL_VERSION},
            "model": {"key": "test/sequence-llama", "revision": "local"},
            "data": {"base": {"dataset": "rows", "field": "input"}},
            "method": {},
        },
        len(completion),
        model="original",
        top_k=2,
    )
    doc = parse_document(raw)
    frame = encode(tokenizer, [row["input"]])
    for i, target in enumerate(row["targets"]):
        assert resolve_position(doc.reads[f"sequence_{i}"].pos, frame, 0) == [
            target["prediction_position"]
        ]
        assert ids[target["token_position"]] == target["token_id"]
    assert row["targets"][-1]["is_eos"]


def test_text_completion_uses_its_contextual_tokenization() -> None:
    row = prepare_sequence(_tokenizer(), "xyz", "bc", example_id="text", split="eval")
    assert row["output_length"] == 1
    assert row["output_0"] == 6


@settings(max_examples=60, deadline=None)
@given(st.lists(st.sampled_from([4, 5, 6, 10]), min_size=1, max_size=8))
def test_recorded_sequence_acceptance_requires_identical_ids(
    completion: list[int],
) -> None:
    tokenizer = _tokenizer()
    expected = tokenizer("xyz")["input_ids"] + completion
    text = "xyz" + tokenizer.decode(completion, clean_up_tokenization_spaces=False)
    if tokenizer(text)["input_ids"] != expected:
        with pytest.raises(ProtocolError, match="recorded token IDs"):
            prepare_sequence(
                tokenizer, "xyz", completion, example_id="property", split="eval"
            )
    else:
        row = prepare_sequence(
            tokenizer, "xyz", completion, example_id="property", split="eval"
        )
        assert tokenizer(row["input"])["input_ids"] == expected
        assert [target["token_id"] for target in row["targets"]] == completion
