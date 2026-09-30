"""``encode`` tokenizes one ``(texts, specials)`` once per tokenizer and
builds every later frame from the memoized host output.

Every point of a campaign encodes the same rows of the same roles into its
own executor, so a ten-point scan paid ten identical tokenizer calls per role.
The memo is keyed by the tokenizer (weakly), the texts and the tokenizer state
the padded output depends on; what it hands out is the same ids, offsets and
first-real cache, in **fresh** tensors — one executor writing rows into its
frame in place must never reach another's.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false

import gc
import weakref
from typing import Any

import pytest
import torch

from causalab.neural.shared import encoding
from causalab.neural.shared.encoding import encode

pytestmark = pytest.mark.unit


class _Tokenizer:
    """A left-padding tokenizer that counts its calls: one id per character."""

    bos_token_id = None
    padding_side = "left"
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        self.calls += 1
        width = max(len(t) for t in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros(len(texts), width, dtype=torch.long)
        offsets = torch.zeros(len(texts), width, 2, dtype=torch.long)
        for row, text in enumerate(texts):
            pad = width - len(text)
            for j, char in enumerate(text):
                ids[row, pad + j] = ord(char)
                mask[row, pad + j] = 1
                offsets[row, pad + j] = torch.tensor([j, j + 1])
        return {"input_ids": ids, "attention_mask": mask, "offset_mapping": offsets}


def test_the_same_texts_tokenize_once_and_frame_equally() -> None:
    tokenizer = _Tokenizer()
    first = encode(tokenizer, ["ab", "cde"])
    second = encode(tokenizer, ["ab", "cde"])
    assert tokenizer.calls == 1
    assert first.texts == second.texts == ("ab", "cde")
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.attention_mask, second.attention_mask)
    assert first.offset_mapping == second.offset_mapping
    assert first.prefix_lengths == second.prefix_lengths == (0, 0)
    assert first.first_reals == second.first_reals == (1, 0)


def test_the_frames_own_their_tensors() -> None:
    tokenizer = _Tokenizer()
    first = encode(tokenizer, ["ab", "cde"])
    second = encode(tokenizer, ["ab", "cde"])
    assert first.input_ids is not second.input_ids
    assert first.attention_mask is not second.attention_mask
    first.input_ids[0, -1] = 7  # a graph worker staging rows into its frame
    assert int(second.input_ids[0, -1]) == ord("b")
    assert int(encode(tokenizer, ["ab", "cde"]).input_ids[0, -1]) == ord("b")


def test_other_texts_or_specials_or_padding_state_tokenize_again() -> None:
    tokenizer = _Tokenizer()
    encode(tokenizer, ["ab"])
    encode(tokenizer, ["ab", "c"])
    assert tokenizer.calls == 2
    encode(tokenizer, ["ab"], add_special_tokens=False)
    assert tokenizer.calls == 3
    tokenizer.padding_side = "right"
    encode(tokenizer, ["ab"])
    assert tokenizer.calls == 4
    # each of those is memoized in turn, under the state it was made in
    encode(tokenizer, ["ab"])
    tokenizer.padding_side = "left"
    encode(tokenizer, ["ab"])
    encode(tokenizer, ["ab", "c"])
    encode(tokenizer, ["ab"], add_special_tokens=False)
    assert tokenizer.calls == 4


def test_two_tokenizers_do_not_share_entries() -> None:
    one, two = _Tokenizer(), _Tokenizer()
    encode(one, ["ab"])
    encode(two, ["ab"])
    assert one.calls == two.calls == 1


def test_a_released_tokenizer_takes_its_entries_with_it() -> None:
    tokenizer = _Tokenizer()
    encode(tokenizer, ["ab"])
    assert tokenizer in encoding._TOKENIZED
    alive = weakref.ref(tokenizer)
    del tokenizer
    gc.collect()
    # the weak key is gone with its owner; other tests' tokenizers may remain
    assert alive() is None
    assert not any(ref() is None for ref in encoding._TOKENIZED.keyrefs())


def test_the_memo_is_bounded_per_tokenizer() -> None:
    tokenizer = _Tokenizer()
    for i in range(encoding._TOKENIZED_PER_TOKENIZER + 5):
        encode(tokenizer, [f"t{i}"])
    assert len(encoding._TOKENIZED[tokenizer]) == encoding._TOKENIZED_PER_TOKENIZER
    # the oldest entries were evicted, the newest kept
    encode(tokenizer, ["t0"])
    assert tokenizer.calls == encoding._TOKENIZED_PER_TOKENIZER + 6
    encode(tokenizer, [f"t{encoding._TOKENIZED_PER_TOKENIZER + 4}"])
    assert tokenizer.calls == encoding._TOKENIZED_PER_TOKENIZER + 6


def test_a_hit_keeps_its_entry_recent() -> None:
    """Least recently *used* goes first: the frame a step keeps coming back
    to survives however many one-off batches pass through the memo."""
    tokenizer = _Tokenizer()
    encode(tokenizer, ["keep"])
    for i in range(encoding._TOKENIZED_PER_TOKENIZER - 1):
        encode(tokenizer, [f"t{i}"])
    assert len(encoding._TOKENIZED[tokenizer]) == encoding._TOKENIZED_PER_TOKENIZER
    encode(tokenizer, ["keep"])  # a hit — refreshed, not evicted next
    calls = tokenizer.calls
    encode(tokenizer, ["one more"])  # evicts t0, the least recently used
    encode(tokenizer, ["keep"])
    assert tokenizer.calls == calls + 1
    encode(tokenizer, ["t0"])
    assert tokenizer.calls == calls + 2


def test_a_refused_batch_is_refused_again() -> None:
    # the refusals are derived from the memoized output on every call, so a
    # batch refused once is refused on the hit as well
    tokenizer = _Tokenizer()
    with pytest.raises(Exception):
        encode(tokenizer, ["ab", ""])  # a row with no token at all
    with pytest.raises(Exception):
        encode(tokenizer, ["ab", ""])
    assert tokenizer.calls == 1
