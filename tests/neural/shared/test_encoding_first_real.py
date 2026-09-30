"""The first-real index of every row is read off the device **once** per
frame (``EncodedBatch.first_reals``).

``first_real(row)`` used to be ``argmax(attention_mask[row]).item()`` — one
device→host round trip per row per position resolution, and the cohort forward
resolves positions for every read and every write at every layer: about 270
synchronizations per forward on the A3B DAS step, while the model itself
issues none. The cache is the same integers, computed for the whole batch in
one reduction, and every constructor path — ``encode``, ``select``, the
cohort's frame concatenation, ``dataclasses.replace`` of another field — keeps
it consistent with the mask.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.cohort import _concat_frames
from causalab.neural.shared.encoding import (
    EncodedBatch,
    first_real_indices,
    refuse_empty_rows,
)
from causalab.protocol.rules.errors import ProtocolError


def _legacy_first_real(mask: torch.Tensor, row: int) -> int:
    """The expression the cache replaces, kept here as the oracle."""
    return int(torch.argmax(mask[row].int()).item())


def _frame(mask: torch.Tensor, **overrides: object) -> EncodedBatch:
    rows, width = mask.shape
    fields: dict[str, object] = dict(
        texts=tuple(f"row {i}" for i in range(rows)),
        input_ids=torch.arange(rows * width).reshape(rows, width),
        attention_mask=mask,
        offset_mapping=tuple(
            tuple((j, j + 1) for j in range(width)) for _ in range(rows)
        ),
        prefix_lengths=tuple(0 for _ in range(rows)),
    )
    fields.update(overrides)
    return EncodedBatch(**fields)  # type: ignore[arg-type]


@st.composite
def left_padded_masks(draw: st.DrawFn) -> torch.Tensor:
    """A left-padded attention mask: per row, some zeros then at least one
    one — what a left-padding tokenizer emits."""
    width = draw(st.integers(min_value=1, max_value=12))
    pads = draw(
        st.lists(st.integers(min_value=0, max_value=width - 1), min_size=1, max_size=8)
    )
    return torch.tensor([[0] * pad + [1] * (width - pad) for pad in pads])


@pytest.mark.property
class TestFirstRealCache:
    @given(mask=left_padded_masks())
    @settings(max_examples=200, deadline=None)
    def test_the_cache_is_the_legacy_expression_row_for_row(
        self, mask: torch.Tensor
    ) -> None:
        frame = _frame(mask)
        expected = tuple(_legacy_first_real(mask, row) for row in range(mask.shape[0]))
        assert frame.first_reals == expected
        assert first_real_indices(mask) == expected
        for row in range(mask.shape[0]):
            assert frame.first_real(row) == expected[row]
            assert frame.content_start(row) == expected[row]

    @given(mask=left_padded_masks(), data=st.data())
    @settings(max_examples=100, deadline=None)
    def test_a_selection_carries_its_rows_entries_over(
        self, mask: torch.Tensor, data: st.DataObject
    ) -> None:
        frame = _frame(mask)
        rows = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=mask.shape[0] - 1),
                min_size=1,
                max_size=6,
            )
        )
        picked = frame.select(rows)
        assert picked.first_reals == tuple(frame.first_reals[i] for i in rows)
        assert picked.first_reals == first_real_indices(picked.attention_mask)

    @given(masks=st.lists(left_padded_masks(), min_size=1, max_size=4))
    @settings(max_examples=100, deadline=None)
    def test_a_concatenation_appends_the_members_entries(
        self, masks: list[torch.Tensor]
    ) -> None:
        width = max(mask.shape[1] for mask in masks)
        # one frame: pad every member to the widest on the left, as a cohort's
        # members are (they share a padded width by construction)
        frames = [
            _frame(torch.nn.functional.pad(mask, (width - mask.shape[1], 0)))
            for mask in masks
        ]
        joined = _concat_frames(frames)
        assert joined.first_reals == tuple(
            index for frame in frames for index in frame.first_reals
        )
        assert joined.first_reals == first_real_indices(joined.attention_mask)


@pytest.mark.unit
class TestFirstRealConstruction:
    def test_replacing_another_field_keeps_the_cache(self) -> None:
        frame = _frame(torch.tensor([[0, 0, 1], [1, 1, 1]]))
        framed = dataclasses.replace(frame, prefix_lengths=(1, 0))
        assert framed.first_reals == frame.first_reals == (2, 0)
        assert framed.content_start(0) == 3
        assert framed.content_start(1) == 0

    def test_a_row_with_no_real_token_reads_zero_as_argmax_does(self) -> None:
        mask = torch.tensor([[0, 0, 0], [0, 1, 1]])
        frame = _frame(mask)
        assert frame.first_reals == (0, 1)
        assert frame.first_reals == tuple(_legacy_first_real(mask, r) for r in (0, 1))

    def test_a_cache_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="first_reals"):
            _frame(torch.tensor([[0, 1], [1, 1]]), first_reals=(1,))

    def test_a_cache_that_disagrees_with_the_mask_is_refused_on_the_cpu(self) -> None:
        """On the CPU the check costs no synchronization, so a caller that
        replaced the mask and carried a stale cache is caught by the CPU
        suite rather than by a wrong position on an accelerator."""
        with pytest.raises(ValueError, match="disagrees"):
            _frame(torch.tensor([[0, 1], [1, 1]]), first_reals=(0, 0))

    def test_the_cache_is_not_part_of_the_frames_identity(self) -> None:
        field = {f.name: f for f in dataclasses.fields(EncodedBatch)}["first_reals"]
        assert field.compare is False
        assert field.repr is False


@pytest.mark.unit
class TestEmptyRows:
    def test_a_row_that_encodes_to_no_token_is_refused_by_row(self) -> None:
        """The frame's arithmetic assumes a first real token per row — an
        empty row's cached index would read 0, as a full row's does, and the
        two would share a mask signature — so the encode refuses it, naming
        the row, on the host tensors before anything moves."""
        with pytest.raises(ProtocolError, match=r"row 1 \(''\) encodes to no token"):
            refuse_empty_rows(("a b", ""), torch.tensor([[1, 1], [0, 0]]))
        refuse_empty_rows(("a b", "c"), torch.tensor([[1, 1], [0, 1]]))
