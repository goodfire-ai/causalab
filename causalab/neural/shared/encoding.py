"""Wrap protocol position frames in device tensors.

``EncodedBatch`` retains the ``PositionFrame`` fields and supplies
``input_ids`` and ``attention_mask`` for model forwards. ``encode`` runs
protocol encoding and then builds this wrapper. ``first_real_indices``
reads the first valid token of each row from the mask.

Protocol resolvers are re-exported here. ``resolve_position`` also handles
positions in the engine's generated continuation frame.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import torch

from causalab.neural.shared.generated import (
    Continuation,
    continuation_frame,
    resolve_steps,
)
from causalab.protocol.positions import encoding as _positions
from causalab.protocol.positions.encoding import (
    Offsets,
    PositionFrame,
    Segments,
    candidate_runs,
    column_value,
    constituent_candidate_runs,
    refuse_double_bos,
    refuse_empty_rows,
    select_field,
    variable_value,
)
from causalab.protocol.schema import PositionSpec

__all__ = [
    "Continuation",
    "EncodedBatch",
    "candidate_runs",
    "column_value",
    "constituent_candidate_runs",
    "continuation_frame",
    "encode",
    "first_real_indices",
    "refuse_double_bos",
    "refuse_empty_rows",
    "resolve_position",
    "resolve_steps",
    "select_field",
    "variable_value",
]

#: the protocol layer's tokenizer memo, read by the engine's tests through
#: this module's name for one beat
_TOKENIZED = _positions._TOKENIZED  # pyright: ignore[reportPrivateUsage]
_TOKENIZED_PER_TOKENIZER = _positions._TOKENIZED_PER_TOKENIZER  # pyright: ignore[reportPrivateUsage]


@dataclasses.dataclass(frozen=True)
class EncodedBatch:
    """One left-padded batch on a device, plus its position frame.

    The per-row fields are the protocol frame's ([`PositionFrame`][]): ``texts``, ``offset_mapping``,
    ``prefix_lengths``, ``segments`` and the ``first_reals`` cache; the ids
    and mask are tensors on the bundle's device. A frame's tensors are the
    executor's to hold — [`encode`][] and [`from_frame`][] give every
    batch its own storage, so one executor writing rows into its frame in
    place (a graph worker staging tokens) never reaches another's.
    """

    texts: tuple[str, ...]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: Offsets
    prefix_lengths: tuple[int, ...]  # chat-prefix token counts; 0 = plain text
    #: Per row, each declared segment's **candidate** char spans in
    #: ``texts[row]`` (§2.2.1); empty for a document with no ``segments``.
    segments: Segments = ()
    #: Per row, the padded index of its first real token — the ``argmax`` of
    #: the mask row — read off the device **once** for the whole batch, so
    #: [`first_real`][], [`content_start`][] and every position
    #: resolution built on them are host arithmetic. Resolved per row per
    #: read and per write at every layer, the per-call round trip used to be
    #: most of the cohort forward's synchronizations. Derived from
    #: ``attention_mask`` when left empty; [`select`][] and the cohort's
    #: frame concatenation pass theirs through, since a row's index does not
    #: change when it travels. ``dataclasses.replace`` copies this field like
    #: any other, so a caller replacing ``attention_mask`` through it must
    #: pass ``first_reals=()`` explicitly to have it re-derived — or, better,
    #: build the new frame as a row selection (``graph_cohort.slotted_frame``
    #: is the worked example). On the CPU, where the check costs no
    #: synchronization, the constructor refuses a cache the mask disagrees
    #: with.
    first_reals: tuple[int, ...] = dataclasses.field(
        default=(), compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.first_reals:
            object.__setattr__(
                self, "first_reals", first_real_indices(self.attention_mask)
            )
            return
        rows = int(self.attention_mask.shape[0])
        if len(self.first_reals) != rows:
            raise ValueError(
                f"first_reals carries {len(self.first_reals)} entries for a "
                f"{rows}-row attention mask"
            )
        if self.attention_mask.device.type == "cpu" and (
            self.first_reals != first_real_indices(self.attention_mask)
        ):
            raise ValueError(
                "first_reals disagrees with attention_mask — a frame whose mask "
                "was replaced must leave the cache empty to be re-derived"
            )

    @classmethod
    def from_frame(cls, frame: PositionFrame, device: str = "cpu") -> "EncodedBatch":
        """The protocol frame's ids and mask as fresh tensors on ``device``,
        every per-row field carried over (the first-real cache included: it
        is a fact of the rows, derived once, torch-free)."""
        return cls(
            texts=frame.texts,
            input_ids=torch.tensor(frame.token_ids, dtype=torch.long, device=device),
            attention_mask=torch.tensor(
                frame.attention_mask, dtype=torch.long, device=device
            ),
            offset_mapping=frame.offset_mapping,
            prefix_lengths=frame.prefix_lengths,
            segments=frame.segments,
            first_reals=frame.first_reals,
        )

    @property
    def padded_len(self) -> int:
        return int(self.input_ids.shape[1])

    def first_real(self, row: int) -> int:
        """First real token of ``row`` in the padded frame — past the left
        padding, any chat prefix included."""
        return self.first_reals[row]

    def content_start(self, row: int) -> int:
        """First real token of ``row`` in the padded frame, past any prefix."""
        return self.first_real(row) + self.prefix_lengths[row]

    def row_ids(self, row: int) -> tuple[int, ...]:
        """The padded ids of one row, on the host — one copy of the batch's
        ids on first use (the ledger reads every addressed token of every
        row), then host indexing."""
        host = self.__dict__.get("_host_ids")
        if host is None:
            host = tuple(tuple(int(t) for t in r) for r in self.input_ids.tolist())
            object.__setattr__(self, "_host_ids", host)
        return host[row]

    def position_ids(self) -> torch.Tensor:
        """Left-pad position ids: ``cumsum(mask) - 1``, clamped at 0 — the
        plain-forward convention (RoPE is shift-blind, absolute embeddings
        like GPT-2's ``wpe`` are not, so this must always be passed)."""
        return (self.attention_mask.cumsum(dim=1) - 1).clamp(min=0)

    def select(self, indices: Sequence[int]) -> "EncodedBatch":
        """The rows ``indices`` of this batch, in that order, **in this frame**:
        the same padded width, every per-row field sliced in step.

        A fit's minibatch is a selection of its point's frame rather than a
        fresh encode of its rows (spec §4, "Cohorts"): minibatches of several
        points then share one frame and concatenate into one forward, and a
        row's position indices — resolved against the frame — hold whichever
        selection it travels in."""
        if not indices:
            raise ValueError("a selection names at least one row")
        rows = list(indices)
        index = torch.tensor(rows, dtype=torch.long, device=self.input_ids.device)
        return EncodedBatch(
            texts=tuple(self.texts[i] for i in rows),
            input_ids=self.input_ids.index_select(0, index),
            attention_mask=self.attention_mask.index_select(0, index),
            offset_mapping=tuple(self.offset_mapping[i] for i in rows),
            prefix_lengths=tuple(self.prefix_lengths[i] for i in rows),
            segments=tuple(self.segments[i] for i in rows) if self.segments else (),
            first_reals=tuple(self.first_reals[i] for i in rows),
        )


def first_real_indices(attention_mask: torch.Tensor) -> tuple[int, ...]:
    """Per row of a left-padded mask, the index of its first real token: the
    ``argmax`` of the row (its first maximal entry; a row with no real token
    reads 0, as ``argmax`` of zeros does). One reduction over the batch and
    one host read, however many rows."""
    return tuple(int(i) for i in attention_mask.int().argmax(dim=1).tolist())


def encode(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: str = "cpu",
    add_special_tokens: bool = True,
    segments: Sequence[Mapping[str, tuple[tuple[int, int], ...]]] = (),
) -> EncodedBatch:
    """The protocol layer's [`encode`][causalab.protocol.positions.encoding.encode] — left padding, the tokenizer's own specials, the offset mapping,
    the empty-row and double-BOS refusals, memoized per tokenizer — wrapped
    onto ``device``."""
    frame = _positions.encode(
        tokenizer, texts, add_special_tokens=add_special_tokens, segments=segments
    )
    return EncodedBatch.from_frame(frame, device)


def resolve_position(
    spec: PositionSpec,
    batch: EncodedBatch,
    row: int,
    *,
    dataset_row: Mapping[str, Any] | None = None,
    field: str | None = None,
    continuation: Continuation | None = None,
) -> list[int]:
    """Resolve one position spec for one row into padded-frame indices, or
    into decode-step indices when the spec selects the continuation frame —
    the protocol resolver for the prompt frame, [`resolve_steps`][] for a ``generated`` spec."""
    if spec.generated is not None:
        if continuation is None:
            raise _positions.ProtocolError(
                "P2",
                "a generated position needs the decode's continuation — the "
                "frame it addresses does not exist until the model has run",
            )
        return resolve_steps(
            spec, continuation, row, dataset_row=dataset_row, field=field
        )
    return _positions.resolve_position(
        spec, batch, row, dataset_row=dataset_row, field=field
    )
