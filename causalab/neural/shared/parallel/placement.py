"""Where a tapped tensor lives across ranks (``docs/model_parallelism.md`` §4).

A placement is data decided at plan time — from the family's parallel plan row
for the module a tap sits on — and never inferred from a tensor's value or
shape. That is what lets every rank answer "is this fragmented?" identically
without a collective, the invariant the whole SPMD design rests on (§3, "never
branch on rank").

Groups are named by **mesh axis**, not by a process-group object, so this
module stays free of ``torch.distributed`` and a placement can be compared,
hashed and written into a test's expectation table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# The mesh axes a group can be named over (§2) are the geometry's vocabulary,
# defined once in the torch-free protocol layer (``protocol/parallel.py``) and
# re-exported here for the seams that name groups by axis.
from causalab.protocol.parallel import AXES as AXES
from causalab.protocol.parallel import Axis as Axis


@dataclass(frozen=True)
class Replicated:
    """Every rank holds the whole tensor. ``whole`` and ``fragment`` are the identity."""


@dataclass(frozen=True)
class Sharded:
    """Each rank of ``group`` holds one contiguous chunk of ``axis``, in rank order.

    ``whole`` is an all-gather along ``axis``; ``fragment`` is this rank's chunk.
    A colwise projection's output is ``Sharded(axis=-1, group="tensor")``; the
    head axis of an attention-interior slot is ``Sharded(axis=1, group="tensor")``.

    ``slots`` says the axis is ``slots`` equal contiguous runs, *each* sharded
    over the group — the routed experts' token-major view under tensor
    parallelism (§6.3): ``(tokens, top_k · d_e)`` is ``top_k`` runs of
    ``d_e``, and each rank holds ``d_e / tp`` of every run. ``whole`` gathers
    within each run and ``fragment`` keeps this rank's chunk of each; one slot
    is the plain shard.

    ``repeat`` says every chunk is held by ``repeat`` consecutive ranks — the
    axis is ``size / repeat`` chunks and rank ``r`` holds chunk ``r //
    repeat`` — the K/V projections' output under KV-head replication (§6.6),
    where ``tp / num_kv_heads`` ranks read the same KV head. ``whole`` gathers
    in rank order and keeps every ``repeat``-th chunk (exact: the gathered
    copies of one chunk are equal by construction); ``fragment`` is chunk
    ``r // repeat``. One is the plain shard; a slotted shard is never
    repeated.
    """

    axis: int
    group: Axis = "tensor"
    slots: int = 1
    repeat: int = 1

    def __post_init__(self) -> None:
        if self.slots < 1:
            raise ValueError(f"Sharded: slots must be at least 1, got {self.slots}")
        if self.repeat < 1:
            raise ValueError(f"Sharded: repeat must be at least 1, got {self.repeat}")
        if self.slots > 1 and self.repeat > 1:
            raise ValueError(
                f"Sharded: slots={self.slots} and repeat={self.repeat} do not "
                "compose; a slotted shard is held once per rank"
            )


@dataclass(frozen=True)
class ExpertLocal:
    """Token-major routed-expert slots: this rank's experts carry values, the rest are zero.

    Every slot is owned by exactly one rank of ``group``, so ``whole`` is an
    all-reduce sum with no rounding beyond adding zeros, and ``fragment`` keeps
    the slots whose expert this rank owns (§6.3).
    """

    group: Axis = "expert"


@dataclass(frozen=True)
class PartialSum:
    """Each rank of ``group`` holds one summand of the tensor; the sum is the value.

    A rowwise projection's *own* output before its all-reduce — the routed
    experts' down-projection under tensor parallelism (§6.3), where the
    module's all-reduce runs only after the routing weights are applied.
    ``whole`` is the all-reduce sum. ``fragment`` hands the edited tensor to
    the group's first rank and zeros to every other, so the summands add back
    to exactly the edit (``x + 0 == x`` bit for bit) through the module's own
    reduction.
    """

    group: Axis = "tensor"


#: What a context rank holds of its chunk *within* the chunk — the
#: model-parallel placements, which a sequence chunk composes with (§8.4).
SequenceInner = Union[Replicated, Sharded, ExpertLocal, PartialSum]


@dataclass(frozen=True)
class SequenceSharded:
    """Each rank of ``group`` holds one contiguous chunk of the position axis (§8.4).

    The chunks are [`sequence_chunks`][causalab.protocol.parallel.sequence_chunks] of
    the forward's padded length — equal, the remainder on the last rank —
    so ``whole`` and ``fragment`` need the frame the forward runs in
    (``parallel/context.py:SequenceFrame``), never the tensor alone.
    ``inner`` is what the rank holds of its chunk within the chunk: a
    colwise output under ``cp=2, tp=2`` is
    ``SequenceSharded(1, inner=Sharded(-1))`` — ``whole`` is the inner whole
    (the tensor-group gather) then the gather along positions, ``fragment``
    the chunk of positions then the inner fragment. ``flat`` says ``axis``
    is a flattened ``(batch, position)`` pair (the experts' token-major
    view), unfolded with the frame's row count before the chunking.
    """

    axis: int = 1
    group: Axis = "context"
    inner: SequenceInner = field(default_factory=Replicated)
    flat: bool = False


#: What a stage's ranks hold of a tensor the stage owns — every placement
#: but another stage: pipeline placement composes with the model-parallel
#: placements and the sequence chunk (§4, "compose, do not replace") and
#: never nests.
Interior = Union[Replicated, Sharded, ExpertLocal, PartialSum, SequenceSharded]


@dataclass(frozen=True)
class StageLocal:
    """Only pipeline stage ``stage`` computes this tensor (§6.5).

    ``inner`` is what the owner's ranks hold of it *within* the stage — a
    colwise output on stage 1 is ``StageLocal(1, inner=Sharded(-1))`` — so
    ``whole`` is the owner's inner ``whole`` followed by a broadcast from the
    owner over ``group``, and ``fragment`` keeps the inner fragment on the
    owner and nothing elsewhere. The default inner is [`Replicated`][]:
    ``StageLocal(s)`` is a tensor the owner stage holds whole.
    """

    stage: int
    group: Axis = "pipeline"
    inner: Interior = field(default_factory=Replicated)


Placement = Union[
    Replicated, Sharded, ExpertLocal, PartialSum, StageLocal, SequenceSharded
]

#: The placement of a module no parallel-plan row names.
REPLICATED = Replicated()
