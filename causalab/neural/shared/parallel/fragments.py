"""``whole`` / ``fragment`` over every placement (``docs/model_parallelism.md`` §4, §6.3, §7).

Around every hook body the executor makes the tapped tensor *whole* — the
global tensor, identical on every rank — runs the read or write on it
unchanged, and *fragments* the result back to what this rank holds. The pair
is pure given a [`Collective`][]; what each placement means:

===================  ==========================================  ================================
placement            ``whole``                                   ``fragment``
===================  ==========================================  ================================
``Replicated``       the tensor                                  the tensor
``Sharded``          all-gather along ``axis`` over ``group``    this rank's contiguous chunk
                     (within each of its ``slots`` runs; every  (of each run; chunk
                     ``repeat``-th gathered chunk kept)          ``rank // repeat``)
``ExpertLocal``      all-reduce sum over ``group``               the slots whose expert this rank owns, zeros elsewhere
``PartialSum``       all-reduce sum over ``group``               the tensor on the group's first rank, zeros elsewhere
``StageLocal``       the owner's inner ``whole``, then a broadcast  the inner ``fragment`` on the owner; an **empty** tensor elsewhere
``SequenceSharded``  the inner ``whole``, then the frame's gather  this rank's chunk of positions, then the inner ``fragment``
                     along the position axis
===================  ==========================================  ================================

*Positions need the frame.* A ``SequenceSharded`` chunk is
``sequence_chunks(padded_len, cp)[rank]`` — equal chunks, the remainder on the
last rank — and neither the local extent alone nor the whole extent alone
says where a rank's chunk lies when the frame is uneven, so both functions
take the forward's [`SequenceFrame`][] (the executor binds one
per forward). The gather pads uneven chunks to the widest and cuts them back,
so it is exact. A flat token axis (the experts' token-major view) is unfolded
by the frame's row count first, chunked or gathered by position, and folded
back. **Without a frame** the chunks are taken as equal — the plain
``Sharded`` rule on the position axis, an extent the group does not divide
refused — which is the frame's own answer whenever the group divides the
frame; a flat axis, whose row count only the frame knows, is refused by name.
A missing frame can therefore never land a wrong chunk silently: an uneven
frame makes the members' extents disagree, which the collective refuses.

*Non-owners under ``StageLocal``* hold ``tensor.new_empty((0, *tensor.shape[1:]))``:
a tensor of the right dtype and device whose leading extent is zero. Shapes
stay typed (``ndim`` and the trailing extents survive), nothing downstream
has to test for ``None``, and ``whole`` recognises the non-owner by rank —
never by the value — so the two ranks arrive at the same broadcast.

*The exact sum.* ``ExpertLocal``'s all-reduce adds each slot's value to zeros
from every other rank, and ``x + 0.0 == x`` bit for bit for every finite ``x``.
That rests on ownership being a partition — every slot owned by exactly one
rank — which [`expert_slot_mask`][] gives because an expert id has one
owner, ``id // (num_experts / ep)``. ``PartialSum``'s *fragment* rests on the
same fact from the other side: the edit on one rank plus zeros on the rest
sums back to the edit exactly, whatever reduction order the collective uses.
Its *whole* is a genuine sum of partial products, exact only up to the
reduction order — the band the smoke tier measures.

*Under grad the pair is autograd-aware* (§7, ``autograd.py``). A tensor
that requires grad — the write math's output around a trained featurizer,
or a tap downstream of one — takes the same collectives through the
Functions of ``torch.autograd`` whose forward is the raw value bit for bit:
``whole`` of a gather placement hands back this rank's slice of the
gradient, ``whole`` of a sum placement the gradient unchanged; ``fragment``
of a gather placement all-gathers the ranks' gradient slices and ``fragment``
of a sum placement all-reduce-sums their masked summands, so every rank's
backward carries the **full** ``∂L/∂edited`` into the replicated write math
and every rank's featurizer gradient is the full gradient — the invariant
``agreements.average_gradients`` rests on at a sharded site. The routing
reads ``torch.is_grad_enabled()`` and ``requires_grad`` and nothing
rank-local, so every rank records the same Functions and reaches the same
backward collectives; a tensor without grad, and every tensor under
``no_grad``, runs the raw path unchanged.

*A group of one is the identity.* Every function here returns its argument
untouched when ``collective.size(group) == 1``; [`Fragments`][] caches the
sizes at construction so that at world 1 the whole seam costs one attribute
lookup and never reaches the collective — the path today's single-device
engine runs.

Refusals are [`PlacementError`][], named after what did not fit, and
deliberately not a [`ProtocolError`][causalab.protocol.rules.errors.ProtocolError] — the
same reasoning as [`LayoutError`][causalab.neural.shared.layout.LayoutError]: a
placement is set by the plan table, never written by a document author, so a
mismatch is an internal invariant broken, not a rule a document violated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

from causalab.neural.shared.parallel.autograd import (
    edit_fragment,
    edit_summand,
    gather_for_edit,
    sum_for_edit,
)
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import (
    AXES,
    Axis,
    ExpertLocal,
    PartialSum,
    Placement,
    Replicated,
    SequenceSharded,
    Sharded,
    StageLocal,
)

if TYPE_CHECKING:
    from causalab.neural.shared.parallel.context import SequenceFrame

__all__ = [
    "NO_SLOT",
    "Fragments",
    "PlacementError",
    "expert_slot_mask",
    "fragment",
    "reconstruct_routing",
    "remap_routing",
    "whole",
]

#: The executor's marker for a slot the router filled with nothing; the value
#: a slot no rank owns reconstructs to (§6.3).
NO_SLOT = -1

_Chunked = Union[Sharded, SequenceSharded]


class PlacementError(ValueError):
    """A tensor, routing table or group does not fit the placement it was handed.

    ``placement`` is the placement in question (``None`` for the routing
    helpers, which take the group's size directly); the message names the
    mismatch in the placement's own terms.
    """

    def __init__(self, placement: Placement | None, message: str) -> None:
        self.placement = placement
        where = f"{placement}: " if placement is not None else ""
        super().__init__(f"{where}{message}")


# --------------------------------------------------------------------------- #
# whole / fragment
# --------------------------------------------------------------------------- #


def whole(
    tensor: torch.Tensor,
    placement: Placement,
    collective: Collective,
    *,
    frame: "SequenceFrame | None" = None,
) -> torch.Tensor:
    """The global tensor, identical on every rank of the placement's group.
    ``frame`` is the forward's position frame a ``SequenceSharded`` placement
    gathers by (module docstring)."""
    if isinstance(placement, Replicated):
        return tensor
    if isinstance(placement, StageLocal):
        # the owner's ranks make the tensor whole within the stage first, then
        # the stage broadcasts it; a stage group of one is just the inner whole
        size = collective.size(placement.group)
        if size == 1:
            return whole(tensor, placement.inner, collective, frame=frame)
        _check_stage(placement, size)
        owner = collective.rank(placement.group) == placement.stage
        payload = (
            whole(tensor, placement.inner, collective, frame=frame) if owner else None
        )
        return collective.broadcast(payload, placement.stage, placement.group)
    if isinstance(placement, SequenceSharded):
        # the chunk is made whole within itself first (the tensor-group
        # gather, the expert sum), then the chunks are gathered by position
        inner = whole(tensor, placement.inner, collective)
        if collective.size(placement.group) == 1:
            return inner
        return _sequence_whole(inner, placement, collective, frame)
    size = collective.size(placement.group)
    if size == 1:
        return tensor
    if isinstance(placement, (ExpertLocal, PartialSum)):
        if _differentiable(tensor):
            return sum_for_edit(tensor, placement.group, collective)
        return collective.all_reduce_sum(tensor, placement.group)
    axis = _axis_index(placement, tensor.dim())
    slots = placement.slots if isinstance(placement, Sharded) else 1
    repeat = placement.repeat if isinstance(placement, Sharded) else 1
    if slots == 1:
        return _gather(tensor, axis, size, repeat, placement, collective)
    # a slotted shard: gather within each run of the axis, in rank order
    runs = _slotted(tensor, axis, slots, placement)
    return _gather(runs, axis + 1, size, 1, placement, collective).flatten(
        axis, axis + 1
    )


def fragment(
    tensor: torch.Tensor,
    placement: Placement,
    collective: Collective,
    *,
    routing: torch.Tensor | None = None,
    num_experts: int | None = None,
    frame: "SequenceFrame | None" = None,
) -> torch.Tensor:
    """This rank's part of a global tensor — the inverse of [`whole`][].

    ``ExpertLocal`` needs the **global** routing table ``(…, top_k)`` and
    ``num_experts``: the slots are contiguous runs of ``feature / top_k`` in the
    token-major feature axis, and which of them this rank keeps is a fact about
    the routing, not about the tensor. ``frame`` is the forward's position
    frame a ``SequenceSharded`` placement chunks by; the routing table, which
    leads with the same positions, is chunked alongside.
    """
    if isinstance(placement, Replicated):
        return tensor
    if isinstance(placement, StageLocal):
        size = collective.size(placement.group)
        if size > 1:
            _check_stage(placement, size)
            if collective.rank(placement.group) != placement.stage:
                return tensor.new_empty((0, *tensor.shape[1:]))
        return fragment(
            tensor,
            placement.inner,
            collective,
            routing=routing,
            num_experts=num_experts,
            frame=frame,
        )
    if isinstance(placement, SequenceSharded):
        if collective.size(placement.group) > 1:
            tensor = _sequence_fragment(tensor, placement, collective, frame)
            if routing is not None:
                routing = _sequence_fragment(routing, placement, collective, frame)
        return fragment(
            tensor,
            placement.inner,
            collective,
            routing=routing,
            num_experts=num_experts,
        )
    size = collective.size(placement.group)
    if size == 1:
        return tensor
    rank = collective.rank(placement.group)
    if isinstance(placement, ExpertLocal):
        if routing is None:
            raise PlacementError(
                placement, "fragment needs the global routing table (…, top_k)"
            )
        if num_experts is None:
            raise PlacementError(
                placement, "fragment needs num_experts to decide slot ownership"
            )
        keep = _keep_mask(
            tensor, expert_slot_mask(routing, num_experts, rank, size), placement
        )
        if _differentiable(tensor):
            return edit_summand(tensor, keep, placement.group, collective)
        return torch.where(keep, tensor, tensor.new_zeros(()))
    if isinstance(placement, PartialSum):
        # the group's first rank carries the whole edit; the others add zeros
        if _differentiable(tensor):
            return edit_summand(tensor, None, placement.group, collective)
        return tensor if rank == 0 else torch.zeros_like(tensor)
    axis = _axis_index(placement, tensor.dim())
    slots = placement.slots if isinstance(placement, Sharded) else 1
    repeat = placement.repeat if isinstance(placement, Sharded) else 1
    if slots > 1:
        runs = _slotted(tensor, axis, slots, placement)
        return _narrow(runs, axis + 1, size, 1, placement, collective).flatten(
            axis, axis + 1
        )
    return _narrow(tensor, axis, size, repeat, placement, collective)


def _differentiable(tensor: torch.Tensor) -> bool:
    """Whether the pair goes through ``autograd.py`` (module docstring): a
    read of grad mode and of the tensor, never of the rank."""
    return torch.is_grad_enabled() and tensor.requires_grad


def _gather(
    tensor: torch.Tensor,
    axis: int,
    size: int,
    repeat: int,
    placement: _Chunked,
    collective: Collective,
) -> torch.Tensor:
    """The whole axis from this rank's chunk — every ``repeat``-th gathered
    chunk kept (§6.6) — through the autograd pair under grad."""
    distinct = _chunks(size, repeat, placement)
    if _differentiable(tensor):
        chunks = _ranges(tensor.shape[axis] * distinct, axis, size, repeat, placement)
        return gather_for_edit(tensor, axis, placement.group, collective, chunks=chunks)
    gathered = collective.all_gather(tensor, axis, placement.group)
    if repeat == 1:
        return gathered
    # a repeated shard: ``repeat`` consecutive ranks hold one chunk, so the
    # gather carries every chunk ``repeat`` times in rank order; keep the
    # first copy of each
    copies = gathered.chunk(distinct * repeat, dim=axis)
    return torch.cat(copies[::repeat], dim=axis)


def _narrow(
    tensor: torch.Tensor,
    axis: int,
    size: int,
    repeat: int,
    placement: _Chunked,
    collective: Collective,
) -> torch.Tensor:
    """This rank's chunk of the whole axis — chunk ``rank // repeat`` of
    ``size / repeat`` (§6.6) — through the autograd pair under grad."""
    chunks = _ranges(tensor.shape[axis], axis, size, repeat, placement)
    if _differentiable(tensor):
        return edit_fragment(tensor, axis, placement.group, collective, chunks=chunks)
    mine = chunks[collective.rank(placement.group)]
    return tensor.narrow(axis, mine.start, len(mine))


def _chunks(size: int, repeat: int, placement: Sharded) -> int:
    """How many distinct chunks a repeated shard has over a group of
    ``size``: ``size / repeat``, refused when the group is not whole
    repeats."""
    if size % repeat:
        raise PlacementError(
            placement,
            f"the group's {size} ranks are not whole repeats of {repeat}",
        )
    return size // repeat


def _frameless(placement: SequenceSharded) -> None:
    if placement.flat:
        raise PlacementError(
            placement,
            "a flat (batch · position) axis is chunked by position only with the "
            "forward's SequenceFrame (context.py), which knows the row count; "
            "none was bound",
        )


def _sequence_whole(
    tensor: torch.Tensor,
    placement: SequenceSharded,
    collective: Collective,
    frame: "SequenceFrame | None",
) -> torch.Tensor:
    """The chunks gathered by position along the placement's axis; a flat
    token axis is unfolded by the frame's rows first and folded back. With
    no frame the chunks are equal (module docstring)."""
    axis = _axis_index(placement, tensor.dim())
    if frame is None:
        _frameless(placement)
        size = collective.size(placement.group)
        return _gather(tensor, axis, size, 1, placement, collective)
    if not placement.flat:
        return _frame_gather(tensor, axis, frame, placement, collective)
    unfolded = frame.unflatten(tensor, axis, local=True)
    return _frame_gather(unfolded, axis + 1, frame, placement, collective).flatten(
        axis, axis + 1
    )


def _frame_gather(
    tensor: torch.Tensor,
    axis: int,
    frame: "SequenceFrame",
    placement: SequenceSharded,
    collective: Collective,
) -> torch.Tensor:
    """The frame's gather — every chunk padded to the widest and cut back —
    through the autograd pair under grad, the frame's own refusal of an
    extent that is not this rank's chunk kept."""
    if not _differentiable(tensor):
        return frame.gather(tensor, axis)
    frame.check_local(tensor, axis)
    return gather_for_edit(
        tensor, axis, placement.group, collective, chunks=frame.chunks
    )


def _sequence_fragment(
    tensor: torch.Tensor,
    placement: SequenceSharded,
    collective: Collective,
    frame: "SequenceFrame | None",
) -> torch.Tensor:
    """This rank's chunk of positions along the placement's axis — the
    inverse of `_sequence_whole`."""
    axis = _axis_index(placement, tensor.dim())
    if frame is None:
        _frameless(placement)
        size = collective.size(placement.group)
        return _narrow(tensor, axis, size, 1, placement, collective)
    if not placement.flat:
        return _frame_fragment(tensor, axis, frame, placement, collective)
    unfolded = frame.unflatten(tensor, axis, local=False)
    return _frame_fragment(unfolded, axis + 1, frame, placement, collective).flatten(
        axis, axis + 1
    )


def _frame_fragment(
    tensor: torch.Tensor,
    axis: int,
    frame: "SequenceFrame",
    placement: SequenceSharded,
    collective: Collective,
) -> torch.Tensor:
    """The frame's chunk of positions through the autograd pair under grad,
    the frame's own refusal of an extent that is not the whole frame kept."""
    if not _differentiable(tensor):
        return frame.fragment(tensor, axis)
    frame.check_whole(tensor, axis)
    return edit_fragment(tensor, axis, placement.group, collective, chunks=frame.chunks)


def _ranges(
    extent: int, axis: int, size: int, repeat: int, placement: Placement
) -> tuple[range, ...]:
    """Each rank's contiguous chunk of a whole ``axis`` of ``extent``: rank
    ``r`` holds chunk ``r // repeat`` of the ``size / repeat`` equal chunks —
    the table ``autograd.py`` gathers and narrows by."""
    distinct = _chunks(size, repeat, placement)
    if extent % distinct:
        raise PlacementError(
            placement,
            f"axis {axis} has extent {extent}, which the group's {distinct} ranks do not divide",
        )
    width = extent // distinct
    return tuple(
        range((rank // repeat) * width, (rank // repeat + 1) * width)
        for rank in range(size)
    )


def _chunk(
    tensor: torch.Tensor, axis: int, rank: int, size: int, placement: Placement
) -> torch.Tensor:
    """Rank ``rank``'s contiguous chunk of ``axis`` among ``size``."""
    mine = _ranges(tensor.shape[axis], axis, size, 1, placement)[rank]
    return tensor.narrow(axis, mine.start, len(mine))


def _slotted(
    tensor: torch.Tensor, axis: int, slots: int, placement: Sharded
) -> torch.Tensor:
    """``axis`` unfolded into ``(slots, extent / slots)``: the runs a slotted
    shard gathers and chunks within."""
    extent = tensor.shape[axis]
    if extent % slots:
        raise PlacementError(
            placement,
            f"axis {axis} has extent {extent}, which is not {slots} runs of one width",
        )
    return tensor.unflatten(axis, (slots, extent // slots))


def _axis_index(placement: _Chunked, ndim: int) -> int:
    axis = placement.axis
    if not -ndim <= axis < ndim:
        raise PlacementError(placement, f"axis {axis} is outside a {ndim}-D tensor")
    return axis % ndim


def _check_stage(placement: StageLocal, size: int) -> None:
    if not 0 <= placement.stage < size:
        raise PlacementError(
            placement, f"stage {placement.stage} is outside the group's {size} ranks"
        )


def _keep_mask(
    tensor: torch.Tensor, mask: torch.Tensor, placement: ExpertLocal
) -> torch.Tensor:
    """The ownership mask ``(…, top_k)`` expanded over the slots' width to
    ``tensor``'s shape: ``where(keep, tensor, 0)`` keeps this rank's slots."""
    lead, top_k = tuple(mask.shape[:-1]), mask.shape[-1]
    if tuple(tensor.shape[:-1]) != lead:
        raise PlacementError(
            placement,
            f"the routing table {tuple(mask.shape)} leads with {lead} but the "
            f"tensor {tuple(tensor.shape)} does not",
        )
    feature = tensor.shape[-1]
    if feature % top_k:
        raise PlacementError(
            placement,
            f"feature axis {feature} is not top_k={top_k} slots of one width",
        )
    d_expert = feature // top_k
    return mask.unsqueeze(-1).expand(*lead, top_k, d_expert).reshape(tensor.shape)


# --------------------------------------------------------------------------- #
# Expert ownership and the routing table (§6.3)
# --------------------------------------------------------------------------- #


def _local_expert_count(num_experts: int, size: int) -> int:
    if size < 1 or num_experts % size:
        raise PlacementError(
            None,
            f"num_experts={num_experts} is not divided by the expert group's {size} ranks",
        )
    return num_experts // size


def _check_rank(rank: int, size: int) -> None:
    if not 0 <= rank < size:
        raise PlacementError(None, f"rank {rank} is outside a group of {size}")


def expert_slot_mask(
    routing: torch.Tensor, num_experts: int, rank: int, size: int
) -> torch.Tensor:
    """Which slots of a global routing table ``(…, top_k)`` rank ``rank`` owns.

    Rank ``r`` of ``size`` owns experts ``[r · n, (r + 1) · n)`` with
    ``n = num_experts / size`` — transformers' layout, ``EpRouterParallel``'s
    ``index // num_local_experts == ep_rank``. A [`NO_SLOT`][] entry floors
    to ``-1`` and is owned by nobody.
    """
    num_local = _local_expert_count(num_experts, size)
    _check_rank(rank, size)
    return torch.div(routing, num_local, rounding_mode="floor") == rank


def remap_routing(
    global_table: torch.Tensor, num_experts: int, rank: int, size: int
) -> torch.Tensor:
    """The local table transformers' ``EpRouterParallel`` hands this rank's experts.

    Owned slots become local ids (``global mod num_local``); every other slot
    becomes the sentinel ``num_local``, which the grouped dispatch skips.
    transformers spells it ``masked_fill(non_local, -1) → fmod(num_local) →
    masked_fill(== -1, num_local)`` with a special case for ``num_local == 1``
    (``fmod(-1, 1)`` is ``0`` and would turn the sentinel into expert 0); this
    spelling never takes the remainder of a sentinel, so it needs no special
    case and produces the same table entry for entry.
    """
    num_local = _local_expert_count(num_experts, size)
    owned = expert_slot_mask(global_table, num_experts, rank, size)
    return torch.where(
        owned, global_table % num_local, torch.full_like(global_table, num_local)
    )


def _routing_contribution(
    local_table: torch.Tensor, num_local: int, rank: int
) -> torch.Tensor:
    """This rank's summand: ``global_id + 1`` on the slots it owns, ``0`` elsewhere.

    The ``+ 1`` is load-bearing. Expert 0 belongs to rank 0 and would
    contribute ``0`` on its slots — the same as every rank contributes on a
    slot it does not own — so without the offset the sum could not tell
    "routed to expert 0" from "routed to nothing". Offset by one, every owned
    slot contributes at least ``1``, and exactly the slots **no** rank owns
    sum to ``0``.
    """
    if bool((local_table < 0).any()) or bool((local_table > num_local).any()):
        raise PlacementError(
            None,
            f"local routing table has entries outside [0, {num_local}] "
            f"(the local ids and the sentinel {num_local})",
        )
    owned = local_table != num_local
    global_ids = local_table.to(torch.int64) + rank * num_local + 1
    return torch.where(owned, global_ids, torch.zeros_like(global_ids))


def _routing_from_sum(summed: torch.Tensor) -> torch.Tensor:
    """Undo the offset: ``0`` (no owner) becomes [`NO_SLOT`][]."""
    return summed - 1


def reconstruct_routing(
    local_table: torch.Tensor,
    num_experts: int,
    collective: Collective,
    *,
    group: Axis = "expert",
) -> torch.Tensor:
    """The global routing table from each rank's ``EpRouterParallel`` output.

    Each rank contributes `_routing_contribution`, an integer all-reduce
    sums them (exact — integers), and the offset comes off. A slot that was
    the sentinel on every rank — nobody owned it — reconstructs to
    [`NO_SLOT`][], the marker the executor already refuses on; nothing else
    does, because every owned slot summed to at least ``1``.
    """
    size = collective.size(group)
    num_local = _local_expert_count(num_experts, size)
    contribution = _routing_contribution(local_table, num_local, collective.rank(group))
    summed = (
        contribution if size == 1 else collective.all_reduce_sum(contribution, group)
    )
    return _routing_from_sum(summed).to(local_table.dtype)


# --------------------------------------------------------------------------- #
# The object the executor holds
# --------------------------------------------------------------------------- #


class Fragments:
    """[`whole`][causalab.neural.shared.parallel.fragments.whole] and [`fragment`][causalab.neural.shared.parallel.fragments.fragment] bound to one collective, with the
    fast path: a placement whose group has one member is the identity without
    a call on the collective. Sizes are read once, at construction, so world 1
    costs one attribute lookup per tap."""

    __slots__ = ("_collective", "_sizes", "_solo")

    def __init__(self, collective: Collective) -> None:
        self._collective = collective
        self._sizes: dict[Axis, int] = {axis: collective.size(axis) for axis in AXES}
        self._solo = all(size == 1 for size in self._sizes.values())

    @property
    def collective(self) -> Collective:
        return self._collective

    def size(self, axis: Axis) -> int:
        """The cached group size on ``axis`` — no call on the collective."""
        return self._sizes[axis]

    def _is_identity(self, placement: Placement) -> bool:
        if self._solo or isinstance(placement, Replicated):
            return True
        if isinstance(placement, (StageLocal, SequenceSharded)):
            # a group of one still holds its inner placement
            return self._sizes[placement.group] == 1 and self._is_identity(
                placement.inner
            )
        return self._sizes[placement.group] == 1

    def whole(
        self,
        tensor: torch.Tensor,
        placement: Placement,
        *,
        frame: "SequenceFrame | None" = None,
    ) -> torch.Tensor:
        if self._is_identity(placement):
            return tensor
        return whole(tensor, placement, self._collective, frame=frame)

    def fragment(
        self,
        tensor: torch.Tensor,
        placement: Placement,
        *,
        routing: torch.Tensor | None = None,
        num_experts: int | None = None,
        frame: "SequenceFrame | None" = None,
    ) -> torch.Tensor:
        if self._is_identity(placement):
            return tensor
        return fragment(
            tensor,
            placement,
            self._collective,
            routing=routing,
            num_experts=num_experts,
            frame=frame,
        )
