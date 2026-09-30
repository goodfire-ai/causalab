"""Autograd-aware collectives for the tap path (``docs/model_parallelism.md`` §7).

The [`Collective`][] protocol is deliberately not
differentiable: ``all_gather`` fills fresh buffers and ``all_reduce_sum``
reduces a clone, so a tensor that crosses it leaves the graph (a gather) or
keeps an identity backward (a sum). A fit needs more. The executor makes a
tapped tensor whole around every hook body, runs the write on the whole, and
fragments the result back to what this rank holds — and at a sharded site the
featurizer inside that write sits *between* two collectives. This module is
the one place autograd meets them: each function below is a
``torch.autograd.Function`` whose forward is exactly the raw collective —
bit for bit the value ``fragments.py`` computes without grad — and whose
backward is the derivative the pairing calls for.

**The pairing.** Every rank computes the same loss on replicated logits, and
the model on rank ``r`` consumes only ``fragment_r(edited)``, its own slice
(a gather placement) or its own summand (a sum placement). Written as one
computation, ``L = f(fragment_0(edited), …, fragment_{n-1}(edited))`` with
``edited`` the featurizer's replicated output, so the true gradient of the
featurizer's parameters carries ``∂L/∂edited = Σ_r ∂L/∂fragment_r`` scattered
back into place. Rank ``r``'s backward alone holds only its own term. Two
choices make every rank hold the whole:

- ``fragment``'s backward **all-gathers** the ranks' gradient slices
  ([`edit_fragment`][]) or **all-reduce-sums** their masked summands
  ([`edit_summand`][]). Each rank then sees the full ``∂L/∂edited``, the
  write math above it is replicated, and so every rank's featurizer gradient
  is the **full** gradient — identical on every rank, which is what lets
  ``agreements.average_gradients``' mean stay exact at every site, sharded
  or not: ``(g + g) / 2 == g``.
- ``whole``'s backward hands back **this rank's slice** of the incoming
  gradient ([`gather_for_edit`][]) — the true ``∂L/∂x_r``, because ``x_r``
  appears once in the real computation and the full downstream gradient is
  already in hand — or, for a sum, the gradient unchanged
  ([`sum_for_edit`][]: the local Jacobian of ``Σ_r x_r`` is the identity).

A repeated shard (``Sharded.repeat``, §6.6) holds one chunk on ``repeat``
consecutive ranks, each consuming it downstream of its own heads; the
gathered gradient slices of one chunk are therefore *partials* and
[`edit_fragment`][] sums them in rank order. The uneven chunks of a
sequence frame (§8.4) are padded to the widest for the wire and cut back, in
both directions, so the gradient of a padded position is dropped exactly as
its value was. Both are spelled by the caller as ``chunks``: rank ``r``'s
positions of the whole axis, duplicates naming a repeated chunk.

**The faithful pair.** A gather whose consumer is *not* replicated — each
rank weighs the whole by something of its own, the context-parallel KV
gather — needs the true ``∂L/∂x_r = (Σ_r' ∂L_{r'}/∂whole)_r``:
[`gather_reduce_scatter`][], an all-gather forward and an all-reduce-sum
then this rank's slice in backward. Exposed for the context branch; nothing
in ``fragments.py`` wires it.

**Point to point.** [`send_with_grad`][] returns its argument so the
sender's graph continues, and receives the gradient from ``dst`` in backward
(added to whatever the sender's own continuation contributed);
[`recv_with_grad`][] attaches the received tensor to ``link``, a tensor on
this rank's graph, and sends the gradient back to ``src`` in backward — the
pipeline's handoff. Exposed for the pipeline and context branches.

**The handoff protocol** ([`Handoff`][]). Whether the gradient crosses
back is a decision both ends must make alike, or one side's backward waits
for a message the other never sends — a deadlock, not an error. Two
protocols exist, and each pair names its own: ``Handoff.ALWAYS`` — the
gradient crosses whether or not the sent tensor requires one, through a
zero-size leaf standing in as the graph's input on the side that has
nothing to train; the pipeline's residual (§6.5), whose stage above sends
the gradient back regardless of what the stage below trains — and
``Handoff.IF_REQUIRED`` — the gradient crosses iff the tensor requires
one, both ends reading that off the same replicated value; the DeltaNet
handoff (§6.4), where the sender's final state and the receiver's link are
the kernel's value on either side of the boundary. A **link** — one
collective, one axis, one ordered pair of ranks — **agrees its protocol
once**, at its first handoff: the two ends exchange their protocol's code
(one small send and receive each) and a mismatch is refused on **both**
ends as [`HandoffMismatch`][], naming the axis, the two stages or chunks
and the two protocols. The agreement is remembered for the collective's
lifetime, so every later handoff on the link costs nothing beyond the
tensor itself, and a second protocol asked of an agreed link is refused
locally by name. The check covers the choice of protocol, which is what a
caller can get wrong; under ``IF_REQUIRED`` the two ends' ``requires_grad``
are the same replicated fact by §3, and a rank whose differs is the
divergence the simulator refuses by call site and a real backend hangs on,
like every other rank-local branch.

*Where the collective lives.* Each Function keeps the collective and the
integer facts it needs (the dimension, this rank's slice, the axis) on
``ctx`` — the same shape as ``sharding._SumGradient`` keeping its process
group. An autograd graph is never pickled, so nothing here has to be; the
Function classes are module-level, importable by name, which is what a
spawned rank needs to *build* one.

*Collectives in backward.* Every rank's backward reaches these collectives
in the same order because every rank runs the same graph (§3): a Function is
recorded on a rank exactly when its input requires grad, and the write math
that decides that is replicated. A rank whose ``requires_grad`` differed
would diverge at the first backward collective — the simulator refuses it by
call site, a real backend hangs — so the routing in ``fragments.py`` reads
``requires_grad`` and nothing rank-local.

Refusals are [`AutogradError`][]: a ``chunks`` table not one range per
rank, or a tensor whose extent is not what its rank's range says — an
internal invariant broken, never a document's fault (the same reasoning as
[`PlacementError`][causalab.neural.shared.parallel.fragments.PlacementError]).
"""

from __future__ import annotations

import dataclasses
import enum
import weakref
from typing import Any, Sequence

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis

__all__ = [
    "AutogradError",
    "Chunks",
    "Handoff",
    "HandoffMismatch",
    "carry",
    "edit_fragment",
    "edit_summand",
    "equal_chunks",
    "gather_for_edit",
    "gather_reduce_scatter",
    "recv_with_grad",
    "send_with_grad",
    "sum_for_edit",
]

#: Rank ``r``'s positions of the whole axis, one range per group-local rank
#: (§4). Equal chunks are the plain shard; duplicates name a repeated chunk
#: (§6.6); unequal lengths are a sequence frame's chunks (§8.4).
Chunks = tuple[range, ...]


class AutogradError(ValueError):
    """A chunk table or a tensor does not fit the Function it was handed."""


class Handoff(enum.IntEnum):
    """The gradient protocol of a point-to-point handoff (module docstring):
    the integer is the code that crosses the wire when a link agrees."""

    #: The gradient crosses back regardless of the sent tensor — the
    #: pipeline's residual (§6.5).
    ALWAYS = 1
    #: The gradient crosses back iff the tensor requires one, both ends
    #: reading the same replicated fact — the DeltaNet handoff (§6.4).
    IF_REQUIRED = 2


class HandoffMismatch(AutogradError):
    """The two ends of a link run different [`Handoff`][] protocols — or
    one end asks a second protocol of a link already agreed. Raised on both
    ends when the link agrees (each compares the other's code to its own),
    naming the axis, the two ranks as stages or chunks, and both protocols;
    the deadlock that would otherwise follow in backward never starts."""


def equal_chunks(extent: int, size: int) -> Chunks:
    """``size`` equal chunks of a whole axis of ``extent``, in rank order.

    Raises:
        AutogradError: ``size`` does not divide ``extent``.
    """
    if size < 1 or extent % size:
        raise AutogradError(
            f"an axis of extent {extent} is not {size} equal chunks; the caller "
            "spells uneven or repeated chunks explicitly"
        )
    width = extent // size
    return tuple(range(rank * width, (rank + 1) * width) for rank in range(size))


# --------------------------------------------------------------------------- #
# the wire layout of a chunked axis
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Pieces:
    """How the ranks' pieces of ``dim`` travel: every piece padded to the
    widest for the all-gather, cut back to its range after; ``distinct`` the
    ranges in order of first appearance with the ranks that hold each."""

    dim: int
    chunks: Chunks

    @property
    def widest(self) -> int:
        return max(len(chunk) for chunk in self.chunks)

    @property
    def uniform(self) -> bool:
        """Every rank holds its own chunk of one width: the gathered buffer
        *is* the whole, no cut and no copy."""
        first = self.chunks[0]
        return all(len(chunk) == len(first) for chunk in self.chunks) and len(
            {(chunk.start, chunk.stop) for chunk in self.chunks}
        ) == len(self.chunks)

    @property
    def distinct(self) -> tuple[tuple[range, tuple[int, ...]], ...]:
        holders: dict[tuple[int, int], list[int]] = {}
        order: list[range] = []
        for rank, chunk in enumerate(self.chunks):
            key = (chunk.start, chunk.stop)
            if key not in holders:
                holders[key] = []
                order.append(chunk)
            holders[key].append(rank)
        return tuple(
            (chunk, tuple(holders[(chunk.start, chunk.stop)])) for chunk in order
        )

    @property
    def whole_extent(self) -> int:
        return sum(len(chunk) for chunk, _ in self.distinct)


def _pieces(
    dim: int,
    chunks: Sequence[range] | None,
    tensor: torch.Tensor,
    size: int,
    whole: bool,
) -> _Pieces:
    """The chunk table for ``tensor`` — ``chunks`` as given, else ``size``
    equal chunks of the whole axis (``tensor`` holding the whole, or one
    chunk of it)."""
    dim = dim % tensor.dim()
    if chunks is None:
        extent = tensor.shape[dim] if whole else tensor.shape[dim] * size
        table = equal_chunks(extent, size)
    else:
        table = tuple(chunks)
        if len(table) != size:
            raise AutogradError(
                f"chunks names {len(table)} rank(s) but the group has {size}"
            )
    return _Pieces(dim, table)


def _check_extent(tensor: torch.Tensor, dim: int, expected: int, what: str) -> None:
    if tensor.shape[dim] != expected:
        raise AutogradError(
            f"{what}: axis {dim} has extent {tensor.shape[dim]}, not {expected}"
        )


def _padded(tensor: torch.Tensor, dim: int, to: int) -> torch.Tensor:
    """``tensor`` extended with zeros along ``dim`` to extent ``to``."""
    extent = tensor.shape[dim]
    if extent == to:
        return tensor.contiguous()
    pad_shape = list(tensor.shape)
    pad_shape[dim] = to - extent
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim).contiguous()


def _assemble(gathered: torch.Tensor, pieces: _Pieces, *, reduce: bool) -> torch.Tensor:
    """The whole axis out of the gathered, padded pieces: each distinct range
    cut from its first holder's slot — or, ``reduce``, the sum over every
    holder's slot in rank order (the partials of a repeated chunk)."""
    dim, widest = pieces.dim, pieces.widest
    parts: list[torch.Tensor] = []
    for chunk, holders in pieces.distinct:
        slots = [
            gathered.narrow(dim, holder * widest, len(chunk)) for holder in holders
        ]
        part = slots[0]
        if reduce:
            for other in slots[1:]:
                part = part + other
        parts.append(part)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


# --------------------------------------------------------------------------- #
# the pairs
# --------------------------------------------------------------------------- #


class _GatherForEdit(torch.autograd.Function):
    """Forward the raw all-gather (padded and cut where the chunks are
    uneven, one copy of a repeated chunk kept); backward this rank's slice
    of the incoming gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        collective: Collective,
        axis: Axis,
        pieces: _Pieces,
    ) -> torch.Tensor:
        rank = collective.rank(axis)
        mine = pieces.chunks[rank]
        _check_extent(tensor, pieces.dim, len(mine), "gather_for_edit")
        ctx.dim, ctx.mine = pieces.dim, mine
        gathered = collective.all_gather(
            _padded(tensor, pieces.dim, pieces.widest), pieces.dim, axis
        )
        if pieces.uniform:
            return gathered
        return _assemble(gathered, pieces, reduce=False)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        mine: range = ctx.mine
        return grad.narrow(ctx.dim, mine.start, len(mine)), None, None, None


class _EditFragment(torch.autograd.Function):
    """Forward this rank's slice; backward the ranks' gradient slices
    all-gathered into the whole gradient (a repeated chunk's partials
    summed in rank order)."""

    @staticmethod
    def forward(
        ctx: Any,
        edited: torch.Tensor,
        collective: Collective,
        axis: Axis,
        pieces: _Pieces,
    ) -> torch.Tensor:
        rank = collective.rank(axis)
        mine = pieces.chunks[rank]
        _check_extent(edited, pieces.dim, pieces.whole_extent, "edit_fragment")
        ctx.collective, ctx.axis, ctx.pieces = collective, axis, pieces
        return edited.narrow(pieces.dim, mine.start, len(mine)).clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        pieces: _Pieces = ctx.pieces
        gathered = ctx.collective.all_gather(
            _padded(grad, pieces.dim, pieces.widest), pieces.dim, ctx.axis
        )
        if pieces.uniform:
            return gathered, None, None, None
        return _assemble(gathered, pieces, reduce=True), None, None, None


class _SumForEdit(torch.autograd.Function):
    """Forward the raw all-reduce sum; backward the identity."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, collective: Collective, axis: Axis
    ) -> torch.Tensor:
        return collective.all_reduce_sum(tensor, axis)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad, None, None


class _EditSummand(torch.autograd.Function):
    """Forward this rank's summand — the kept slots, or the whole edit on the
    group's first rank and zeros elsewhere; backward the masked gradients
    all-reduce-summed, the whole gradient since the kept sets partition."""

    @staticmethod
    def forward(
        ctx: Any,
        edited: torch.Tensor,
        collective: Collective,
        axis: Axis,
        keep: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.collective, ctx.axis, ctx.keep = collective, axis, keep
        ctx.first = collective.rank(axis) == 0
        if keep is None:
            return edited.clone() if ctx.first else torch.zeros_like(edited)
        return torch.where(keep, edited, edited.new_zeros(()))

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        keep: torch.Tensor | None = ctx.keep
        if keep is None:
            mine = grad if ctx.first else torch.zeros_like(grad)
        else:
            mine = torch.where(keep, grad, grad.new_zeros(()))
        return (
            ctx.collective.all_reduce_sum(mine.contiguous(), ctx.axis),
            None,
            None,
            None,
        )


class _GatherReduceScatter(torch.autograd.Function):
    """Forward the raw all-gather of equal chunks; backward the gradients
    all-reduce-summed, then this rank's slice."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, collective: Collective, axis: Axis, dim: int
    ) -> torch.Tensor:
        dim = dim % tensor.dim()
        ctx.collective, ctx.axis, ctx.dim = collective, axis, dim
        ctx.mine = equal_chunks(
            tensor.shape[dim] * collective.size(axis), collective.size(axis)
        )[collective.rank(axis)]
        return collective.all_gather(tensor.contiguous(), dim, axis)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        total = ctx.collective.all_reduce_sum(grad.contiguous(), ctx.axis)
        mine: range = ctx.mine
        return total.narrow(ctx.dim, mine.start, len(mine)), None, None, None


def _leaf(device: torch.device) -> torch.Tensor:
    """A zero-size leaf that requires a gradient: what puts a node on a rank's
    graph when the tensor it sends requires none (``send_with_grad``), or
    when the tensor it receives attaches to nothing of its own
    (``recv_with_grad`` under ``Handoff.ALWAYS``)."""
    return torch.zeros(0, device=device, requires_grad=True)


# --------------------------------------------------------------------------- #
# the link's agreement (module docstring)
# --------------------------------------------------------------------------- #

#: One ordered link on an axis: ``(axis, sender's group-local rank,
#: receiver's group-local rank)``.
_Link = tuple[Axis, int, int]

#: The protocol each link of a collective agreed, kept for the collective's
#: lifetime: the world's collective is one object per process, the
#: simulator's one per rank per run.
_AGREED: weakref.WeakKeyDictionary[Collective, dict[_Link, Handoff]] = (
    weakref.WeakKeyDictionary()
)

#: What a group-local rank is called on each axis in a refusal.
_MEMBER = {"pipeline": "stage", "context": "chunk"}


def _member(axis: Axis, rank: int) -> str:
    return f"{_MEMBER.get(axis, 'rank')} {rank}"


def _agree_handoff(
    collective: Collective,
    axis: Axis,
    src: int,
    dst: int,
    handoff: Handoff,
    device: torch.device,
) -> None:
    """The link ``src → dst`` on ``axis`` agrees ``handoff`` (module
    docstring), once: on its first handoff the two ends exchange their
    codes — the sender sends then receives, the receiver receives then
    sends, so the two blocking point-to-points pair up — and compare; the
    agreement is remembered on the collective. Called by both ends at the
    same handoff, since both ends run the same program (§3).

    Raises:
        HandoffMismatch: the other end runs another protocol, or this end
            asks a second protocol of a link already agreed.
    """
    me = collective.rank(axis)
    sender = me == src
    links = _AGREED.setdefault(collective, {})
    link: _Link = (axis, src, dst)
    agreed = links.get(link)
    if agreed is not None:
        if agreed != handoff:
            raise HandoffMismatch(
                f"the {axis} handoff {_member(axis, src)} → {_member(axis, dst)} "
                f"agreed the {agreed.name} protocol at its first handoff and "
                f"{_member(axis, me)} now asks {handoff.name}; one link runs one "
                "protocol (docs/model_parallelism.md §7)"
            )
        return
    mine = torch.tensor([int(handoff)], dtype=torch.int64, device=device)
    peer = dst if sender else src
    if sender:
        collective.send(mine, peer, axis)
        theirs = collective.recv((1,), torch.int64, device, peer, axis)
    else:
        theirs = collective.recv((1,), torch.int64, device, peer, axis)
        collective.send(mine, peer, axis)
    code = int(theirs.item())
    if code != int(handoff):
        try:
            other = Handoff(code).name
        except ValueError:
            other = f"an unknown protocol (code {code})"
        raise HandoffMismatch(
            f"the {axis} handoff {_member(axis, src)} → {_member(axis, dst)}: "
            f"{_member(axis, me)} ({'the sender' if sender else 'the receiver'}) runs "
            f"the {handoff.name} protocol while {_member(axis, peer)} runs {other}; "
            "the gradient would cross back on one end alone and the other would "
            "wait in backward forever — both ends must send and receive it under "
            "one protocol (docs/model_parallelism.md §7)"
        )
    links[link] = handoff


class _SendWithGrad(torch.autograd.Function):
    """Forward the raw send, the tensor returned so the graph continues;
    backward the gradient received from ``dst`` added to the local one.
    ``link`` is the tensor itself — the node recorded iff it requires a
    gradient — or, for a sender whose peer sends a gradient back regardless
    (``Handoff.ALWAYS``), a zero-size leaf that does, so the node is
    recorded and the receive runs for a residual nothing on this rank
    trains."""

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        link: torch.Tensor,
        collective: Collective,
        dst: int,
        axis: Axis,
    ) -> torch.Tensor:
        ctx.collective, ctx.dst, ctx.axis = collective, dst, axis
        collective.send(tensor.contiguous(), dst, axis)
        return tensor

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        received = ctx.collective.recv(
            tuple(grad.shape), grad.dtype, grad.device, ctx.dst, ctx.axis
        )
        return grad + received, None, None, None, None


class _RecvWithGrad(torch.autograd.Function):
    """Forward the raw receive, attached to ``link``; backward the gradient
    sent to ``src``, nothing to ``link``."""

    @staticmethod
    def forward(
        ctx: Any,
        link: torch.Tensor,
        collective: Collective,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: Axis,
    ) -> torch.Tensor:
        ctx.collective, ctx.src, ctx.axis = collective, src, axis
        return collective.recv(shape, dtype, device, src, axis)

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[None, None, None, None, None, None, None]:
        ctx.collective.send(grad.contiguous(), ctx.src, ctx.axis)
        return None, None, None, None, None, None, None


# --------------------------------------------------------------------------- #
# the public functions
# --------------------------------------------------------------------------- #


class _Carry(torch.autograd.Function):
    """The tensor as is, with ``dependency`` on its backward path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, tensor: torch.Tensor, dependency: torch.Tensor
    ) -> torch.Tensor:
        return tensor.view_as(tensor)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        # ``None`` reaches the dependency's node as a materialized zero
        # gradient: the node runs (a send's backward receives the peer's),
        # and contributes nothing of its own
        return grad, None


def gather_for_edit(
    tensor: torch.Tensor,
    dim: int,
    axis: Axis,
    collective: Collective,
    *,
    chunks: Sequence[range] | None = None,
) -> torch.Tensor:
    """The whole axis from every rank's piece — ``whole`` of a gather
    placement — whose backward is this rank's slice of the gradient.

    ``chunks`` is each rank's range of the whole ``dim`` ([`Chunks`][]);
    ``None`` is ``size`` equal chunks of the local extent. Repeated ranges
    keep one copy; uneven ones travel padded.
    """
    pieces = _pieces(dim, chunks, tensor, collective.size(axis), whole=False)
    return _GatherForEdit.apply(tensor, collective, axis, pieces)


def edit_fragment(
    edited: torch.Tensor,
    dim: int,
    axis: Axis,
    collective: Collective,
    *,
    chunks: Sequence[range] | None = None,
) -> torch.Tensor:
    """This rank's slice of the edited whole — ``fragment`` of a gather
    placement — whose backward all-gathers the ranks' gradient slices into
    the whole gradient (``chunks`` as for [`gather_for_edit`][]; a repeated
    chunk's partials are summed)."""
    pieces = _pieces(dim, chunks, edited, collective.size(axis), whole=True)
    return _EditFragment.apply(edited, collective, axis, pieces)


def sum_for_edit(
    tensor: torch.Tensor, axis: Axis, collective: Collective
) -> torch.Tensor:
    """The all-reduce sum — ``whole`` of a sum placement, and the model's own
    reduction of partial products — whose backward is the identity."""
    return _SumForEdit.apply(tensor, collective, axis)


def edit_summand(
    edited: torch.Tensor,
    keep: torch.Tensor | None,
    axis: Axis,
    collective: Collective,
) -> torch.Tensor:
    """This rank's summand of the edited whole — ``fragment`` of a sum
    placement: ``where(keep, edited, 0)`` for the slots this rank owns
    (``ExpertLocal``), or with ``keep=None`` the edit on the group's first
    rank and zeros elsewhere (``PartialSum``) — whose backward all-reduce-sums
    the masked gradients into the whole gradient."""
    return _EditSummand.apply(edited, collective, axis, keep)


def gather_reduce_scatter(
    tensor: torch.Tensor, dim: int, axis: Axis, collective: Collective
) -> torch.Tensor:
    """The faithful gather (module docstring): the whole axis from equal
    chunks, whose backward sums every rank's gradient and keeps this rank's
    slice — for a consumer that differs per rank."""
    return _GatherReduceScatter.apply(tensor, collective, axis, dim)


def send_with_grad(
    tensor: torch.Tensor,
    dst: int,
    axis: Axis,
    collective: Collective,
    *,
    handoff: Handoff = Handoff.IF_REQUIRED,
) -> torch.Tensor:
    """Send ``tensor`` to group-local ``dst`` and return it, so the sender's
    graph continues; in backward the gradient ``dst`` sends back is added to
    the sender's own. A sender with no local continuation drives its backward
    from the returned tensor with a zero gradient.

    Whether the node exists must agree with the peer's [`recv_with_grad`][]
    — the [`Handoff`][] protocol, agreed by the link at its first handoff
    and a mismatch refused on both ends (module docstring). Under
    ``IF_REQUIRED`` (the default) the node is recorded iff ``tensor``
    requires a gradient — the DeltaNet handoff, where both ends read that
    off the same kernel value (§6.4). Under ``ALWAYS`` it is recorded
    whether or not ``tensor`` requires one, through a zero-size leaf standing
    in as the graph's input — the pipeline's residual, whose stage above
    sends the gradient back regardless of what the stage below trains
    (§6.5). Nothing is recorded under ``no_grad`` either way.

    Raises:
        HandoffMismatch: the receiver runs the other protocol.
    """
    _agree_handoff(collective, axis, collective.rank(axis), dst, handoff, tensor.device)
    always = handoff is Handoff.ALWAYS
    link = tensor if tensor.requires_grad or not always else _leaf(tensor.device)
    return _SendWithGrad.apply(tensor, link, collective, dst, axis)


def recv_with_grad(
    link: torch.Tensor | None,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    src: int,
    axis: Axis,
    collective: Collective,
    *,
    handoff: Handoff = Handoff.IF_REQUIRED,
) -> torch.Tensor:
    """Receive a ``shape`` / ``dtype`` tensor on ``device`` from group-local
    ``src``, attached to ``link`` — a tensor on this rank's graph, so the
    receive is recorded and its backward runs — and in backward send the
    gradient to ``src``. ``link`` itself receives no gradient.

    The [`Handoff`][] protocol must be the sender's (module docstring).
    Under ``IF_REQUIRED`` (the default) ``link`` is required and the
    backward sends iff it requires a gradient, as the sender's node exists
    iff its tensor does. Under ``ALWAYS`` the backward sends regardless:
    ``link`` may be ``None`` — or a tensor requiring no gradient — and a
    zero-size leaf stands in, so the node is recorded on a stage that
    trains nothing above the boundary too (§6.5).

    Raises:
        HandoffMismatch: the sender runs the other protocol.
        AutogradError: no ``link`` under ``IF_REQUIRED``.
    """
    if handoff is Handoff.ALWAYS:
        if link is None or not link.requires_grad:
            link = _leaf(device)
    elif link is None:
        raise AutogradError(
            "recv_with_grad under Handoff.IF_REQUIRED needs a link: the tensor "
            "whose requires_grad says whether the gradient crosses back"
        )
    _agree_handoff(collective, axis, src, collective.rank(axis), handoff, device)
    return _RecvWithGrad.apply(link, collective, shape, dtype, device, src, axis)


def carry(tensor: torch.Tensor, dependency: torch.Tensor) -> torch.Tensor:
    """``tensor`` unchanged, with ``dependency``'s graph on its backward path.
    Two uses, one Function: a tensor whose only consumer is a peer — the
    final state a rank sends on (``context.handoff_state``) — is carried onto
    an output that reaches the loss, so its [`send_with_grad`][] backward
    is reached; and a value another rank computed — the broadcast logits, a
    broadcast capture (``stages.attach_received``) — is put on this rank's
    graph behind the residual it sent, bit for bit, so ``loss.backward()``
    on a stage that did not compute the loss reaches the boundary. The
    dependency's node runs with a materialized zero gradient and contributes
    nothing of its own; several values carried onto one dependency reach it
    once. The identity when the dependency has no graph."""
    if not dependency.requires_grad:
        return tensor
    return _Carry.apply(tensor, dependency)
