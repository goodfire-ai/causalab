"""Context (sequence) parallelism: the frame one forward runs in and the three
places its positions cross ranks (``docs/model_parallelism.md`` §6.4, §8.4).

Under ``cp = c`` every rank of a context group runs the model over one
contiguous chunk of the **padded** position axis — the chunks are
[`sequence_chunks`][] of the padded length,
equal with the remainder on the last rank, a pure function of two integers
every rank computes alike. Embeddings, norms, MLPs, routed experts and the
residual adds are position-local and need nothing. Three things are not:

- **a tapped tensor** (§4): its ``SequenceSharded`` placement gathers the
  chunks along the position axis before the read or write and keeps this
  rank's chunk after (``fragments.py``), through [`SequenceFrame.gather`][]
  / [`SequenceFrame.fragment`][]. The chunks are uneven when ``c`` does
  not divide the frame, and a collective's all-gather wants equal shapes, so
  the gather pads every chunk to the widest, gathers, and cuts each rank's
  piece back to its own length — exact, since padding is dropped;
- **attention**: keys and values are gathered over the group along the
  position axis *inside* the eager attention function the engine already
  owns (``attention_interface.py``), the query stays local, and the causal
  mask is the whole frame's rows at this rank's chunk against the full key
  axis ([`SequenceFrame.attention_mask`][] — the same mask transformers
  builds for the whole frame, sliced). All-gather KV, not ring attention:
  the registered tasks' prompts are short and the code is one gather;
- **the DeltaNet recurrence**: the chunked gated-delta kernel takes an
  ``initial_state``, so rank ``r`` receives the final state of rank ``r − 1``,
  runs its chunk, and sends its own final state on
  ([`handoff_state`][]); the causal conv1d before it needs the previous
  chunk's last ``kernel − 1`` inputs, handed over the same way
  ([`chunked_conv`][]). The DeltaNet layers are therefore **sequential**
  across the group — memory relief, no speed-up there; the attention layers
  and everything position-local run in parallel.

**The gradient crosses where the forward does** (§7; ``autograd.py``). A
tap's gather and fragment are the *tap pair*, which ``fragments.py`` routes
a ``SequenceSharded`` tap through under grad over this frame's ``chunks``
(the frame's own ``gather`` / ``fragment`` are the raw collectives) — the
loss every rank computes is the same scalar over the same whole tensor (the
logits are a tap), so the whole's gradient is this rank's chunk and the
fragment's the all-gather of the chunks' gradients, which hands every rank
the **full** gradient of the edit's parameters. The KV gather is the *faithful* one
([`SequenceFrame.gather_faithful`][]): the gathered keys feed this rank's
queries alone, so chunk ``j``'s key gradient is the all-reduce sum over the
ranks' queries, then the chunk. The DeltaNet handoffs travel with their
gradient (``send_with_grad`` / ``recv_with_grad`` under
``Handoff.IF_REQUIRED``: the gradient crosses back iff the state requires
one, both ends reading that off the kernel's value; the link agrees the
protocol at its first handoff and a mismatch is refused on both chunks by
name — ``autograd.py``): the state rank ``r + 1`` received carries back
what its chunk made of it, in reverse chunk order by construction. Every
backward collective mirrors a forward one, so the
sequences agree on every rank; a tensor no local consumer reads (the final
state on a sending rank) is [`carry`][]-ed onto the output so
its send's backward is reached.

The frame is **bound per forward** ([`activate`][]), thread-locally: the
attention registry entry and the kernel globals are process-wide while
installed, and the executor enters the frame around the model call so the
wrappers read the frame of the forward they are inside — a rank is a
process in production, and a thread in the simulator, so the thread-local
is the one binding that serves both.

Refusals are ``P4`` naming ``--parallel.context``: a frame shorter than the
group ([`sequence_chunks`][]), a decode, a conv
width the chunks cannot feed. A local extent that is not this rank's chunk
is a [`PlacementError`][causalab.neural.shared.parallel.fragments.PlacementError] — an internal invariant, never a
document's fault.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from typing import Callable, Iterator

import torch

from causalab.neural.shared.parallel.autograd import (
    carry,
    edit_fragment,
    gather_for_edit,
    gather_reduce_scatter,
    recv_with_grad,
    send_with_grad,
)
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import sequence_chunks

__all__ = [
    "AXIS",
    "SequenceError",
    "SequenceFrame",
    "activate",
    "chunked_conv",
    "current",
    "handoff_state",
    "local_causal_mask",
]

#: The mesh axis the positions are split over.
AXIS: Axis = "context"


class SequenceError(ValueError):
    """A tensor does not fit the frame it was handed to: its position extent
    is neither this rank's chunk nor the whole frame, or a row count the
    flat token axis does not unfold by. An internal invariant broken, never
    a rule a document violated — the same reasoning as ``PlacementError``."""


def local_causal_mask(
    attention_mask: torch.Tensor, chunk: range, dtype: torch.dtype
) -> torch.Tensor:
    """The additive causal mask of the whole padded frame at the query rows
    ``chunk``, ``(batch, 1, len(chunk), padded_len)``: ``0`` where key
    position ``k`` is at most the query position and is a real token of the
    row (``attention_mask[b, k]``), the dtype's minimum elsewhere — exactly
    transformers' eager mask (``masking_utils.eager_mask``: the boolean
    causal-and-padding mask, ``where(mask, 0, finfo.min)``) for the whole
    frame, sliced to these rows. A fully padded query row stays all-minimum,
    as transformers leaves it, so the softmax over it is what the whole frame
    computes too.
    """
    if attention_mask.dim() != 2:
        raise SequenceError(
            f"the attention mask is (batch, padded_len), got {tuple(attention_mask.shape)}"
        )
    padded_len = attention_mask.shape[1]
    if not 0 <= chunk.start <= chunk.stop <= padded_len:
        raise SequenceError(
            f"query chunk {chunk} is outside a frame of {padded_len} positions"
        )
    device = attention_mask.device
    queries = torch.arange(chunk.start, chunk.stop, device=device)
    keys = torch.arange(padded_len, device=device)
    causal = keys[None, :] <= queries[:, None]  # (q_local, padded_len)
    allowed = causal[None, :, :] & attention_mask.to(torch.bool)[:, None, :]
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    floor = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.where(allowed, zero, floor).unsqueeze(1)


@dataclasses.dataclass(frozen=True)
class SequenceFrame:
    """One forward's position frame as the context group holds it: the
    collective the chunks travel over and the whole frame's 2-D attention
    mask, ``(rows, padded_len)`` — every rank encodes the same batch, so
    every rank builds the same frame without a collective.

    Raises:
        ParseError: ``P4`` at ``--parallel.context`` — the frame is shorter
            than the group (``sequence_chunks``).
    """

    collective: Collective
    mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.mask.dim() != 2:
            raise SequenceError(
                f"the frame's mask is (rows, padded_len), got {tuple(self.mask.shape)}"
            )
        sequence_chunks(self.padded_len, self.size)  # the frame fits the group

    # -- geometry -------------------------------------------------------------

    @property
    def rows(self) -> int:
        return int(self.mask.shape[0])

    @property
    def padded_len(self) -> int:
        return int(self.mask.shape[1])

    @property
    def size(self) -> int:
        return self.collective.size(AXIS)

    @property
    def rank(self) -> int:
        return self.collective.rank(AXIS)

    @property
    def chunks(self) -> tuple[range, ...]:
        return sequence_chunks(self.padded_len, self.size)

    @property
    def chunk(self) -> range:
        """This rank's positions of the padded frame."""
        return self.chunks[self.rank]

    @property
    def first(self) -> bool:
        return self.rank == 0

    @property
    def last(self) -> bool:
        return self.rank == self.size - 1

    # -- the batch --------------------------------------------------------------

    def narrow(self, tensor: torch.Tensor) -> torch.Tensor:
        """A ``(rows, padded_len, …)`` batch tensor — ids, mask, position ids
        — at this rank's chunk of positions."""
        return self.fragment(tensor, 1)

    # -- whole / fragment along a position axis ---------------------------------

    def check_whole(self, tensor: torch.Tensor, axis: int) -> int:
        """``axis`` normalised, refused unless its extent is the whole frame."""
        axis = self._axis(tensor, axis)
        extent = tensor.shape[axis]
        if extent != self.padded_len:
            raise SequenceError(
                f"axis {axis} has extent {extent}, not the frame's {self.padded_len} "
                "positions"
            )
        return axis

    def check_local(self, tensor: torch.Tensor, axis: int) -> int:
        """``axis`` normalised, refused unless its extent is this rank's chunk."""
        axis = self._axis(tensor, axis)
        extent = tensor.shape[axis]
        if extent != len(self.chunk):
            raise SequenceError(
                f"axis {axis} has extent {extent}, not this rank's chunk of "
                f"{len(self.chunk)} positions ({self.chunk} of {self.padded_len})"
            )
        return axis

    def fragment(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """This rank's chunk of a tensor whole along ``axis``: the raw narrow,
        or under grad the tap pair's [`edit_fragment`][] over
        [`chunks`][] (§7), whose backward all-gathers the ranks' chunk
        gradients — the one chunk table ``autograd.py`` pads uneven chunks by.
        ``fragments.py`` reaches this method with no-grad tensors alone and
        routes a differentiable tap through the same function itself, so a
        tap is wrapped once whichever way it arrives."""
        axis = self.check_whole(tensor, axis)
        if self.size > 1 and _differentiable(tensor):
            return edit_fragment(
                tensor, axis, AXIS, self.collective, chunks=self.chunks
            )
        return tensor.narrow(axis, self.chunk.start, len(self.chunk))

    def gather(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """The whole frame along ``axis`` from every rank's chunk, in position
        order — padded to the widest chunk for the all-gather and cut back,
        so uneven chunks gather exactly: the raw all-gather, or under grad the
        tap pair's [`gather_for_edit`][] over [`chunks`][]
        (§7), whose backward is this rank's chunk of the gradient (see
        [`fragment`][] for the one-wrapping rule)."""
        axis = self.check_local(tensor, axis)
        if self.size > 1 and _differentiable(tensor):
            return gather_for_edit(
                tensor, axis, AXIS, self.collective, chunks=self.chunks
            )
        return self._gather(tensor, axis, None)

    def gather_faithful(self, tensor: torch.Tensor, axis: int) -> torch.Tensor:
        """[`gather`][] for a downstream that differs per rank (the
        attention's queries against the gathered keys, §8.4): through
        [`gather_reduce_scatter`][], whose backward all-reduce-sums
        the gradient over the group and keeps this rank's chunk — the padding
        makes the chunks equal for it, and is cut after."""
        return self._gather(tensor, axis, gather_reduce_scatter)

    def _gather(
        self,
        tensor: torch.Tensor,
        axis: int,
        function: Callable[[torch.Tensor, int, Axis, Collective], torch.Tensor] | None,
    ) -> torch.Tensor:
        axis = self.check_local(tensor, axis)
        if self.size == 1:
            return tensor
        chunks = self.chunks
        widest = max(len(chunk) for chunk in chunks)
        local = self._padded(tensor, axis, widest).contiguous()
        if function is None:
            gathered = self.collective.all_gather(local, axis, AXIS)
        else:
            gathered = function(local, axis, AXIS, self.collective)
        if all(len(chunk) == widest for chunk in chunks):
            return gathered
        pieces = [
            gathered.narrow(axis, rank * widest, len(chunk))
            for rank, chunk in enumerate(chunks)
        ]
        return torch.cat(pieces, dim=axis)

    @staticmethod
    def _padded(tensor: torch.Tensor, axis: int, widest: int) -> torch.Tensor:
        """``tensor`` zero-padded along ``axis`` to ``widest`` positions."""
        extent = tensor.shape[axis]
        if extent >= widest:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[axis] = widest - extent
        return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=axis)

    def unflatten(
        self, tensor: torch.Tensor, axis: int, *, local: bool
    ) -> torch.Tensor:
        """A flattened ``(rows · positions)`` axis unfolded into ``(rows,
        positions)`` — this rank's chunk when ``local``, the whole frame
        otherwise — so a token-major view chunks and gathers by position."""
        axis = self._axis(tensor, axis)
        positions = len(self.chunk) if local else self.padded_len
        if tensor.shape[axis] != self.rows * positions:
            raise SequenceError(
                f"axis {axis} has extent {tensor.shape[axis]}, not {self.rows} rows × "
                f"{positions} positions"
            )
        return tensor.unflatten(axis, (self.rows, positions))

    # -- attention --------------------------------------------------------------

    def attention_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """The whole frame's eager causal mask at this rank's query rows,
        ``(rows, 1, len(chunk), padded_len)`` ([`local_causal_mask`][])."""
        return local_causal_mask(self.mask, self.chunk, dtype)

    # -- refusals ---------------------------------------------------------------

    def check_conv_width(self, kernel_size: int) -> None:
        """A causal conv of ``kernel_size`` taps needs the previous chunk's
        last ``kernel_size − 1`` inputs, so every chunk must hold at least
        that many positions — else the prefix would span two ranks.

        Raises:
            ProtocolError: ``P4`` naming ``--parallel.context``.
        """
        needed = kernel_size - 1
        shortest = min(len(chunk) for chunk in self.chunks)
        if shortest >= needed:
            return
        raise ProtocolError(
            "P4",
            f"--parallel.context: cp={self.size} over a frame of {self.padded_len} "
            f"padded positions leaves a chunk of {shortest}, shorter than the "
            f"DeltaNet conv width {kernel_size} − 1 = {needed} it must hand to the "
            "next rank (docs/model_parallelism.md §6.4); run with fewer context "
            "ranks or longer prompts",
        )

    @staticmethod
    def _axis(tensor: torch.Tensor, axis: int) -> int:
        if not -tensor.dim() <= axis < tensor.dim():
            raise SequenceError(f"axis {axis} is outside a {tensor.dim()}-D tensor")
        return axis % tensor.dim()


def _differentiable(tensor: torch.Tensor) -> bool:
    """Whether a backward can reach ``tensor`` — the same rule ``fragments.py``
    routes by."""
    return torch.is_grad_enabled() and tensor.requires_grad


# --------------------------------------------------------------------------- #
# the DeltaNet handoffs (§6.4)
# --------------------------------------------------------------------------- #


def handoff_state(
    frame: SequenceFrame,
    run: Callable[[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    link: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a recurrence over this rank's chunk **after** the rank below it:
    receive the final state of rank ``r − 1`` (none on the first rank — the
    kernel's own zero state), ``run(initial_state) -> (out, final_state)``
    over the local chunk, and send the final state to rank ``r + 1`` (none
    from the last). ``shape`` / ``dtype`` / ``device`` describe the state
    that crosses, ``(batch, heads, d_k, d_v)`` in the kernel's float32.
    Sequential across the group by construction; a group of one runs the
    recurrence as it is.

    ``link`` puts the handoff on the graph (module docstring): a tensor of
    this rank's forward — the kernel's value — whose ``requires_grad`` says
    whether the received state is differentiable; the state sent on is
    carried onto ``out`` so its send's backward is reached. With no link
    the state crosses raw on both ends, without a gradient, as an inference
    forward's does — every rank passes a link or none, off the same fact."""
    incoming: torch.Tensor | None = None
    if not frame.first:
        if link is None:
            incoming = frame.collective.recv(shape, dtype, device, frame.rank - 1, AXIS)
        else:
            incoming = recv_with_grad(
                link, shape, dtype, device, frame.rank - 1, AXIS, frame.collective
            )
    out, final = run(incoming)
    if not frame.last:
        if final is None:
            raise SequenceError(
                "the recurrence returned no final state to hand to the next rank"
            )
        outgoing = final.to(dtype).contiguous()
        if link is None:
            # the raw path on both ends alike: the rank above receives raw
            # too, and the graded pair's link agreement would otherwise meet
            # its plain receive (§3: one path, chosen from the same fact)
            frame.collective.send(outgoing, frame.rank + 1, AXIS)
        else:
            sent = send_with_grad(outgoing, frame.rank + 1, AXIS, frame.collective)
            out = carry(out, sent)
    return out, final


def chunked_conv(
    frame: SequenceFrame,
    conv: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *args: object,
    **kwargs: object,
) -> torch.Tensor:
    """The causal conv1d over this rank's chunk with the previous chunk's
    tail as its history: ``hidden_states`` is channels-first ``(batch,
    channels, positions)``; rank ``r`` receives the last ``kernel − 1``
    inputs of rank ``r − 1`` (the first rank has none — the conv's own zero
    padding), sends its own last ``kernel − 1`` inputs to ``r + 1``,
    runs ``conv`` over the prefixed chunk and keeps the outputs of its own
    positions — each of which then saw exactly the inputs the whole frame's
    conv would have. Both handoffs travel with their gradient (module
    docstring): the received history is on the graph through the chunk's
    own inputs, and the tail sent on is what the conv reads its own last
    inputs through, so its send's backward is reached.

    Raises:
        ProtocolError: ``P4`` — a chunk shorter than the conv's history.
    """
    kernel_size = int(weight.shape[-1])
    history = kernel_size - 1
    if frame.size == 1 or history == 0:
        return conv(hidden_states, weight, *args, **kwargs)
    frame.check_conv_width(kernel_size)
    positions = hidden_states.shape[-1]
    prefixed = hidden_states
    if not frame.first:
        prefix = recv_with_grad(
            hidden_states,
            (*hidden_states.shape[:-1], history),
            hidden_states.dtype,
            hidden_states.device,
            frame.rank - 1,
            AXIS,
            frame.collective,
        )
        prefixed = torch.cat([prefix, hidden_states], dim=-1)
    if not frame.last:
        tail = send_with_grad(
            hidden_states[..., -history:].contiguous(),
            frame.rank + 1,
            AXIS,
            frame.collective,
        )
        prefixed = torch.cat([prefixed[..., :-history], tail], dim=-1)
    out = conv(prefixed, weight, *args, **kwargs)
    return out[..., -positions:]


# --------------------------------------------------------------------------- #
# the frame of the forward in flight
# --------------------------------------------------------------------------- #

_ACTIVE = threading.local()


def current() -> SequenceFrame | None:
    """The frame the forward in flight on this thread runs in, ``None`` at
    world 1 or outside a forward."""
    return getattr(_ACTIVE, "frame", None)


@contextlib.contextmanager
def activate(frame: SequenceFrame | None) -> Iterator[None]:
    """Bind ``frame`` as the forward's frame for the duration; ``None`` binds
    nothing (world 1, or a geometry without a context axis) and costs one
    attribute read on exit."""
    previous = current()
    _ACTIVE.frame = frame
    try:
        yield
    finally:
        _ACTIVE.frame = previous
