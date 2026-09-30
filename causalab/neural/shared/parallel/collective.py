"""The one interface through which the engine talks to other ranks (``docs/model_parallelism.md`` §3).

Every collective the engine needs is a method here, named by mesh axis; the
production [`TorchCollective`][] resolves an axis to a ``torch.distributed``
group through a [`Mesh`][], the world-1 [`Solo`][] is the
identity, and the tests' ``SimulatedWorld`` runs many ranks in one process
behind the same protocol (§10.1–10.2). The three are held to one conformance
suite, ``tests/neural/shared/parallel/collective_contract.py``.

Contract, for every implementation:

- ``all_gather`` concatenates the members' tensors **in rank order** along
  ``dim``; every member receives the same result.
- ``all_reduce_sum`` returns a **new** tensor equal on every member; the
  argument is not mutated.
- ``broadcast`` is called by every member of the group; non-source members
  pass ``None`` and receive the source's tensor (shape and dtype travel with
  it). The source receives its own tensor back.
- ``send`` / ``recv`` are point to point between two ranks of one axis.
- ``src`` and ``dst`` are **group-local** indices on ``axis`` — what ``rank(axis)``
  returns and what a ``StageLocal.stage`` names — never global ranks. A
  production implementation over ``torch.distributed`` translates them through
  the group (``get_global_rank``); the simulator already reads them this way.
- ``agree_*`` are the three host-side agreements (§3): every member receives
  the ``min`` / ``any`` / ``sum`` of the members' values as a Python scalar,
  so control flow that depends on device state cannot diverge between ranks.
- ``rank`` and ``size`` describe this process's position on an axis; at world
  1 every axis has size 1 and rank 0.

**When a peer is gone** (§3 "when a rank dies"): every ``torch.distributed``
call of the [`TorchCollective`][] runs inside [`inside`][causalab.neural.shared.parallel.watchdog.inside],
so the rank's heartbeat can say which collective it was waiting in, and a
backend error there — the group's timeout (``CAUSALAB_COLLECTIVE_TIMEOUT``),
a communicator aborted under a dead peer, gloo's read failing the instant a
peer's socket closes — first asks the process's heartbeat
(``heartbeat.running().hold()``): a peer lost within the bound is refused
by name there and the process ends; otherwise the error is re-rendered as
[`CollectiveFailed`][], a refusal naming the axis, the op, this rank and
the group, never a bare backend traceback.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from causalab.neural.shared.parallel import heartbeat, watchdog
from causalab.neural.shared.parallel.placement import Axis
from causalab.protocol.rules.errors import ProtocolError

if TYPE_CHECKING:
    from causalab.neural.shared.parallel.heartbeat import Heartbeat
    from causalab.neural.shared.parallel.mesh import Mesh


@runtime_checkable
class Collective(Protocol):
    """One rank's view of a group per mesh axis (§3): its position, the
    tensor collectives, the scalar agreements; ``device`` is where its
    tensors live."""

    @property
    def device(self) -> torch.device:
        """The device every tensor handed to this collective must sit on
        — the backend's (``cpu`` under ``gloo``, this process's ``cuda``
        device under NCCL), ``cpu`` for a collective moving no bytes. A
        record or a vector born to cross the collective is made here.
        Requiring the device on the protocol prevents callers from silently
        creating CPU tensors for a CUDA collective."""
        ...

    def rank(self, axis: Axis) -> int: ...

    def size(self, axis: Axis) -> int: ...

    def all_gather(
        self, tensor: torch.Tensor, dim: int, axis: Axis
    ) -> torch.Tensor: ...

    def all_reduce_sum(self, tensor: torch.Tensor, axis: Axis) -> torch.Tensor: ...

    def broadcast(
        self, tensor: torch.Tensor | None, src: int, axis: Axis
    ) -> torch.Tensor: ...

    def send(self, tensor: torch.Tensor, dst: int, axis: Axis) -> None: ...

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: Axis,
    ) -> torch.Tensor: ...

    def agree_min(self, value: int, axis: Axis) -> int: ...

    def agree_any(self, value: bool, axis: Axis) -> bool: ...

    def agree_sum(self, value: int, axis: Axis) -> int: ...

    def barrier(self, axis: Axis) -> None: ...


class Solo:
    """World 1: every axis has one member, and every collective is the identity.

    This is what today's single-device engine runs under, so the code paths
    that take a [`Collective`][] are exercised by the whole existing suite.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def rank(self, axis: Axis) -> int:
        return 0

    def size(self, axis: Axis) -> int:
        return 1

    def all_gather(self, tensor: torch.Tensor, dim: int, axis: Axis) -> torch.Tensor:
        return tensor

    def all_reduce_sum(self, tensor: torch.Tensor, axis: Axis) -> torch.Tensor:
        return tensor

    def broadcast(
        self, tensor: torch.Tensor | None, src: int, axis: Axis
    ) -> torch.Tensor:
        if tensor is None:
            raise ValueError(
                "Solo.broadcast: the single rank is the source and must pass its tensor"
            )
        return tensor

    def send(self, tensor: torch.Tensor, dst: int, axis: Axis) -> None:
        raise ValueError("Solo.send: a world of one rank has no peer to send to")

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: Axis,
    ) -> torch.Tensor:
        raise ValueError("Solo.recv: a world of one rank has no peer to receive from")

    def agree_min(self, value: int, axis: Axis) -> int:
        return value

    def agree_any(self, value: bool, axis: Axis) -> bool:
        return value

    def agree_sum(self, value: int, axis: Axis) -> int:
        return value

    def barrier(self, axis: Axis) -> None:
        return None


SOLO = Solo()


# --------------------------------------------------------------------------- #
# TorchCollective — the production implementation over the mesh's groups
# --------------------------------------------------------------------------- #


class CollectiveError(ValueError):
    """A collective was called outside its contract: a peer index outside the
    group or naming this rank, a source passing ``None`` or a non-source
    passing a tensor, a tensor off the collective's device, members
    disagreeing on shape or dtype. Raised before ``torch.distributed`` is
    touched wherever the rank can tell alone, so a misuse is a refusal by
    name and never a hang."""


class CaptureUnsafe(CollectiveError):
    """A host-bound collective (``agree_*``, ``barrier``, ``broadcast``) was
    called while this rank's stream is capturing a CUDA graph. Raised before
    ``torch.distributed`` is touched, instead of letting the capture fail with
    a CUDA error that names no collective (``docs/cuda_graphs.md``
    "Multi-rank execution")."""


def capturing(device: torch.device) -> bool:
    """Whether this thread's current CUDA stream is being captured into a
    CUDA graph — a forward's hooks and the autograd device thread's backward
    alike. The current device is the collective's (``launcher.join_group``
    pins every rank to ``cuda:LOCAL_RANK``); ``device`` only rules out a
    backend off CUDA, where it is never true."""
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


class CollectiveFailed(ProtocolError):
    """The backend failed inside a collective and no peer's death explains
    it: the group's timeout passed with a peer missing, or the communicator
    was aborted. ``P4`` at ``--parallel``, naming the op, the axis, this
    rank, the group's ranks and the backend's own words; ``watch`` is the
    heartbeat that was asked and named nobody — a dead peer would have been
    refused by name before this — ``None`` where no watchdog runs
    (module docstring); [`watched`][] records which."""

    def __init__(
        self,
        *,
        op: str,
        axis: Axis,
        rank: int,
        world: int,
        members: tuple[int, ...],
        cause: BaseException,
        watch: Heartbeat | None,
    ) -> None:
        self.op = op
        self.axis = axis
        self.rank = rank
        self.members = members
        self.watched = watch is not None
        first_line = str(cause).strip().splitlines()[0] if str(cause).strip() else ""
        if watch is not None:
            peers = (
                "no peer went silent for the grace after the failure "
                f"({watchdog.RANK_GRACE_VARIABLE}={watch.settings.grace:g}), so the "
                "rank watchdog names none"
            )
        else:
            peers = "a peer may have exited; no rank watchdog runs in this process"
        super().__init__(
            "P4",
            f"{op} on axis {axis!r} failed on rank {rank} of {world} (group ranks "
            f"{list(members)}): {type(cause).__name__}: {first_line} — {peers}; a "
            "hang without a death is bounded by "
            f"{watchdog.COLLECTIVE_TIMEOUT_VARIABLE} (docs/model_parallelism.md §3)",
            path="--parallel",
        )


#: The dtypes a header can spell (``broadcast``: shape and dtype travel with
#: the tensor). Position is the code on the wire; append, never reorder.
DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
    torch.complex128,
)

#: The most dimensions a header can carry.
MAX_DIMS = 8

#: A header is ``[ndim, *shape padded with -1 to MAX_DIMS, dtype code]``.
HEADER_WIDTH = MAX_DIMS + 2


def encode_header(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The shape and dtype of ``tensor`` as one int64 vector on ``device``.

    Raises:
        CollectiveError: more than [`MAX_DIMS`][] dimensions, or a dtype
            outside [`DTYPES`][].
    """
    if tensor.dim() > MAX_DIMS:
        raise CollectiveError(
            f"a tensor of {tensor.dim()} dimensions cannot cross the collective; "
            f"the header carries at most {MAX_DIMS}"
        )
    if tensor.dtype not in DTYPES:
        raise CollectiveError(
            f"dtype {tensor.dtype} has no header code; the codes are "
            f"{', '.join(str(d) for d in DTYPES)}"
        )
    header = torch.full((HEADER_WIDTH,), -1, dtype=torch.int64)
    header[0] = tensor.dim()
    if tensor.dim():
        header[1 : 1 + tensor.dim()] = torch.tensor(tensor.shape, dtype=torch.int64)
    header[-1] = DTYPES.index(tensor.dtype)
    return header.to(device)


def decode_header(header: torch.Tensor) -> tuple[tuple[int, ...], torch.dtype]:
    """The inverse of [`encode_header`][]."""
    values = [int(v) for v in header.tolist()]
    ndim = values[0]
    return tuple(values[1 : 1 + ndim]), DTYPES[values[-1]]


def _device_of(device_type: str) -> torch.device:
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(device_type)


class TorchCollective:
    """The protocol over a [`Mesh`][]'s process groups (§3).

    Tensors live on the collective's device — the one the mesh's backend
    runs on (``cpu`` under ``gloo``, this process's ``cuda`` device under
    ``nccl``) — and a tensor elsewhere is refused by name. An axis of size
    one makes every method the identity without touching
    ``torch.distributed``.

    Every collective that carries a tensor first exchanges a small header
    (shape and dtype): ``broadcast`` needs it so a non-source can allocate
    without knowing the shape, and ``all_gather`` / ``all_reduce_sum`` use it
    to refuse members that disagree — the ``SimulatedWorld`` refuses those as
    a divergence, and under a real backend a mismatch would otherwise hang or
    corrupt rather than fail. The cost is one small collective per call.

    **Inside CUDA graph capture** (`capturing`) the header, a host copy,
    cannot run, so ``all_gather`` and ``all_reduce_sum`` skip it. Every rank
    captures the same fixed sequence after an eager warm-up of it in which
    the headers were checked. ``broadcast`` (a non-source sizes its buffer
    from the header), ``agree_*`` and ``barrier`` raise [`CaptureUnsafe`][]
    while capturing. ``send`` / ``recv`` carry no header.
    """

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.device = _device_of(mesh.device_type)

    # -- position -------------------------------------------------------------

    def rank(self, axis: Axis) -> int:
        return self.mesh.local_rank(axis)

    def size(self, axis: Axis) -> int:
        return self.mesh.size(axis)

    # -- plumbing -------------------------------------------------------------

    def _peer(self, index: int, axis: Axis, what: str) -> int:
        """The global rank of group-local ``index`` on ``axis``, refusing an
        index outside the group or naming this rank."""
        size = self.size(axis)
        if isinstance(index, bool) or not 0 <= index < size:
            raise CollectiveError(
                f"{what}={index!r} is outside the {axis} group of {size} ranks"
            )
        if index == self.rank(axis):
            raise CollectiveError(
                f"{what}={index} names this rank itself on axis {axis!r}; "
                "point to point needs two ranks"
            )
        return self.mesh.global_rank(axis, index)

    def _on_device(self, tensor: torch.Tensor, what: str) -> None:
        """The tensor is on the collective's device — type **and** ordinal:
        under ``nccl`` every rank is pinned to ``cuda:LOCAL_RANK``
        (``launcher.join_group``), and a tensor on another ordinal would
        make the backend copy or fail inside the call; refused by name
        here instead."""
        if tensor.device != self.device:
            raise CollectiveError(
                f"{what}: tensor on {tensor.device} but the collective runs on "
                f"{self.device} (the {self.mesh.device_type} backend's device)"
            )

    @contextmanager
    def _inside(self, axis: Axis, op: str) -> Iterator[None]:
        """Around one ``torch.distributed`` call: registered for the
        heartbeat, a backend failure held for the heartbeat's word and then
        re-rendered as [`CollectiveFailed`][] (module docstring). Only the
        call itself is inside — an allocation's out-of-memory is the
        caller's to retry, not a collective failure."""
        with watchdog.inside(axis, op):
            try:
                yield
            except torch.OutOfMemoryError:
                raise
            except RuntimeError as error:
                # gloo's timeout is a RuntimeError; NCCL's DistBackendError
                # is one of its subclasses
                watch = heartbeat.running()
                if watch is not None:
                    # a dead peer is refused by name here, within the bound,
                    # and the process ends; only a failure nobody's death
                    # explains comes back
                    watch.hold()
                raise CollectiveFailed(
                    op=op,
                    axis=axis,
                    rank=self.mesh.rank,
                    world=self.mesh.geometry.world,
                    members=self.mesh.ranks(axis),
                    cause=error,
                    watch=watch,
                ) from error

    def _agree_header(
        self, tensor: torch.Tensor, group: ProcessGroup, axis: Axis, what: str
    ) -> None:
        """Every member's shape and dtype must equal this rank's."""
        mine = encode_header(tensor, self.device)
        headers = [torch.empty_like(mine) for _ in range(self.size(axis))]
        with self._inside(axis, what):
            dist.all_gather(headers, mine, group=group)
        for local, header in enumerate(headers):
            if not torch.equal(header, mine):
                shape, dtype = decode_header(header)
                raise CollectiveError(
                    f"{what} on axis {axis!r}: rank "
                    f"{self.mesh.global_rank(axis, local)} holds {shape} {dtype} "
                    f"while rank {self.mesh.rank} holds {tuple(tensor.shape)} "
                    f"{tensor.dtype}; members must agree"
                )

    def _refuse_inside_capture(self, axis: Axis, op: str) -> None:
        """A host-bound collective on a capturing stream is [`CaptureUnsafe`][]
        (class docstring)."""
        if capturing(self.device):
            raise CaptureUnsafe(
                f"{op} on axis {axis!r} needs a value on the host, which CUDA "
                "graph capture cannot record; it must run outside the captured "
                "region (docs/cuda_graphs.md)"
            )

    def _agree(self, value: int, op: ReduceOp.RedOpType, axis: Axis) -> int:
        group = self.mesh.group(axis)
        if group is None:
            return value
        self._refuse_inside_capture(axis, "agree")
        cell = torch.tensor([value], dtype=torch.int64, device=self.device)
        with self._inside(axis, "agree"):
            dist.all_reduce(cell, op=op, group=group)
        return int(cell.item())

    # -- collectives ----------------------------------------------------------

    def all_gather(self, tensor: torch.Tensor, dim: int, axis: Axis) -> torch.Tensor:
        group = self.mesh.group(axis)
        if group is None:
            return tensor
        self._on_device(tensor, "all_gather")
        if not capturing(self.device):  # the warm-up checked it (class docstring)
            self._agree_header(tensor, group, axis, "all_gather")
        local = tensor.contiguous()
        chunks = [torch.empty_like(local) for _ in range(self.size(axis))]
        with self._inside(axis, "all_gather"):
            dist.all_gather(chunks, local, group=group)  # in group-rank order
        return torch.cat(chunks, dim=dim)

    def all_reduce_sum(self, tensor: torch.Tensor, axis: Axis) -> torch.Tensor:
        group = self.mesh.group(axis)
        if group is None:
            return tensor
        self._on_device(tensor, "all_reduce_sum")
        if not capturing(self.device):  # the warm-up checked it (class docstring)
            self._agree_header(tensor, group, axis, "all_reduce_sum")
        total = tensor.clone(memory_format=torch.contiguous_format)
        with self._inside(axis, "all_reduce_sum"):
            dist.all_reduce(total, op=ReduceOp.SUM, group=group)
        return total

    def broadcast(
        self, tensor: torch.Tensor | None, src: int, axis: Axis
    ) -> torch.Tensor:
        size = self.size(axis)
        if isinstance(src, bool) or not 0 <= src < size:
            raise CollectiveError(
                f"src={src!r} is outside the {axis} group of {size} ranks"
            )
        is_source = src == self.rank(axis)
        group = self.mesh.group(axis)
        source = self.mesh.global_rank(axis, src)
        if tensor is None:
            if is_source:
                raise CollectiveError("the broadcast source must pass its tensor")
            # a non-source: the group has at least two members, so ``group``
            # is a process group and the header tells what to allocate
            self._refuse_inside_capture(axis, "broadcast")
            header = torch.empty(HEADER_WIDTH, dtype=torch.int64, device=self.device)
            with self._inside(axis, "broadcast"):
                dist.broadcast(header, src=source, group=group)
            shape, dtype = decode_header(header)
            out = torch.empty(shape, dtype=dtype, device=self.device)
            with self._inside(axis, "broadcast"):
                dist.broadcast(out, src=source, group=group)
            return out
        if not is_source:
            raise CollectiveError(
                "a non-source of a broadcast passes None, not a tensor: shape and "
                "dtype travel from the source"
            )
        if group is None:
            return tensor
        self._refuse_inside_capture(axis, "broadcast")
        self._on_device(tensor, "broadcast")
        header = encode_header(tensor, self.device)
        payload = tensor.contiguous()
        with self._inside(axis, "broadcast"):
            dist.broadcast(header, src=source, group=group)
            dist.broadcast(payload, src=source, group=group)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int, axis: Axis) -> None:
        peer = self._peer(dst, axis, "dst")
        self._on_device(tensor, "send")
        payload = tensor.contiguous()
        with self._inside(axis, "send"):
            dist.send(payload, dst=peer, group=self.mesh.group(axis))

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: Axis,
    ) -> torch.Tensor:
        peer = self._peer(src, axis, "src")
        out = torch.empty(tuple(shape), dtype=dtype, device=device)
        self._on_device(out, "recv")
        with self._inside(axis, "recv"):
            dist.recv(out, src=peer, group=self.mesh.group(axis))
        return out

    def agree_min(self, value: int, axis: Axis) -> int:
        return self._agree(value, ReduceOp.MIN, axis)

    def agree_any(self, value: bool, axis: Axis) -> bool:
        return bool(self._agree(int(value), ReduceOp.MAX, axis))

    def agree_sum(self, value: int, axis: Axis) -> int:
        return self._agree(value, ReduceOp.SUM, axis)

    def barrier(self, axis: Axis) -> None:
        group = self.mesh.group(axis)
        if group is not None:
            self._refuse_inside_capture(axis, "barrier")
            with self._inside(axis, "barrier"):
                dist.barrier(group=group)
