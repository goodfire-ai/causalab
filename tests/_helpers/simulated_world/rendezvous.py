"""A rendezvous: one collective, every member of the group arrived at it.

An arrival carries a `Signature` — what must agree across the group
(op, call site, shape, dtype and the op's arguments) — and a payload; the
rendezvous compares each later signature with the first and, once every
member is in, applies the op on CPU in **fixed rank order**, so the simulated
numerics are deterministic and bit-reproducible under every schedule.

What leaves a rendezvous is **detached**, as it is from a real backend: a
production all-gather fills fresh buffers and an all-reduce a clone that the
backend then overwrites, so no gradient flows through a raw collective and
autograd meets them only in ``parallel/autograd.py``'s Functions. A simulator
that concatenated the members' live tensors would be strictly more forgiving
than production — a ``whole`` that detaches would still look grad-connected
here — and could not find the bug the pairing exists to fix (§7). The one
exception mirrors production too: a broadcast's source receives its *own*
tensor back (``TorchCollective.broadcast`` returns the argument), so the
owner's graph continues through it.
"""

from __future__ import annotations

import os
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import torch

from causalab.neural.shared.parallel.placement import Axis
from tests._helpers.simulated_world.errors import Divergence, Waiting

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep
_ROOT = str(Path(__file__).resolve().parents[3]) + os.sep

#: The signature fields two arrivals must agree on. (The call-site mutation
#: test removes ``call_site`` here and shows the divergence test go red.)
COMPARED_FIELDS: tuple[str, ...] = ("op", "call_site", "shape", "dtype", "detail")


def call_site() -> str:
    """``file:line`` of the nearest caller outside the simulator package,
    relative to the repository root when it lies inside it."""
    frame = inspect.currentframe()
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_PACKAGE):
            if filename.startswith(_ROOT):
                filename = filename[len(_ROOT) :]
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "<unknown>"


class Event(NamedTuple):
    """One transcript line: a rank's ``step``-th collective."""

    step: int
    rank: int
    axis: Axis
    op: str
    call_site: str
    shape: tuple[int, ...] | None


@dataclass(frozen=True)
class Signature:
    op: str
    call_site: str | None
    shape: tuple[int, ...] | None
    dtype: torch.dtype | None
    detail: tuple[Any, ...]

    def differences(self, other: Signature) -> tuple[str, ...]:
        return tuple(
            name
            for name in COMPARED_FIELDS
            if getattr(self, name) != getattr(other, name)
        )


@dataclass(eq=False)
class Arrival:
    rank: int
    axis: Axis
    key: tuple[Any, ...]
    members: tuple[int, ...]
    op: str
    call_site: str
    signature: Signature
    payload: Any
    step: int
    shape: tuple[int, ...] | None
    result: Any = None
    indices: tuple[int, int] = field(default=(-1, -1))

    def waiting(self) -> Waiting:
        return Waiting(self.rank, self.axis, self.op, self.call_site)

    def event(self) -> Event:
        return Event(
            self.step, self.rank, self.axis, self.op, self.call_site, self.shape
        )


def reduce_sum_in_rank_order(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """The fixed-order sum: sequential adds in ascending rank, a new tensor."""
    total = tensors[0].clone()
    for tensor in tensors[1:]:
        total = total + tensor
    return total


def gather_in_rank_order(tensors: Sequence[torch.Tensor], dim: int) -> torch.Tensor:
    return torch.cat(list(tensors), dim=dim)


class Rendezvous:
    def __init__(self, first: Arrival) -> None:
        self.first = first
        self.arrivals: dict[int, Arrival] = {first.rank: first}

    @property
    def members(self) -> tuple[int, ...]:
        return self.first.members

    def join(self, arrival: Arrival) -> Divergence | None:
        fields = self.first.signature.differences(arrival.signature)
        if fields:
            return Divergence(
                axis=arrival.axis,
                group=self.members,
                rank=arrival.rank,
                op=arrival.op,
                call_site=arrival.call_site,
                first_rank=self.first.rank,
                first_op=self.first.op,
                first_call_site=self.first.call_site,
                fields=fields,
            )
        self.arrivals[arrival.rank] = arrival
        return None

    def complete(self) -> bool:
        return set(self.arrivals) == set(self.members)

    def resolve(self) -> None:
        """Apply the op and store every member's result on its arrival.

        What crosses the wire carries no autograd history, as under a real
        backend, whose receivers fill fresh buffers: a received tensor is a
        **detached** copy of the sender's, never a view into another rank's
        graph (a rank's ``backward`` walking into a peer's graph is the bug
        that would otherwise follow — the ranks are threads of one process).
        What a rank gets back of its own — the broadcast source's tensor,
        the reduction's result built on its own clone — keeps its own
        history, exactly as ``TorchCollective`` returns them.
        """
        ordered = [self.arrivals[rank] for rank in sorted(self.arrivals)]
        signature = self.first.signature
        match signature.op:
            case "all_gather":
                gathered = gather_in_rank_order(
                    [a.payload for a in ordered], signature.detail[0]
                ).detach()
                for arrival in ordered:
                    arrival.result = gathered.clone()
            case "all_reduce_sum":
                total = reduce_sum_in_rank_order([a.payload for a in ordered]).detach()
                for arrival in ordered:
                    # the real one: this rank's differentiable clone, summed in place
                    result = arrival.payload.clone()
                    with torch.no_grad():
                        result.copy_(total)
                    arrival.result = result
            case "broadcast":
                origin = self.arrivals[self.members[signature.detail[0]]]
                source = origin.payload
                for arrival in ordered:
                    arrival.result = (
                        source if arrival is origin else source.detach().clone()
                    )
            case "send/recv":
                sender, receiver = (self.arrivals[rank] for rank in self.members)
                receiver.result = sender.payload.detach().clone().to(receiver.payload)
            case "agree_min":
                self._agree(ordered, min(a.payload for a in ordered))
            case "agree_any":
                self._agree(ordered, any(a.payload for a in ordered))
            case "agree_sum":
                self._agree(ordered, sum(a.payload for a in ordered))
            case "barrier":
                pass
            case other:  # pragma: no cover - the collective builds every op
                raise AssertionError(f"unknown collective {other!r}")

    @staticmethod
    def _agree(ordered: Sequence[Arrival], value: Any) -> None:
        for arrival in ordered:
            arrival.result = value
