"""One rank's view of the world behind the ``Collective`` protocol.

``src`` / ``dst`` are **group-local** indices on the axis — what ``rank(axis)``
returns — since that is the only coordinate a placement knows (a
``StageLocal`` names its owner by stage, not by global rank).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch

from causalab.neural.shared.parallel.placement import Axis
from tests._helpers.simulated_world.errors import Misuse, RankKilled
from tests._helpers.simulated_world.rendezvous import Arrival, Signature, call_site

if TYPE_CHECKING:
    from tests._helpers.simulated_world.world import Run

Groups = Mapping[Axis, tuple[tuple[int, ...], ...]]


class RankCollective:
    def __init__(
        self, rank: int, groups: Groups, run: Run, kill_at: int | None
    ) -> None:
        self._rank = rank
        self._groups = groups
        self._run = run
        self._kill_at = kill_at
        self.calls = 0
        #: the simulator moves CPU tensors between threads of one process
        self.device = torch.device("cpu")

    # -- position -----------------------------------------------------------------

    def _group(self, axis: Axis) -> tuple[int, ...]:
        groups = self._groups.get(axis)
        if groups is None:
            raise Misuse(
                f"rank {self._rank}: the layout has no groups on axis {axis!r}"
            )
        for group in groups:
            if self._rank in group:
                return group
        raise AssertionError(f"rank {self._rank} is in no {axis} group")  # validated

    def _peer(self, index: int, group: tuple[int, ...], axis: Axis, what: str) -> int:
        if not 0 <= index < len(group):
            raise Misuse(
                f"rank {self._rank}: {what}={index} is outside the {axis} group of "
                f"{len(group)} ranks"
            )
        return group[index]

    def rank(self, axis: Axis) -> int:
        return self._group(axis).index(self._rank)

    def size(self, axis: Axis) -> int:
        return len(self._group(axis))

    # -- collectives --------------------------------------------------------------

    def all_gather(self, tensor: torch.Tensor, dim: int, axis: Axis) -> torch.Tensor:
        group = self._group(axis)
        site = call_site()
        shape = tuple(tensor.shape)
        return self._arrive(
            axis,
            key=(axis, group),
            members=group,
            op="all_gather",
            site=site,
            signature=Signature("all_gather", site, shape, tensor.dtype, (dim,)),
            payload=tensor,
            shape=shape,
        )

    def all_reduce_sum(self, tensor: torch.Tensor, axis: Axis) -> torch.Tensor:
        group = self._group(axis)
        site = call_site()
        shape = tuple(tensor.shape)
        return self._arrive(
            axis,
            key=(axis, group),
            members=group,
            op="all_reduce_sum",
            site=site,
            signature=Signature("all_reduce_sum", site, shape, tensor.dtype, ()),
            payload=tensor,
            shape=shape,
        )

    def broadcast(
        self, tensor: torch.Tensor | None, src: int, axis: Axis
    ) -> torch.Tensor:
        group = self._group(axis)
        site = call_site()
        source = self._peer(src, group, axis, "src")
        if source == self._rank and tensor is None:
            raise Misuse(
                f"rank {self._rank}: the broadcast source must pass its tensor"
            )
        if source != self._rank and tensor is not None:
            raise Misuse(
                f"rank {self._rank}: a non-source of a broadcast passes None, "
                f"not a tensor"
            )
        return self._arrive(
            axis,
            key=(axis, group),
            members=group,
            op="broadcast",
            site=site,
            # shape and dtype travel from the source; receivers cannot declare them
            signature=Signature("broadcast", site, None, None, (src,)),
            payload=tensor,
            shape=tuple(tensor.shape) if tensor is not None else None,
        )

    def send(self, tensor: torch.Tensor, dst: int, axis: Axis) -> None:
        group = self._group(axis)
        site = call_site()
        me = group.index(self._rank)
        peer = self._peer(dst, group, axis, "dst")
        if peer == self._rank:
            raise Misuse(f"rank {self._rank}: send to itself")
        shape = tuple(tensor.shape)
        self._arrive(
            axis,
            # the group is part of the key: two pairs at the same local indices
            # in two groups of one axis are two rendezvous, not one
            key=(axis, group, "p2p", me, dst),
            members=(self._rank, peer),
            op="send",
            site=site,
            signature=Signature("send/recv", None, shape, tensor.dtype, (me, dst)),
            payload=tensor,
            shape=shape,
        )

    def recv(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        src: int,
        axis: Axis,
    ) -> torch.Tensor:
        group = self._group(axis)
        site = call_site()
        me = group.index(self._rank)
        peer = self._peer(src, group, axis, "src")
        if peer == self._rank:
            raise Misuse(f"rank {self._rank}: recv from itself")
        return self._arrive(
            axis,
            key=(axis, group, "p2p", src, me),
            members=(peer, self._rank),
            op="recv",
            site=site,
            signature=Signature("send/recv", None, tuple(shape), dtype, (src, me)),
            payload=device,
            shape=tuple(shape),
        )

    def agree_min(self, value: int, axis: Axis) -> int:
        return self._agree("agree_min", value, axis, call_site())

    def agree_any(self, value: bool, axis: Axis) -> bool:
        return self._agree("agree_any", value, axis, call_site())

    def agree_sum(self, value: int, axis: Axis) -> int:
        return self._agree("agree_sum", value, axis, call_site())

    def barrier(self, axis: Axis) -> None:
        group = self._group(axis)
        site = call_site()
        self._arrive(
            axis,
            key=(axis, group),
            members=group,
            op="barrier",
            site=site,
            signature=Signature("barrier", site, None, None, ()),
            payload=None,
            shape=None,
        )

    # -- plumbing -----------------------------------------------------------------

    def _agree(self, op: str, value: Any, axis: Axis, site: str) -> Any:
        group = self._group(axis)
        return self._arrive(
            axis,
            key=(axis, group),
            members=group,
            op=op,
            site=site,
            signature=Signature(op, site, (), None, ()),
            payload=value,
            shape=(),
        )

    def _arrive(
        self,
        axis: Axis,
        *,
        key: tuple[Any, ...],
        members: tuple[int, ...],
        op: str,
        site: str,
        signature: Signature,
        payload: Any,
        shape: tuple[int, ...] | None,
    ) -> Any:
        self.calls += 1
        if self.calls == self._kill_at:
            raise RankKilled(self._rank, self.calls, site)
        arrival = Arrival(
            rank=self._rank,
            axis=axis,
            key=key,
            members=members,
            op=op,
            call_site=site,
            signature=signature,
            payload=payload,
            step=self.calls,
            shape=shape,
        )
        return self._run.arrive(arrival)
