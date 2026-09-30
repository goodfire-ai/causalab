"""Who publishes a run, and which points a replica runs (``docs/model_parallelism.md``
§3, §8.3, §9) — the torch-free half of the SPMD launcher.

Under SPMD every rank runs the same document over the same Python control
flow; **rank 0 of each data-parallel replica publishes** and every other
rank computes the same result and discards it. Data parallelism over points
shards the campaign's points across the replicas — each replica runs a
contiguous shard — and the publishing rank of replica 0 joins the shards by
point digest into the campaign's outputs.

The one seam through which every writer learns what to do is the
[`Publisher`][] a run's `ExecutionRequest`
carries: whether this process's shard leaves it at all (``publish``), which
replica it is (``replica`` of ``replicas``), the word the receipt records
(``launcher``), and ``gather`` — every publishing rank hands its shard in,
and the joiner (the publishing rank of replica 0, [`is_joiner`][])
receives every replica's shard in replica order and is the only process that
writes. [`SOLO`][] is world 1: it publishes, joins, and its gather is the
identity, which is what today's single-device engine runs under and what the
whole existing suite proves bit-identical.

The production publisher over ``torch.distributed`` lives in
``causalab/neural/shared/parallel/launcher.py``; this module stays free of
torch and of every engine so the point-shard arithmetic runs where
``--points`` does, in [`causalab.protocol.run.run_protocol`][causalab.protocol.pipeline.run_protocol].
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, Sequence, TypeVar, runtime_checkable

from causalab.protocol.rules.errors import ProtocolError

__all__ = [
    "LAUNCHERS",
    "SOLO",
    "Publisher",
    "Solo",
    "is_joiner",
    "point_shard",
    "point_shards",
]

#: The closed vocabulary of ``execution.parallel.launcher`` (§3, §9): ``solo``
#: — this process alone, no ``torch.distributed`` initialised, every world-1
#: run; ``spawned`` — one of the ``world`` local children the CLI's parent
#: process started; ``joined`` — a process launched into a ``WORLD_SIZE``
#: group by ``torchrun``, Slurm or another launcher.
LAUNCHERS: tuple[str, ...] = ("solo", "spawned", "joined")

T = TypeVar("T")


@runtime_checkable
class Publisher(Protocol):
    """This process's place in the run, as every writer reads it (§3).

    ``launcher`` is one of [`LAUNCHERS`][]. ``replica`` is this process's
    index on the data axis, of ``replicas``. ``publish`` is whether this
    rank is the publishing rank of its replica — rank 0 of the replica's
    model, pipeline and context groups — so its shard leaves the process;
    a rank that does not publish computes and discards. ``gather`` is
    called by every publishing rank, once per join point, with its shard's
    payload: the joiner receives every replica's payload in replica order,
    every other publishing rank receives ``None``.
    """

    @property
    def launcher(self) -> str: ...

    @property
    def replica(self) -> int: ...

    @property
    def replicas(self) -> int: ...

    @property
    def publish(self) -> bool: ...

    def gather(self, payload: T) -> Sequence[T] | None: ...


@dataclasses.dataclass(frozen=True)
class Solo:
    """World 1: one replica, this process publishes and joins, and the
    gather hands the payload straight back."""

    launcher: str = LAUNCHERS[0]
    replica: int = 0
    replicas: int = 1
    publish: bool = True

    def gather(self, payload: T) -> Sequence[T] | None:
        return (payload,)


#: The world-1 publisher every request carries unless a launcher set another.
SOLO = Solo()


def is_joiner(publisher: Publisher) -> bool:
    """Whether ``publisher`` is the one process that writes the campaign's
    outputs: the publishing rank of replica 0."""
    return bool(publisher.publish) and publisher.replica == 0


def point_shards(selected: range, replicas: int) -> tuple[range, ...]:
    """``replicas`` contiguous shards of ``selected`` in order, sizes
    differing by at most one with the first shards longer — the workflow's
    ``fan_out.over.shards`` arithmetic (``workflow/fan_out.py``), so a data-
    parallel run and a declared fan-out cut a campaign the same way and no
    replica is ever idle.

    Raises:
        ProtocolError: ``P4`` naming ``--parallel.data`` — more replicas
            than selected points; a replica holds at least one point.
    """
    n_points = len(selected)
    if replicas < 1:
        raise ValueError(f"replicas must be at least 1, got {replicas}")
    if replicas > n_points:
        raise ProtocolError(
            "P4",
            f"dp={replicas} replicas over {n_points} selected point"
            f"{'s' if n_points != 1 else ''} — every replica runs at least one "
            "point, so the data axis is at most the point count (shard a "
            "smaller campaign, or select more points)",
            path="--parallel.data",
        )
    base, extra = divmod(n_points, replicas)
    shards: list[range] = []
    start = selected.start
    for index in range(replicas):
        size = base + (1 if index < extra else 0)
        shards.append(range(start, start + size))
        start += size
    return tuple(shards)


def point_shard(selected: range, replica: int, replicas: int) -> range:
    """The shard of ``selected`` replica ``replica`` runs ([`point_shards`][]).

    Raises:
        ValueError: ``replica`` is outside ``range(replicas)``.
    """
    if not 0 <= replica < replicas:
        raise ValueError(f"replica {replica} is outside range({replicas})")
    return point_shards(selected, replicas)[replica]
