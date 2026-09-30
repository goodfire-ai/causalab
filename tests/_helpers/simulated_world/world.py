"""``SimulatedWorld``: many ranks in one process behind ``Collective``.

The world is a layout — the rank groups per mesh axis — a schedule and the
scripted faults; each ``run`` executes one rank program on every rank under
the baton scheduler and either returns the results in rank order or raises
one of the refusals in `.errors`.
"""

from __future__ import annotations

from typing import Callable, Generic, Mapping, TypeVar

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import AXES, Axis
from tests._helpers.simulated_world.collective import Groups, RankCollective
from tests._helpers.simulated_world.errors import (
    Abandoned,
    LayoutError,
    RankFailed,
    RankKilled,
    Waiting,
)
from tests._helpers.simulated_world.rendezvous import Arrival, Event, Rendezvous
from tests._helpers.simulated_world.schedule import Pick, Schedule, check_schedule
from tests._helpers.simulated_world.scheduler import Abort, Baton

T = TypeVar("T")


def validate_groups(
    groups: Mapping[Axis, tuple[tuple[int, ...], ...]], world: int
) -> Groups:
    """Every axis' groups partition ``range(world)``; each group is strictly
    increasing (rank order is the gather order). Groups come back sorted by
    their first rank."""
    if world < 1:
        raise LayoutError(f"world must be at least 1, got {world}")
    layout: dict[Axis, tuple[tuple[int, ...], ...]] = {}
    for axis, axis_groups in groups.items():
        if axis not in AXES:
            raise LayoutError(f"unknown axis {axis!r}; the axes are {AXES}")
        seen: dict[int, tuple[int, ...]] = {}
        for group in axis_groups:
            if not group or list(group) != sorted(set(group)):
                raise LayoutError(
                    f"{axis} group {group} must be non-empty and strictly increasing"
                )
            for rank in group:
                if not 0 <= rank < world:
                    raise LayoutError(
                        f"{axis} group {group}: rank {rank} outside world {world}"
                    )
                if rank in seen:
                    raise LayoutError(
                        f"rank {rank} is in two {axis} groups: {seen[rank]} and {group}"
                    )
                seen[rank] = tuple(group)
        missing = sorted(set(range(world)) - set(seen))
        if missing:
            raise LayoutError(f"ranks {missing} are in no {axis} group")
        layout[axis] = tuple(
            sorted((tuple(g) for g in axis_groups), key=lambda g: g[0])
        )
    return layout


def groups_for(
    world: int,
    *,
    data: int = 1,
    pipeline: int = 1,
    context: int = 1,
    tensor: int = 1,
    expert: int = 1,
) -> Groups:
    """The mesh layout of ``docs/model_parallelism.md`` §2: ``(data, pipeline,
    context, model)`` outermost first, ``tensor`` and ``expert`` contiguous
    sub-groups of the ``model`` dimension (``model = max(tensor, expert)``)."""
    model = max(tensor, expert)
    if data * pipeline * context * model != world:
        raise LayoutError(
            f"world {world} != data {data} * pipeline {pipeline} * context {context} "
            f"* model {model}"
        )
    if model % tensor or model % expert:
        raise LayoutError(
            f"tensor {tensor} and expert {expert} must divide model {model}"
        )

    def rank(d: int, p: int, c: int, m: int) -> int:
        return ((d * pipeline + p) * context + c) * model + m

    blocks = [
        (d, p, c) for d in range(data) for p in range(pipeline) for c in range(context)
    ]

    def sub_groups(size: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(rank(d, p, c, m) for m in range(start, start + size))
            for d, p, c in blocks
            for start in range(0, model, size)
        )

    layout: dict[Axis, tuple[tuple[int, ...], ...]] = {
        "model": sub_groups(model),
        "tensor": sub_groups(tensor),
        "expert": sub_groups(expert),
        "context": tuple(
            tuple(rank(d, p, c, m) for c in range(context))
            for d in range(data)
            for p in range(pipeline)
            for m in range(model)
        ),
        "pipeline": tuple(
            tuple(rank(d, p, c, m) for p in range(pipeline))
            for d in range(data)
            for c in range(context)
            for m in range(model)
        ),
        "data": tuple(
            tuple(rank(d, p, c, m) for d in range(data))
            for p in range(pipeline)
            for c in range(context)
            for m in range(model)
        ),
    }
    return validate_groups(layout, world)


class SimulatedWorld:
    """``world`` ranks, grouped per axis, interleaved by ``schedule`` — a seed
    or a tape, `.schedule` — (module docstring of the package).
    ``timeout`` is the longest a rank may compute between two collectives
    before the wall-clock guard reports a hang."""

    def __init__(
        self,
        groups: Mapping[Axis, tuple[tuple[int, ...], ...]],
        *,
        world: int,
        schedule: Schedule,
        timeout: float = 10.0,
    ) -> None:
        self.groups = validate_groups(groups, world)
        self.world = world
        self.schedule: int | tuple[int, ...] = check_schedule(schedule)
        self.timeout = timeout
        self.slow_ranks: set[int] = set()
        self.kills: dict[int, int] = {}
        #: the last run's merged, schedule-ordered transcript
        self.transcript: list[Event] = []
        #: the last run's per-rank transcripts
        self.transcripts: list[list[Event]] = [[] for _ in range(world)]
        #: the last run's hand-offs of the baton, in order
        self.picks: list[Pick] = []

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.world:
            raise LayoutError(f"rank {rank} outside world {self.world}")

    def slow(self, rank: int) -> None:
        """``rank`` is scheduled last at every rendezvous."""
        self._check_rank(rank)
        self.slow_ranks.add(rank)

    def kill(self, rank: int, at_call: int) -> None:
        """``rank`` raises `RankKilled` at its ``at_call``-th collective (1-based)."""
        self._check_rank(rank)
        if at_call < 1:
            raise LayoutError(f"at_call is 1-based, got {at_call}")
        self.kills[rank] = at_call

    def group_of(self, axis: Axis, rank: int) -> tuple[int, ...]:
        for group in self.groups[axis]:
            if rank in group:
                return group
        raise LayoutError(f"rank {rank} is in no {axis} group")

    def run(self, program: Callable[[int, Collective], T]) -> list[T]:
        """Run ``program(rank, collective)`` on every rank; results in rank order."""
        execution: Run[T] = Run(self, program)
        self.transcript = execution.transcript
        self.transcripts = execution.transcripts
        self.picks = execution.picks
        return execution.execute()


class Run(Generic[T]):
    """One execution of a program over a world: the rendezvous table, the
    transcripts and the thread bodies."""

    def __init__(
        self, world: SimulatedWorld, program: Callable[[int, Collective], T]
    ) -> None:
        self._world = world
        self._program = program
        self._baton = Baton(
            world.world,
            schedule=world.schedule,
            slow=frozenset(world.slow_ranks),
            timeout=world.timeout,
        )
        self.picks = self._baton.picks
        self._pending: dict[tuple[object, ...], Rendezvous] = {}
        self._results: dict[int, T] = {}
        self.transcript: list[Event] = []
        self.transcripts: list[list[Event]] = [[] for _ in range(world.world)]

    def execute(self) -> list[T]:
        bodies = {rank: self._body_for(rank) for rank in range(self._world.world)}
        self._baton.run_all(bodies)
        return [self._results[rank] for rank in range(self._world.world)]

    def _body_for(self, rank: int) -> Callable[[], None]:
        collective = RankCollective(
            rank, self._world.groups, self, self._world.kills.get(rank)
        )

        def body() -> None:
            self._body(rank, collective)

        return body

    def _body(self, rank: int, collective: RankCollective) -> None:
        baton = self._baton
        try:
            baton.start(rank)
            result = self._program(rank, collective)
        except Abort:
            return
        except RankKilled as killed:
            with baton.cv:
                killed.abandoned = self._waiting_on(rank)
                baton.fail(killed)
            return
        except Exception as error:  # the seam: a rank's failure, typed by rank
            with baton.cv:
                baton.fail(RankFailed(rank, error))
            return
        with baton.cv:
            self._results[rank] = result
            waiting = self._waiting_on(rank)
            if waiting:
                baton.fail(Abandoned(rank, waiting))
                return
        baton.finish(rank)

    # -- under ``baton.cv`` ---------------------------------------------------

    def _waiting_on(self, rank: int) -> tuple[Waiting, ...]:
        """Who is parked at a rendezvous ``rank`` belongs to and has not reached."""
        return tuple(
            arrival.waiting()
            for rendezvous in self._pending.values()
            if rank in rendezvous.members
            for arrival in rendezvous.arrivals.values()
        )

    def _record(self, arrival: Arrival) -> None:
        event = arrival.event()
        arrival.indices = (len(self.transcript), len(self.transcripts[arrival.rank]))
        self.transcript.append(event)
        self.transcripts[arrival.rank].append(event)

    def _fill_shapes(self, rendezvous: Rendezvous) -> None:
        """A broadcast receiver learns its shape only at completion."""
        for arrival in rendezvous.arrivals.values():
            if arrival.shape is None and hasattr(arrival.result, "shape"):
                merged, own = arrival.indices
                event = self.transcript[merged]._replace(
                    shape=tuple(arrival.result.shape)
                )
                self.transcript[merged] = event
                self.transcripts[arrival.rank][own] = event

    def arrive(self, arrival: Arrival) -> object:
        """A rank reaches a collective: join or open its rendezvous, complete it
        when the group is in, then yield the baton until the result is ready."""
        baton = self._baton
        with baton.cv:
            self._record(arrival)
            finished = [member for member in arrival.members if member in baton.done]
            if finished:
                pending = self._pending.get(arrival.key)
                others = (
                    tuple(a.waiting() for a in pending.arrivals.values())
                    if pending
                    else ()
                )
                baton.fail(Abandoned(finished[0], others + (arrival.waiting(),)))
                raise Abort()
            rendezvous = self._pending.get(arrival.key)
            if rendezvous is None:
                rendezvous = Rendezvous(arrival)
                self._pending[arrival.key] = rendezvous
            else:
                divergence = rendezvous.join(arrival)
                if divergence is not None:
                    baton.fail(divergence)
                    raise Abort()
            parked: Waiting | None = arrival.waiting()
            if rendezvous.complete():
                del self._pending[arrival.key]
                rendezvous.resolve()
                self._fill_shapes(rendezvous)
                baton.wake(rendezvous.members)
                parked = None
        baton.yield_baton(arrival.rank, parked)
        return arrival.result
