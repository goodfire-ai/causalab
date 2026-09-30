"""The simulator's refusals (``docs/model_parallelism.md`` §10.2).

Every way a simulated world can fail is a typed exception here, raised to
the caller of ``SimulatedWorld.run`` naming the ranks and call sites involved
— never a hang of the test process.
"""

from __future__ import annotations

from dataclasses import dataclass

from causalab.neural.shared.parallel.placement import Axis


@dataclass(frozen=True)
class Waiting:
    """One rank parked at a collective that has not completed."""

    rank: int
    axis: Axis
    op: str
    call_site: str

    def __str__(self) -> str:
        return f"rank {self.rank} at {self.op}[{self.axis}] ({self.call_site})"


class SimulationError(Exception):
    """Base of everything the simulator raises."""


class LayoutError(SimulationError, ValueError):
    """The rank groups handed to ``SimulatedWorld`` are not a partition per axis."""


class Misuse(SimulationError, ValueError):
    """A rank called the protocol outside its contract (a non-source passing a
    tensor to ``broadcast``, a peer outside the group, an axis the layout has
    no groups for). Raised inside the calling rank, so it surfaces as
    `RankFailed`."""


class MeterScriptExhausted(SimulationError):
    """A rank probed its ``SimulatedMeter`` more often than its script has readings."""


class ScheduleError(SimulationError, ValueError):
    """The schedule handed to a simulation is neither a seed nor a tape of
    non-negative integers."""


class Unfinished(SimulationError):
    """A simulated clock reached its horizon with ranks still running."""

    def __init__(self, ranks: tuple[int, ...], until: float) -> None:
        self.ranks = ranks
        self.until = until
        super().__init__(
            f"ranks {list(ranks)} were still running when the clock reached {until}"
        )


class Refusal(SimulationError):
    """A refusal ``SimulatedWorld.run`` raises in place of results."""


class Divergence(Refusal):
    """A group member arrived at a different collective than the group's first arrival."""

    def __init__(
        self,
        *,
        axis: Axis,
        group: tuple[int, ...],
        rank: int,
        op: str,
        call_site: str,
        first_rank: int,
        first_op: str,
        first_call_site: str,
        fields: tuple[str, ...],
    ) -> None:
        self.axis = axis
        self.group = group
        self.rank = rank
        self.op = op
        self.call_site = call_site
        self.first_rank = first_rank
        self.first_op = first_op
        self.first_call_site = first_call_site
        self.fields = fields
        super().__init__(
            f"rank {rank} arrived at {op} ({call_site}) on {axis} group {group}, "
            f"where rank {first_rank} first arrived at {first_op} ({first_call_site}); "
            f"differing: {', '.join(fields)}"
        )


class Hang(Refusal):
    """Every unfinished rank is parked and no group is complete."""

    def __init__(
        self, waiting: tuple[Waiting, ...], *, timed_out: bool = False
    ) -> None:
        self.waiting = waiting
        self.timed_out = timed_out
        why = "the wall-clock guard fired" if timed_out else "no rank is runnable"
        parked = "; ".join(str(w) for w in waiting) or "none parked"
        super().__init__(f"hang: {why}; waiting: {parked}")


class Abandoned(Refusal):
    """A rank finished while others still wait on a group it belongs to."""

    def __init__(self, finished: int, waiting: tuple[Waiting, ...]) -> None:
        self.finished = finished
        self.waiting = waiting
        parked = "; ".join(str(w) for w in waiting)
        super().__init__(f"rank {finished} finished while others wait on it: {parked}")


class RankKilled(Refusal):
    """The scripted ``kill``: raised inside the rank at its n-th collective and
    reported by ``run`` with the ranks it left waiting."""

    def __init__(
        self,
        rank: int,
        call: int,
        call_site: str,
        abandoned: tuple[Waiting, ...] = (),
    ) -> None:
        self.rank = rank
        self.call = call
        self.call_site = call_site
        self.abandoned = abandoned
        left = "; ".join(str(w) for w in abandoned) or "none"
        super().__init__(
            f"rank {rank} killed at its collective #{call} ({call_site}); "
            f"waiting on it: {left}"
        )


class RankFailed(Refusal):
    """A rank's program raised; the original is the ``__cause__``."""

    def __init__(self, rank: int, error: BaseException) -> None:
        self.rank = rank
        self.error = error
        super().__init__(f"rank {rank} raised {type(error).__name__}: {error}")
        self.__cause__ = error
