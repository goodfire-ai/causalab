"""A scheduled world of ranks in one process behind ``Collective`` (``docs/model_parallelism.md`` §10.2).

``SimulatedWorld(groups, world=n, schedule=s).run(program)`` runs
``program(rank, collective)`` on every rank and returns the results in rank
order. Ranks are threads, but exactly one runs at a time: the scheduler hands
a baton to one runnable rank, and a rank gives it back only inside a
collective call, where it is parked until every member of its group has
arrived at the *same* collective — same op, call site, shape and dtype — and
the op has been applied on CPU in fixed rank order. Which parked rank resumes
next is the one thing the schedule decides: a seed draws it at random, a
**tape** of integers names it switch by switch and falls back to the lowest
rank when spent (`.schedule`), so hypothesis can draw and shrink an
interleaving (``tests/_helpers/parallel_strategies.py:schedules``). Every
hand-off is recorded in ``world.picks``.

**Why switching only at collectives is enough.** §10.2 speaks of pre-empting
a rank after a drawn number of steps. Between two collectives, however, a
causalab rank runs rank-local, deterministic code — its inputs are the
document, the data and the tensors the last collective returned, none of
which another rank can change — so the *state* a rank reaches at its next
collective does not depend on where it was interrupted, only on what it
received. Every interleaving of the rank-local stretches is therefore
observationally equivalent to running each stretch atomically, and the whole
space of schedules a pre-emptive scheduler could explore collapses to the
order in which parked ranks resume — exactly what the schedule decides.
Cooperative switching loses nothing and buys determinism: no rank-local code
ever runs concurrently, so the simulator needs no locks in the program under
test and two runs at one schedule are byte-identical.

What the world refuses, as typed exceptions naming ranks and call sites
(`.errors`), never as a hang of the test process: a member arriving at a
different collective than its group's first arrival (`Divergence`);
every unfinished rank parked with no complete group (`Hang`, also the
wall-clock guard's verdict); a rank finishing while others wait on it
(`Abandoned`); the scripted ``kill`` (`RankKilled`); a program
that raised (`RankFailed`, the original as ``__cause__``).

Fault injection: `SimulatedMeter` scripts memory readings and OOMs per
rank; ``world.slow(rank)`` schedules a rank last at every rendezvous;
``world.kill(rank, at_call=n)`` kills it at its n-th collective. Every rank
appends ``(step, rank, axis, op, call_site, shape)`` to its transcript;
``world.transcript`` is the merged, schedule-ordered list.

The heartbeat protocol has its own driver over the same schedules,
`.heartbeat`: ``world`` monitors over one fake store and one clock,
ticked in the scheduled order.
"""

from __future__ import annotations

from tests._helpers.simulated_world.errors import (
    Abandoned,
    Divergence,
    Hang,
    LayoutError,
    MeterScriptExhausted,
    Misuse,
    RankFailed,
    RankKilled,
    Refusal,
    ScheduleError,
    SimulationError,
    Unfinished,
    Waiting,
)
from tests._helpers.simulated_world.faults import RankMeter, SimulatedMeter
from tests._helpers.simulated_world.rendezvous import Event
from tests._helpers.simulated_world.schedule import Pick, Schedule
from tests._helpers.simulated_world.world import (
    SimulatedWorld,
    groups_for,
    validate_groups,
)

__all__ = [
    "Abandoned",
    "Divergence",
    "Event",
    "Hang",
    "LayoutError",
    "MeterScriptExhausted",
    "Misuse",
    "Pick",
    "RankFailed",
    "RankKilled",
    "RankMeter",
    "Refusal",
    "Schedule",
    "ScheduleError",
    "SimulatedMeter",
    "SimulatedWorld",
    "SimulationError",
    "Unfinished",
    "Waiting",
    "groups_for",
    "validate_groups",
]
