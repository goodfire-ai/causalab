"""The rank watchdog's rules (``docs/model_parallelism.md`` §3 "when a rank
dies", §11): the two settings, the liveness verdict, the refusal — torch-free,
so the deterministic tests run the rules without a process group. The thread
over the rendezvous store is [`.heartbeat`][causalab.neural.shared.parallel.heartbeat].

**Why.** A rank that exits mid-run — an exception, an out-of-memory kill, a
``kill -9`` — leaves the others blocked in their next collective. A spawn
parent ([`.spawn`][causalab.neural.shared.parallel.spawn]) learns of the exit at once and terminates the
siblings, but a ``torchrun`` world has no parent of ours, and across nodes
the failed node's agent stops only its own workers: the rest wait for the
backend's timeout — ten minutes under NCCL, thirty under gloo — and then
abort, without a word about which rank went. Two settings bound that:

- [`RANK_GRACE_VARIABLE`][] (``CAUSALAB_RANK_GRACE``, default
  [`RANK_GRACE_S`][] seconds): every rank beats on the rendezvous store
  [`BEATS_PER_GRACE`][] times a grace and watches its peers; a peer that
  has been *gone* for a grace while this rank still runs — no beat, a
  non-zero exit status it wrote itself, or the store it hosted unreachable
  — is a refusal by name ([`RankLost`][]) and the rank exits with
  [`LOST_STATUS`][]. A peer that finished cleanly (``done:0``) is never
  lost. The grace is the time to refusal after a death; it must cover the
  longest a healthy rank may go without beating (a thread starved by a
  host-side sync is still a thread; 30 s is far above it) and be short
  enough that a dead world costs minutes, not the backend's timeout.
- [`COLLECTIVE_TIMEOUT_VARIABLE`][] (``CAUSALAB_COLLECTIVE_TIMEOUT``,
  default [`COLLECTIVE_TIMEOUT_S`][] seconds): the timeout of the
  default process group **and of every mesh group** — ``new_group`` takes
  torch's global default otherwise, not the default group's — the backstop
  for a hang without a death (ranks at different collectives, a wedged
  peer), which no heartbeat can see since every rank is still beating. It
  must exceed the longest gap between one rank's collective and its
  slowest peer's arrival: the model load sits between the mesh's group
  creation and the first forward. Allow enough time for checkpoint reads
  with a cold page cache. A grace at or above the timeout is refused, since
  the watchdog would then never speak first. The same timeout bounds a
  peer that **never arrives** ([`Liveness`][]'s ``arrival``): a rank
  that dies before it connects to the store cannot be told from one still
  starting, and startup skew is a gap of the kind the timeout bounds, so a
  peer never seen is lost as ``unreached`` once the timeout has passed — by
  name, where the store's own wait for workers named a count.

**Three ways a peer is gone, and a fourth.** ``silent``: it beat and
stopped — a kill, an OOM, a ``SIGSTOP`` (a stopped process's heartbeat
thread stops with it, so a stop reads exactly like a death and the
survivor names it the same way; only the process left behind differs,
and that is the launcher's to reap). ``exited``: it wrote its own failure
status. ``unreachable``: the store rank 0 hosts has failed for a grace —
rank 0 died, or is stopped (a black-holed host is noticed one store
timeout, the grace, later than a dead one). ``unreached``: it never beat at
all, and the timeout has passed. A rank **wedged** — its main thread
stuck while its heartbeat thread beats — is none of these: no heartbeat
can see it, and the collective timeout is what ends its peers.

Both are read once per process from the environment ([`Settings.from_environment`][]), typed, a malformed value refused by name
([`WatchdogSetting`][]). [`inside`][] / [`current`][] register the
collective this rank is inside — every ``TorchCollective`` call passes
through it — so the refusal can say what the rank was waiting in.

**A collective that fails** (``CollectiveFailed`` in [`.collective`][causalab.neural.shared.parallel.collective]) may
be a peer's death seen from the other side: under gloo a peer's closed
socket fails the survivor's read at once, before a single beat is missed,
where NCCL holds until its timeout. The backend's words name no rank, so
the survivor asks the heartbeat (``Heartbeat.hold``) before raising the
collective's own refusal: a peer lost within the bound ([`Settings.bound`][],
the grace plus [`BOUND_BEATS`][] beats) is the named refusal, and the
collective's refusal stands only when every peer is proven alive after
the failure — [`Liveness.every_peer_alive_since`][]: its beat advanced
twice after it, since one advance may have been written before the death
and read after — or the bound passes with nobody lost.

**When the store goes with rank 0.** Every exit of rank 0 takes the store
— its refusal of a lost peer included — so a survivor that ticks after it
reads nothing more. What it already holds still counts: a peer whose
silence had reached the grace when the store went is lost by its own
silence and named at that very tick ([`Liveness.verdict`][], the
observations before the failure); a peer whose silence is younger than
that may be alive behind a dead store, and is not named — the host is,
``unreachable``, a grace after the store went. So in a world of several
survivors of one death, each names the victim or, when its last look at
the victim was one beat younger than rank 0's, the host that refused first;
both are true, and every survivor has exited within two graces and three
beats of the death.
"""

from __future__ import annotations

import dataclasses
import math
import os
from contextlib import contextmanager
from typing import Iterator, Literal, Mapping

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import Axis, ParallelGeometry, format_geometry

__all__ = [
    "BEATS_PER_GRACE",
    "BOUND_BEATS",
    "Blame",
    "CompletionTimeout",
    "COLLECTIVE_TIMEOUT_S",
    "COLLECTIVE_TIMEOUT_VARIABLE",
    "LOST_STATUS",
    "RANK_GRACE_S",
    "RANK_GRACE_VARIABLE",
    "Beat",
    "Done",
    "Liveness",
    "Lost",
    "MalformedSignal",
    "RankLost",
    "Settings",
    "Signal",
    "StoreHost",
    "WatchdogSetting",
    "Why",
    "current",
    "decode",
    "describe",
    "encode",
    "inside",
    "parse_seconds",
]

#: The timeout of every process group, in seconds (module docstring).
COLLECTIVE_TIMEOUT_VARIABLE = "CAUSALAB_COLLECTIVE_TIMEOUT"
COLLECTIVE_TIMEOUT_S = 600.0

#: How long a peer may be gone before this rank refuses, in seconds.
RANK_GRACE_VARIABLE = "CAUSALAB_RANK_GRACE"
RANK_GRACE_S = 30.0

#: Beats (and peer checks) per grace: the verdict lands within one tenth of
#: a grace after the grace has passed.
BEATS_PER_GRACE = 10

#: Beats past the grace within which a refusal lands after a death — the
#: death falls between beats, a survivor notices at its next tick and the
#: verdict lands on a tick: ``1.3 × grace`` ([`Settings.bound`][]).
BOUND_BEATS = 3

#: The exit status of a rank that refuses because a peer was lost — a
#: victim, distinct on purpose from whatever status the peer itself had.
LOST_STATUS = 1


class WatchdogSetting(ProtocolError):
    """A watchdog variable holds something that is not a duration in
    seconds — not a number, not finite, not positive — or the grace is at
    or above the timeout. ``P4`` at ``--parallel``, named after the
    variable so the operator's fix is one line."""

    def __init__(self, message: str) -> None:
        super().__init__("P4", message, path="--parallel")


def parse_seconds(name: str, text: str | None, default: float) -> float:
    """``text`` as a positive, finite number of seconds; ``default`` for an
    unset or blank variable.

    Raises:
        WatchdogSetting: anything else, naming ``name`` and the text.
    """
    if text is None or not text.strip():
        return default
    try:
        seconds = float(text)
    except ValueError:
        raise WatchdogSetting(
            f"{name}={text!r} is not a number of seconds "
            "(docs/model_parallelism.md §11)"
        ) from None
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise WatchdogSetting(
            f"{name}={text!r} is not a positive, finite number of seconds "
            "(docs/model_parallelism.md §11)"
        )
    return seconds


@dataclasses.dataclass(frozen=True)
class Settings:
    """The watchdog's two durations, in seconds (module docstring).

    Raises:
        WatchdogSetting: a non-positive or non-finite value, or a grace at
            or above the timeout.
    """

    timeout: float = COLLECTIVE_TIMEOUT_S
    grace: float = RANK_GRACE_S

    def __post_init__(self) -> None:
        for name, value in (
            (COLLECTIVE_TIMEOUT_VARIABLE, self.timeout),
            (RANK_GRACE_VARIABLE, self.grace),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise WatchdogSetting(
                    f"{name}={value!r} is not a positive, finite number of seconds"
                )
        if self.grace >= self.timeout:
            raise WatchdogSetting(
                f"{RANK_GRACE_VARIABLE}={self.grace:g} is not below "
                f"{COLLECTIVE_TIMEOUT_VARIABLE}={self.timeout:g}: the grace is how "
                "long a lost rank goes unnamed and the timeout the backstop for a "
                "hang without a death, so the grace must be the shorter "
                "(docs/model_parallelism.md §11)"
            )

    @property
    def interval(self) -> float:
        """Seconds between beats: a grace over [`BEATS_PER_GRACE`][]."""
        return self.grace / BEATS_PER_GRACE

    @property
    def bound(self) -> float:
        """Seconds after a death by which a survivor has refused: the grace
        plus [`BOUND_BEATS`][] beats."""
        return self.grace + BOUND_BEATS * self.interval

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] = os.environ) -> Settings:
        """The settings ``environ`` spells, the defaults where it is silent.

        Raises:
            WatchdogSetting: a malformed value ([`parse_seconds`][]), or a
                grace at or above the timeout.
        """
        return cls(
            timeout=parse_seconds(
                COLLECTIVE_TIMEOUT_VARIABLE,
                environ.get(COLLECTIVE_TIMEOUT_VARIABLE),
                COLLECTIVE_TIMEOUT_S,
            ),
            grace=parse_seconds(
                RANK_GRACE_VARIABLE, environ.get(RANK_GRACE_VARIABLE), RANK_GRACE_S
            ),
        )


# --------------------------------------------------------------------------- #
# the signals on the store
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Beat:
    """A rank is alive: its ``count``-th beat."""

    count: int


@dataclasses.dataclass(frozen=True)
class Done:
    """A rank has finished with ``status`` (``0`` cleanly)."""

    status: int


Signal = Beat | Done


class MalformedSignal(ValueError):
    """A liveness key holds neither ``beat:<n>`` nor ``done:<status>``."""


def encode(signal: Signal) -> str:
    """``beat:<n>`` / ``done:<status>``: what a rank writes under its key."""
    if isinstance(signal, Beat):
        return f"beat:{signal.count}"
    return f"done:{signal.status}"


def decode(text: str | bytes) -> Signal:
    """The inverse of [`encode`][].

    Raises:
        MalformedSignal: anything else.
    """
    raw = text.decode() if isinstance(text, bytes) else text
    kind, _, number = raw.partition(":")
    if kind in ("beat", "done") and number.isdigit():
        return Beat(int(number)) if kind == "beat" else Done(int(number))
    raise MalformedSignal(
        f"liveness key holds {raw!r}, neither 'beat:<n>' nor 'done:<status>'"
    )


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #

Why = Literal["silent", "exited", "unreachable", "unreached"]

#: Who hosts the rendezvous store (``launcher.store_host``): rank 0's own
#: process (a spawn, a bare ``torchrun``-shaped launch), or rank 0's
#: ``torchrun`` agent — which exits when rank 0 fails, and outlives rank 0
#: stopped. The refusal for an unreachable store says which.
StoreHost = Literal["rank", "agent"]


@dataclasses.dataclass(frozen=True)
class Lost:
    """A peer this rank has given up on: ``silent`` — no new beat for a
    grace (``silence`` seconds since its last); ``exited`` — it wrote a
    non-zero ``status`` a grace ago and this rank is still running;
    ``unreachable`` — the store rank 0 hosts has failed for a grace, so
    rank 0 is gone; ``unreached`` — it never beat, and the arrival bound (the
    collective timeout, module docstring) has passed (``silence`` seconds
    since this rank began watching)."""

    rank: int
    why: Why
    status: int | None = None
    silence: float = 0.0


@dataclasses.dataclass(frozen=True)
class _Seen:
    """What a peer's key last read, when that changed, when it had changed
    before that (``None`` until a second change), and whether the peer has
    ever been seen — a key that later reads as nothing (a malformed byte)
    is a silence of a peer that arrived, bounded by the grace, not an
    absence."""

    signal: Signal | None
    changed: float
    before: float | None = None
    arrived: bool = False


class Liveness:
    """One rank's view of its peers: per peer the last signal seen and when
    it last changed; [`verdict`][] is the grace rule of the module
    docstring — the lowest lost rank, ``None`` when none is. A peer never
    seen is silent since ``now`` at construction and bounded by ``arrival``
    — the collective timeout in production, where the store has no barrier
    and a peer not yet seen may still be starting (module docstring); the
    grace when not given, for a watch that begins once every peer is
    known to be there."""

    def __init__(
        self,
        *,
        rank: int,
        world: int,
        grace: float,
        now: float,
        arrival: float | None = None,
    ) -> None:
        self.rank = rank
        self.world = world
        self.grace = grace
        self.arrival = grace if arrival is None else arrival
        self._seen: dict[int, _Seen] = {
            peer: _Seen(None, now) for peer in range(world) if peer != rank
        }
        self._unreachable_since: float | None = None

    @property
    def reachable(self) -> bool:
        """No store operation has failed: the watch can still see its peers."""
        return self._unreachable_since is None

    @property
    def settled(self) -> bool:
        """The store's host finished cleanly and the store is gone: nothing
        is left to watch, and nothing is lost."""
        host = self._seen.get(0)
        return self._unreachable_since is not None and (
            host is not None and host.signal == Done(0)
        )

    @property
    def all_finished(self) -> bool:
        """Every peer has published a successful terminal status."""
        return all(seen.signal == Done(0) for seen in self._seen.values())

    def observe(self, peer: int, signal: Signal | None, now: float) -> None:
        """``peer``'s key reads ``signal`` (``None``: not written yet) at
        ``now``; a change restarts its silence."""
        seen = self._seen[peer]
        if now < seen.changed:
            return  # an older in-flight tick cannot roll back newer observations
        if signal != seen.signal:
            self._seen[peer] = _Seen(
                signal,
                now,
                before=seen.changed,
                arrived=seen.arrived or signal is not None,
            )

    def every_peer_alive_since(self, since: float) -> bool:
        """Every peer is proven alive after ``since``: its beat has advanced
        twice at observations strictly after it — the second advance was
        written after the first was read, so after ``since``; one advance
        alone may predate a death and have been read after — or it has
        finished cleanly. A peer that wrote a failure status, or nothing, is
        not alive."""
        for seen in self._seen.values():
            if seen.signal == Done(0):
                continue
            if not isinstance(seen.signal, Beat):
                return False
            if seen.before is None or seen.before <= since:
                return False
        return True

    def unreachable(self, now: float) -> None:
        """A store operation failed at ``now``: the host is gone, or the
        network is; the first failure starts the clock."""
        if self._unreachable_since is None:
            self._unreachable_since = now

    def verdict(self, now: float) -> Lost | None:
        """The lowest lost peer at ``now`` (module docstring): a failed exit
        or a silence past its bound — the grace for a peer that has ever
        been seen, ``arrival`` for one never seen — judged on what this rank
        has read;
        once the store is unreachable, a silence that had not reached its
        bound when the store went is the store's, not the peer's, and the
        host is named instead, a grace after the store went. A store whose
        host finished cleanly is settled: nothing is lost."""
        gone_at = self._unreachable_since
        if gone_at is not None and self.settled:
            return None
        for peer in sorted(self._seen):
            seen = self._seen[peer]
            silence = now - seen.changed
            if isinstance(seen.signal, Done):
                if seen.signal.status == 0 or silence < self.grace:
                    continue
                return Lost(peer, "exited", status=seen.signal.status)
            bound = self.grace if seen.arrived else self.arrival
            if silence < bound:
                continue
            if gone_at is not None and gone_at - seen.changed < bound:
                continue  # silent only since the store went: the store's silence
            if not seen.arrived:
                return Lost(peer, "unreached", silence=silence)
            return Lost(peer, "silent", silence=silence)
        if gone_at is not None and now - gone_at >= self.grace:
            return Lost(0, "unreachable")
        return None


#: What the heartbeat says of a collective failure: the peer whose loss
#: explains it, or ``"collective"`` — nobody lost: every peer proven alive
#: after the failure, or the bound passed — so the collective's own refusal
#: stands (module docstring).
Blame = Lost | Literal["collective"]


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #


class CompletionTimeout(ProtocolError):
    """A live rank did not reach clean shutdown within the collective timeout."""

    def __init__(self, timeout: float) -> None:
        super().__init__(
            "P4",
            f"rank 0 waited {timeout:g} s for peers to finish; "
            f"clean shutdown is bounded by {COLLECTIVE_TIMEOUT_VARIABLE}",
            path="--parallel",
        )


class RankLost(ProtocolError):
    """A peer was lost ([`Lost`][]) while this rank still ran: ``P4`` at
    ``--parallel``, naming the peer and why, this rank and the collective
    it was waiting in, the geometry and the grace."""

    def __init__(self, lost: Lost, message: str) -> None:
        self.lost = lost
        super().__init__("P4", message, path="--parallel")


def describe(
    lost: Lost,
    *,
    rank: int,
    world: int,
    geometry: ParallelGeometry,
    settings: Settings,
    where: tuple[Axis, str] | None,
    store_host: StoreHost = "rank",
) -> RankLost:
    """The refusal for ``lost`` as seen from ``rank``; ``where`` is the
    collective this rank is inside ([`current`][]), if any; ``store_host``
    who hosted the store an ``unreachable`` loss names."""
    if lost.why == "exited":
        what = f"rank {lost.rank} of {world} exited with status {lost.status}"
    elif lost.why == "silent":
        what = (
            f"rank {lost.rank} of {world} has sent no heartbeat for "
            f"{lost.silence:.0f} s"
        )
    elif lost.why == "unreached":
        what = (
            f"rank {lost.rank} of {world} never reached the rendezvous in "
            f"{lost.silence:.0f} s (a rank that dies before it connects cannot be "
            f"told from one still starting, so the wait for it is "
            f"{COLLECTIVE_TIMEOUT_VARIABLE}'s, not the grace's)"
        )
    elif store_host == "agent":
        what = (
            f"the rendezvous store, hosted by rank {lost.rank}'s torchrun agent, is "
            f"unreachable (the agent exits when rank {lost.rank} fails, or its node "
            "is gone)"
        )
    else:
        what = (
            f"rank {lost.rank} of {world}, the rendezvous store's host, is unreachable"
        )
    waiting = f", waiting in {where[1]} on axis {where[0]!r}" if where else ""
    return RankLost(
        lost,
        f"{what} and rank {rank} is still running{waiting}: the world cannot "
        f"complete, so this rank exits (--parallel {format_geometry(geometry)}; "
        f"{RANK_GRACE_VARIABLE}={settings.grace:g}; a hang without a death is "
        f"bounded by {COLLECTIVE_TIMEOUT_VARIABLE}={settings.timeout:g}; "
        "docs/model_parallelism.md §3)",
    )


# --------------------------------------------------------------------------- #
# where this rank is
# --------------------------------------------------------------------------- #

#: The collectives the process is inside, innermost last — process-wide,
#: not thread-local, on purpose: the rank's one main thread calls every
#: collective and the heartbeat thread reads what it is waiting in.
_STACK: list[tuple[Axis, str]] = []


def current() -> tuple[Axis, str] | None:
    """The ``(axis, op)`` of the collective the process is inside, if any.
    Read by the heartbeat thread while the main thread pushes and pops, so
    it is one indexing, not a test and then an index — a pop between the
    two would raise on the thread that must not die."""
    try:
        return _STACK[-1]
    except IndexError:
        return None


@contextmanager
def inside(axis: Axis, op: str) -> Iterator[None]:
    """Register ``op`` on ``axis`` as the collective the process is inside
    for the block's duration; the heartbeat reads it for the refusal."""
    _STACK.append((axis, op))
    try:
        yield
    finally:
        _STACK.pop()
