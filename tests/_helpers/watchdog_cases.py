"""The rank watchdog's failure cases as one table (``docs/model_parallelism.md``
§3 "when a rank dies", §10.6): what a rank does wrong, what its survivor
must print, with which status, and by when — torch-free, so the CPU guard
(``tests/_helpers/test_watchdog_cases.py``) holds the rules and the two
real worlds apply them unchanged: the gloo smoke
(``tests/neural/engines/pytorch_hooks/test_rank_watchdog_cases_run.py``)
and the NCCL golden (``tests/golden/test_parallel_watchdog.py``).

A `Case` is a `dying_rank` mode on a victim
rank under a [`Settings`][causalab.neural.shared.parallel.watchdog.Settings]
(``None`` for the mode when the harness ``SIGKILL``s the victim from
outside once the run is under way). Its rules are functions of the
settings, never numbers: the survivor's exit lands within `Case.bound`
of the act, prints one of `Case.phrases` (each alternative a tuple of
substrings that must all appear — the collective's own refusal has three)
and none of `Case.forbidden`, never a Python traceback, and exits
with `Case.survivor_status`; the victim exits with
`Case.victim_status`, or never (a stopped process, ``None``), or on
its own after the survivor (a wedged rank, whose heartbeat names the host
that left). `check` is the comparison against an `Observed`
world, a list of problems by name — empty when the case holds.

The bounds, from the watchdog's rules: a death or a stop is the heartbeat's
(``settings.bound``: the grace plus three beats) — a stopped **host**
included: its store acknowledges and never replies, the survivor's tick
stalls, and the watch counts the store unreachable from the tick's start
once a grace has passed without an answer (``heartbeat.STALL_S``), the
same bound as a dead host; a wedge is the collective timeout's, plus the
hold (the bound) for the survivor to prove every peer alive; a rank unreached
from the rendezvous is the timeout's plus one beat; each plus
`EXIT_SLACK_S` for the process to print and exit, and the startup
case plus `STARTUP_SKEW_S` for the two processes' start to differ.
"""

from __future__ import annotations

import dataclasses
import signal
from typing import Literal, Sequence

from causalab.neural.shared.parallel.launcher import nccl_abort_delay
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    LOST_STATUS,
    Settings,
)
from tests._helpers.dying_rank import Mode

__all__ = [
    "CASES",
    "EXIT_SLACK_S",
    "RUN_AHEAD_S",
    "STARTUP_SKEW_S",
    "Backend",
    "Case",
    "Observed",
    "TRACEBACK",
    "case_named",
    "check",
]

Backend = Literal["gloo", "nccl"]

#: Seconds for a process to print its refusal and exit after its verdict.
EXIT_SLACK_S = 3.0

#: The survivor's run-ahead: a wedged peer stops at *its* collective, but the
#: survivor keeps computing until it reaches the *next* one, and the timeout's
#: clock starts there, not at the mark. Allow four seconds for the
#: survivor to reach that collective, including a cold first forward.
RUN_AHEAD_S = 4.0
#: Seconds two ranks' starts may differ before the store exists (their
#: torch imports, the document's parse): the startup case's act is the
#: victim's, the clock the survivor's.
STARTUP_SKEW_S = 3.0
#: The exit status the harness's ``SIGKILL`` leaves.
KILLED = -signal.SIGKILL
#: What a survivor must never print: the run's failure is a refusal.
TRACEBACK = "Traceback (most recent call last)"

#: The words NCCL's own watchdog prints when it takes the process down on a
#: timed-out kernel, and the signal it ends with: **the expected outcome of
#: a wedge under NCCL** (§3) — a collective is asynchronous there, the
#: caller never blocks in it, and the watchdog thread's teardown is the only
#: thing that ends the rank (measured: ``-6`` at the timeout plus torch's
#: 60 s sleep, cut to 2 s by ``launcher.nccl_environment``). Under gloo the
#: survivor blocks in the collective and refuses by name itself.
NCCL_TEARDOWN = "Watchdog caught collective operation timeout"
NCCL_TEARDOWN_STATUS = -signal.SIGABRT


@dataclasses.dataclass(frozen=True)
class Case:
    """One failure of one rank (module docstring)."""

    name: str
    mode: Mode | None
    victim: int
    settings: Settings
    world: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.victim < self.world:
            raise ValueError(f"victim {self.victim} is outside a world of {self.world}")

    @property
    def survivor(self) -> int:
        """The rank whose refusal is checked: the lowest survivor."""
        return next(rank for rank in range(self.world) if rank != self.victim)

    @property
    def host_gone(self) -> bool:
        """The victim hosts the store: the survivor names the host."""
        return self.victim == 0

    def survivor_status(self, backend: Backend = "gloo") -> int:
        """``LOST_STATUS`` for a refusal the heartbeat draws; the CLI's ``1``
        for the collective's own refusal — the same number, on purpose; for
        a wedge under NCCL the watchdog's ``SIGABRT`` (module docstring)."""
        if self.mode == "wedge" and backend == "nccl":
            return NCCL_TEARDOWN_STATUS
        return LOST_STATUS

    @property
    def victim_exits(self) -> bool:
        """Whether the victim process ends: a stopped one never does."""
        return self.mode != "stop"

    @property
    def victim_status(self) -> int | None:
        """The victim's exit status: the dying rank's ``3``, the harness's
        ``SIGKILL``, ``LOST_STATUS`` for a wedged rank that names the host
        that left, ``None`` for a stopped rank that never exits."""
        if self.mode is None:
            return KILLED
        if self.mode == "stop":
            return None
        if self.mode == "wedge":
            return LOST_STATUS
        return 3

    @property
    def victim_after_survivor(self) -> bool:
        """The victim exits only after the survivor did (a wedged rank,
        named by nobody, names the host that left)."""
        return self.mode == "wedge"

    def bound(self, backend: Backend = "gloo") -> float:
        """Seconds from the act to the survivor's exit (module docstring): a
        wedge is the timeout's plus the survivor's run-ahead to its next
        collective (`RUN_AHEAD_S`) plus — under gloo — the hold, or —
        under NCCL — the watchdog's abort delay (``launcher.nccl_abort_delay``,
        2.1 s at the launcher's setting; the abort needs no hold, it names
        nobody)."""
        s = self.settings
        if self.mode == "wedge" and backend == "nccl":
            return s.timeout + nccl_abort_delay({}) + RUN_AHEAD_S + EXIT_SLACK_S
        if self.mode == "wedge":
            return s.timeout + s.bound + RUN_AHEAD_S + EXIT_SLACK_S
        if self.mode == "exit-before-join":
            return s.timeout + s.interval + STARTUP_SKEW_S + EXIT_SLACK_S
        return s.bound + EXIT_SLACK_S

    def victim_bound(self) -> float:
        """Seconds from the survivor's exit to a wedged victim's own exit:
        the host's store went with the survivor, so the victim's watch names
        it a grace and three beats later."""
        return self.settings.bound + EXIT_SLACK_S

    def phrases(self, backend: Backend = "gloo") -> tuple[tuple[str, ...], ...]:
        """The alternatives the survivor's stderr must satisfy: each a tuple
        of substrings that must all appear. One for a named loss; for the
        wedge under gloo the collective's own refusal — ``CollectiveFailed``
        inside a ``TorchCollective`` call, ``BackendFailed`` where the
        timed-out collective was transformers' own (the tensor-parallel
        styles' all-reduce, the first thing a survivor's forward hits) —
        and under NCCL the watchdog's own words with its teardown line
        (`NCCL_TEARDOWN`), the only ending a hang has there."""
        w = self.world
        if self.mode == "wedge" and backend == "nccl":
            return ((NCCL_TEARDOWN, "taking the entire process down"),)
        if self.mode == "wedge":
            return (
                (
                    f"failed on rank {self.survivor} of {w}",
                    "so the rank watchdog names none",
                    COLLECTIVE_TIMEOUT_VARIABLE,
                ),
            )
        if self.mode == "exit-before-join":
            return ((f"rank {self.victim} of {w} never reached the rendezvous",),)
        if self.host_gone:
            return ((f"rank 0 of {w}, the rendezvous store's host, is unreachable",),)
        return ((f"rank {self.victim} of {w} has sent no heartbeat",),)

    def forbidden(self, backend: Backend = "gloo") -> tuple[str, ...]:
        """Substrings the survivor must not print: a Python traceback,
        always — except the wedge under NCCL, where torch's C++ teardown
        prints its own stack and no Python frame of ours ever runs; the
        ``waiting in`` clause where the survivor was in no collective of
        ours (a death right after the group formed: it is loading, or in
        ``new_group``)."""
        if self.mode == "wedge" and backend == "nccl":
            return ()
        if self.mode == "exit-after-join":
            return (TRACEBACK, "waiting in")
        return (TRACEBACK,)

    def victim_phrases(self) -> tuple[tuple[str, ...], ...]:
        """The alternatives a wedged victim's stderr must satisfy once it
        exits on its own: the survivor's ``leave`` wrote ``done:1`` before
        its exit took the store, so the victim's heartbeat read either the
        status or the store gone — ``exited with status 1``, or ``the
        rendezvous store's host, is unreachable``."""
        if self.mode == "wedge":
            w, r = self.world, self.survivor
            return (
                (f"rank {r} of {w} exited with status 1",),
                (f"rank {r} of {w}, the rendezvous store's host, is unreachable",),
            )
        return ()

    @property
    def act(self) -> str:
        """The moment the lag is measured from, in words."""
        if self.mode is None:
            return f"SIGKILL of rank {self.victim} once the run is under way"
        return f"rank {self.victim} {self.mode} (its mark)"


#: A short grace and a long timeout: a refusal inside the bound is the
#: heartbeat's and cannot be the timeout's.
QUICK = Settings(timeout=120.0, grace=3.0)
#: A short timeout for the cases the timeout ends — long enough for two
#: ranks to load the tiny fixture and reach their first collective together
#: (the timeout must exceed the ranks' skew at a collective, §11).
TIMED = Settings(timeout=15.0, grace=3.0)
#: The startup case: no load sits before the store, a shorter timeout serves.
STARTUP = Settings(timeout=10.0, grace=2.0)

CASES: tuple[Case, ...] = (
    Case("stop", "stop", 1, QUICK),
    Case("wedge", "wedge", 1, TIMED),
    Case("host-stop", "stop", 0, QUICK),
    Case("host-exit", "exit", 0, QUICK),
    Case("host-kill", None, 0, QUICK),
    Case("exit-before-join", "exit-before-join", 1, STARTUP),
    Case("exit-after-join", "exit-after-join", 1, QUICK),
)


def case_named(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(
        f"no watchdog case named {name!r}; the cases are {[c.name for c in CASES]}"
    )


@dataclasses.dataclass(frozen=True)
class Observed:
    """What a real world showed: the survivor's status, stderr and lag
    (seconds from the act to its exit), the victim's status (``None``: still
    there), its stderr, and its own lag after the survivor's exit where it
    exited (``None`` otherwise)."""

    survivor_status: int
    survivor_stderr: str
    lag: float
    victim_status: int | None
    victim_stderr: str = ""
    victim_lag: float | None = None

    def record(self) -> dict[str, object]:
        """The observation as JSON-able fields, the stderrs' tails."""
        return {
            "survivor_status": self.survivor_status,
            "lag_s": round(self.lag, 3),
            "victim_status": self.victim_status,
            "victim_lag_s": None
            if self.victim_lag is None
            else round(self.victim_lag, 3),
            "survivor_stderr_tail": self.survivor_stderr[-2000:],
            "victim_stderr_tail": self.victim_stderr[-2000:],
        }


def _satisfied(text: str, alternatives: Sequence[Sequence[str]]) -> bool:
    return any(all(word in text for word in words) for words in alternatives)


def check(case: Case, observed: Observed, backend: Backend = "gloo") -> list[str]:
    """How ``observed`` falls short of ``case`` under ``backend`` (module
    docstring): every rule by name, ``[]`` when the case holds."""
    problems: list[str] = []
    if observed.survivor_status != case.survivor_status(backend):
        problems.append(
            f"the survivor exited {observed.survivor_status}, not "
            f"{case.survivor_status(backend)}"
        )
    bound = case.bound(backend)
    if not observed.lag <= bound:
        problems.append(
            f"the survivor exited {observed.lag:.2f} s after the act; the bound is "
            f"{bound:.2f} s ({case.act})"
        )
    if observed.lag < 0.0:
        problems.append(f"the survivor exited {-observed.lag:.2f} s before the act")
    err = observed.survivor_stderr
    if not _satisfied(err, case.phrases(backend)):
        problems.append(
            "the survivor's stderr names none of "
            f"{[' + '.join(words) for words in case.phrases(backend)]}"
        )
    for word in case.forbidden(backend):
        if word in err:
            problems.append(f"the survivor's stderr contains {word!r}")
    if observed.victim_status != case.victim_status:
        what = (
            "is still there"
            if observed.victim_status is None
            else f"exited {observed.victim_status}"
        )
        want = (
            "never exit" if case.victim_status is None else f"exit {case.victim_status}"
        )
        problems.append(f"the victim {what}; it should {want}")
    if case.victim_phrases() and not _satisfied(
        observed.victim_stderr, case.victim_phrases()
    ):
        problems.append(
            "the victim's stderr names none of "
            f"{[' + '.join(words) for words in case.victim_phrases()]}"
        )
    if case.victim_after_survivor and observed.victim_lag is not None:
        if not observed.victim_lag <= case.victim_bound():
            problems.append(
                f"the victim exited {observed.victim_lag:.2f} s after the survivor; the "
                f"bound is {case.victim_bound():.2f} s"
            )
    return problems
