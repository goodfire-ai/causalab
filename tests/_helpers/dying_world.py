"""A ``torchrun``-shaped world with one rank going wrong on purpose — the
harness the rank watchdog's real worlds share (``docs/model_parallelism.md``
§10.6; ``tests/_helpers/watchdog_cases.py`` for the rules it observes):
one subprocess per rank of ``python -m tests._helpers.dying_rank run …``,
the group variables preset, the victim and its mode in the environment,
and the moment of the act taken from the victim's mark file (or from the
harness's own ``SIGKILL``, sent once the run is under way).

Torch-free: the pytest process launching a world loads nothing. Every
rank's stderr goes to ``<stderr_dir>/rank<r>.stderr`` — a file, so a case
that times out or is killed leaves diagnostics available. A rank still
running when the harness finishes — a ``SIGSTOP``ped victim,
which nobody else reaps: not ``multiprocessing``, not a ``torchrun`` agent
waiting on exits — is ``SIGKILL``ed by `DyingWorld.reap`, which
names the ranks it had to end, so a test pins that the process it left
behind was the one the case says.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
)
from tests._helpers import dying_rank
from tests._helpers.watchdog_cases import Case, Observed

__all__ = ["DyingWorld", "Exit", "rank_environment"]

REPO = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class Exit:
    when: float
    code: int


def rank_environment(
    case: Case,
    rank: int,
    port: int,
    mark: Path,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Rank ``rank``'s environment for ``case``: the group variables and
    the rendezvous on the loopback, the watchdog's settings, the caches
    offline, and — on the victim — the dying rank's variables."""
    env = {
        **(os.environ if base is None else base),
        "WORLD_SIZE": str(case.world),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        RANK_GRACE_VARIABLE: f"{case.settings.grace:g}",
        COLLECTIVE_TIMEOUT_VARIABLE: f"{case.settings.timeout:g}",
    }
    env.pop(dying_rank.RANK_VARIABLE, None)
    env.pop(dying_rank.MODE_VARIABLE, None)
    if case.mode is not None:
        env[dying_rank.RANK_VARIABLE] = str(case.victim)
        env[dying_rank.MODE_VARIABLE] = case.mode
        env[dying_rank.STATUS_VARIABLE] = "3"
        env[dying_rank.MARK_VARIABLE] = str(mark)
    return env


class DyingWorld:
    """The launched ranks of one case (module docstring)."""

    def __init__(
        self,
        argv: Sequence[str],
        case: Case,
        *,
        port: int,
        mark: Path,
        under_way: Path,
        stderr_dir: Path,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        self.case = case
        self.mark = mark
        self.under_way = under_way
        self.stderr_dir = stderr_dir
        stderr_dir.mkdir(parents=True, exist_ok=True)
        self.procs: dict[int, subprocess.Popen[str]] = {}
        for rank in range(case.world):
            with self.stderr_path(rank).open("w") as sink:
                self.procs[rank] = subprocess.Popen(
                    [sys.executable, "-m", dying_rank.__name__, *argv],
                    cwd=REPO,
                    env=rank_environment(case, rank, port, mark, base_environment),
                    stdout=subprocess.DEVNULL,
                    stderr=sink,
                    text=True,
                )

    def stderr_path(self, rank: int) -> Path:
        return self.stderr_dir / f"rank{rank}.stderr"

    # -- waiting ----------------------------------------------------------------

    def wait(self, ranks: Sequence[int], deadline: float) -> dict[int, Exit]:
        """Each of ``ranks``' exit, polled; ranks still running at
        ``deadline`` are reported with ``code`` ``None``-less: raised as
        `TimeoutError` naming them (the caller reaps)."""
        exits: dict[int, Exit] = {}
        while len(exits) < len(ranks):
            for rank in ranks:
                if rank not in exits and self.procs[rank].poll() is not None:
                    exits[rank] = Exit(time.monotonic(), self.procs[rank].returncode)
            if len(exits) < len(ranks) and time.monotonic() > deadline:
                alive = sorted(set(ranks) - set(exits))
                raise TimeoutError(
                    f"ranks {alive} were still running at the deadline; their stderr "
                    f"is under {self.stderr_dir}: "
                    + " | ".join(
                        f"rank {rank}: {self.stderr(rank)[-600:]!r}" for rank in alive
                    )
                )
            time.sleep(0.02)
        return exits

    def wait_for_act(self, deadline: float) -> float:
        """When the act happened: the victim's mark appearing, or — with no
        mode — the harness's ``SIGKILL``, sent once the run is under way
        (its event stream open)."""
        marker = self.mark if self.case.mode is not None else self.under_way
        while not marker.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{marker.name} never appeared: the run never got there"
                )
            for rank, proc in self.procs.items():
                if proc.poll() is not None and rank != self.case.victim:
                    raise RuntimeError(
                        f"rank {rank} exited {proc.returncode} before the act: "
                        + self.stderr(rank)
                    )
            time.sleep(0.01)
        if self.case.mode is None:
            self.procs[self.case.victim].send_signal(signal.SIGKILL)
        return time.monotonic()

    def observe(self, deadline_s: float) -> Observed:
        """Run the case to its end: the act, the survivor's exit and the
        victim's fate, within ``deadline_s`` of now (module docstring)."""
        case = self.case
        deadline = time.monotonic() + deadline_s
        act = self.wait_for_act(deadline)
        survivors = [rank for rank in range(case.world) if rank != case.victim]
        exits = self.wait(survivors, deadline)
        survivor = exits[case.survivor]
        victim_status: int | None = None
        victim_lag: float | None = None
        if case.victim_exits:
            victim = self.wait(
                [case.victim], max(deadline, survivor.when + case.victim_bound())
            )[case.victim]
            victim_status = victim.code
            if case.victim_after_survivor:
                victim_lag = victim.when - survivor.when
        else:
            time.sleep(0.2)  # a stopped process: still there, not reaped by anyone
            victim_status = self.procs[case.victim].poll()
        return Observed(
            survivor_status=survivor.code,
            survivor_stderr=self.stderr(case.survivor),
            lag=survivor.when - act,
            victim_status=victim_status,
            victim_stderr=self.stderr(case.victim) if victim_status is not None else "",
            victim_lag=victim_lag,
        )

    # -- the end ------------------------------------------------------------------

    def stderr(self, rank: int) -> str:
        """A rank's stderr so far — the file it writes, readable while the
        rank runs and after it is gone."""
        return self.stderr_path(rank).read_text(errors="replace")

    def reap(self) -> list[int]:
        """``SIGKILL`` every rank still there and wait for all; the ranks
        that had to be killed, in order — a stopped victim is the one a
        case leaves behind."""
        left: list[int] = []
        for rank, proc in self.procs.items():
            if proc.poll() is None:
                left.append(rank)
                proc.send_signal(signal.SIGKILL)
        for proc in self.procs.values():
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass
        return left
