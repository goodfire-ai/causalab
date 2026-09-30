"""The rank watchdog in a real world (``docs/model_parallelism.md`` §3 "when
a rank dies", §10.6): the tiny Llama at ``tp=2`` over ``gloo`` with rank 1
dying mid-forward, on both launch paths.

- **Joined** (``torchrun``'s shape: the group variables preset, one process
  per rank, no parent of ours): rank 1 exits with status 3 after its first
  collective (``tests/_helpers/dying_rank.py``), or is ``SIGKILL``ed from
  here once the run is under way. Rank 0, blocked in its next collective,
  must exit with [`LOST_STATUS`][causalab.neural.shared.parallel.watchdog.LOST_STATUS]
  and the refusal naming rank 1 on its stderr **within the documented
  bound** of the death — the grace plus three beats (``1.3 × grace``), plus
  the process's own exit — where before the watchdog it sat until the gloo
  group's thirty-minute timeout. The grace is set short here (``3 s``) and
  the collective timeout long (``120 s``), so a refusal inside the bound is
  the heartbeat's and cannot be the timeout's.
- **Spawned**: the parent's ``ProcessContext.join`` already terminates the
  siblings the moment a child exits and names the rank and its status; the
  test pins that the whole run ends far inside the collective timeout, so
  the parent's report is what a user sees, not a wait.

The assertion allows the documented ``1.3 × grace`` plus three seconds
for process exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from causalab.neural.shared.parallel.launcher import spawn
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    LOST_STATUS,
    RANK_GRACE_VARIABLE,
    Settings,
)
from causalab.protocol.parallel import ParallelGeometry
from causalab.neural.shared.parallel.spawn import reserve_port
from tests._helpers import dying_rank
from tests.neural.engines.pytorch_hooks import (
    test_tensor_expert_parallel_run as boundary,
)
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[4]

#: The watchdog's settings for these worlds: a short grace so the test is
#: quick, a long timeout so the refusal cannot be the backend's.
GRACE_S = 3.0
TIMEOUT_S = 120.0
#: The documented bound on the refusal after a death — the grace plus
#: three beats — and the slack for a process to print and exit.
BOUND_S = GRACE_S + 3 * Settings(TIMEOUT_S, GRACE_S).interval + 3.0
#: How long a whole world may take before the test gives up on it.
DEADLINE_S = 120.0


@pytest.fixture(scope="module")
def document(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("watchdog")
    return boundary._document(tmp, TINY_LLAMA)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv(RANK_GRACE_VARIABLE, str(GRACE_S))
    monkeypatch.setenv(COLLECTIVE_TIMEOUT_VARIABLE, str(TIMEOUT_S))
    monkeypatch.delenv(dying_rank.RANK_VARIABLE, raising=False)


@dataclass
class Exit:
    when: float
    code: int


@pytest.fixture
def port() -> Iterator[int]:
    """The ranks' rendezvous port, held by this process for the test's life
    (``spawn.reserve_port``: the launcher's own hold)."""
    with reserve_port() as hold:
        yield hold.port


def _rank_environment(rank: int, world: int, port: int, **extra: str) -> dict[str, str]:
    return {
        **os.environ,
        "WORLD_SIZE": str(world),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        **extra,
    }


def _launch(argv: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
    """One ``torchrun``-style rank of the dying-rank entry."""
    return subprocess.Popen(
        [sys.executable, "-m", dying_rank.__name__, *argv],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_all(
    procs: dict[int, subprocess.Popen[str]], deadline: float
) -> dict[int, Exit]:
    """Every rank's exit time and status, polled; ranks still running at
    the deadline are killed and the test fails naming them."""
    exits: dict[int, Exit] = {}
    while len(exits) < len(procs):
        for rank, proc in procs.items():
            if rank not in exits and proc.poll() is not None:
                exits[rank] = Exit(time.monotonic(), proc.returncode)
        if time.monotonic() > deadline:
            alive = sorted(set(procs) - set(exits))
            for rank in alive:
                procs[rank].kill()
            pytest.fail(f"ranks {alive} were still running at the deadline")
        time.sleep(0.05)
    return exits


def _stderr(proc: subprocess.Popen[str]) -> str:
    assert proc.stderr is not None
    return proc.stderr.read()


def _assert_named(err: str, geometry: str) -> None:
    assert "refused: [P4] at --parallel rank 1 of 2 has sent no heartbeat" in err, err
    assert "rank 0 is still running" in err and geometry in err, err
    assert f"{RANK_GRACE_VARIABLE}={GRACE_S:g}" in err, err


# --------------------------------------------------------------------------- #
# the joined path
# --------------------------------------------------------------------------- #


def test_joined_survivor_refuses_by_name_within_the_bound_of_a_ranks_exit(
    document: Path, tmp_path: Path, port: int
) -> None:
    argv = boundary._argv(document, tmp_path / "out", "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    procs = {
        rank: _launch(
            argv,
            _rank_environment(
                rank,
                2,
                port,
                **{dying_rank.RANK_VARIABLE: "1", dying_rank.STATUS_VARIABLE: "3"},
            ),
        )
        for rank in (0, 1)
    }
    exits = _wait_all(procs, time.monotonic() + DEADLINE_S)
    err0, err1 = _stderr(procs[0]), _stderr(procs[1])
    assert exits[1].code == 3, err1
    assert "dying rank 1: exiting 3 after its first collective" in err1
    assert exits[0].code == LOST_STATUS, err0
    _assert_named(err0, "tp=2")
    lag = exits[0].when - exits[1].when
    print(f"joined, exit 3: rank 0 refused {lag:.2f} s after rank 1's exit")
    assert GRACE_S - 2 * Settings(TIMEOUT_S, GRACE_S).interval <= lag <= BOUND_S, (
        f"rank 0 exited {lag:.2f} s after rank 1 (bound {BOUND_S:.1f} s): {err0}"
    )


def test_joined_survivor_refuses_by_name_within_the_bound_of_a_sigkill(
    document: Path, tmp_path: Path, port: int
) -> None:
    """The death from outside: ``SIGKILL`` once the run is under way (the
    joiner has opened its event stream), no exit status written anywhere."""
    out = tmp_path / "out"
    argv = boundary._argv(document, out, "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    procs = {rank: _launch(argv, _rank_environment(rank, 2, port)) for rank in (0, 1)}
    started = time.monotonic()
    while not (out / "events.jsonl").exists():
        if procs[1].poll() is not None or time.monotonic() - started > DEADLINE_S:
            for proc in procs.values():
                proc.kill()
            pytest.fail(
                "the run ended or timed out before its event stream opened: "
                + _stderr(procs[0])
                + _stderr(procs[1])
            )
        time.sleep(0.02)
    procs[1].send_signal(signal.SIGKILL)
    killed = time.monotonic()
    exits = _wait_all(procs, time.monotonic() + DEADLINE_S)
    err0 = _stderr(procs[0])
    assert exits[1].code == -signal.SIGKILL
    assert exits[0].code == LOST_STATUS, err0
    _assert_named(err0, "tp=2")
    lag = exits[0].when - killed
    print(f"joined, SIGKILL: rank 0 refused {lag:.2f} s after the kill")
    assert lag <= BOUND_S, f"rank 0 exited {lag:.2f} s after the kill: {err0}"


# --------------------------------------------------------------------------- #
# the spawn path
# --------------------------------------------------------------------------- #


def test_spawn_parent_names_the_dead_rank_without_waiting_for_the_timeout(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv(dying_rank.RANK_VARIABLE, "1")
    monkeypatch.setenv(dying_rank.STATUS_VARIABLE, "3")
    argv = boundary._argv(document, tmp_path / "out", "--parallel", "tp=2")  # pyright: ignore[reportPrivateUsage]
    started = time.monotonic()
    code = spawn(ParallelGeometry(tensor=2), argv, entry=dying_rank.entry)
    wall = time.monotonic() - started
    err = capsys.readouterr().err
    assert code == 3
    assert "refused: rank 1 of 2 exited with status 3 (--parallel " in err
    assert "tp=2" in err
    print(f"spawn: the whole world ended {wall:.2f} s after its start")
    # the parent terminated rank 0 the moment rank 1 exited: the whole world
    # — start, load, death, report — ends far inside the collective timeout
    # and inside the grace, so neither the backend nor the heartbeat spoke
    assert wall < min(TIMEOUT_S, GRACE_S + 40.0), f"the spawn took {wall:.1f} s"
