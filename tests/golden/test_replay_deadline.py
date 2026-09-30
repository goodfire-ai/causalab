"""The replay deadline on NCCL (``docs/cuda_graphs.md`` "Hung replays",
``docs/model_parallelism.md`` §3): two ranks under ``torchrun`` replay a
captured NCCL ``all_reduce`` (``tests/_helpers/replay_rank.py``).

The process group's watchdog does not see collectives recorded into a graph:
without the deadline, a rank whose peer skipped the replay and left cleanly
waits in its synchronization until killed. Here that case must end with the
deadline's refusal (``refused: [P4] … a CUDA graph replay … has not
completed`` and ``LOST_STATUS``) within the collective timeout plus two beats.
A healthy world must be untouched: both ranks exit 0, and the per-replay cost
stays under `OVERHEAD_US`.

Both cases are subprocesses only, each bounded by `DEADLINE_S`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from causalab.neural.shared.parallel.spawn import reserve_port
from causalab.neural.shared.parallel.watchdog import (
    BEATS_PER_GRACE,
    COLLECTIVE_TIMEOUT_VARIABLE,
    LOST_STATUS,
    RANK_GRACE_VARIABLE,
)
from tests._helpers.replay_rank import EVENTS_VARIABLE

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices"),
]

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 20.0
GRACE_S = 3.0
#: the refusal lands on the first tick past the timeout: one beat, plus one
#: for the tick in flight and the process's exit
SLACK_S = 2 * GRACE_S / BEATS_PER_GRACE + 1.0
DEADLINE_S = 180.0
#: an event record and a queue append per replay, against a replay that
#: synchronizes over NCCL: generous for a noisy node
OVERHEAD_US = 50.0


def _world(mode: str, tmp_path: Path) -> tuple[int, list[dict[str, Any]], str]:
    env = {
        **os.environ,
        COLLECTIVE_TIMEOUT_VARIABLE: str(TIMEOUT_S),
        RANK_GRACE_VARIABLE: str(GRACE_S),
        EVENTS_VARIABLE: str(tmp_path),
        "PYTHONPATH": str(REPO),
    }
    stdout, stderr = tmp_path / f"{mode}.out", tmp_path / f"{mode}.err"
    # the port stays reserved while the world runs (spawn.reserve_port)
    with reserve_port() as hold, stdout.open("w") as out, stderr.open("w") as err:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            f"--master_port={hold.port}",
            "-m",
            "tests._helpers.replay_rank",
            mode,
        ]
        done = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=out,
            stderr=err,
            timeout=DEADLINE_S,
            check=False,
        )
    events = [
        json.loads(line)
        for path in sorted(tmp_path.glob("rank*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    return done.returncode, events, stderr.read_text()


def test_a_healthy_world_replays_untouched_by_the_bound(tmp_path: Path) -> None:
    status, events, stderr = _world("healthy", tmp_path)
    assert status == 0, stderr[-4000:]
    timed = [event for event in events if event["event"] == "timed"]
    assert sorted(event["rank"] for event in timed) == [0, 1]
    for event in timed:
        print(
            f"rank {event['rank']}: replay median {event['bounded_us']:.1f} µs bounded, "
            f"{event['unbounded_us']:.1f} µs unbounded"
        )
        assert event["bounded_us"] - event["unbounded_us"] < OVERHEAD_US
    assert "refused:" not in stderr


def test_a_replay_whose_peer_left_is_refused_at_the_deadline(tmp_path: Path) -> None:
    started = time.time()
    status, events, stderr = _world("skip", tmp_path)
    assert status != 0, "the world must not succeed with a rank hung"
    unmatched = [event for event in events if event["event"] == "unmatched"]
    assert len(unmatched) == 1 and unmatched[0]["rank"] == 0, events
    assert not [event for event in events if event["event"] == "returned"]
    refusals = [line for line in stderr.splitlines() if "refused: [P4]" in line]
    assert len(refusals) == 1, stderr[-4000:]
    refusal = refusals[0]
    print(refusal)
    assert "a CUDA graph replay enqueued" in refusal
    assert "on rank 0 of 2 has not completed" in refusal
    assert f"{COLLECTIVE_TIMEOUT_VARIABLE}={TIMEOUT_S:g}" in refusal
    # torchrun reports the rank's own status
    assert (
        f"exitcode  : {LOST_STATUS}" in stderr or f"exitcode: {LOST_STATUS}" in stderr
    )
    ended = time.time()
    lag = ended - unmatched[0]["at"]
    print(
        f"refused {lag:.1f} s after the unmatched replay (run {ended - started:.1f} s)"
    )
    assert TIMEOUT_S <= lag < TIMEOUT_S + SLACK_S + 5.0, lag
