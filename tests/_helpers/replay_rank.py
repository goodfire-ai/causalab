"""One rank of the replay deadline's real world (``tests/golden/test_replay_deadline.py``;
``docs/cuda_graphs.md`` "Hung replays", ``docs/model_parallelism.md`` §3).

``python -m tests._helpers.replay_rank <mode>`` under ``torchrun
--nproc_per_node=2``: the rank joins through the production launcher
(``launcher.detect`` / ``enter``, so its heartbeat runs), captures a
[`Replay`][causalab.neural.engines.pytorch_hooks.cuda_graphs.Replay] whose work is one
NCCL ``all_reduce`` over the tensor group — the collective a graph records and
the process group's watchdog never sees — and replays it matched once.

- ``healthy``: both ranks replay `REPLAYS` times, synchronizing after each,
  once with the deadline bounding every replay and once without; each prints
  ``{"event": "timed", …}`` with the per-replay medians, and both exit 0.
- ``skip``: rank 1 skips the next replay and leaves cleanly (status 0).
  Nobody is lost, so only the deadline can end rank 0, which
  replays, prints ``{"event": "unmatched", "at": <epoch>}`` and synchronizes
  for ever. The deadline's refusal and ``LOST_STATUS`` are its expected end.

Each rank appends its events as JSON lines to ``<EVENTS_VARIABLE>/rank<r>.jsonl``
(``torchrun`` interleaves the ranks' stdout within a line).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

from causalab.neural.engines.pytorch_hooks import cuda_graphs
from causalab.neural.shared.parallel import heartbeat, launcher
from causalab.protocol.parallel import parse_geometry

REPLAYS = 200
EVENTS_VARIABLE = "CAUSALAB_TEST_REPLAY_EVENTS"


def _say(rank: int, **fields: object) -> None:
    line = json.dumps({"rank": rank, "at": time.time(), **fields})
    with open(
        os.path.join(os.environ[EVENTS_VARIABLE], f"rank{rank}.jsonl"), "a"
    ) as out:
        out.write(line + "\n")


def _median_replay(replay: cuda_graphs.Replay, device: torch.device) -> float:
    times = []
    for _ in range(REPLAYS):
        started = time.perf_counter()
        replay()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def main(mode: str) -> int:
    geometry = parse_geometry("tp=2")
    launch = launcher.detect(geometry)
    assert isinstance(launch, launcher.Launch), "run under torchrun"
    word = launcher.device_for(launch, "cuda")
    device = torch.device(word)
    publisher = launcher.enter(launch, geometry, word)
    status = 1
    try:
        assert heartbeat.running() is not None, "a launched rank beats"
        assert isinstance(publisher, launcher.RankPublisher)
        group = publisher.mesh.group("tensor")
        cell = torch.ones(1 << 16, device=device)

        def work() -> torch.Tensor:
            dist.all_reduce(cell, group=group)
            return cell

        replay = cuda_graphs.Replay(work, {}, device=device)
        replay()
        torch.cuda.synchronize(device)
        rank = launch.rank
        _say(rank, event="matched")
        if mode == "healthy":
            bounded = _median_replay(replay, device)
            beat = heartbeat.running()
            heartbeat._running = None  # pyright: ignore[reportPrivateUsage]
            try:
                unbounded = _median_replay(replay, device)
            finally:
                heartbeat._running = beat  # pyright: ignore[reportPrivateUsage]
            _say(
                rank,
                event="timed",
                replays=REPLAYS,
                bounded_us=bounded * 1e6,
                unbounded_us=unbounded * 1e6,
            )
        elif mode == "skip":
            if rank == 1:
                _say(rank, event="skipping")
            else:
                replay()
                _say(rank, event="unmatched")
                torch.cuda.synchronize(device)
                _say(rank, event="returned")  # the deadline failed
        else:
            raise SystemExit(f"unknown mode {mode!r}")
        status = 0
        return status
    finally:
        launcher.leave(publisher, status)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
