"""Real processes under ``gloo`` behind the rank-program interface of
``SimulatedWorld.run`` (``docs/model_parallelism.md`` §10.6).

``GlooWorld(geometry).run(program)`` spawns ``geometry.world`` processes
(``torch.multiprocessing.spawn``), each of which joins a ``gloo`` process
group on a free loopback port, builds the production
[`Mesh`][causalab.neural.shared.parallel.mesh.Mesh] through
``Mesh.from_environment`` — ``RANK`` / ``WORLD_SIZE`` are set the way a
launcher sets them — wraps it in a
[`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective], runs
``program(rank, collective)`` and writes its result to a file the parent
loads; the parent returns the results in rank order.

The program and the optional ``mutation`` — applied in the child before the
collective is built, for the hand-written mutation tests — cross a process
boundary, so both are module-level functions, picklable by name.

Nothing here hangs the test run: a rank that raises or exits is
`RankCrashed`, a world still running at the deadline is killed and
reported as `WorldTimedOut`, and every collective inside carries the
process group's own timeout.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import (
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
)

from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.neural.shared.parallel.spawn import reserve_port
from causalab.protocol.parallel import ParallelGeometry

#: How long one collective may wait on its peers before ``gloo`` refuses.
COLLECTIVE_TIMEOUT_S = 20.0
#: How long the parent waits for the whole world before killing it.
WORLD_TIMEOUT_S = 120.0

Program = Callable[[int, Collective], Any]
#: Applied in the child before the collective is built.
Mutation = Callable[[], None]


class GlooError(RuntimeError):
    """Base of what a spawned world can fail with."""


class RankCrashed(GlooError):
    """A rank raised or exited non-zero; the child's traceback is the message."""

    def __init__(self, rank: int, detail: str) -> None:
        self.rank = rank
        super().__init__(f"rank {rank} crashed: {detail}")


class WorldTimedOut(GlooError):
    """Ranks were still running at the deadline and have been killed."""

    def __init__(self, alive: Sequence[int]) -> None:
        self.alive = tuple(alive)
        super().__init__(
            f"the world did not finish within {WORLD_TIMEOUT_S:.0f} s; ranks "
            f"{list(alive)} were still running and have been killed"
        )


def _worker(
    rank: int,
    geometry: ParallelGeometry,
    port: int,
    program: Program,
    out_dir: str,
    mutation: Mutation | None,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(geometry.world)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=geometry.world,
        timeout=datetime.timedelta(seconds=COLLECTIVE_TIMEOUT_S),
    )
    try:
        if mutation is not None:
            mutation()
        collective = TorchCollective(Mesh.from_environment(geometry))
        result = program(rank, collective)
        torch.save(result, Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@dataclass(frozen=True)
class GlooWorld:
    """``geometry.world`` real processes under ``gloo``; see the module docstring."""

    geometry: ParallelGeometry
    mutation: Mutation | None = None

    def run(self, program: Program) -> list[Any]:
        """Run ``program(rank, collective)`` on every rank; results in rank order.

        Raises:
            RankCrashed: a rank raised or exited non-zero.
            WorldTimedOut: the world ran past `WORLD_TIMEOUT_S`.
        """
        world = self.geometry.world
        # the rendezvous port is the launcher's hold (``spawn.reserve_port``):
        # bound for the world's life, so rank 0's store never finds it taken
        with (
            tempfile.TemporaryDirectory(prefix="gloo-world-") as out_dir,
            reserve_port() as hold,
        ):
            context = spawn(
                _worker,
                args=(self.geometry, hold.port, program, out_dir, self.mutation),
                nprocs=world,
                join=False,
            )
            if context is None:  # ``join=False`` always hands the context back
                raise GlooError("torch.multiprocessing.spawn returned no context")
            _join(context)
            return [
                torch.load(Path(out_dir) / f"rank{rank}.pt", weights_only=False)
                for rank in range(world)
            ]


def _join(context: ProcessContext) -> None:
    deadline = time.monotonic() + WORLD_TIMEOUT_S
    try:
        # ``join`` returns as soon as any process exits; loop until all have
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                alive = [i for i, p in enumerate(context.processes) if p.is_alive()]
                _kill(context)
                raise WorldTimedOut(alive)
    except (ProcessRaisedException, ProcessExitedException) as exc:
        _kill(context)
        raise RankCrashed(exc.error_index, str(exc)) from exc


def _kill(context: ProcessContext) -> None:
    for process in context.processes:
        if process.is_alive():
            process.kill()
