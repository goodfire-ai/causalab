"""A world of real processes over ``gloo`` for the smoke tier
(``docs/model_parallelism.md`` §10.6): transformers' DTensor styles call
``torch.distributed`` directly inside ``tp_forward``, so the one seam the
simulated tiers cannot cross is run here, on the tiny fixtures, at
``world ∈ {2, 4}``.

`run_world` spawns ``world`` ranks with ``torch.multiprocessing.spawn``,
each joining one ``gloo`` group over a loopback TCP rendezvous, running the
same rank program with the same payload (SPMD, §3), and handing back a
picklable result — tensors cross by value, as arrays — collected by rank
with its tensors restored. A rank that
raises fails the whole world, with its traceback in the error; a world that
does not finish within ``timeout`` is killed and refused, never hung.

A rank program that loads a model builds its
[`Sharding`][causalab.neural.engines.pytorch_hooks.sharding.Sharding] the way the
engine does — ``Sharding.from_mesh(Mesh.from_environment(geometry))`` over
``neural/shared/parallel/mesh.py`` — so the test and the production path
create the same process groups in the same order; ``RANK`` and
``WORLD_SIZE`` are set in each rank the way a launcher sets them.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import spawn

from causalab.neural.shared.parallel.spawn import reserve_port

__all__ = ["RankProgram", "WorldError", "run_world"]

#: ``(rank, world, payload) -> result``; runs inside the initialised group.
RankProgram = Callable[[int, int, Any], Any]


class WorldError(RuntimeError):
    """A rank failed, or the world did not finish in time."""


def _by_value(value: Any) -> Any:
    """A result with every tensor as a numpy array, recursively through
    dicts, lists and tuples. ``torch.multiprocessing`` ships a tensor across
    the queue through shared memory, and the producing rank then lingers at
    exit until that memory is released — minutes, on macOS — while an array
    pickles by value and the rank exits when its program returns."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _by_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_by_value(v) for v in value)
    if isinstance(value, list):
        return [_by_value(v) for v in value]
    return value


def _as_tensors(value: Any) -> Any:
    """The inverse of `_by_value`: every array back to a tensor."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: _as_tensors(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_as_tensors(v) for v in value)
    if isinstance(value, list):
        return [_as_tensors(v) for v in value]
    return value


def _rank_main(
    rank: int,
    world: int,
    port: int,
    program: RankProgram,
    payload: Any,
    queue: Any,
) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
        )
        result = program(rank, world, payload)
        queue.put((rank, "ok", _by_value(result)))
        dist.barrier()
    except BaseException:  # noqa: BLE001 — the traceback is the payload
        queue.put((rank, "error", traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_world(
    world: int, program: RankProgram, payload: Any, *, timeout: float = 600.0
) -> dict[int, Any]:
    """Run ``program(rank, world, payload)`` on ``world`` ranks over gloo and
    return every rank's result by rank (module docstring).

    Raises:
        WorldError: a rank raised (its traceback quoted), or the world did
            not deliver every result within ``timeout`` seconds.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    hold = reserve_port()  # the launcher's hold: the port is ours for the world's life
    procs = spawn(
        _rank_main,
        args=(world, hold.port, program, payload, queue),
        nprocs=world,
        join=False,
    )
    assert procs is not None  # ``join=False`` returns the context
    results: dict[int, Any] = {}
    errors: dict[int, str] = {}
    try:
        for _ in range(world):
            try:
                rank, status, value = queue.get(timeout=timeout)
            except Exception as err:  # noqa: BLE001 — queue.Empty is the timeout
                raise WorldError(
                    f"world of {world} did not deliver every result within "
                    f"{timeout:.0f}s: have {sorted(results)}, errors {sorted(errors)}"
                ) from err
            if status == "ok":
                results[rank] = _as_tensors(value)
            else:
                errors[rank] = str(value)
                break
        if errors:
            raise WorldError(
                "rank(s) failed:\n"
                + "\n".join(f"--- rank {r} ---\n{t}" for r, t in sorted(errors.items()))
            )
        procs.join(timeout=timeout)
    finally:
        for process in procs.processes:
            if process is not None and process.is_alive():
                process.kill()
        hold.release()
    return results
