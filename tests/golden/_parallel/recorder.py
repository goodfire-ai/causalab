"""The recording entry of the parallel golden's fit runs
(``docs/model_parallelism.md`` §7, §10.6): ``python -m tests.golden._parallel.recorder
run …`` is ``causalab run …`` with the §7 gradient guard recording every
parameter's gradient before its mean, and this rank's peak device memory,
written at exit under `GRADIENTS_VARIABLE` — ``rank<r>.pt`` (the
per-step gradients) and ``rank<r>.json`` (the memory) — and, under
`MEMORY_TRACE_VARIABLE` (or ``--memory-trace DIR`` ahead of the CLI
verb, which sets it for the children), this rank's **memory trace**:
one JSON line per point, ``rank<r>.jsonl``, the bytes allocated and
reserved on its device once the point's outputs are written
([`TraceLine`][causalab.neural.shared.parallel.soak.TraceLine]; the soak rule
``soak.flat_after_warmup`` reads it). The trace wraps the executor's
point loop at its one seam, ``execution._execute_point``, so the
production loop is untouched.

The production spawn path is untouched: a spawned child re-imports the
parent's main module (``multiprocessing``'s ``spawn`` start method, by the
``-m`` name), so the installation at import below runs in every rank before
the CLI does, and the receipt still says ``launcher: spawned``. The world-1
run goes through the same entry so its gradients are recorded the same way
(``RANK`` unset there: rank 0). A rank that touched its device writes its
memory whether or not it trained (the large model's inference documents
measure their peaks through this entry); a process that never trained and
never initialised CUDA — the spawn parent — writes nothing. The rank is
read at save time, after the launcher
set it; ``tests/neural/engines/pytorch_hooks/test_train_parallel_run.py``'s
``_recording_guard``, the same recorder for the gloo smoke, reads it at
install time inside a ``torchrun``-style child where it is already set.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "GRADIENTS_VARIABLE",
    "MEMORY_TRACE_FLAG",
    "MEMORY_TRACE_VARIABLE",
    "device_sample",
    "gradients_path",
    "install",
    "install_trace",
    "memory_path",
    "split_trace_flag",
    "trace_path",
]

#: Where the recorder writes; the smoke tier's spelling (held equal by
#: ``tests/golden/test_parallel_record.py``), so one variable turns on
#: gradient recording under both tiers.
GRADIENTS_VARIABLE = "CAUSALAB_TEST_GRADIENTS_DIR"
#: Where the per-point memory trace goes, one ``rank<r>.jsonl`` per rank;
#: set by ``--memory-trace DIR`` on this entry or by the harness (``runs.run``
#: for every recorded document), inherited by the spawned ranks.
MEMORY_TRACE_VARIABLE = "CAUSALAB_TEST_MEMORY_TRACE_DIR"
MEMORY_TRACE_FLAG = "--memory-trace"
_TRACED = "__memory_trace__"


def gradients_path(directory: Path, rank: int) -> Path:
    return directory / f"rank{rank}.pt"


def memory_path(directory: Path, rank: int) -> Path:
    return directory / f"rank{rank}.json"


def trace_path(directory: Path, rank: int) -> Path:
    return directory / f"rank{rank}.jsonl"


def device_sample() -> tuple[int, int, str | None]:
    """``(allocated, reserved, device)`` on this rank's current CUDA device;
    ``(0, 0, None)`` where no CUDA context exists (the CPU tiers)."""
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return 0, 0, None
    device = torch.cuda.current_device()
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
        f"cuda:{device}",
    )


def install_trace() -> None:
    """Wrap ``execution._execute_point`` so every point this rank finishes
    appends one [`TraceLine`][causalab.neural.shared.parallel.soak.TraceLine]
    to ``trace_path(MEMORY_TRACE_VARIABLE, RANK)``. The rank and the
    directory are read at write time (after the launcher set them); the
    point index is the position in this rank's request, the digest the
    point's own."""
    from causalab.neural.shared import execution as execution_module
    from causalab.neural.shared.parallel.soak import TraceLine

    real = execution_module._execute_point  # pyright: ignore[reportPrivateUsage]
    if getattr(real, _TRACED, False):
        # installed once per process: a spawned child runs this module's
        # top level twice — as ``__mp_main__`` and, through the package's
        # ``__init__`` (``runs`` imports it by name), as itself — and a
        # second wrap would trace every point twice.
        return
    counter = {"point": 0}

    def traced(member: Any, request: Any, **kwargs: Any) -> Any:
        summary = real(member, request, **kwargs)
        allocated, reserved, device = device_sample()
        line = TraceLine(
            rank=int(os.environ.get("RANK", "0")),
            point=counter["point"],
            point_digest=str(getattr(member, "point_digest", "")),
            allocated=allocated,
            reserved=reserved,
            device=device,
        )
        counter["point"] += 1
        directory = Path(os.environ[MEMORY_TRACE_VARIABLE])
        directory.mkdir(parents=True, exist_ok=True)
        with trace_path(directory, line.rank).open("a") as handle:
            handle.write(line.render() + "\n")
        return summary

    setattr(traced, _TRACED, True)
    execution_module._execute_point = traced  # type: ignore[assignment]


def split_trace_flag(argv: list[str]) -> tuple[list[str], str | None]:
    """``["--memory-trace", DIR, "run", …]`` → ``(["run", …], DIR)``: the
    recorder's own flag, ahead of the CLI verb, taken off the arguments the
    CLI sees. Absent, ``(argv, None)``; given without a value, refused."""
    if argv[:1] != [MEMORY_TRACE_FLAG]:
        return list(argv), None
    if len(argv) < 2:
        raise SystemExit(f"{MEMORY_TRACE_FLAG} needs a directory")
    return list(argv[2:]), argv[1]


def _save(records: list[list[Any]]) -> None:
    import torch

    on_device = torch.cuda.is_available() and torch.cuda.is_initialized()
    if not records and not on_device:
        return  # a spawn parent: trains nothing, touches no device
    directory = Path(os.environ[GRADIENTS_VARIABLE])
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if records:
        torch.save(records, gradients_path(directory, rank))
    memory: dict[str, Any] = {
        "rank": rank,
        "steps": len(records),
        "device": None,
        "peak_bytes_allocated": None,
        "peak_bytes_reserved": None,
    }
    if on_device:
        device = torch.cuda.current_device()
        memory.update(
            device=f"cuda:{device}",
            peak_bytes_allocated=torch.cuda.max_memory_allocated(device),
            peak_bytes_reserved=torch.cuda.max_memory_reserved(device),
        )
    memory_path(directory, rank).write_text(
        json.dumps(memory, indent=2, sort_keys=True) + "\n"
    )


def install() -> None:
    """Wrap ``train.average_gradients`` so every call records the parameters'
    gradients (cloned, before the mean) and register the save at exit."""
    from causalab.neural.engines.pytorch_hooks import train as train_module

    real = train_module.average_gradients
    records: list[list[Any]] = []

    def recording(parameters: Any, collective: Any, **kwargs: Any) -> None:
        parameters = list(parameters)
        records.append(
            [p.grad.detach().clone() for p in parameters if p.grad is not None]
        )
        real(parameters, collective, **kwargs)

    train_module.average_gradients = recording  # type: ignore[assignment]
    atexit.register(_save, records)


if __name__ == "__main__":
    # the flag sets the variable before the CLI spawns: the children
    # re-import this module and install from the environment below
    _arguments, _trace = split_trace_flag(sys.argv[1:])
    if _trace is not None:
        os.environ[MEMORY_TRACE_VARIABLE] = _trace

if GRADIENTS_VARIABLE in os.environ:
    install()
if MEMORY_TRACE_VARIABLE in os.environ:
    install_trace()

if __name__ == "__main__":
    from causalab.cli import main

    sys.exit(main(_arguments))
