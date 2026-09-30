"""A rank that goes wrong on purpose — the rank watchdog's real world
(``tests/neural/engines/pytorch_hooks/test_rank_watchdog_run.py``,
``test_rank_watchdog_cases_run.py``, ``tests/golden/test_parallel_watchdog.py``;
``docs/model_parallelism.md`` §3 "when a rank dies", §10.6).

The rank `RANK_VARIABLE` names acts once, in the `MODE_VARIABLE`
way (`MODES`), touching `MARK_VARIABLE`'s path the moment it
does so the harness can time the survivors from it:

- ``exit`` (the default): exits with `STATUS_VARIABLE` (``3``) the
  moment its **first collective** returns — the first of: transformers'
  rowwise style's output reduce (``RowwiseParallel.transform_output_post_forward``,
  the ``all_reduce`` or DTensor redistribute after ``o_proj`` in a Llama
  decoder layer, whichever path the style takes) and ``TorchCollective``'s
  header exchange (a tap's gather) — so after the model is loaded and the
  group and the mesh are up, in the middle of the first forward — through
  ``os._exit``, so no ``finally`` runs, no ``done`` is written and no
  process group is left: what a kill, an OOM or a segfault leaves behind.
  The mark file (`MARK_VARIABLE`) records *where*: the style's
  module by its features (``Linear(in->out)``), or the header's op.
  Hook both collective paths so the failure occurs inside a model layer;
  ``tests/neural/engines/pytorch_hooks/test_dying_rank_placement_run.py``
  checks the placement.
- ``mark``: the same moment, recorded and *not* acted on — the rank writes
  where it would have acted into the mark file and runs on, so a test can
  pin the hook's placement against the run's events.
- ``stop``: ``SIGSTOP``s itself at the same moment. Every thread stops,
  the heartbeat's included, so the peers read a death; the process itself
  stays, and whoever launched it must reap it (a stopped process ignores
  ``SIGTERM`` until continued; ``SIGKILL`` ends it).
- ``wedge``: its main thread blocks forever at the same moment while the
  heartbeat thread beats on — the hang without a death of §3, which only
  the collective timeout ends.
- ``exit-before-join``: exits as ``launcher.join_group`` is entered — after
  ``detect``, before its rendezvous store exists — the startup death whose
  peers never see a beat of its.
- ``exit-after-join``: exits as ``join_group`` returns — the group formed,
  its heartbeat beating, before any collective of the run.

Every other rank runs the production CLI untouched. Two entries, one per
launch path: `entry` is a spawn parent's ``entry``
(``launcher.spawn(geometry, argv, entry=dying_rank.entry)``; picklable by
name, applied in each child before the CLI runs); ``python -m
tests._helpers.dying_rank [--<mode>] run …`` is one ``torchrun``-style
rank with the group variables preset in its environment, a leading flag
naming the mode (`FLAGS`) for a command line typed on a node. The
module imports no torch: the hooks are installed in `arm`, so a
spawn parent that imports it stays torch-free.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence, get_args

__all__ = [
    "FLAGS",
    "MARK_VARIABLE",
    "MODES",
    "MODE_VARIABLE",
    "Mode",
    "RANK_VARIABLE",
    "STATUS_VARIABLE",
    "arm",
    "entry",
    "main",
    "mode_from",
]

RANK_VARIABLE = "CAUSALAB_TEST_DYING_RANK"
STATUS_VARIABLE = "CAUSALAB_TEST_DYING_STATUS"
MODE_VARIABLE = "CAUSALAB_TEST_DYING_MODE"
MARK_VARIABLE = "CAUSALAB_TEST_DYING_MARK"

Mode = Literal["exit", "stop", "wedge", "exit-before-join", "exit-after-join", "mark"]
MODES: tuple[Mode, ...] = get_args(Mode)
#: The command-line spelling of each mode.
FLAGS: dict[str, Mode] = {f"--{mode}": mode for mode in MODES}


def mode_from(environ: Mapping[str, str]) -> Mode:
    """The mode the environment names, ``exit`` when it names none.

    Raises:
        ValueError: a word outside `MODES`.
    """
    word = environ.get(MODE_VARIABLE, "exit")
    if word not in MODES:
        raise ValueError(f"{MODE_VARIABLE}={word!r} is not one of {MODES}")
    return word  # type: ignore[return-value]


def _mark(where: str) -> None:
    """Touch the mark, its content ``where`` and the wall clock (the events
    stream's clock, ``datetime.now(timezone.utc)``), for a placement test."""
    path = os.environ.get(MARK_VARIABLE)
    if path:
        stamp = datetime.now(timezone.utc).isoformat()
        Path(path).write_text(f"{stamp} {where}\n")


def _act(victim: str, mode: Mode, where: str) -> None:
    """Say what is about to happen, mark the moment, and do it."""
    status = int(os.environ.get(STATUS_VARIABLE, "3"))
    verb = {
        "exit": f"exiting {status}",
        "exit-before-join": f"exiting {status}",
        "exit-after-join": f"exiting {status}",
        "stop": "stopping itself (SIGSTOP)",
        "wedge": "blocking its main thread forever",
        "mark": "marking the moment and running on",
    }[mode]
    sys.stderr.write(f"dying rank {victim}: {verb} {where}\n")
    sys.stderr.flush()
    _mark(where)
    if mode == "mark":
        return
    if mode == "stop":
        os.kill(os.getpid(), signal.SIGSTOP)
        # continued from outside: carry on, the peers have judged already
        return
    if mode == "wedge":
        threading.Event().wait()  # the heartbeat thread beats on
        return
    os._exit(status)


def arm() -> Mode | None:
    """If this process is the dying rank, install the mode's hook; returns
    the mode it was armed with, ``None`` when this is not the victim."""
    victim = os.environ.get(RANK_VARIABLE)
    if victim is None or os.environ.get("RANK") != victim:
        return None
    mode = mode_from(os.environ)
    if mode in ("exit-before-join", "exit-after-join"):
        _arm_join(victim, mode)
    else:
        _arm_first_collective(victim, mode)
    return mode


def _arm_first_collective(victim: str, mode: Mode) -> None:
    """Act once, after the first collective of either kind (module
    docstring): the rowwise style's output reduce — hooked on the style
    class, whose ``tp_forward`` looks the method up at call time, so the
    hook holds whichever path (``dist.all_reduce`` or a DTensor
    redistribute) the style takes — or ``TorchCollective``'s header
    exchange."""
    from transformers.distributed.tensor_parallel import RowwiseParallel

    from causalab.neural.shared.parallel.collective import TorchCollective

    acted = False

    def once(where: str) -> None:
        nonlocal acted
        if not acted:
            acted = True
            _act(victim, mode, f"after its first collective ({where})")

    header: Callable[..., None] = TorchCollective._agree_header  # pyright: ignore[reportPrivateUsage]

    def dying_header(self: TorchCollective, *args: object, **kwargs: object) -> None:
        header(self, *args, **kwargs)
        once(
            f"header exchange of {kwargs.get('what', args[3] if len(args) > 3 else '?')}"
        )

    reduce: Callable[..., object] = RowwiseParallel.transform_output_post_forward

    def dying_reduce(
        self: RowwiseParallel, module: object, output: object, mesh: object
    ) -> object:
        result = reduce(self, module, output, mesh)
        features = f"({getattr(module, 'in_features', '?')}->{getattr(module, 'out_features', '?')})"
        once(f"rowwise reduce of {type(module).__name__}{features}")
        return result

    TorchCollective._agree_header = dying_header  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    RowwiseParallel.transform_output_post_forward = dying_reduce  # pyright: ignore[reportAttributeAccessIssue]


def _arm_join(victim: str, mode: Mode) -> None:
    from causalab.neural.shared.parallel import launcher

    original = launcher.join_group

    def dying(*args: object, **kwargs: object) -> object:
        if mode == "exit-before-join":
            _act(victim, mode, "before joining the group (no store, no beat)")
        heartbeat = original(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        _act(victim, mode, "after joining the group, before any collective")
        return heartbeat

    launcher.join_group = dying  # pyright: ignore[reportAttributeAccessIssue]


def entry(argv: Sequence[str]) -> int:
    """The CLI's ``main`` over ``argv``, the dying rank armed first."""
    arm()
    from causalab.cli import main

    return main(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """``[--<mode>] <cli argv>``: a leading flag names the mode for this
    process's environment (the hooks read one source), the rest is the CLI's."""
    rest = list(sys.argv[1:] if argv is None else argv)
    while rest and rest[0] in FLAGS:
        os.environ[MODE_VARIABLE] = FLAGS[rest.pop(0)]
    return entry(rest)


if __name__ == "__main__":
    raise SystemExit(main())
