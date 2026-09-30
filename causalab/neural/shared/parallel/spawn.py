"""The spawn half of the launcher (``docs/model_parallelism.md`` §3): the
parent of a world starts ``world`` local children re-entering the CLI's
``main`` with the same ``argv``, waits, and exits with their status.

Each child is a fresh interpreter (the ``spawn`` start method) handed
``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` / ``MASTER_PORT``
and [`LAUNCHER_VARIABLE`][] in
its own environment — the parent's is untouched — so its ``detect`` reads a
``spawned`` rank. The parent builds no engine, loads no model and **imports
no torch**: the children start as soon as the document is parsed, not after
the parent's own torch import ([`spawn`][]).

**Threads.** Each child defaults to one intra-op thread
(``OMP_NUM_THREADS=1``, [`CHILD_THREADS`][]) to avoid oversubscribing the
host. An explicit value in the parent's environment takes precedence. The
parent sets the default before spawning, because OpenMP reads it during the
child's torch import, and restores its environment afterwards
([`spawn_environment`][]).

**The gloo interface.** Local spawning uses ``127.0.0.1`` for rendezvous.
Unless ``GLOO_SOCKET_IFNAME`` is already set, the parent selects the loopback
interface to avoid hostname-resolution delays when gloo creates process
groups. A ``torchrun`` join retains its own environment: a multi-node
rendezvous must not bind the loopback.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import socket
import sys
from pathlib import Path

import traceback
from multiprocessing import connection
from typing import Any, Callable, Iterable, Mapping, Sequence

from causalab.protocol.parallel import ParallelGeometry

__all__ = [
    "CHILD_THREADS",
    "GLOO_INTERFACE_VARIABLE",
    "LAUNCHER_VARIABLE",
    "THREADS_VARIABLE",
    "PortHold",
    "RESERVED_LOW",
    "NoFreePort",
    "EPHEMERAL_FLOOR",
    "child_environment",
    "loopback_interface",
    "ephemeral_low",
    "reserve_port",
    "spawn",
    "spawn_environment",
]

logger = logging.getLogger(__name__)

#: Set by the spawn parent in each child's environment, to the word
#: ``spawned``, so a child knows it was spawned rather than joined; any
#: other value is refused by name (``launcher.detect``).
LAUNCHER_VARIABLE = "CAUSALAB_LAUNCHER"

#: The intra-op thread count variable torch reads (OpenMP's), and the count
#: each spawned child runs with unless the parent's environment names one
#: (module docstring).
THREADS_VARIABLE = "OMP_NUM_THREADS"
CHILD_THREADS = "1"

#: The interface gloo binds its sockets to. Unset, every ``ProcessGroupGloo``
#: resolves the node's *hostname* to pick one and, on a node whose name does
#: not resolve, waits out the resolver before falling back to the loopback
#: (module docstring): the parent names the loopback for a spawn, whose
#: rendezvous is ``127.0.0.1`` anyway.
GLOO_INTERFACE_VARIABLE = "GLOO_SOCKET_IFNAME"


def loopback_interface(interfaces: Iterable[str]) -> str | None:
    """The loopback interface among ``interfaces`` (``lo`` on Linux, ``lo0``
    on macOS: the name starting with ``lo`` whose remainder is digits), or
    ``None`` when the list names none."""
    for name in interfaces:
        if name.startswith("lo") and (not name[2:] or name[2:].isdigit()):
            return name
    return None


def _interfaces() -> tuple[str, ...]:
    try:
        return tuple(name for _, name in socket.if_nameindex())
    except OSError:
        return ()


def spawn_environment(
    environ: Mapping[str, str], interfaces: Iterable[str] | None = None
) -> dict[str, str]:
    """What the parent adds to its environment for the children's lifetime
    (module docstring): [`THREADS_VARIABLE`][] at [`CHILD_THREADS`][]
    when ``environ`` names no count — a count the user set is the count
    every rank runs with; and [`GLOO_INTERFACE_VARIABLE`][] at the node's
    loopback interface when ``environ`` names none and ``interfaces`` (the
    node's, by default) has one — a spawn's rendezvous is ``127.0.0.1``,
    so the loopback is the interface the children's ``gloo`` groups bind."""
    added: dict[str, str] = {}
    if THREADS_VARIABLE not in environ:
        added[THREADS_VARIABLE] = CHILD_THREADS
    if GLOO_INTERFACE_VARIABLE not in environ:
        loopback = loopback_interface(
            _interfaces() if interfaces is None else interfaces
        )
        if loopback is not None:
            added[GLOO_INTERFACE_VARIABLE] = loopback
    return added


def child_environment(index: int, world: int, port: int) -> dict[str, str]:
    """The variables child ``index`` of ``world`` sets over the environment
    it inherits: the group variables and the rendezvous on ``port``, and the
    spawn mark."""
    return {
        "WORLD_SIZE": str(world),
        "RANK": str(index),
        "LOCAL_RANK": str(index),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        LAUNCHER_VARIABLE: "spawned",
    }


#: How long a terminated child is given to exit before it is killed: a rank
#: blocked in a CUDA driver call, or one that installed its own ``SIGTERM``
#: handler, would otherwise hold the parent's refusal open forever.
TERMINATE_GRACE_S = 5.0

#: The most of a child's traceback that crosses the report pipe — its tail,
#: where the exception is. The parent reads the pipe only after the child has
#: exited, and a message wider than the pipe's buffer (16 KiB on macOS,
#: 64 KiB on Linux) would block the child in ``send`` before it could exit:
#: a deadlock, inherited from ``torch.multiprocessing.spawn``'s error queue.
TRACEBACK_LIMIT = 8 << 10


class PortHold:
    """The rendezvous port the parent chose, and — on Linux — the socket
    that keeps it: bound to the loopback, never listening, ``SO_REUSEADDR``
    set, alive from before the first child starts until the last is reaped
    ([`spawn`][]). The bound socket refuses every explicit bind without
    the flag, while rank 0's ``TCPStore`` listener — ``SO_REUSEADDR``, as
    libuv sets it — binds beside it. The port itself is drawn from
    **below the kernel's ephemeral range** ([`reserve_port`][]): a bound
    but unlistening ``SO_REUSEADDR`` socket does *not* take its port out of
    the pool the kernel assigns to outgoing connections, so an ephemeral
    port — however held — can be handed to some other process's
    ``connect`` before rank 0 listens. BSD (macOS)
    admits a second exact binding only when *both* sockets set
    ``SO_REUSEPORT``, which the store's listener does not, so there the
    hold is the probe alone, released at birth (``held`` is ``False``);
    the CPU smokes run on it, GPU hosts and CI on Linux.
    [`release`][] is idempotent, and the hold is a context manager."""

    def __init__(self, port: int, socket_: "socket.socket | None") -> None:
        self.port = port
        self._socket = socket_

    @property
    def held(self) -> bool:
        return self._socket is not None

    def release(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "PortHold":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"PortHold(port={self.port}, held={self.held})"


#: The lowest port [`reserve_port`][] draws: above the registered
#: services a node runs (sshd, slurm, NFS, the CUDA and NCCL debuggers), below
#: every kernel's ephemeral floor.
RESERVED_LOW = 20000

#: The Linux default ``ip_local_port_range`` floor, used when the file cannot
#: be read; macOS's ``net.inet.ip.portrange.first``.
EPHEMERAL_FLOOR: dict[str, int] = {"linux": 32768, "darwin": 49152}

#: How many candidate ports [`reserve_port`][] tries before refusing.
PORT_CANDIDATES = 64

_PORT_RANGE_FILE = "/proc/sys/net/ipv4/ip_local_port_range"


class NoFreePort(RuntimeError):
    """Every candidate port [`reserve_port`][] drew was taken."""

    def __init__(self, tried: int, low: int, high: int) -> None:
        self.tried = tried
        self.low = low
        self.high = high
        super().__init__(
            f"no free rendezvous port among {tried} candidates in [{low}, {high})"
        )


def ephemeral_low(
    platform: str = sys.platform, read: Callable[[], str] | None = None
) -> int:
    """The first port the kernel hands to outgoing connections: Linux's
    ``ip_local_port_range`` floor (``read`` reads the file; unreadable or
    malformed, the default), macOS's ``portrange.first`` default."""
    default = EPHEMERAL_FLOOR.get(platform, EPHEMERAL_FLOOR["linux"])
    if platform != "linux":
        return default
    try:
        text = read() if read is not None else Path(_PORT_RANGE_FILE).read_text()
        low = int(text.split()[0])
    except (OSError, ValueError, IndexError):
        return default
    return low if RESERVED_LOW < low <= 65535 else default


def reserve_port(
    platform: str = sys.platform,
    *,
    candidates: Sequence[int] | None = None,
    ephemeral: int | None = None,
) -> PortHold:
    """A free loopback port for the children's rendezvous, held for the
    world's life where the platform allows it ([`PortHold`][]). The port
    is drawn from ``[RESERVED_LOW, ephemeral)`` — below the kernel's
    ephemeral floor ([`ephemeral_low`][]), so no outgoing connection can
    be assigned it — the first candidate that binds *exclusively* (no
    ``SO_REUSEADDR`` on the probe: another hold would otherwise bind beside
    ours) and is then re-bound with the flag for the store's listener to
    join. ``candidates`` overrides the draw; every one taken is
    [`NoFreePort`][]."""
    low = RESERVED_LOW
    high = ephemeral if ephemeral is not None else ephemeral_low(platform)
    if candidates is None:
        candidates = random.sample(range(low, high), min(PORT_CANDIDATES, high - low))
    for port in candidates:
        probe = socket.socket()
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            probe.close()
            continue
        probe.close()
        if platform != "linux":
            return PortHold(port, None)
        hold = socket.socket()
        hold.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            hold.bind(("127.0.0.1", port))
        except OSError:
            hold.close()
            continue
        return PortHold(port, hold)
    raise NoFreePort(len(candidates), low, high)


def _child(
    index: int,
    argv: Sequence[str],
    world: int,
    port: int,
    entry: Callable[[Sequence[str]], int],
    report: Any,
) -> None:
    """One spawned child: the group variables and the spawn mark set
    ([`child_environment`][]), then ``entry(argv)`` — the CLI's ``main`` —
    whose non-zero status is the process's; an exception is its traceback,
    sent to the parent over ``report`` (the parent prints it in its
    refusal, as ``torch.multiprocessing.spawn`` did) and status 1."""
    os.environ.update(child_environment(index, world, port))
    try:
        code = entry(list(argv))
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — the child's last word is its traceback
        report.send(_bounded(traceback.format_exc()))
        report.close()
        sys.exit(1)
    report.close()
    if code:
        sys.exit(int(code))


def _bounded(text: str) -> str:
    """``text`` cut to its last [`TRACEBACK_LIMIT`][] bytes, saying so."""
    if len(text) <= TRACEBACK_LIMIT:
        return text
    return (
        f"… (traceback truncated to its last {TRACEBACK_LIMIT} bytes)\n"
        + text[-TRACEBACK_LIMIT:]
    )


def _cli_main(argv: Sequence[str]) -> int:
    from causalab.cli import main

    return main(list(argv))


def spawn(
    geometry: ParallelGeometry,
    argv: Sequence[str],
    *,
    entry: Callable[[Sequence[str]], int] = _cli_main,
    device: str = "cpu",
) -> int:
    """Start ``geometry.world`` local children re-entering ``entry`` — the
    CLI's ``main`` — with the same ``argv``, wait for them, and return their
    status: ``0`` when every child returned ``0``, else the first failed
    child's exit status (``1`` for one that raised, its traceback in the
    parent's refusal, and ``1`` for one a signal killed, the signal named in
    the refusal — the OOM killer's ``SIGKILL`` is the likeliest death of a
    rank run near the card's ceiling, and an operator must tell it from a
    ``sys.exit(1)``), with the rank named on stderr — a ``SIGABRT`` under
    NCCL (``device`` a CUDA word) named as NCCL's watchdog ending a rank
    whose collective timed out, the hang without a death
    (``launcher.describe_child_exit``); the other children are terminated
    once one has failed, rather than left waiting on a collective the
    failed rank never reaches. The parent owns the children's
    lifetime whichever way it leaves: a ``start`` that raised partway, a
    ``KeyboardInterrupt`` or ``SIGTERM`` reaching the parent while it waits
    (a job cancelled under ``nohup`` or ``scancel``, where the signal does
    not reach the children's process group) — every child still alive is
    terminated and reaped (`_reap`) and the rendezvous port released
    ([`PortHold`][]) before the parent's environment is put back, so no
    rank is left holding a device with no parent.

    ``entry`` is a module-level callable (the children are fresh
    interpreters and receive it by reference); each child is handed
    ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR`` /
    ``MASTER_PORT`` and [`LAUNCHER_VARIABLE`][] in its own environment —
    the parent's is untouched.

    The parent is **torch-free**: the standard library's ``spawn`` start
    method starts the children (what ``torch.multiprocessing.spawn`` wraps),
    so the children — which import torch anyway — start as soon as the
    document is parsed instead of after the parent's own torch import.
    """
    world = geometry.world
    hold = reserve_port()
    port = hold.port
    added = spawn_environment(os.environ)
    if added:
        # set before the children start: each child's OpenMP runtime reads
        # the thread count as torch is imported, ahead of any code of ours
        # in the child, and gloo reads its interface as each group is made
        logger.info(
            "each of the %d spawned ranks runs with %s (set a variable to "
            "choose otherwise)",
            world,
            " ".join(f"{name}={value}" for name, value in sorted(added.items())),
        )
        os.environ.update(added)
    context = multiprocessing.get_context("spawn")
    reports = [context.Pipe(duplex=False) for _ in range(world)]
    children = [
        context.Process(
            target=_child,
            args=(index, tuple(argv), world, port, entry, writer),
            daemon=False,
        )
        for index, (_, writer) in enumerate(reports)
    ]
    try:
        for child in children:
            child.start()
        for _, writer in reports:
            writer.close()  # the child's end is the child's
        failed = _wait(children)
    finally:
        _reap(children)
        hold.release()
        for name in added:
            # the parent's environment is left as it was
            os.environ.pop(name, None)
    if failed is None:
        return 0
    index, exitcode = failed
    from causalab.neural.shared.parallel.launcher import (
        backend_for,
        describe_child_exit,
    )
    from causalab.neural.shared.parallel.watchdog import Settings

    print(
        describe_child_exit(
            index=index,
            world=world,
            exitcode=int(exitcode),
            raised=_received(reports[index][0]),
            geometry=geometry,
            backend=backend_for(device),
            settings=Settings.from_environment(),
        ),
        file=sys.stderr,
    )
    return exitcode if exitcode > 0 else 1


def _received(reader: Any) -> str | None:
    """The traceback a child sent before exiting, or ``None`` — a child that
    exited by status closes its end unsent, which reads as end-of-file."""
    try:
        return reader.recv() if reader.poll() else None
    except EOFError:
        return None


def _wait(children: Sequence[Any]) -> tuple[int, int] | None:
    """Wait for every child; the first ``(rank, exitcode)`` to exit non-zero
    — negative for a signal, as ``multiprocessing`` reports it, so the
    caller can name it — the others terminated, or ``None`` when all
    exited 0."""
    pending = {child.sentinel: index for index, child in enumerate(children)}
    while pending:
        for sentinel in connection.wait(list(pending)):
            index = pending.pop(sentinel)
            child = children[index]
            child.join()
            code = child.exitcode
            if code:
                _reap(children)
                return index, int(code)
    return None


def _reap(children: Sequence[Any], grace: float | None = None) -> None:
    """Terminate every started child still alive and wait for all of them,
    bounded: ``SIGTERM``, ``join(grace)`` ([`TERMINATE_GRACE_S`][] unless
    given), then ``SIGKILL`` for one that would not go. A child never
    started has nothing to reap; one already exited is joined at once."""
    limit = TERMINATE_GRACE_S if grace is None else grace
    started = [child for child in children if child.pid is not None]
    for child in started:
        if child.is_alive():
            child.terminate()
    for child in started:
        child.join(limit)
        if child.is_alive():
            child.kill()
            child.join()
