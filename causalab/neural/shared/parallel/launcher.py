"""The SPMD launcher (``docs/model_parallelism.md`` §3, §8.3, §9): what this
process is, how a world of ranks comes to exist, which ranks publish, and
the production [`Publisher`][] over
``torch.distributed``.

**Launch.** If ``WORLD_SIZE`` is in the environment the process joins that
group (``torchrun``, multi-node, Slurm) — a ``joined`` rank. Otherwise a
geometry with ``world > 1`` makes the CLI's process the **parent** of a
spawn: it starts ``world`` local children (``torch.multiprocessing.spawn``),
each re-entering the same CLI ``main`` with the same ``argv`` and
``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` / ``MASTER_*`` set — a ``spawned``
rank, marked by [`LAUNCHER_VARIABLE`][] so the receipt can say so — and
waits, exiting with the children's status. World 1 is ``solo``: no
``torch.distributed`` initialised, today's path. [`detect`][] is that
decision, over an explicit environment mapping so no test reads the process
environment.

**Publishing.** Rank 0 of each data-parallel replica publishes
([`publish_here`][]): the rank at local index 0 on every axis but
``data``. Those ranks are exactly rank 0's data group, so the data-parallel
join gathers over that group — every publishing rank hands its shard in,
the publishing rank of replica 0 (the joiner) receives them all. Under the
``rows`` mode of the data axis (§8.3) every replica computes the whole
campaign and the outputs are identical on every replica by construction, so
the joiner alone publishes and nothing is gathered: [`publishes`][] is
the decision, ``publish_here`` narrowed to replica 0, and the joiner's
gather is the identity.

**One mesh per process.** [`enter`][] joins the process group and builds
this rank's [`Mesh`][] — every process group the geometry needs,
created once — and the [`RankPublisher`][] gathers over *its* data
group; the engine takes the same mesh off the request's publisher for its
collective and its sharding (``serving.process_mesh``), so nothing carves a
second set of groups.

**When a rank dies** (§3, §11; [`.watchdog`][causalab.neural.shared.parallel.watchdog], [`.heartbeat`][causalab.neural.shared.parallel.heartbeat]).
[`join_group`][] builds the rendezvous store itself — the ``TCPStore``
rank 0 hosts on ``MASTER_ADDR:MASTER_PORT``, as ``env://`` would, but
**without the store's wait for workers** — so the rank can keep a client
of it for the heartbeat, which beats from before any peer has arrived, and
initialises the default group and (through [`enter`][]) every mesh group
with the collective timeout the environment names
(``CAUSALAB_COLLECTIVE_TIMEOUT``, 600 s by default; ``new_group`` would
otherwise take torch's 30-minute global default). The heartbeat thread
beats on the store and watches the peers; a peer gone for the grace
(``CAUSALAB_RANK_GRACE``, 30 s) while this rank still runs is a refusal by
name and the rank exits, instead of blocking in its next collective until
the backend's timeout. A peer that **never arrives** — dead before it
connected — cannot be told from one still starting, so it is named
(``never reached the rendezvous``) once the collective timeout has passed:
the same cost the store's own wait for workers had, which named a count
and no rank; a rendezvous step that fails under it (the environment
agreement's read, ``init_process_group``) is held for the heartbeat's word
first ([`join_group`][]), like a collective. [`leave`][] tells the
peers this rank's status first, so a clean finish is never read as a death
and a refusal of this rank's own is named with its status.

The spawn itself — the parent starting the children — is [`.spawn`][causalab.neural.shared.parallel.spawn],
re-exported here as [`spawn`][]; [`LAUNCHER_VARIABLE`][] is its mark.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime
import os
import signal
import sys
import threading
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    get_args,
)

from causalab.neural.shared.parallel import watchdog
from causalab.neural.shared.parallel.environment import agree_environment
from causalab.neural.shared.parallel.spawn import LAUNCHER_VARIABLE, spawn
from causalab.neural.shared.parallel.watchdog import Settings, StoreHost
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    Axis,
    MeshLayout,
    ParallelGeometry,
    format_geometry,
)
from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.lockstep import Lockstep
from causalab.protocol.publish import LAUNCHERS, Publisher

if TYPE_CHECKING:
    from causalab.neural.shared.parallel.mesh import Mesh
    from causalab.neural.shared.parallel.heartbeat import Heartbeat

__all__ = [
    "AGENT_STORE_VARIABLE",
    "Backend",
    "BackendFailed",
    "NCCL_BLOCKING_WAIT_VARIABLES",
    "NCCL_DUMP_WAIT_MS",
    "NCCL_DUMP_WAIT_VARIABLE",
    "NCCL_ERROR_HANDLING_VARIABLE",
    "describe_child_exit",
    "nccl_abort_delay",
    "hosts_store",
    "nccl_environment",
    "store_host",
    "RendezvousFailed",
    "backend_error",
    "backend_for",
    "cuda_visible_count",
    "detect",
    "device_for",
    "enter",
    "GROUP_VARIABLES",
    "hold_for_peer",
    "join_group",
    "Launch",
    "LAUNCHER_VARIABLE",
    "LauncherWord",
    "leave",
    "leave_group",
    "lockstep",
    "NVIDIA_GPUS",
    "Parent",
    "publish_here",
    "publishes",
    "RankPublisher",
    "rendezvous",
    "SOLO_LAUNCH",
    "spawn",
]

LauncherWord = Literal["solo", "spawned", "joined"]
assert get_args(LauncherWord) == LAUNCHERS

Backend = Literal["gloo", "nccl"]

#: The variables a launched rank reads (``torchrun``'s names): the world, this
#: rank, its ordinal on the node, and the rendezvous address the process
#: group initialises from.
GROUP_VARIABLES: tuple[str, ...] = (
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)

#: The axes whose local index decides publishing: every axis but ``data``.
_PUBLISHING_AXES: tuple[Axis, ...] = ("pipeline", "context", "model")

#: ``torchrun`` sets this to ``True`` in every worker when the rendezvous
#: store at ``MASTER_ADDR:MASTER_PORT`` is the **agent's** (the c10d backend
#: by default — ``TORCH_DISABLE_SHARE_RDZV_TCP_STORE`` unset — and the
#: ``static`` backend always): then no rank hosts it, every rank connects as
#: a client, and torch's own ``TCPStore`` master would fall back to a client
#: on the failed bind, logging so. The store's lifetime is then the agent's:
#: it goes when rank 0 fails (``--max-restarts 0`` exits the agent) and
#: stays when rank 0 is stopped ([`store_host`][]; §3 "two nodes").
AGENT_STORE_VARIABLE = "TORCHELASTIC_USE_AGENT_STORE"

#: NCCL's watchdog thread sleeps four times this many milliseconds between
#: detecting a timed-out collective and ending the process
#: ([`nccl_environment`][]): ``TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC``,
#: 15 000 by default — sixty seconds of nothing, after the timeout.
NCCL_DUMP_WAIT_VARIABLE = "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"
NCCL_DUMP_WAIT_MS = 500
#: The watchdog thread polls its works every 100 ms (``kWatchdogThreadSleepMillis``).
NCCL_WATCHDOG_POLL_S = 0.1

#: The two NCCL settings whose values make a hang undetectable or unending,
#: refused by name ([`nccl_environment`][]): the blocking wait creates no
#: watchdog thread at all, and error handling ``0`` (none) or ``2`` (clean
#: up only) never ends the process.
NCCL_BLOCKING_WAIT_VARIABLES: tuple[str, ...] = (
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT",
)
NCCL_ERROR_HANDLING_VARIABLE = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
NCCL_TEARDOWN_MODES: tuple[str, ...] = ("1", "3")


def store_host(environ: Mapping[str, str]) -> StoreHost:
    """Who hosts the rendezvous store under ``environ``: rank 0's own
    process, or — [`AGENT_STORE_VARIABLE`][] ``True`` — rank 0's
    ``torchrun`` agent."""
    return "agent" if environ.get(AGENT_STORE_VARIABLE) == "True" else "rank"


def hosts_store(rank: int, environ: Mapping[str, str]) -> bool:
    """Whether ``rank`` builds the store as its master: rank 0, unless the
    agent already hosts it ([`store_host`][]) — then every rank is a
    client, by name rather than by torch's silent fallback."""
    return rank == 0 and store_host(environ) == "rank"


def nccl_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What a NCCL rank sets in its environment before any process group
    exists (§3 "a hang without a death", §11): [`NCCL_DUMP_WAIT_VARIABLE`][]
    at [`NCCL_DUMP_WAIT_MS`][] unless ``environ`` spells it — a user's
    choice stands, as with ``OMP_NUM_THREADS`` for a spawn.

    **Why.** NCCL collectives are asynchronous: the host enqueues the kernel
    and moves on, so a collective that never completes — a wedged peer — is
    seen by NCCL's *watchdog thread*, never by the caller, and the only
    mechanism that ends the rank is that thread tearing the process down.
    After detecting a timeout, the watchdog waits
    ``4 × TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`` before aborting, even when
    timeout dumps are disabled. The default here is 500 ms; the expected
    abort delay is computed by [`nccl_abort_delay`][].
    ``TORCH_NCCL_BLOCKING_WAIT`` is refused because it disables the watchdog
    needed to detect a hung collective. Error-handling modes ``0`` and ``2``
    are also refused: neither terminates the rank after a failed collective.

    Raises:
        ProtocolError: ``P4`` at ``--parallel`` — the blocking wait, or an
            error-handling mode that never ends a hung rank.
    """
    for name in NCCL_BLOCKING_WAIT_VARIABLES:
        value = environ.get(name)
        if value is not None and value.strip().lower() not in (
            "",
            "0",
            "false",
            "n",
            "no",
            "off",
        ):
            raise _refuse(
                f"{name}={value!r} makes a hang undetectable under NCCL: with the "
                "blocking wait torch creates no watchdog thread and the caller does "
                "not block either, so a collective a wedged peer never joins times "
                "out nowhere (measured: nothing fired in 240 s; docs/model_parallelism.md "
                "§3) — unset it; the launcher bounds NCCL's teardown instead"
            )
    mode = environ.get(NCCL_ERROR_HANDLING_VARIABLE)
    if mode is not None and mode.strip() not in NCCL_TEARDOWN_MODES:
        raise _refuse(
            f"{NCCL_ERROR_HANDLING_VARIABLE}={mode!r} never ends a hung rank under "
            "NCCL: the watchdog aborts nothing (0) or the communicator alone (2) and "
            "the process runs on with the collective's incomplete data (measured "
            "under 2: the survivor never exited; docs/model_parallelism.md §3) — "
            "leave it at torch's default, which tears the process down"
        )
    if NCCL_DUMP_WAIT_VARIABLE in environ:
        return {}
    return {NCCL_DUMP_WAIT_VARIABLE: str(NCCL_DUMP_WAIT_MS)}


def nccl_abort_delay(environ: Mapping[str, str]) -> float:
    """Seconds between a NCCL group's timeout and the watchdog's abort of
    the process under ``environ`` (module docstring of
    [`nccl_environment`][]): four dump waits and one poll — 2.1 s at the
    launcher's setting, 60.1 s at torch's default."""
    text = environ.get(NCCL_DUMP_WAIT_VARIABLE)
    wait_ms = NCCL_DUMP_WAIT_MS if text is None or not text.strip() else int(text)
    return 4 * wait_ms / 1000.0 + NCCL_WATCHDOG_POLL_S


def describe_child_exit(
    *,
    index: int,
    world: int,
    exitcode: int,
    raised: str | None,
    geometry: ParallelGeometry,
    backend: Backend,
    settings: Settings,
) -> str:
    """The spawn parent's one line for a child that failed (``spawn.py``):
    its traceback, its signal or its status — and, for a ``SIGABRT`` under
    NCCL, what that signal is: NCCL's watchdog ending a rank whose
    collective timed out while its peers were alive, the hang without a
    death (§3), whose own words (``Watchdog caught collective operation
    timeout``) stand above this line in the child's stderr."""
    where = f"(--parallel {format_geometry(geometry)})"
    if raised:
        return f"refused: rank {index} of {world} raised:\n{raised} {where}"
    if exitcode < 0:
        name = signal.Signals(-exitcode).name
        if backend == "nccl" and name == "SIGABRT":
            return (
                f"refused: [P4] at --parallel rank {index} of {world} was ended by "
                "NCCL's watchdog (SIGABRT): a collective timed out after "
                f"{watchdog.COLLECTIVE_TIMEOUT_VARIABLE}={settings.timeout:g} s while "
                "its peers were alive — a hang without a death, which no heartbeat "
                "names; NCCL's own words stand above (docs/model_parallelism.md §3) "
                f"{where}"
            )
        return f"refused: rank {index} of {world} was killed by {name} {where}"
    return f"refused: rank {index} of {world} exited with status {exitcode} {where}"


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Launch:
    """What this process is: the launcher word the receipt records, this
    rank of ``world``, and its ordinal on the node (the CUDA device it runs
    on under ``--device cuda``).

    Raises:
        ValueError: a word outside [`LAUNCHERS`][], a rank outside the
            world, a negative local rank, or ``solo`` with a world above 1.
    """

    word: LauncherWord
    rank: int
    world: int
    local_rank: int

    def __post_init__(self) -> None:
        if self.word not in LAUNCHERS:
            raise ValueError(f"launcher {self.word!r} is not one of {LAUNCHERS}")
        if self.world < 1:
            raise ValueError(f"world must be at least 1, got {self.world}")
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {self.rank} is outside range({self.world})")
        if self.local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {self.local_rank}")
        if self.word == "solo" and self.world != 1:
            raise ValueError("a solo launch is a world of one rank")


@dataclasses.dataclass(frozen=True)
class Parent:
    """This process is the parent of a spawn: no group in the environment
    and a geometry of ``world`` ranks. It starts the children and exits with
    their status, never loading a model."""

    world: int


#: World 1: this process alone.
SOLO_LAUNCH = Launch("solo", 0, 1, 0)


def _refuse(message: str) -> ProtocolError:
    return ProtocolError("P4", message, path="--parallel")


class RendezvousFailed(ProtocolError):
    """The rendezvous failed on this rank and no peer's death explains it:
    the store could not be reached (rank 0 never started hosting it), or a
    step over it — the environment agreement, ``init_process_group`` —
    raised past the heartbeat's word. ``P4`` at ``--parallel``, naming the
    rank, the step and the backend's own words."""

    def __init__(self, *, rank: int, world: int, step: str, cause: BaseException):
        self.rank = rank
        self.step = step
        super().__init__(
            "P4",
            f"{step} failed on rank {rank} of {world}: {_first_line(cause)} — no "
            "peer was lost within the bound, so the rank watchdog names none; a "
            f"rank that never arrives costs {watchdog.COLLECTIVE_TIMEOUT_VARIABLE} "
            "(docs/model_parallelism.md §3)",
            path="--parallel",
        )


class BackendFailed(ProtocolError):
    """A ``torch.distributed`` error raised **outside** this package's
    collectives — transformers' own ``all_reduce`` inside a tensor-parallel
    style, NCCL's watchdog surfacing a timed-out kernel at the next sync —
    that no peer's death explains: the heartbeat was asked and named nobody.
    ``P4`` at ``--parallel``, naming the rank, the backend's words and the
    timeout that bounds a hang without a death ([`hold_for_peer`][])."""

    def __init__(self, *, rank: int, world: int, grace: float, cause: BaseException):
        self.rank = rank
        super().__init__(
            "P4",
            f"a collective outside this package's calls failed on rank {rank} of "
            f"{world}: {_first_line(cause)} — no peer went silent for the grace "
            f"after the failure ({watchdog.RANK_GRACE_VARIABLE}={grace:g}), so the "
            "rank watchdog names none; a hang without a death is bounded by "
            f"{watchdog.COLLECTIVE_TIMEOUT_VARIABLE} (docs/model_parallelism.md §3)",
            path="--parallel",
        )


def _first_line(error: BaseException) -> str:
    text = str(error).strip()
    first = text.splitlines()[0] if text else ""
    return f"{type(error).__name__}: {first}"


def _integer(environ: Mapping[str, str], name: str) -> int:
    """``environ[name]`` as a non-negative integer, refused by name."""
    text = environ.get(name)
    if text is None:
        raise _refuse(
            f"the environment sets WORLD_SIZE but not {name} — a launched rank "
            f"carries every one of {list(GROUP_VARIABLES[:3])} (torchrun sets them)"
        )
    if not text.isdigit():
        raise _refuse(
            f"{name}={text!r} in the environment is not a non-negative integer"
        )
    return int(text)


def detect(
    geometry: ParallelGeometry, environ: Mapping[str, str] | None = None
) -> Launch | Parent:
    """What this process is under ``geometry`` (§3).

    ``world == 1`` is [`SOLO_LAUNCH`][] — inside a one-process group too,
    since world 1 never initialises ``torch.distributed``; a larger group in
    the environment with no geometry to match it is refused (two launched
    processes would both run and both write the whole campaign). Above
    world 1, ``WORLD_SIZE`` in the environment makes this a ``joined`` rank
    — or a ``spawned`` one when [`LAUNCHER_VARIABLE`][] says so — with
    every disagreement refused by name; no ``WORLD_SIZE`` makes this the
    [`Parent`][] of a spawn. ``environ`` defaults to the process
    environment.

    Raises:
        ProtocolError: ``P4`` naming ``--parallel`` and the variable — a
            ``WORLD_SIZE`` that disagrees with ``geometry.world``, a
            missing or malformed ``RANK`` / ``LOCAL_RANK``, a rank outside
            the world, a foreign [`LAUNCHER_VARIABLE`][] word.
    """
    env = os.environ if environ is None else environ
    world_text = env.get("WORLD_SIZE")
    if geometry.world == 1:
        if world_text is not None and world_text != "1":
            raise _refuse(
                f"the environment sets WORLD_SIZE={world_text}, a launched group, "
                f"but --parallel {format_geometry(geometry)} is a world of 1 — "
                "every process of the group would run and publish the whole "
                "campaign; name the geometry the group was launched for"
            )
        return SOLO_LAUNCH
    if world_text is None:
        return Parent(world=geometry.world)
    world = _integer(env, "WORLD_SIZE")
    if world != geometry.world:
        raise _refuse(
            f"the environment sets WORLD_SIZE={world} but --parallel "
            f"{format_geometry(geometry)} is a world of {geometry.world} ranks — "
            "the launched group and the geometry must agree"
        )
    rank = _integer(env, "RANK")
    if rank >= world:
        raise _refuse(f"RANK={rank} in the environment is outside WORLD_SIZE={world}")
    local_rank = _integer(env, "LOCAL_RANK")
    word_text = env.get(LAUNCHER_VARIABLE)
    if word_text is None:
        word: LauncherWord = "joined"
    elif word_text == "spawned":
        word = "spawned"
    else:
        raise _refuse(
            f"{LAUNCHER_VARIABLE}={word_text!r} in the environment is not a word "
            "this launcher sets — only a spawn parent sets it, to 'spawned'"
        )
    return Launch(word, rank, world, local_rank)


def backend_for(device: str) -> Backend:
    """``nccl`` for a CUDA device word, ``gloo`` for everything else — how
    ``--device cpu`` with ``world > 1`` runs the CPU test tier over every
    axis on the tiny fixtures (§3)."""
    return "nccl" if device.split(",")[0].strip().startswith("cuda") else "gloo"


#: Where the NVIDIA driver lists the node's GPUs on Linux: one entry per
#: device, read by the spawn parent without importing torch.
NVIDIA_GPUS = Path("/proc/driver/nvidia/gpus")


def visible_devices(
    device: str,
    environ: Mapping[str, str] | None = None,
    gpus: Path = NVIDIA_GPUS,
) -> int | None:
    """How many devices of ``device``'s kind this node shows: the CUDA count
    for a CUDA word, ``None`` for any other word — there is nothing to bound
    a CPU rank by. The count is read **without torch** where it can be —
    the driver's listing (``gpus``, one entry per device, Linux) is the
    node's device set, and ``CUDA_VISIBLE_DEVICES``, when set, can only
    narrow it ([`cuda_visible_count`][], CUDA's own enumeration rule) —
    and from ``torch.cuda.device_count()`` otherwise: the spawn parent asks
    before any child starts, and its torch import would sit serially ahead
    of every child's (``spawn.py``)."""
    if not device.strip().startswith("cuda"):
        return None
    env = os.environ if environ is None else environ
    listed = len(list(gpus.iterdir())) if gpus.is_dir() else 0
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return cuda_visible_count(visible, listed or None)
    if listed:
        return listed
    import torch  # noqa: PLC0415 — the pure verbs stay torch-free

    return torch.cuda.device_count()


def cuda_visible_count(spelling: str, listed: int | None) -> int:
    """How many devices CUDA enumerates for ``CUDA_VISIBLE_DEVICES=spelling``
    on a node with ``listed`` devices (``None`` when the count is unknown):
    the entries in order, **stopping at the first invalid one** — CUDA's
    documented rule, under which ``0,1,2,3`` on a two-GPU node is two
    devices, not four, ``-1`` (the spelling for "no devices") is none, and a
    repeated ordinal is not a second device. An ordinal at or past the
    listed count is invalid where the count is known; a UUID entry
    (``GPU-…``, ``MIG-…``) cannot be checked without the driver and counts.
    """
    seen: set[str] = set()
    count = 0
    for raw in spelling.split(","):
        entry = raw.strip()
        if not entry or entry in seen:
            break
        if not entry.startswith(("GPU-", "MIG-")):
            if not entry.isdigit():
                break
            if listed is not None and int(entry) >= listed:
                break
        seen.add(entry)
        count += 1
    return count


def check_spawn_devices(
    geometry: ParallelGeometry, device: str, *, visible: int | None
) -> None:
    """A spawn puts one rank on each device of this node (§3): a CUDA world
    larger than the devices the node shows is refused **before** any child
    starts. A multi-node world joins a
    ``torchrun`` group instead, where every rank's ``LOCAL_RANK`` is a
    device on its own node.

    Raises:
        ProtocolError: ``P4`` — a CUDA word, a known count, and
            ``geometry.world`` above it.
    """
    if not visible or not device.strip().startswith("cuda"):
        # no count, or no CUDA at all (the CPU-only tiers stub the group and
        # pin the cuda:LOCAL_RANK mapping): torch refuses the word itself
        return
    if geometry.world > visible:
        raise _refuse(
            f"--parallel {format_geometry(geometry)} spawns a world of "
            f"{geometry.world} ranks, one per CUDA device, and this node shows "
            f"{visible}: use a geometry of at most {visible} ranks here, or launch "
            "the ranks across nodes with torchrun (docs/model_parallelism.md §3)"
        )


def device_for(launch: Launch, device: str, *, visible: int | None = None) -> str:
    """The device this rank runs on: ``cuda:LOCAL_RANK`` for a CUDA word
    under a launched world, the word itself otherwise.

    Raises:
        ProtocolError: ``P4`` — a comma list of devices above world 1; a
            rank is one device, and placing one rank's layers across several
            is not served under a geometry.
    """
    if launch.world == 1:
        return device
    if "," in device:
        raise _refuse(
            f"--device {device!r} places one process's layers across several "
            "devices, and under --parallel a rank is one device — name one "
            "device (cuda runs each rank on cuda:LOCAL_RANK)"
        )
    if device.strip().startswith("cuda"):
        if visible and launch.local_rank >= visible:
            raise _refuse(
                f"LOCAL_RANK={launch.local_rank} names no CUDA device on this node, "
                f"which shows {visible}: a rank is one device (cuda:LOCAL_RANK), so "
                "a launcher must place at most that many ranks here"
            )
        return f"cuda:{launch.local_rank}"
    return device


def publish_here(launch: Launch, layout: MeshLayout) -> bool:
    """Whether this rank publishes (§3): rank 0 of its data-parallel replica
    — local index 0 on ``pipeline``, ``context`` and ``model`` (so on
    ``tensor`` and ``expert`` too). Everything else computes and discards.

    Raises:
        ValueError: ``launch.rank`` is outside the layout's world.
    """
    return all(layout.rank_in(launch.rank, axis) == 0 for axis in _PUBLISHING_AXES)


def publishes(launch: Launch, geometry: ParallelGeometry) -> bool:
    """Whether this rank's outputs leave the process under ``geometry``
    (§3, §8.3): [`publish_here`][] — rank 0 of each data replica — over
    points, where every replica holds a shard to hand the joiner; over
    **rows**, where every replica holds the whole campaign, rank 0 of
    replica 0 alone.

    Raises:
        ValueError: ``launch.rank`` is outside the geometry's world.
    """
    layout = MeshLayout(geometry)
    if not publish_here(launch, layout):
        return False
    return geometry.data_mode != "rows" or layout.rank_in(launch.rank, "data") == 0


# --------------------------------------------------------------------------- #
# the process group
# --------------------------------------------------------------------------- #


def rendezvous(
    launch: Launch,
    geometry: ParallelGeometry,
    settings: Settings,
    environ: Mapping[str, str] | None = None,
) -> tuple[object, Heartbeat]:
    """The rendezvous store and this rank's heartbeat over it (module
    docstring): the ``TCPStore`` on ``MASTER_ADDR:MASTER_PORT`` — hosted by
    rank 0 (or, under ``torchrun``, by rank 0's agent already:
    [`hosts_store`][], every rank then a client), joined by every other
    rank, its timeout the collective's, so a slow rank's arrival at
    ``new_group`` or NCCL's communicator exchange is bounded by that and
    not by the grace; **no wait for workers**, so the heartbeat below is
    the watch over who arrives and names a rank that never does — and a
    second, client connection of it with the grace as its timeout for the
    [`Heartbeat`][], started here so a rank that dies during
    the group's own initialisation is named too, told who hosts the store
    so an unreachable store is named as the host it had.

    Raises:
        ProtocolError: ``P4`` — ``MASTER_ADDR`` or ``MASTER_PORT`` missing or
            malformed.
        RendezvousFailed: the store could not be reached within the
            collective timeout — rank 0 never started hosting it. Without
            the store there is no heartbeat to name it, so this is the one
            startup death that costs the timeout with no rank named (§11).
    """
    import torch.distributed as dist

    from causalab.neural.shared.parallel.heartbeat import Heartbeat

    env = os.environ if environ is None else environ
    host = env.get("MASTER_ADDR")
    port_text = env.get("MASTER_PORT")
    if host is None or port_text is None:
        raise _refuse(
            "the environment sets WORLD_SIZE but not MASTER_ADDR and MASTER_PORT — "
            "a launched rank rendezvous on the address they name (torchrun sets "
            "them; the spawn parent does)"
        )
    if not port_text.isdigit():
        raise _refuse(f"MASTER_PORT={port_text!r} in the environment is not a port")
    port = int(port_text)
    hosted_by = store_host(env)
    try:
        store = dist.TCPStore(
            host,
            port,
            launch.world,
            hosts_store(launch.rank, env),
            timeout=datetime.timedelta(seconds=settings.timeout),
            wait_for_workers=False,
        )
        watch = dist.TCPStore(
            host,
            port,
            launch.world,
            False,
            timeout=datetime.timedelta(seconds=settings.grace),
            wait_for_workers=False,
        )
    except RuntimeError as error:
        # torch's DistNetworkError: the host is not listening and did not
        # start to within the timeout
        who = "rank 0" if hosted_by == "rank" else "rank 0's torchrun agent"
        raise RendezvousFailed(
            rank=launch.rank,
            world=launch.world,
            step=f"the rendezvous on {host}:{port}, which {who} hosts,",
            cause=error,
        ) from error
    heartbeat = Heartbeat(
        watch,
        rank=launch.rank,
        world=launch.world,
        settings=settings,
        geometry=geometry,
        store_host=hosted_by,
    )
    heartbeat.start()
    return store, heartbeat


def join_group(
    launch: Launch,
    backend: Backend,
    *,
    geometry: ParallelGeometry,
    settings: Settings,
) -> Heartbeat | None:
    """Initialise ``torch.distributed`` for this rank: ``gloo`` or ``nccl``
    over the store [`rendezvous`][] builds from the environment's
    ``MASTER_ADDR`` / ``MASTER_PORT``, with ``settings.timeout`` as the
    default group's timeout, a CUDA rank pinned to ``cuda:LOCAL_RANK``
    first and its NCCL environment set ([`nccl_environment`][]: the
    watchdog's post-timeout sleep cut from 60 s to 2 s, so a hang ends the
    rank at the timeout and not a minute later; a setting that would make
    a hang undetectable refused by name); the rank's heartbeat, beating from before the group exists; the
    environment-derived settings agreed over the store before the group is
    built (``parallel/environment.py``: a world whose ranks spell a timeout,
    the grace, the context waiver or the gradient agreement differently is
    refused by name here, on every rank, instead of diverging inside the
    group). A ``solo`` launch initialises nothing and has no heartbeat.

    A backend error in either step — the agreement's read of a peer's key
    timing out, the group's rendezvous — is held for the heartbeat's word
    first (``Heartbeat.hold``): a peer that never arrived is named there
    and the process ends; only a failure nobody's absence explains comes
    back, as [`RendezvousFailed`][].

    Raises:
        ProtocolError: ``P4`` — the ranks disagree on an agreed setting.
        RendezvousFailed: the rendezvous failed with no peer lost.
    """
    if launch.word == "solo":
        return None
    import torch
    import torch.distributed as dist

    if backend == "nccl":
        # read by ProcessGroupNCCL's monitor as each group is built: before
        # the first
        os.environ.update(nccl_environment(os.environ))
        torch.cuda.set_device(launch.local_rank)
    store, heartbeat = rendezvous(launch, geometry, settings)
    step = "the environment agreement over the rendezvous store"
    try:
        agree_environment(store, rank=launch.rank, world=launch.world)  # type: ignore[arg-type]
        step = f"init_process_group({backend!r})"
        dist.init_process_group(
            backend,
            store=store,
            rank=launch.rank,
            world_size=launch.world,
            timeout=datetime.timedelta(seconds=settings.timeout),
        )
    except RuntimeError as error:
        # a peer that never arrived, or died arriving, is refused by name
        # here and the process ends; anything else is the rendezvous's own
        heartbeat.hold()
        raise RendezvousFailed(
            rank=launch.rank, world=launch.world, step=step, cause=error
        ) from error
    return heartbeat


def leave_group(bound: float | None = None) -> bool:
    """Tear the process group down, if one was initialised, and say whether
    it finished: within ``bound`` seconds when one is given — the teardown
    then runs on a daemon thread the caller waits on, and a teardown that
    has not returned by then is left behind (``False``) for the caller to
    exit past — unbounded otherwise.

    **Why bounded.** ``destroy_process_group`` under NCCL finalizes the
    communicators, which flushes against the peers: a peer alive but never
    leaving — its main thread wedged in Python while its heartbeat beats —
    holds the finalize forever, after this rank has already decided its
    status — the gloo wedge's survivor, or a rank whose store failed, on
    its way out with its refusal still unprinted (``main`` prints it after
    ``_launched``'s ``finally``, which is this teardown). A teardown that
    returns takes milliseconds; one that does not is the hang without a
    death, which no heartbeat names. (Under NCCL a wedge's survivor never
    reaches here: NCCL's watchdog tears it down, [`nccl_environment`][].)"""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True
    if bound is None:
        dist.destroy_process_group()
        return True
    done = threading.Event()

    def teardown() -> None:
        try:
            dist.destroy_process_group()
        except Exception as err:  # noqa: BLE001 — the exit is decided; the words are all that is left
            sys.stderr.write(
                f"the process group's teardown raised {type(err).__name__}: {err}\n"
            )
        finally:
            done.set()

    threading.Thread(target=teardown, name="causalab-teardown", daemon=True).start()
    return done.wait(bound)


def _exit_at_shutdown(status: int) -> None:
    """Arrange the process's end at interpreter shutdown regardless of a
    thread stuck in the backend: ``os._exit(status)`` after the streams are
    flushed, from an ``atexit`` hook — so ``main`` still prints its refusal
    and returns its status first, and no C++ destructor or NCCL teardown
    waits on a peer after that."""

    def end() -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)

    atexit.register(end)


class RankPublisher:
    """The [`Publisher`][] of a launched rank
    over the process's one [`Mesh`][]: its replica on the data
    axis, whether it publishes, and a gather over the mesh's data group to
    the group's first rank — the publishing rank of replica 0 — through
    the ``TorchCollective`` over the mesh (``parallel/objects.py:gather_objects``,
    the payload pickled over the tensor collectives, so the hand-off is
    held to the collective contract and runs under the tests' simulator).
    It creates no process group of its own; the mesh did, once, for every
    axis.

    Only publishing ranks ever call ``gather``: a data group's members all
    publish or none do, since publishing is decided by the coordinates the
    group shares — except under the ``rows`` mode (§8.3), where the joiner
    publishes alone and its gather hands its own payload straight back,
    nothing to join. With one replica the mesh has no data group and the
    gather is the identity.

    ``heartbeat`` is the rank's watch over its peers (module docstring),
    ``None`` where none runs; [`leave`][] finishes it.

    Raises:
        ValueError: ``launch.rank`` is not the mesh's rank.
    """

    def __init__(
        self, launch: Launch, mesh: Mesh, heartbeat: Heartbeat | None = None
    ) -> None:
        if launch.rank != mesh.rank or launch.world != mesh.geometry.world:
            raise ValueError(
                f"the launch is rank {launch.rank} of {launch.world} but the mesh "
                f"was built for rank {mesh.rank} of {mesh.geometry.world}"
            )
        self._launch = launch
        self.mesh = mesh
        self.heartbeat = heartbeat
        self.launcher: str = launch.word
        self.replica: int = mesh.local_rank("data")
        self.replicas: int = mesh.geometry.data
        self.publish: bool = publishes(launch, mesh.geometry)
        self._alone: bool = mesh.geometry.data_mode == "rows"

    def gather(self, payload: T) -> Sequence[T] | None:
        if not self.publish:
            raise AssertionError(
                f"rank {self._launch.rank} does not publish and has nothing to gather"
            )
        if self._alone:
            # rows (§8.3): the joiner publishes alone, nothing to join
            return (payload,)
        if self.mesh.group("data") is None:
            return (payload,)
        from causalab.neural.shared.parallel.collective import TorchCollective
        from causalab.neural.shared.parallel.objects import gather_objects

        # the joiner is the data group's first member (local index 0): it
        # receives every replica's shard, the others hand theirs over
        return gather_objects(payload, "data", TorchCollective(self.mesh))


def enter(launch: Launch, geometry: ParallelGeometry, device: str) -> Publisher:
    """A launched rank's entry: read the watchdog's settings once, join the
    process group for ``device``'s backend, build this rank's one mesh over
    it — every group with the collective timeout — and the publisher over
    the mesh. The CLI calls this once, after [`detect`][], before any
    engine is built; the engine reads the mesh off the publisher.

    Raises:
        ProtocolError: ``P4`` — a malformed watchdog setting
            ([`WatchdogSetting`][causalab.neural.shared.parallel.watchdog.WatchdogSetting]), a missing rendezvous
            address.
    """
    from causalab.neural.shared.parallel.mesh import Mesh

    settings = Settings.from_environment()
    heartbeat = join_group(
        launch, backend_for(device), geometry=geometry, settings=settings
    )
    mesh = Mesh(geometry, launch.rank, timeout=settings.timeout)
    return RankPublisher(launch, mesh, heartbeat)


def hold_for_peer(error: BaseException) -> BackendFailed | None:
    """A ``RuntimeError`` escaping a rank's run may be a backend failure
    raised outside this package's collectives — transformers' own
    ``dist.all_reduce`` inside its tensor-parallel styles is one (on Linux,
    gloo's read fails the instant a peer dies), NCCL's watchdog surfacing a
    timed-out kernel at the next sync another. Hold for the heartbeat's
    word as [`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective] does: a dead peer is
    refused by name here, within the bound, and the process ends. When
    nobody is lost and the error is the backend's ([`backend_error`][]:
    torch's ``DistError`` — an aborted communicator surfacing at a later
    call, a store gone — or gloo's bare ``RuntimeError``)
    it is re-rendered as [`BackendFailed`][] for the caller to raise —
    a refusal, never a bare backend traceback; any other error is the
    caller's to re-raise as it was (``None``). An out-of-memory error is
    the rank's own and holds nobody; so does a process with no heartbeat
    running."""
    from causalab.neural.shared.parallel import heartbeat

    if not isinstance(error, RuntimeError) or _is_out_of_memory(error):
        return None
    watch = heartbeat.running()
    if watch is None:
        return None
    watch.hold()
    if not backend_error(error):
        return None
    return BackendFailed(
        rank=watch.rank, world=watch.world, grace=watch.settings.grace, cause=error
    )


def _is_out_of_memory(error: BaseException) -> bool:
    """``torch.OutOfMemoryError`` without importing torch here."""
    return any(klass.__name__ == "OutOfMemoryError" for klass in type(error).__mro__)


def backend_error(error: BaseException) -> bool:
    """Whether ``error`` is the communication backend's, without importing
    torch: ``torch.distributed.DistError`` by name in its class hierarchy
    (``DistBackendError`` — an aborted communicator surfacing at a later
    call —, ``DistNetworkError``, ``DistStoreError``), or
    a bare ``RuntimeError`` whose words are **gloo's**: gloo raises plain
    ``RuntimeError``s stamped with its source path (``[…/gloo/transport/
    tcp/pair.cc:…] Read error … Connection reset by peer`` on Linux,
    ``[…/gloo/transport/uv/unbound_buffer.cc:…] Timed out waiting 15000ms
    for recv operation to complete`` on macOS)."""
    if any(klass.__name__ == "DistError" for klass in type(error).__mro__):
        return True
    return isinstance(error, RuntimeError) and "gloo" in str(error)


def leave(
    publisher: Publisher,
    status: int = 0,
    *,
    hard_exit: Callable[[int], None] = _exit_at_shutdown,
) -> None:
    """The matching exit: tell the peers this rank finished with ``status``
    (the heartbeat's ``done``), then leave the process group — waiting no
    longer than the grace for the teardown ([`leave_group`][]): one that
    has not returned by then is written up as the hang it is, and
    ``hard_exit(status)`` arranges the process's end past it
    (`_exit_at_shutdown`; the tests hand in a recorder). A teardown
    with every peer gone takes milliseconds. Before clean shutdown, rank 0
    keeps the store alive until all peers finish, bounded by the collective
    timeout. A nonzero status bypasses this coordination."""
    settings = Settings()
    if isinstance(publisher, RankPublisher) and publisher.heartbeat is not None:
        settings = publisher.heartbeat.settings
        if status == 0:
            try:
                publisher.heartbeat.complete()
            except watchdog.CompletionTimeout:
                publisher.heartbeat.finish(1)
                if not leave_group(bound=settings.grace):
                    hard_exit(1)
                raise
        publisher.heartbeat.finish(status)
    if not leave_group(bound=settings.grace):
        sys.stderr.write(
            "the process group's teardown did not return within "
            f"{watchdog.RANK_GRACE_VARIABLE}={settings.grace:g} s: a peer is alive "
            "but never left — a hang without a death, which no heartbeat names — so "
            f"this rank exits with status {status} without waiting for it "
            "(docs/model_parallelism.md §3)\n"
        )
        sys.stderr.flush()
        hard_exit(status)


def lockstep(publisher: Publisher) -> Lockstep:
    """The workflow runner's [`Lockstep`][]
    for this rank (§3, §11): over the mesh of a [`RankPublisher`][] — a
    [`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective] over the same groups the engine
    gathers over, so the joiner's decisions travel on the process's one mesh
    — and the world-1 identity for any other publisher."""
    if isinstance(publisher, RankPublisher):
        from causalab.neural.shared.parallel.collective import TorchCollective
        from causalab.neural.shared.parallel.lockstep import CollectiveLockstep

        return CollectiveLockstep(TorchCollective(publisher.mesh))
    return SOLO_LOCKSTEP
