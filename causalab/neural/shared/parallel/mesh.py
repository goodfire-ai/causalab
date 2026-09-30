"""The process groups of a geometry (``docs/model_parallelism.md`` §2–3).

A [`Mesh`][] is one rank's view of the ``torch.distributed`` groups a
[`ParallelGeometry`][] needs: one
``ProcessGroup`` per group of every axis whose size is above one, named by
[`Axis`][] and resolved through the integer
[`MeshLayout`][], so the mesh holds no rank
arithmetic of its own.

**Every rank creates every group.** ``torch.distributed.new_group`` is
collective over the whole world: each process must call it once for each
group, in the same order, whether or not it is a member. [`group_table`][]
is that order — a pure function of the layout, so it is identical on every
rank and testable without a process group — and [`Mesh`][] walks it. Two
axes whose partitions coincide (``tensor`` and ``model`` when ``tp`` is the
whole model group) share their groups rather than creating a second set.

**An axis of size one has no group.** ``group(axis)`` is ``None`` there and
the collective short-circuits; ``device_mesh`` refuses, since a parallel plan
is applied only on an axis above one.

**The device mesh** transformers' parallel styles take
(``style.shard_param(module, name, mesh)``) is a **1-D** mesh over this
rank's group, built with ``DeviceMesh.from_group`` on the group already
created. ``DeviceMesh(device_type, torch.tensor(ranks))`` would call
``new_group`` itself, with a *different* rank list on each group's members
and the same auto-generated group name on all of them — the store keys
collide and the ranks hang. ``init_device_mesh`` would build a fresh world
mesh. Neither is what the plan needs.

The device rule is one line: collectives run on the tensor's own device with
the backend the default process group was initialised with — ``gloo`` for
CPU worlds (the smoke tier), ``nccl`` for CUDA.

**Every group carries the collective timeout** (§3 "when a rank dies", §11):
``new_group`` with no timeout takes torch's *global* default — thirty
minutes under gloo, ten under NCCL — not the default group's, so a hang in
a mesh group would outlast the launcher's bound; ``Mesh(…, timeout=)`` hands
the launcher's seconds to every ``new_group``, and ``from_environment``
reads the same setting the launcher does (``CAUSALAB_COLLECTIVE_TIMEOUT``).
"""

from __future__ import annotations

import datetime
import functools
import os
from typing import Callable, Sequence

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from causalab.protocol.rules.errors import ParseError, ProtocolError
from causalab.protocol.parallel import (
    AXES,
    Axis,
    MeshLayout,
    ParallelGeometry,
    format_geometry,
)

__all__ = [
    "DeviceMeshFactory",
    "GroupFactory",
    "Mesh",
    "MeshError",
    "device_type_of",
    "group_table",
    "torch_device_mesh",
]

#: Builds one process group over the given global ranks, or ``None`` when
#: this rank is not a member (``new_group`` hands non-members a placeholder).
#: The production factory is [`torch_new_group`][]; a test hands in a
#: recorder (``docs/model_parallelism.md`` §10.1, the seam precedent of
#: ``Meter``).
GroupFactory = Callable[[Sequence[int]], "ProcessGroup | None"]

#: Builds the 1-D device mesh over one process group on a device type — what
#: transformers' parallel styles take. The production factory is
#: [`torch_device_mesh`][]; a test hands in the fake of
#: ``tests/_helpers/device_mesh.py``, so a sharding can be built with no
#: process group (``docs/model_parallelism.md`` §10.1).
DeviceMeshFactory = Callable[[ProcessGroup, str], DeviceMesh]

#: The device a backend's collectives run on.
_DEVICE_OF_BACKEND: dict[str, str] = {"gloo": "cpu", "nccl": "cuda"}


class MeshError(ValueError):
    """The mesh was asked for something its geometry does not have: the group
    or device mesh of an axis of size one, a group-local index outside the
    group, a backend with no device rule. Not a
    [`ProtocolError`][]: a mesh is built by the
    launcher from a checked geometry, so this is an internal invariant broken,
    never a rule a document violated."""


def device_type_of(backend: str) -> str:
    """``cpu`` for ``gloo``, ``cuda`` for ``nccl``; any other backend is refused
    by name — the device rule is deliberately this short."""
    device_type = _DEVICE_OF_BACKEND.get(backend.lower())
    if device_type is None:
        raise MeshError(
            f"backend {backend!r} has no device rule; the mesh runs on "
            f"{', '.join(f'{b} ({d})' for b, d in _DEVICE_OF_BACKEND.items())}"
        )
    return device_type


def torch_new_group(
    ranks: Sequence[int], timeout: float | None = None
) -> ProcessGroup | None:
    """``torch.distributed.new_group`` over ``ranks`` with ``timeout``
    seconds (torch's global default when ``None``): the group when this
    rank is a member, ``None`` otherwise (torch hands non-members the
    ``NON_GROUP_MEMBER`` placeholder, which no collective may take)."""
    group = dist.new_group(
        ranks=list(ranks),
        timeout=None if timeout is None else datetime.timedelta(seconds=timeout),
    )
    return group if isinstance(group, ProcessGroup) else None


def torch_device_mesh(group: ProcessGroup, device_type: str) -> DeviceMesh:
    """``DeviceMesh.from_group`` over a group already created (module
    docstring: never ``DeviceMesh(…)`` or ``init_device_mesh``, which would
    create groups of their own)."""
    return DeviceMesh.from_group(group, device_type)


def group_table(layout: MeshLayout) -> tuple[tuple[Axis, tuple[int, ...]], ...]:
    """Every ``(axis, members)`` a rank creates a process group for, in the
    order every rank creates them: axes in [`AXES`][] order, each axis's
    groups by first member (``MeshLayout.groups``). An axis of size one is
    absent; an axis whose partition repeats an earlier axis's is absent too,
    since it shares those groups."""
    table: list[tuple[Axis, tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for axis in AXES:
        groups = layout.groups(axis)
        if len(groups[0]) == 1 or groups in seen:
            continue
        seen.add(groups)
        table.extend((axis, members) for members in groups)
    return tuple(table)


def _environment_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ParseError(
            "P4",
            f"{name} is not set: a world above one is launched with RANK and "
            "WORLD_SIZE in the environment (torchrun, or the engine's own spawn)",
            path="--parallel",
        )
    if not raw.isdigit():
        raise ParseError(
            "P4",
            f"{name} must be a non-negative integer, got {raw!r}",
            path="--parallel",
        )
    return int(raw)


class Mesh:
    """This rank's process groups over a geometry.

    ``Mesh(geometry, rank)`` after ``torch.distributed.init_process_group``
    creates the groups of [`group_table`][] through
    ``torch.distributed.new_group``, each with ``timeout`` seconds (module
    docstring; torch's default when ``None``); ``groups`` names another
    factory (a test's recorder), in which case no process group is
    consulted and ``device_type`` defaults to ``cpu``; ``device_meshes``
    names the [`DeviceMeshFactory`][] ``device_mesh`` builds through
    ([`torch_device_mesh`][] when ``None``).

    Raises:
        ProtocolError: ``P4`` — no default process group is initialised, or
            its world or this process's rank disagrees with the arguments.
        ParseError: ``P4`` — ``tensor`` or ``expert`` does not divide the
            model group (``MeshLayout``'s refusal).
    """

    def __init__(
        self,
        geometry: ParallelGeometry,
        rank: int,
        *,
        groups: GroupFactory | None = None,
        device_type: str | None = None,
        timeout: float | None = None,
        device_meshes: DeviceMeshFactory | None = None,
    ) -> None:
        self.geometry = geometry
        self.layout = MeshLayout(geometry)
        if isinstance(rank, bool) or not 0 <= rank < self.layout.world:
            raise MeshError(
                f"rank {rank!r} is outside range({self.layout.world}) of geometry "
                f"{format_geometry(geometry)}"
            )
        self.rank = rank
        if groups is None:
            self._check_default_group()
            factory: GroupFactory = functools.partial(torch_new_group, timeout=timeout)
            device_type = device_type or device_type_of(dist.get_backend())
        else:
            factory = groups
        self.device_type = device_type or "cpu"
        self._device_meshes: DeviceMeshFactory = device_meshes or torch_device_mesh
        self._groups: dict[Axis, ProcessGroup | None] = {axis: None for axis in AXES}
        self._meshes: dict[Axis, DeviceMesh] = {}
        self._create_groups(factory)

    def _check_default_group(self) -> None:
        if not dist.is_initialized():
            raise ProtocolError(
                "P4",
                "no default process group is initialised: call "
                "torch.distributed.init_process_group before building the mesh, "
                f"or run at the default geometry (asked for "
                f"{format_geometry(self.geometry)})",
                path="--parallel",
            )
        world = dist.get_world_size()
        if world != self.geometry.world:
            raise ProtocolError(
                "P4",
                f"the process group has {world} ranks but --parallel "
                f"{format_geometry(self.geometry)} spans {self.geometry.world}",
                path="--parallel",
            )
        if dist.get_rank() != self.rank:
            raise ProtocolError(
                "P4",
                f"this process is rank {dist.get_rank()} of the process group, "
                f"not {self.rank}",
                path="--parallel",
            )

    def _create_groups(self, factory: GroupFactory) -> None:
        # Every rank walks the whole table (the ``new_group`` contract) and
        # keeps the one group per partition it is a member of, so axes with
        # identical partitions share their group.
        mine: dict[tuple[tuple[int, ...], ...], ProcessGroup] = {}
        for axis, members in group_table(self.layout):
            group = factory(members)
            if self.rank not in members:
                continue
            if group is None:
                raise MeshError(
                    f"no process group came back for {axis} group {members}, of "
                    f"which rank {self.rank} is a member"
                )
            mine[self.layout.groups(axis)] = group
        for axis in AXES:
            self._groups[axis] = mine.get(self.layout.groups(axis))

    @classmethod
    def from_environment(cls, geometry: ParallelGeometry) -> Mesh:
        """The mesh of the process ``RANK`` / ``WORLD_SIZE`` describe (§3),
        every group with the collective timeout the environment names
        (``CAUSALAB_COLLECTIVE_TIMEOUT``, the launcher's setting).

        Raises:
            ProtocolError: ``P4`` — a variable missing or malformed,
                ``WORLD_SIZE`` disagreeing with ``geometry.world``, a
                malformed timeout, or no default process group initialised.
        """
        from causalab.neural.shared.parallel.watchdog import Settings

        rank = _environment_int("RANK")
        world = _environment_int("WORLD_SIZE")
        if world != geometry.world:
            raise ParseError(
                "P4",
                f"WORLD_SIZE={world} disagrees with --parallel "
                f"{format_geometry(geometry)}, which spans {geometry.world} ranks",
                path="--parallel",
            )
        return cls(geometry, rank, timeout=Settings.from_environment().timeout)

    # -- position -------------------------------------------------------------

    def size(self, axis: Axis) -> int:
        """The number of ranks in this rank's group on ``axis``."""
        return len(self.ranks(axis))

    def ranks(self, axis: Axis) -> tuple[int, ...]:
        """The global ranks of this rank's group on ``axis``, in rank order —
        group-local index ``i`` is ``ranks(axis)[i]``."""
        return self.layout.group_of(self.rank, axis)

    def local_rank(self, axis: Axis) -> int:
        """This rank's group-local index on ``axis`` — what ``Collective.rank``
        returns and what a ``StageLocal.stage`` names."""
        return self.layout.rank_in(self.rank, axis)

    def global_rank(self, axis: Axis, local: int) -> int:
        """The global rank at group-local index ``local`` of this rank's group
        on ``axis``.

        Raises:
            MeshError: ``local`` is outside the group.
        """
        ranks = self.ranks(axis)
        if isinstance(local, bool) or not 0 <= local < len(ranks):
            raise MeshError(
                f"index {local!r} is outside the {axis} group of {len(ranks)} ranks "
                f"(geometry {format_geometry(self.geometry)})"
            )
        return ranks[local]

    # -- groups ---------------------------------------------------------------

    def group(self, axis: Axis) -> ProcessGroup | None:
        """This rank's process group on ``axis``; ``None`` when the axis has
        size one, where every collective is the identity."""
        return self._groups[axis]

    def device_mesh(self, axis: Axis) -> DeviceMesh:
        """A 1-D ``DeviceMesh`` over this rank's group on ``axis`` — what
        transformers' parallel styles take. Built once per axis, through the
        factory the mesh was given.

        Raises:
            MeshError: the axis has size one; a plan is applied only on an
                axis above one, so no mesh is ever asked for there.
        """
        mesh = self._meshes.get(axis)
        if mesh is None:
            group = self.group(axis)
            if group is None:
                raise MeshError(
                    f"axis {axis!r} has size 1 in geometry "
                    f"{format_geometry(self.geometry)}: no process group and no "
                    "device mesh; a parallel plan is applied only on an axis above one"
                )
            mesh = self._device_meshes(group, self.device_type)
            self._meshes[axis] = mesh
        return mesh
