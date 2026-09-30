"""Applying the registry's parallel plan over sub-meshes
(``docs/model_parallelism.md`` §5.2, §6.5, §10.8).

[`Sharding`][] is what a rank knows about its place: the geometry, its
global rank, the collective it talks to its groups through and, per plan
axis whose size is above one, the 1-D ``DeviceMesh`` over its group on that
axis — built by the mesh module (``neural/shared/parallel/mesh.py``,
``Mesh.device_mesh(axis)``) and handed in, never read from process-global
state. [`Sharding.from_mesh`][] is the one constructor the engine uses:
the process's one [`Mesh`][]
yields the collective the hooks gather through and the meshes the plan is
applied over, so the two are the same process groups by construction (§3).

[`apply_plan`][] is the ~40-line loop of transformers'
``apply_tensor_parallelism``, over the registry's table **as rewritten for
the geometry** (``ParallelPlan.for_geometry``: the K/V projections
``kv_replicated`` above the KV heads, §6.6) instead of the model's own
``tp_plan``, and over a [`Styles`][] instead of transformers'
registry: for every module, each of its parameters is validated and sharded
by the [`Style`][causalab.neural.engines.pytorch_hooks.styles.Style] its row names over this rank's
[`Group`][] on the row's axis, then the style's forward wrap is
installed on the module. The production styles are
[`TransformersStyles`][causalab.neural.engines.pytorch_hooks.styles.dtensor.TransformersStyles] — transformers' own objects
over the sharding's meshes, DTensor placeholders on the parameters — and
the tests' simulated tiers hand in
[`FragmentStyles`][causalab.neural.engines.pytorch_hooks.styles.fragment.FragmentStyles], the same partition on plain
tensors over any collective. A row on an axis of size one is not applied —
its group is this rank alone, and the tensor is whole. The repository's own
``kv_replicated`` leaves the weight a plain parameter, narrows the output
to this rank's KV head and then divides the mixer's ``num_key_value_groups``
by its repeat, once per mixer. Under ``pipeline > 1`` the layers outside
this stage's range ([`stage_layers`][]) become [`StageStandIn`][] —
transformers' ``PipelineIdentityLayer`` carrying the replaced meta module
as its shadow, so the site resolver and the stream table still see the tree
of a block another stage owns (``shared/parallel/standin.py``) — the
embedding lives on the first stage, the final norm and the head on the
last; transformers' naive stage forward is **not** installed — the executor
owns the stage forward (``stages.py``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers.distributed.pipeline_parallel import PipelineIdentityLayer

from causalab.neural.engines.pytorch_hooks.kv_replication import KvReplicated
from causalab.neural.engines.pytorch_hooks.styles import Group, Styles
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.standin import Shadowed
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    Axis,
    MeshLayout,
    ParallelGeometry,
    format_geometry,
    stage_layers,
)
from causalab.protocol.registry import (
    PLAN_AXES,
    STYLES,
    ParallelPlan,
    PlanAxis,
    PlanRow,
    family_for,
    walk,
)

if TYPE_CHECKING:
    from causalab.neural.shared.parallel.mesh import Mesh

__all__ = [
    "Sharding",
    "StageStandIn",
    "apply_plan",
    "place_stage",
    "stage_layers",
]


@dataclasses.dataclass(frozen=True)
class Sharding:
    """This rank's place in a geometry (module docstring). A value by
    ``(geometry, rank)`` — the loader's cache key — with the meshes and the
    collective outside equality: one process is one rank, and its groups
    for a geometry are fixed for the life of the process group. The
    collective's group sizes on the active plan axes must be the
    geometry's; a mesh, where given, must be 1-D over the axis's group.
    A missing mesh is refused when a style needs one (the transformers
    tier), not here: the fragment tier needs none."""

    geometry: ParallelGeometry
    rank: int
    meshes: Mapping[Axis, DeviceMesh] = dataclasses.field(
        default_factory=dict, compare=False, hash=False
    )
    collective: Collective = dataclasses.field(default=SOLO, compare=False, hash=False)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not 0 <= self.rank < self.geometry.world:
            raise ProtocolError(
                "P4",
                f"rank {self.rank!r} is outside range({self.geometry.world}) of "
                f"geometry {format_geometry(self.geometry)}",
            )
        for axis in PLAN_AXES:
            size = getattr(self.geometry, axis)
            if size == 1:
                continue
            held = self.collective.size(axis)
            if held != size:
                raise ProtocolError(
                    "P4",
                    f"geometry {format_geometry(self.geometry)} shards over a "
                    f"{axis} group of {size}, but the collective's {axis} group "
                    f"has {held} rank(s)",
                )
            mesh = self.meshes.get(axis)
            if mesh is not None and (mesh.ndim != 1 or mesh.size() != size):
                raise ProtocolError(
                    "P4",
                    f"the {axis!r} mesh is {mesh.ndim}-D over {mesh.size()} rank(s); "
                    f"geometry {format_geometry(self.geometry)} wants a 1-D mesh "
                    f"over this rank's group of {size}",
                )

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> Sharding:
        """This rank's sharding over the process's one mesh: the
        ``TorchCollective`` over it, and a 1-D device mesh per plan axis
        whose size is above one (``Mesh.device_mesh``), none for an axis of
        size one — the mesh has no group there and no row is applied."""
        from causalab.neural.shared.parallel.collective import TorchCollective

        geometry = mesh.geometry
        return cls(
            geometry,
            mesh.rank,
            meshes={
                axis: mesh.device_mesh(axis)
                for axis in PLAN_AXES
                if getattr(geometry, axis) > 1
            },
            collective=TorchCollective(mesh),
        )

    @property
    def layout(self) -> MeshLayout:
        return MeshLayout(self.geometry)

    @property
    def stage(self) -> int:
        """This rank's pipeline stage."""
        return self.layout.rank_in(self.rank, "pipeline")

    @property
    def active_axes(self) -> tuple[PlanAxis, ...]:
        """The plan axes above one — the rows applied."""
        return tuple(axis for axis in PLAN_AXES if getattr(self.geometry, axis) > 1)

    def group(self, axis: PlanAxis) -> Group:
        """This rank's group on ``axis``."""
        return Group.of(self, axis)


def _check_rows(plan: ParallelPlan, sharding: Sharding) -> None:
    """The plan rules ``check`` states (``protocol/parallel.py``), restated
    at the seam that would otherwise compute a wrong number: every active
    axis has rows, and every row on it is a served style."""
    for axis in sharding.active_axes:
        rows = plan.rows_on(axis)
        if not rows:
            raise ProtocolError(
                "P4",
                f"--parallel.{axis}: the parallel plan has no {axis}-axis row, so "
                f"nothing would be sharded over the {axis} group of "
                f"{getattr(sharding.geometry, axis)}",
            )
        unserved = sorted(
            f"{pattern} → {row.style}"
            for pattern, row in rows.items()
            if not row.served
        )
        if unserved:
            raise ProtocolError(
                "P4",
                f"--parallel.{axis}: the parallel plan names styles this engine "
                f"does not serve ({', '.join(unserved)}); the served styles are "
                f"{', '.join(sorted(STYLES))}",
            )


def apply_plan(
    model: torch.nn.Module,
    plan: ParallelPlan,
    sharding: Sharding,
    styles: Styles | None = None,
) -> None:
    """Shard ``model`` by ``plan`` for this rank (module docstring): every
    parameter a row names validated and sharded by the row's style, the
    row's style installed on every module a row names, over this rank's
    group on the row's axis; then the pipeline placement. Rows on an axis
    of size one are left alone. ``styles`` is the tier the styles come from
    — transformers' DTensor styles over the sharding's meshes when
    ``None``, the production default.

    Raises:
        ProtocolError: ``P4`` — an active axis with no row or an unserved
            style (the geometry check's rules, restated), or a style's own
            refusal of a parameter (a gathered output the tensor group does
            not divide), naming the module.
    """
    if styles is None:
        from causalab.neural.engines.pytorch_hooks.styles.dtensor import (
            TransformersStyles,
        )

        styles = TransformersStyles(sharding)
    _check_rows(plan, sharding)
    active = sharding.active_axes
    prefix = getattr(model, "base_model_prefix", None)

    def row_for(path: str) -> PlanRow | None:
        row = plan.style_for(path, prefix=prefix)
        return row if row is not None and row.axis in active else None

    mixers: dict[str, KvReplicated] = {}
    for name, module in model.named_modules():
        for parameter_name, _ in list(module.named_parameters(recurse=False)):
            full = f"{name}.{parameter_name}" if name else parameter_name
            row = row_for(full) or (row_for(name) if name else None)
            if row is None:
                continue
            style = styles.style(row)
            group = sharding.group(row.axis)
            try:
                style.validate(module, parameter_name, group, path=full)
            except ValueError as err:
                raise ProtocolError(
                    "P4",
                    f"--parallel.{row.axis}: style {row.style!r} refuses parameter "
                    f"{full!r} over a group of {group.size}: {err}",
                ) from err
            style.shard(module, parameter_name, group)
        if not name:
            continue
        row = row_for(name)
        if row is None:
            continue
        style = styles.style(row)
        style.install(
            module, sharding.group(row.axis), expert_parallel=row.axis == "expert"
        )
        if isinstance(style, KvReplicated):
            # the mixer above a replicated K/V projection repeats the one
            # held KV head over its local query heads (§6.6): once per mixer
            mixers[name.rpartition(".")[0]] = style
    for mixer_path, style in mixers.items():
        style.repeat_locally(walk(model, mixer_path), mixer_path)
    if sharding.geometry.pipeline > 1:
        place_stage(model, sharding)


class StageStandIn(Shadowed, PipelineIdentityLayer):
    """The identity standing in for a module another pipeline stage holds:
    transformers' ``PipelineIdentityLayer`` — so a census of the placed tree
    reads it as one — whose attributes fall through to the replaced module
    ([`Shadowed`][]). The shadow
    is the meta instance the plan was applied to: no tensor of it is ever
    materialised, and nothing registers it as a child."""


def _replace(model: torch.nn.Module, path: str) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = walk(model, parent_path) if parent_path else model
    if parent is None or not hasattr(parent, leaf):
        raise ProtocolError(
            "P4",
            f"the family's tree addresses {path!r}, which this model "
            f"({type(model).__name__}) does not have",
        )
    setattr(parent, leaf, StageStandIn(getattr(parent, leaf)))


def place_stage(model: torch.nn.Module, sharding: Sharding) -> None:
    """Keep this stage's layers and replace the rest by a [`StageStandIn`][]
    (module docstring): the layers outside [`stage_layers`][], the
    embedding off the first stage, the final norm and head off the last —
    through the family's tree addresses, so any registered tree is placed
    the same way.

    Raises:
        ProtocolError: ``P4`` — a model whose head is tied to its embedding
            (one tensor the first stage and the last would both hold).
    """
    geometry = sharding.geometry
    if getattr(model.config, "tie_word_embeddings", False):
        raise ProtocolError(
            "P4",
            "--parallel.pipeline: the model ties its head to its embedding "
            "(tie_word_embeddings), one tensor the first stage and the last "
            f"would both hold; pp={geometry.pipeline} places them apart",
        )
    adapter = family_for(model)
    blocks: Any = adapter.blocks_of(model)
    keep = stage_layers(geometry, sharding.rank, len(blocks))
    for index in range(len(blocks)):
        if index not in keep:
            blocks[index] = StageStandIn(blocks[index])
    stage = sharding.stage
    if stage != 0:
        _replace(model, adapter.tree.embedding)
    if stage != geometry.pipeline - 1:
        _replace(model, adapter.tree.final_norm)
        _replace(model, adapter.tree.lm_head)
