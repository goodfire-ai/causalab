"""The styles as transformers applies them (the package docstring's first
tier, production).

[`TransformersStyles`][] names transformers' ``TensorParallelLayer``
objects (``ALL_PARALLEL_STYLES``) through the [`Styles`][] protocol,
each over the 1-D ``DeviceMesh`` of its row's axis: ``validate_param`` /
``shard_param`` wrap a parameter as a DTensor placeholder, ``install_forward``
installs the style's DTensor redistributes. Nothing is changed about what
they do — the GPU goldens hold this tier byte for byte — only two facts the
repository adds are here beside them: ``MoeExpertsParallel`` is told whether
the experts are expert-parallel (and the module is marked for the lean
experts path), and the ``ep_router`` row on the expert axis gets
[`sum_router_gradient`][]. The repository's own ``kv_replicated`` is the
same `KvReplicated` the fragment tier uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    MoeExpertsParallel,
)

from causalab.neural.engines.pytorch_hooks.experts_path import EXPERT_PARALLEL_MARK
from causalab.neural.engines.pytorch_hooks.kv_replication import KvReplicated
from causalab.neural.engines.pytorch_hooks.partial_gradient import summed_over
from causalab.neural.engines.pytorch_hooks.styles import Group, Style, StyleError
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.placement import Axis
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import format_geometry
from causalab.protocol.registry import KV_REPLICATED, PlanRow

if TYPE_CHECKING:
    from causalab.neural.engines.pytorch_hooks.sharding import Sharding

__all__ = ["TransformersStyles", "sum_router_gradient"]


def sum_router_gradient(
    module: torch.nn.Module, axis: Axis, collective: Collective
) -> None:
    """Make the router's input gradient whole under expert parallelism
    (``docs/model_parallelism.md`` §6.3, §7).

    transformers' ``ep_router`` style is forward-only slicing: the gate runs
    replicated, and each rank keeps the scores of its own experts and zeroes
    the rest. In backward, then, the gradient that flows from this rank's
    experts through the router into the residual stream is this rank's
    experts' share alone — a **partial** sum over the expert group that the
    experts' style never sums (it all-reduces the gradient of its *own*
    input, and under ``is_expert_parallel`` deliberately not the routing
    weights'). At inference nothing reads it; a fit of a featurizer below
    the router does. Summing the router's input gradient over the group
    gives every rank the full gradient without changing the forward."""
    forward = module.forward

    def routed_forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return forward(summed_over(hidden_states, axis, collective), *args, **kwargs)

    module.forward = routed_forward


class _Library:
    """One of transformers' styles behind the protocol, over the mesh of
    its row's axis."""

    def __init__(
        self,
        name: str,
        style: Any,
        meshes: Mapping[Axis, DeviceMesh],
        geometry_text: str,
        collective: Collective,
    ) -> None:
        self.name = name
        self.style = style
        self._meshes = meshes
        self._geometry = geometry_text
        self.collective = collective

    def _mesh(self, group: Group) -> DeviceMesh:
        mesh = self._meshes.get(group.axis)
        if mesh is None:
            raise ProtocolError(
                "P4",
                f"geometry {self._geometry} shards over the {group.axis} axis but no "
                f"{group.axis!r} mesh was given",
            )
        if mesh.ndim != 1 or mesh.size() != group.size:
            raise ProtocolError(
                "P4",
                f"the {group.axis!r} mesh is {mesh.ndim}-D over {mesh.size()} rank(s); "
                f"geometry {self._geometry} wants a 1-D mesh over this rank's group "
                f"of {group.size}",
            )
        return mesh

    def validate(
        self, module: torch.nn.Module, parameter: str, group: Group, *, path: str
    ) -> None:
        self.style.validate_param(
            module, parameter, self._mesh(group), parameter_name=path
        )

    def shard(self, module: torch.nn.Module, parameter: str, group: Group) -> None:
        self.style.shard_param(module, parameter, self._mesh(group))

    def install(
        self, module: torch.nn.Module, group: Group, *, expert_parallel: bool
    ) -> None:
        mesh = self._mesh(group)
        if isinstance(self.style, MoeExpertsParallel):
            self.style.install_forward(module, mesh, is_expert_parallel=expert_parallel)
            if expert_parallel:
                # the router now writes sentinel ids for other ranks' experts,
                # and this loader left the weight local: say so, since the
                # lean experts path cannot read it off the shapes
                setattr(module, EXPERT_PARALLEL_MARK, True)
        else:
            self.style.install_forward(module, mesh)
        if self.name == "ep_router" and expert_parallel:
            sum_router_gradient(module, group.axis, self.collective)


class TransformersStyles:
    """The [`Styles`][] of a rank's `Sharding`: its
    meshes for the library's styles, its collective for the repository's
    two sums (module docstring).

    Raises:
        StyleError: a row naming a style transformers' registry lacks.
    """

    def __init__(self, sharding: Sharding) -> None:
        self._meshes = sharding.meshes
        self._geometry = format_geometry(sharding.geometry)
        self.collective = sharding.collective

    def style(self, row: PlanRow) -> Style:
        if row.style == KV_REPLICATED:
            return KvReplicated(row.repeat, self.collective)
        if row.style not in ALL_PARALLEL_STYLES:
            raise StyleError(
                f"parallel style {row.style!r} is not in transformers' registry; "
                f"the styles are {sorted(ALL_PARALLEL_STYLES)}"
            )
        return _Library(
            row.style,
            ALL_PARALLEL_STYLES[row.style],
            self._meshes,
            self._geometry,
            self.collective,
        )
