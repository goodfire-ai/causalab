"""What the reference engine serves at ``world > 1``, and where a rank's one
mesh comes from (``docs/model_parallelism.md`` §3, §8.2).

Two questions the engine asks before any weights load, each a ``P4``
naming ``--parallel`` or ``--parallel.<axis>``. Every axis is served:
``pipeline`` stages (§6.5, ``pytorch_hooks/stages.py``), ``tensor`` and
``expert`` (the plan is applied over the rank's meshes at load and every
module-boundary tap is made whole through the collective), ``data`` in both
modes (§8.3) and ``context`` (§8.4, ``parallel/context.py``) — so no axis is
refused here; what an axis cannot serve (a decode under ``cp > 1``, under
``pp > 1``) is refused by name where the document or the frame is known.

- [`check_collective`][] — the collective handed in has the geometry's
  group sizes on every axis, so a placement's group and the mesh cannot
  disagree;
- [`process_mesh`][] — **one mesh per process** (§3): the
  [`Mesh`][] the launcher built for this rank's publisher, or,
  for a process launched some other way into an initialised
  ``torch.distributed`` group, ``Mesh.from_environment``; refused by name
  where no group is initialised. The engine derives both its
  [`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective] and the loader's ``Sharding`` from
  it, so the hooks gather over exactly the groups the plan was applied over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.launcher import RankPublisher
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.protocol.rules.errors import ParseError
from causalab.protocol.parallel import AXES, ParallelGeometry, format_geometry
from causalab.protocol.publish import Publisher

if TYPE_CHECKING:
    from causalab.neural.shared.parallel.placement import Axis

__all__ = ["check_collective", "process_mesh"]

_FLAG = "--parallel"


def _refuse(axis: str | None, message: str) -> ParseError:
    path = _FLAG if axis is None else f"{_FLAG}.{axis}"
    return ParseError("P4", message, path=path)


def check_collective(geometry: ParallelGeometry, collective: Collective) -> None:
    """The collective's group sizes are the geometry's on every axis.

    Raises:
        ParseError: ``P4`` naming the first axis that disagrees.
    """
    expected: dict[Axis, int] = {
        "data": geometry.data,
        "pipeline": geometry.pipeline,
        "context": geometry.context,
        "model": geometry.model,
        "tensor": geometry.tensor,
        "expert": geometry.expert,
    }
    for axis in AXES:
        size = collective.size(axis)
        if size != expected[axis]:
            raise _refuse(
                axis if axis != "model" else None,
                f"the collective has {size} ranks on the {axis} axis, but "
                f"{format_geometry(geometry)} asks for {expected[axis]}",
            )


def process_mesh(geometry: ParallelGeometry, publisher: Publisher) -> Mesh:
    """This process's one mesh over ``geometry`` (module docstring): the
    launcher's, riding on its [`RankPublisher`][], else one
    built from the environment over the initialised process group.

    Raises:
        ParseError: ``P4`` — the publisher's mesh was built for another
            geometry, or no ``torch.distributed`` group is initialised.
    """
    if isinstance(publisher, RankPublisher):
        mesh = publisher.mesh
        if mesh.geometry != geometry:
            raise _refuse(
                None,
                f"the launcher built this rank's mesh for "
                f"{format_geometry(mesh.geometry)}, but the engine runs under "
                f"{format_geometry(geometry)}",
            )
        return mesh
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise _refuse(
            None,
            f"{format_geometry(geometry)} asks for a world of {geometry.world} "
            "ranks, and no torch.distributed process group is initialised in this "
            "process: launch through the SPMD launcher (docs/model_parallelism.md "
            "§3) or hand the engine a collective",
        )
    return Mesh.from_environment(geometry)
