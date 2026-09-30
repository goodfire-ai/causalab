"""The one fake for the ``DeviceMesh`` seam (``docs/model_parallelism.md``
§10.1): what [`Sharding`][causalab.neural.engines.pytorch_hooks.sharding.Sharding]
reads off a plan mesh — one dimension, the group's size — and what
[`Mesh`][causalab.neural.shared.parallel.mesh.Mesh] hands its factory.

``Mesh(…, device_meshes=fake_device_meshes)`` builds a `FakeDeviceMesh`
per group in place of ``DeviceMesh.from_group``, so a mesh over the
recording group factory (``tests/neural/shared/parallel/test_mesh.py:Recorder``)
yields shardings with no process group in the test. The transformers styles
never run over a fake: they need DTensor's real mesh, the ``gloo`` tier's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

__all__ = ["FakeDeviceMesh", "fake_device_meshes"]


@dataclass(frozen=True)
class FakeDeviceMesh:
    """A 1-D stand-in over ``ranks``."""

    ranks: tuple[int, ...]
    device_type: str = "cpu"
    ndim: int = 1

    def size(self, mesh_dim: int | None = None) -> int:
        return len(self.ranks)

    @property
    def mesh(self) -> torch.Tensor:
        return torch.tensor(self.ranks, dtype=torch.int64)


def fake_device_meshes(group: Any, device_type: str) -> FakeDeviceMesh:
    """The factory: the group's ``ranks`` (a recorder's ``FakeGroup``) or, for
    a real group, ``torch.distributed``'s member list."""
    ranks: Sequence[int]
    if hasattr(group, "ranks"):
        ranks = group.ranks
    else:
        import torch.distributed as dist

        ranks = dist.get_process_group_ranks(group)
    return FakeDeviceMesh(tuple(int(r) for r in ranks), device_type)
