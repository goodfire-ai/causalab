"""``Sharding.from_mesh`` — one mesh per process (``docs/model_parallelism.md``
§3, §5.2).

The engine builds one [`Mesh`][causalab.neural.shared.parallel.mesh.Mesh]
per rank and derives both its collective and its [`Sharding`][causalab.neural.engines.pytorch_hooks.sharding.Sharding] from it,
so the loader's plan meshes and the hooks' collective groups are the same
process groups by construction. ``property``: over every mesh geometry and
every rank, the sharding carries exactly one 1-D device mesh per plan axis
whose size is above one, none for an axis of size one, and never asks the
mesh for a size-one axis. ``unit``: the two ends spelled by hand. The mesh is
built through its ``groups=`` seam and ``device_mesh`` stubbed, so no
process group exists in the test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.sharding import Sharding
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.registry import PLAN_AXES

from tests._helpers.geometries import mesh_geometries
from tests.neural.shared.parallel.test_mesh import Recorder

_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@dataclasses.dataclass(frozen=True)
class _DeviceMesh:
    """What ``Sharding`` reads off a device mesh: its rank and its size."""

    axis: str
    ranks: int
    ndim: int = 1

    def size(self) -> int:
        return self.ranks


@pytest.fixture
def stubbed_device_mesh(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``Mesh.device_mesh`` answering a 1-D stand-in over the axis's group,
    recording every axis it was asked for."""
    asked: list[str] = []

    def device_mesh(self: Mesh, axis: Any) -> _DeviceMesh:
        asked.append(axis)
        return _DeviceMesh(axis, self.size(axis))

    monkeypatch.setattr(Mesh, "device_mesh", device_mesh)
    return asked


class TestFromMeshProperties:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(geometry=mesh_geometries(), data=st.data())
    def test_one_mesh_per_plan_axis_above_one_and_none_for_size_one(
        self, geometry: ParallelGeometry, data: st.DataObject, stubbed_device_mesh
    ) -> None:
        rank = data.draw(st.integers(0, geometry.world - 1))
        stubbed_device_mesh.clear()
        mesh = Mesh(geometry, rank, groups=Recorder())
        sharding = Sharding.from_mesh(mesh)
        active = {axis for axis in PLAN_AXES if getattr(geometry, axis) > 1}
        assert set(sharding.meshes) == active
        for axis, device_mesh in sharding.meshes.items():
            assert device_mesh.size() == getattr(geometry, axis)
            assert device_mesh.ndim == 1
        assert sharding.geometry == geometry and sharding.rank == rank
        assert set(sharding.active_axes) == active
        # a size-one axis has no group and no device mesh, and none was asked for
        assert set(stubbed_device_mesh) == active


class TestFromMesh:
    pytestmark = pytest.mark.unit

    def test_world_one_carries_no_mesh(self, stubbed_device_mesh) -> None:
        sharding = Sharding.from_mesh(Mesh(ONE, 0, groups=Recorder()))
        assert sharding == Sharding(ONE, 0, meshes={})
        assert dict(sharding.meshes) == {} and stubbed_device_mesh == []

    def test_tp2_ep4_carries_a_tensor_and_an_expert_mesh(
        self, stubbed_device_mesh
    ) -> None:
        geometry = ParallelGeometry(tensor=2, expert=4)
        sharding = Sharding.from_mesh(Mesh(geometry, 3, groups=Recorder()))
        assert set(sharding.meshes) == {"tensor", "expert"}
        assert sharding.meshes["tensor"].size() == 2
        assert sharding.meshes["expert"].size() == 4
        assert sharding.rank == 3 and sharding.stage == 0

    def test_a_pipeline_only_geometry_carries_no_plan_mesh(
        self, stubbed_device_mesh
    ) -> None:
        sharding = Sharding.from_mesh(
            Mesh(ParallelGeometry(pipeline=2), 1, groups=Recorder())
        )
        assert dict(sharding.meshes) == {} and sharding.stage == 1
