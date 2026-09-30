"""``Mesh``: the process groups of a geometry (``docs/model_parallelism.md`` §2–3).

The unit tier builds meshes through a recording group factory — no process
group exists in the test process — and holds them to ``MeshLayout``: every
rank creates the same groups in the same order (``torch.distributed``'s
contract for ``new_group``), this rank's group on every axis is its layout
group, ``global_rank`` / ``local_rank`` round-trip, a size-one axis has no
group, and ``from_environment`` refuses a world that disagrees with the
geometry or has no process group. The production factories run in the test
process too, over torch's ``fake`` backend — a default group of ``world``
ranks in one process, no sockets, collectives that move nothing — so the
default-group checks (world, rank), ``torch_new_group`` with the mesh's
timeout on every group and ``DeviceMesh.from_group`` per axis are judged
without a spawn (§10.7's ``Mesh.__init__`` over real groups). The smoke tier
spawns world 4 under ``gloo`` at ``tp=2,ep=4`` and checks the real groups,
the 1-D device mesh per axis and an all-reduce over each.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence, cast

import pytest
import torch
import torch.distributed as dist
from hypothesis import HealthCheck, given, settings
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

import causalab.neural.shared.parallel.mesh as mesh_module
from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.mesh import (
    Mesh,
    MeshError,
    device_type_of,
    group_table,
)
from causalab.neural.shared.parallel.placement import AXES
from causalab.neural.shared.parallel.watchdog import COLLECTIVE_TIMEOUT_VARIABLE
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, MeshLayout, ParallelGeometry
from tests._helpers.device_mesh import FakeDeviceMesh, fake_device_meshes
from tests._helpers.geometries import mesh_geometries
from tests._helpers.gloo_world import GlooWorld

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

TP2_EP4 = ParallelGeometry(tensor=2, expert=4)


@dataclass(frozen=True)
class FakeGroup:
    """What the recording factory hands back: the ranks it was asked for."""

    ranks: tuple[int, ...]


class Recorder:
    """A [`GroupFactory`][causalab.neural.shared.parallel.mesh.GroupFactory] that records every call, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, ranks: Sequence[int]) -> ProcessGroup:
        self.calls.append(tuple(ranks))
        return cast(ProcessGroup, FakeGroup(tuple(ranks)))


def _fake_mesh(geometry: ParallelGeometry, rank: int) -> tuple[Mesh, Recorder]:
    recorder = Recorder()
    return Mesh(geometry, rank, groups=recorder), recorder


def _members(mesh: Mesh, axis: Any) -> tuple[int, ...] | None:
    group = mesh.group(axis)
    return None if group is None else cast(FakeGroup, group).ranks


# --------------------------------------------------------------------------- #
# property: the mesh against the layout, every geometry, every rank
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestMeshProperties:
    @_SETTINGS
    @given(geometry=mesh_geometries(bound=3))
    def test_every_rank_creates_the_same_groups_in_the_same_order(
        self, geometry: ParallelGeometry
    ) -> None:
        layout = MeshLayout(geometry)
        expected = [members for _, members in group_table(layout)]
        for rank in range(geometry.world):
            _, recorder = _fake_mesh(geometry, rank)
            assert recorder.calls == expected, rank

    @_SETTINGS
    @given(geometry=mesh_geometries(bound=3))
    def test_this_ranks_group_on_every_axis_is_its_layout_group(
        self, geometry: ParallelGeometry
    ) -> None:
        layout = MeshLayout(geometry)
        for rank in range(geometry.world):
            mesh, _ = _fake_mesh(geometry, rank)
            for axis in AXES:
                group = layout.group_of(rank, axis)
                assert mesh.ranks(axis) == group
                assert mesh.size(axis) == len(group)
                assert mesh.local_rank(axis) == layout.rank_in(rank, axis)
                assert mesh.global_rank(axis, mesh.local_rank(axis)) == rank
                assert [mesh.global_rank(axis, i) for i in range(len(group))] == list(
                    group
                )
                if len(group) == 1:
                    assert mesh.group(axis) is None
                else:
                    assert _members(mesh, axis) == group

    @_SETTINGS
    @given(geometry=mesh_geometries(bound=3))
    def test_the_group_table_is_every_axis_partition_once(
        self, geometry: ParallelGeometry
    ) -> None:
        layout = MeshLayout(geometry)
        table = group_table(layout)
        by_axis: dict[str, list[tuple[int, ...]]] = {}
        for axis, members in table:
            by_axis.setdefault(axis, []).append(members)
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for axis in AXES:
            groups = layout.groups(axis)
            if len(groups[0]) == 1 or groups in seen:
                assert axis not in by_axis, axis
            else:
                assert tuple(by_axis[axis]) == groups, axis
            seen.add(groups)
        # axes in AXES order, groups by first member within an axis
        order = [AXES.index(axis) for axis, _ in table]
        assert order == sorted(order)


# --------------------------------------------------------------------------- #
# unit: the hand-spelled cases and the refusals
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestMesh:
    def test_the_group_table_spelled_by_hand(self) -> None:
        # model = 4 is one group; tensor carves it in two; expert repeats model
        assert group_table(MeshLayout(TP2_EP4)) == (
            ("model", (0, 1, 2, 3)),
            ("tensor", (0, 1)),
            ("tensor", (2, 3)),
        )
        # dp=2, pp=2: data is the strided axis, pipeline the contiguous one
        assert group_table(MeshLayout(ParallelGeometry(data=2, pipeline=2))) == (
            ("data", (0, 2)),
            ("data", (1, 3)),
            ("pipeline", (0, 1)),
            ("pipeline", (2, 3)),
        )

    def test_axes_with_one_partition_share_their_groups(self) -> None:
        mesh, recorder = _fake_mesh(ParallelGeometry(tensor=2, expert=2), 1)
        assert recorder.calls == [(0, 1)]
        assert mesh.group("tensor") is mesh.group("expert") is mesh.group("model")

    def test_a_size_one_axis_has_no_group_and_no_device_mesh(self) -> None:
        mesh, recorder = _fake_mesh(ONE, 0)
        assert recorder.calls == []
        for axis in AXES:
            assert mesh.group(axis) is None
            assert mesh.size(axis) == 1 and mesh.local_rank(axis) == 0
            with pytest.raises(MeshError, match=f"axis '{axis}' has size 1"):
                mesh.device_mesh(axis)
        mesh, _ = _fake_mesh(TP2_EP4, 2)
        assert mesh.group("data") is None
        with pytest.raises(MeshError, match="axis 'data' has size 1"):
            mesh.device_mesh("data")

    def test_a_local_index_outside_the_group_is_refused(self) -> None:
        mesh, _ = _fake_mesh(TP2_EP4, 2)
        assert mesh.global_rank("tensor", 1) == 3
        for bad in (2, -1, True):
            with pytest.raises(MeshError, match="outside the tensor group of 2 ranks"):
                mesh.global_rank("tensor", bad)

    def test_a_rank_outside_the_world_is_refused(self) -> None:
        for bad in (4, -1, True):
            with pytest.raises(MeshError, match="outside range\\(4\\)"):
                _fake_mesh(TP2_EP4, bad)

    def test_the_device_rule_is_gloo_cpu_nccl_cuda(self) -> None:
        assert device_type_of("gloo") == "cpu"
        assert device_type_of("nccl") == "cuda"
        assert device_type_of("NCCL") == "cuda"
        with pytest.raises(MeshError, match="'mpi' has no device rule"):
            device_type_of("mpi")
        mesh, _ = _fake_mesh(TP2_EP4, 0)
        assert mesh.device_type == "cpu"
        assert (
            Mesh(TP2_EP4, 0, groups=Recorder(), device_type="cuda").device_type
            == "cuda"
        )

    def test_without_a_process_group_the_mesh_is_refused(self) -> None:
        assert not dist.is_initialized()
        with pytest.raises(ProtocolError, match="init_process_group") as err:
            Mesh(ParallelGeometry(tensor=2), 0)
        assert err.value.code == "P4"

    def test_a_factory_handing_a_member_nothing_is_refused_by_name(self) -> None:
        """``new_group`` hands non-members a placeholder and members a group;
        a factory that hands this rank nothing for a group it belongs to has
        broken the contract, named with the axis, the members and the rank."""
        with pytest.raises(
            MeshError,
            match=r"no process group came back for model group \(0, 1, 2, 3\), "
            r"of which rank 0 is a member",
        ):
            Mesh(TP2_EP4, 0, groups=lambda ranks: None)

    def test_the_device_mesh_is_built_once_per_axis_through_the_factory(self) -> None:
        """The one fake for the ``DeviceMesh`` seam (§10.1): the factory gets
        this rank's group on the axis and the mesh's device type, and its
        answer is kept — a second ask is the same object, not a second build."""
        mesh = Mesh(TP2_EP4, 2, groups=Recorder(), device_meshes=fake_device_meshes)
        tensor = mesh.device_mesh("tensor")
        assert isinstance(tensor, FakeDeviceMesh)
        assert tensor.ranks == (2, 3) and tensor.device_type == "cpu"
        assert mesh.device_mesh("tensor") is tensor
        assert mesh.device_mesh("model").ranks == (0, 1, 2, 3)
        assert mesh.device_mesh("expert").ranks == (0, 1, 2, 3)
        cuda = Mesh(
            TP2_EP4,
            0,
            groups=Recorder(),
            device_type="cuda",
            device_meshes=fake_device_meshes,
        )
        assert cuda.device_mesh("tensor").device_type == "cuda"


@pytest.mark.unit
class TestFromEnvironment:
    def test_a_missing_variable_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(ProtocolError, match="RANK is not set") as err:
            Mesh.from_environment(ParallelGeometry(tensor=2))
        assert err.value.code == "P4"
        monkeypatch.setenv("RANK", "0")
        monkeypatch.delenv("WORLD_SIZE")
        with pytest.raises(ProtocolError, match="WORLD_SIZE is not set"):
            Mesh.from_environment(ParallelGeometry(tensor=2))

    def test_a_malformed_variable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "zero")
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(ProtocolError, match="RANK must be a non-negative integer"):
            Mesh.from_environment(ParallelGeometry(tensor=2))

    def test_a_world_that_disagrees_with_the_geometry_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "3")
        with pytest.raises(ProtocolError, match="WORLD_SIZE=3 disagrees") as err:
            Mesh.from_environment(ParallelGeometry(tensor=2))
        assert err.value.code == "P4"
        assert "tp=2" in str(err.value) and "spans 2 ranks" in str(err.value)

    def test_an_agreeing_world_without_a_process_group_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        assert not dist.is_initialized()
        with pytest.raises(ProtocolError, match="no default process group") as err:
            Mesh.from_environment(ParallelGeometry(tensor=2))
        assert err.value.code == "P4"

    @pytest.mark.parametrize(
        ("environ", "words"),
        [
            ({"WORLD_SIZE": "2"}, ("RANK is not set", "torchrun")),
            ({"RANK": "x", "WORLD_SIZE": "2"}, ("RANK must be a non-negative",)),
            ({"RANK": "0", "WORLD_SIZE": "3"}, ("WORLD_SIZE=3 disagrees",)),
            ({"RANK": "1", "WORLD_SIZE": "2"}, ("no default process group",)),
        ],
    )
    def test_every_refusal_is_p4_at_the_parallel_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        environ: dict[str, str],
        words: tuple[str, ...],
    ) -> None:
        """The receipt names the flag: each refusal renders as
        ``[P4] at --parallel …`` whichever variable was wrong."""
        for name in ("RANK", "WORLD_SIZE"):
            monkeypatch.delenv(name, raising=False)
        for name, value in environ.items():
            monkeypatch.setenv(name, value)
        with pytest.raises(ProtocolError) as err:
            Mesh.from_environment(ParallelGeometry(tensor=2))
        assert (err.value.code, err.value.path) == ("P4", "--parallel")
        assert str(err.value).startswith("[P4] at --parallel ")
        for word in words:
            assert word in str(err.value), word


# --------------------------------------------------------------------------- #
# unit: the production factories over torch's fake backend, in this process
# --------------------------------------------------------------------------- #

Join = Callable[[int, int], None]


@pytest.fixture
def fake_group() -> Iterator[Join]:
    """``join(world, rank)``: a default process group of ``world`` ranks in
    which this process is ``rank`` — torch's ``fake`` backend, which needs
    no peer and no socket and whose collectives move nothing. ``new_group``
    over it hands members a ``ProcessGroup`` and non-members the placeholder,
    as ``gloo`` does, so [`Mesh`][causalab.neural.shared.parallel.mesh.Mesh]'s production path runs here; torn
    down after the test, since every other test asserts no group exists."""
    from torch.testing._internal.distributed.fake_pg import FakeStore

    def join(world: int, rank: int) -> None:
        dist.init_process_group("fake", rank=rank, world_size=world, store=FakeStore())

    try:
        yield join
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spy_on_new_group(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[int, ...], float | None]]:
    """Every ``(members, timeout)`` the mesh hands [`torch_new_group`][causalab.neural.shared.parallel.mesh.torch_new_group],
    which still creates the group."""
    seen: list[tuple[tuple[int, ...], float | None]] = []
    real = mesh_module.torch_new_group

    def spy(ranks: Sequence[int], timeout: float | None = None) -> ProcessGroup | None:
        seen.append((tuple(ranks), timeout))
        return real(ranks, timeout)

    monkeypatch.setattr(mesh_module, "torch_new_group", spy)
    return seen


@pytest.mark.unit
class TestMeshOverAFakeGroup:
    def test_the_backend_without_a_device_rule_is_refused_by_name(
        self, fake_group: Join
    ) -> None:
        """The device rule is deliberately two lines long; a backend outside
        it is refused before any group is carved, unless the caller names
        the device type itself."""
        fake_group(4, 1)
        with pytest.raises(
            MeshError,
            match=r"backend 'fake' has no device rule; the mesh runs on "
            r"gloo \(cpu\), nccl \(cuda\)",
        ):
            Mesh(TP2_EP4, 1)
        assert Mesh(TP2_EP4, 1, device_type="cuda").device_type == "cuda"

    def test_a_group_of_another_world_or_rank_is_refused(
        self, fake_group: Join
    ) -> None:
        fake_group(2, 0)
        with pytest.raises(ProtocolError) as world:
            Mesh(TP2_EP4, 0, device_type="cpu")
        assert (world.value.code, world.value.path) == ("P4", "--parallel")
        assert (
            "the process group has 2 ranks but --parallel dp=1,pp=1,cp=1,tp=2,ep=4 "
            "spans 4" in str(world.value)
        )
        dist.destroy_process_group()
        fake_group(4, 3)
        with pytest.raises(ProtocolError) as rank:
            Mesh(TP2_EP4, 1, device_type="cpu")
        assert (rank.value.code, rank.value.path) == ("P4", "--parallel")
        assert "this process is rank 3 of the process group, not 1" in str(rank.value)

    def test_the_real_factory_creates_this_ranks_groups_with_the_timeout(
        self, fake_group: Join, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every rank walks the whole table through ``new_group``, each call
        with the mesh's timeout (module docstring: torch's global default
        would outlast the launcher's bound); this rank keeps a real group
        for every axis it is a member of, one per partition, and builds the
        1-D ``DeviceMesh`` over it through ``DeviceMesh.from_group`` once."""
        fake_group(4, 1)
        seen = _spy_on_new_group(monkeypatch)
        mesh = Mesh(TP2_EP4, 1, device_type="cpu", timeout=7.5)
        assert seen == [(members, 7.5) for _, members in group_table(mesh.layout)]
        assert mesh.group("data") is None
        for axis in ("tensor", "expert", "model"):
            group = mesh.group(axis)
            assert isinstance(group, ProcessGroup), axis
            assert tuple(dist.get_process_group_ranks(group)) == mesh.ranks(axis)
        assert mesh.group("expert") is mesh.group("model")
        device_mesh = mesh.device_mesh("tensor")
        assert isinstance(device_mesh, DeviceMesh)
        assert device_mesh.mesh.tolist() == [0, 1] and device_mesh.ndim == 1
        assert device_mesh.device_type == "cpu"
        assert tuple(dist.get_process_group_ranks(device_mesh.get_group())) == (0, 1)
        assert mesh.device_mesh("tensor") is device_mesh
        assert mesh.device_mesh("model").mesh.tolist() == [0, 1, 2, 3]

    def test_no_timeout_leaves_every_group_to_torchs_default(
        self, fake_group: Join, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_group(2, 1)
        seen = _spy_on_new_group(monkeypatch)
        Mesh(ParallelGeometry(tensor=2), 1, device_type="cpu")
        assert seen == [((0, 1), None)]

    def test_from_environment_hands_the_collective_timeout_to_every_group(
        self, fake_group: Join, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The launcher's setting reaches every mesh group (module docstring);
        the fake backend is given the CPU rule for this test alone, since
        ``from_environment`` names no device type."""
        fake_group(4, 2)
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv(COLLECTIVE_TIMEOUT_VARIABLE, "42")
        monkeypatch.setitem(mesh_module._DEVICE_OF_BACKEND, "fake", "cpu")
        seen = _spy_on_new_group(monkeypatch)
        mesh = Mesh.from_environment(TP2_EP4)
        assert mesh.rank == 2 and mesh.device_type == "cpu"
        assert seen == [(members, 42.0) for _, members in group_table(mesh.layout)]
        assert mesh.ranks("tensor") == (2, 3)


# --------------------------------------------------------------------------- #
# smoke: the real groups under gloo
# --------------------------------------------------------------------------- #


def _mesh_program(rank: int, c: Collective) -> dict[str, Any]:
    """Per axis: the group's members, an all-reduce of ``rank`` over it, and
    the 1-D device mesh over it."""
    assert isinstance(c, TorchCollective)
    mesh = c.mesh
    out: dict[str, Any] = {}
    for axis in AXES:
        group = mesh.group(axis)
        members = None if group is None else tuple(dist.get_process_group_ranks(group))
        total = int(c.all_reduce_sum(torch.tensor([rank]), axis).item())
        if group is None:
            device_mesh = None
        else:
            dm = mesh.device_mesh(axis)
            device_mesh = (
                dm.mesh.tolist(),
                dm.ndim,
                dm.device_type,
                tuple(dist.get_process_group_ranks(dm.get_group())),
                mesh.device_mesh(axis) is dm,
            )
        out[axis] = (members, total, device_mesh)
    return out


@pytest.mark.smoke
class TestMeshUnderGloo:
    def test_world_four_at_tp2_ep4(self) -> None:
        layout = MeshLayout(TP2_EP4)
        results = GlooWorld(TP2_EP4).run(_mesh_program)
        assert len(results) == 4
        for rank, result in enumerate(results):
            for axis in AXES:
                group = layout.group_of(rank, axis)
                members, total, device_mesh = result[axis]
                assert total == sum(group), (rank, axis)
                if len(group) == 1:
                    assert members is None and device_mesh is None, (rank, axis)
                    continue
                assert members == group, (rank, axis)
                assert device_mesh == (list(group), 1, "cpu", group, True), (rank, axis)
            assert len(result["tensor"][0]) == 2
            assert len(result["expert"][0]) == 4
            assert result["model"][0] == result["expert"][0]
