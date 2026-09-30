"""The SPMD launcher (``docs/model_parallelism.md`` §3, §8.3, §9): what this
process is, which ranks publish, and which points a replica runs.

``unit``: ``detect`` in every environment shape — world 1 is ``solo`` with
or without a one-process group in the environment; ``WORLD_SIZE`` and its
two companions make a ``joined`` rank, or a ``spawned`` one when the spawn
parent marked it; no group in the environment makes this process the parent
of a spawn — each refusal (a disagreeing ``WORLD_SIZE``, a missing or
malformed ``RANK`` / ``LOCAL_RANK``, a rank outside the world, a foreign
launcher word) beside its twin; the backend and the device per rank.
``property``: ``publish_here`` over the mesh strategies — exactly ``data``
ranks publish, one per replica, and they are the ranks at local index 0 on
every non-data axis. ``RankPublisher`` takes the process's one ``Mesh`` and
gathers over *its* data group — it carves no group of its own (one mesh per
process, §3). The deterministic-simulation scenario: a ``dp=2,tp=2``
world where every rank computes its publishing verdict and its point shard
through the simulated collective, asserting the publishing set, the union
of the shards, and that the join refuses a duplicate digest by name — under
every drawn schedule, since a rank-uniform program is schedule independent.
The publisher's own gather runs under the same simulator: the
``TorchCollective`` it builds over the mesh is stood in for by the rank's
simulated collective, so the joiner receiving every replica's payload over
points, the joiner alone with no collective over rows (§8.3), and a
non-publishing rank refused by name are judged without ``gloo``.
"""

from __future__ import annotations

import os

from pathlib import Path
import datetime
from typing import Any, Sequence, Iterator

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import causalab.neural.shared.parallel.collective as collective_module
from causalab.neural.shared.join import ShardOutput, join_shards
from causalab.neural.shared.results import MetricTable
from causalab.neural.shared.parallel.collective import Collective, TorchCollective
from causalab.neural.shared.parallel.heartbeat import key_for
from causalab.neural.shared.parallel.launcher import (
    hold_for_peer,
    LAUNCHER_VARIABLE,
    SOLO_LAUNCH,
    Launch,
    Parent,
    RankPublisher,
    backend_for,
    detect,
    device_for,
    lockstep,
    publish_here,
    check_spawn_devices,
    rendezvous,
)
from causalab.neural.shared.parallel.lockstep import CollectiveLockstep
from causalab.neural.shared.parallel.mesh import Mesh
from causalab.neural.shared.parallel.watchdog import (
    COLLECTIVE_TIMEOUT_VARIABLE,
    RANK_GRACE_VARIABLE,
    Settings,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.lockstep import SOLO as SOLO_LOCKSTEP
from causalab.protocol.parallel import format_geometry, parse_geometry
from causalab.protocol.estimand import metric_record_identity
from causalab.protocol.parallel import ONE, MeshLayout, ParallelGeometry
from causalab.protocol.publish import point_shard

from tests._helpers.geometries import mesh_geometries
from tests._helpers.parallel_strategies import schedules, seeds
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for
from tests.neural.shared.parallel.test_mesh import Recorder

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

TWO = ParallelGeometry(data=2)
FOUR = ParallelGeometry(data=2, tensor=2)


def _joined(world: int, rank: int, local_rank: int | None = None) -> dict[str, str]:
    return {
        "WORLD_SIZE": str(world),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank if local_rank is None else local_rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #


class TestDetect:
    pytestmark = pytest.mark.unit

    def test_world_one_is_solo_with_an_empty_environment(self) -> None:
        assert detect(ONE, {}) == SOLO_LAUNCH
        assert SOLO_LAUNCH == Launch("solo", 0, 1, 0)

    def test_world_one_is_solo_inside_a_one_process_group(self) -> None:
        """``torchrun --nproc_per_node=1`` with no ``--parallel``: today's
        path, no ``torch.distributed`` initialised (§2)."""
        assert detect(ONE, _joined(1, 0)) == SOLO_LAUNCH

    def test_world_one_refuses_a_larger_group_in_the_environment(self) -> None:
        """Two ``torchrun`` processes and no geometry would both run the
        whole campaign and both write it: refused naming WORLD_SIZE."""
        with pytest.raises(ProtocolError) as err:
            detect(ONE, _joined(2, 0))
        assert err.value.code == "P4"
        assert "WORLD_SIZE" in str(err.value) and "--parallel" in str(err.value)

    def test_no_group_in_the_environment_makes_this_the_parent(self) -> None:
        assert detect(TWO, {}) == Parent(world=2)
        assert detect(FOUR, {"MASTER_ADDR": "x"}) == Parent(world=4)

    def test_a_group_in_the_environment_is_joined(self) -> None:
        assert detect(TWO, _joined(2, 1)) == Launch("joined", 1, 2, 1)
        assert detect(FOUR, _joined(4, 3, local_rank=1)) == Launch("joined", 3, 4, 1)

    def test_the_spawn_parents_mark_makes_a_spawned_rank(self) -> None:
        env = {**_joined(2, 1), LAUNCHER_VARIABLE: "spawned"}
        assert detect(TWO, env) == Launch("spawned", 1, 2, 1)

    def test_a_foreign_launcher_word_is_refused(self) -> None:
        with pytest.raises(ProtocolError) as err:
            detect(TWO, {**_joined(2, 0), LAUNCHER_VARIABLE: "torchrun"})
        assert LAUNCHER_VARIABLE in str(err.value) and "torchrun" in str(err.value)

    def test_a_world_size_disagreeing_with_the_geometry_is_refused(self) -> None:
        with pytest.raises(ProtocolError) as err:
            detect(TWO, _joined(4, 0))
        assert err.value.code == "P4"
        message = str(err.value)
        assert "WORLD_SIZE=4" in message and "world of 2" in message
        assert "--parallel" in message

    @pytest.mark.parametrize("value", ["two", "", "2.0", "-2"])
    def test_a_malformed_world_size_is_refused_naming_it(self, value: str) -> None:
        with pytest.raises(ProtocolError) as err:
            detect(TWO, {**_joined(2, 0), "WORLD_SIZE": value})
        assert "WORLD_SIZE" in str(err.value)

    def test_a_missing_rank_is_refused_naming_the_variable(self) -> None:
        env = _joined(2, 0)
        del env["RANK"]
        with pytest.raises(ProtocolError) as err:
            detect(TWO, env)
        assert "RANK" in str(err.value)

    def test_a_missing_local_rank_is_refused_naming_the_variable(self) -> None:
        env = _joined(2, 0)
        del env["LOCAL_RANK"]
        with pytest.raises(ProtocolError) as err:
            detect(TWO, env)
        assert "LOCAL_RANK" in str(err.value)

    @pytest.mark.parametrize("rank", ["2", "-1", "one"])
    def test_a_rank_outside_the_world_is_refused(self, rank: str) -> None:
        with pytest.raises(ProtocolError) as err:
            detect(TWO, {**_joined(2, 0), "RANK": rank})
        assert "RANK" in str(err.value)

    def test_a_negative_local_rank_is_refused(self) -> None:
        with pytest.raises(ProtocolError) as err:
            detect(TWO, {**_joined(2, 0), "LOCAL_RANK": "-1"})
        assert "LOCAL_RANK" in str(err.value)

    def test_detect_reads_the_process_environment_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", LAUNCHER_VARIABLE):
            monkeypatch.delenv(name, raising=False)
        assert detect(TWO) == Parent(world=2)

    @pytest.mark.parametrize(
        ("geometry", "environ"),
        [
            (ONE, _joined(2, 0)),
            (TWO, _joined(4, 0)),
            (TWO, {**_joined(2, 0), "WORLD_SIZE": "two"}),
            (TWO, {k: v for k, v in _joined(2, 0).items() if k != "RANK"}),
            (TWO, {**_joined(2, 0), "RANK": "2"}),
            (TWO, {**_joined(2, 0), LAUNCHER_VARIABLE: "torchrun"}),
        ],
    )
    def test_every_refusal_is_p4_at_the_parallel_flag(
        self, geometry: ParallelGeometry, environ: dict[str, str]
    ) -> None:
        """The receipt names the flag: whichever variable disagreed, the
        refusal renders as ``[P4] at --parallel …``."""
        with pytest.raises(ProtocolError) as err:
            detect(geometry, environ)
        assert (err.value.code, err.value.path) == ("P4", "--parallel")
        assert str(err.value).startswith("[P4] at --parallel ")


class TestBackendAndDevice:
    pytestmark = pytest.mark.unit

    def test_cpu_and_mps_use_gloo_and_cuda_uses_nccl(self) -> None:
        assert backend_for("cpu") == "gloo"
        assert backend_for("mps") == "gloo"
        assert backend_for("cuda") == "nccl"
        assert backend_for("cuda:1") == "nccl"

    def test_a_cuda_rank_runs_on_its_local_ordinal(self) -> None:
        launch = Launch("spawned", 3, 4, 1)
        assert device_for(launch, "cuda") == "cuda:1"
        assert device_for(launch, "cuda:0") == "cuda:1"
        assert device_for(launch, "cpu") == "cpu"
        assert device_for(SOLO_LAUNCH, "cuda:2") == "cuda:2"

    def test_a_device_list_is_refused_above_world_one(self) -> None:
        """A rank is one device; placing one rank's layers across several
        devices under a geometry is not served yet."""
        with pytest.raises(ProtocolError) as err:
            device_for(Launch("joined", 0, 2, 0), "cuda:0,cuda:1")
        assert "--device" in str(err.value) and "--parallel" in str(err.value)
        assert device_for(SOLO_LAUNCH, "cuda:0,cuda:1") == "cuda:0,cuda:1"

    def test_a_cuda_world_larger_than_the_visible_devices_is_refused_before_spawning(
        self,
    ) -> None:
        """Refuse a CUDA world larger than the device count in the parent,
        before a child can select an invalid device ordinal. CPU worlds
        and unknown device counts pass this check."""
        geometry = parse_geometry("tp=2,ep=4")
        with pytest.raises(ProtocolError) as err:
            check_spawn_devices(geometry, "cuda", visible=2)
        assert "4" in str(err.value) and "2" in str(err.value)
        assert "torchrun" in str(err.value)
        check_spawn_devices(parse_geometry("tp=2"), "cuda", visible=2)
        check_spawn_devices(geometry, "cpu", visible=0)
        check_spawn_devices(geometry, "cuda", visible=None)

    def test_a_cuda_rank_beyond_the_visible_devices_is_refused(self) -> None:
        """A joined rank whose ``LOCAL_RANK`` names no device on this node."""
        with pytest.raises(ProtocolError) as err:
            device_for(Launch("joined", 2, 4, 2), "cuda", visible=2)
        assert "LOCAL_RANK" in str(err.value) and "2" in str(err.value)
        assert device_for(Launch("joined", 1, 4, 1), "cuda", visible=2) == "cuda:1"
        assert device_for(Launch("joined", 2, 4, 2), "cpu", visible=0) == "cpu"
        assert device_for(Launch("joined", 2, 4, 2), "cuda", visible=None) == "cuda:2"


# --------------------------------------------------------------------------- #
# publish_here
# --------------------------------------------------------------------------- #


class TestPublishHere:
    pytestmark = pytest.mark.unit

    def test_solo_publishes(self) -> None:
        assert publish_here(SOLO_LAUNCH, MeshLayout(ONE))

    def test_every_rank_of_a_pure_data_world_publishes(self) -> None:
        layout = MeshLayout(ParallelGeometry(data=3))
        assert [publish_here(Launch("spawned", r, 3, r), layout) for r in range(3)] == [
            True,
            True,
            True,
        ]

    def test_only_the_first_rank_of_each_model_group_publishes(self) -> None:
        layout = MeshLayout(FOUR)  # ranks (d, m): 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
        verdicts = [publish_here(Launch("spawned", r, 4, r), layout) for r in range(4)]
        assert verdicts == [True, False, True, False]

    def test_a_launch_outside_the_layout_is_refused(self) -> None:
        with pytest.raises(ValueError):
            publish_here(Launch("spawned", 4, 4, 0), MeshLayout(FOUR))


class TestPublishHereProperty:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(mesh_geometries())
    def test_exactly_one_rank_per_replica_publishes(
        self, geometry: ParallelGeometry
    ) -> None:
        """§3: rank 0 of each data-parallel replica publishes — ``data``
        ranks in all, one per data coordinate, each at local index 0 on
        ``pipeline``, ``context`` and ``model``."""
        layout = MeshLayout(geometry)
        publishing = [
            rank
            for rank in range(geometry.world)
            if publish_here(Launch("joined", rank, geometry.world, rank), layout)
        ]
        assert len(publishing) == geometry.data
        assert sorted(layout.rank_in(rank, "data") for rank in publishing) == list(
            range(geometry.data)
        )
        for rank in publishing:
            for axis in ("pipeline", "context", "model", "tensor", "expert"):
                assert layout.rank_in(rank, axis) == 0
        # and they are exactly rank 0's data group — the group the join gathers over
        assert tuple(publishing) == layout.group_of(0, "data")


# --------------------------------------------------------------------------- #
# the simulation scenario: dp=2, tp=2
# --------------------------------------------------------------------------- #

N_POINTS = 7
DIGESTS = tuple(f"{i:064x}" for i in range(N_POINTS))


def _program(rank: int, collective: Collective) -> tuple[bool, range, int]:
    """One rank's launcher decisions, then the barrier every publishing
    rank's join would wait at — rank-uniform, so the simulator's divergence
    check would refuse any rank that decided differently."""
    layout = MeshLayout(FOUR)
    launch = Launch("spawned", rank, FOUR.world, rank)
    publish = publish_here(launch, layout)
    replica = layout.rank_in(rank, "data")
    assert replica == collective.rank("data")
    shard = point_shard(range(N_POINTS), replica, FOUR.data)
    collective.barrier("data")
    return publish, shard, replica


def _shard_output(shard: range) -> ShardOutput:
    table = MetricTable()
    digests = tuple(DIGESTS[i] for i in shard)
    for index, digest in zip(shard, digests):
        table.add(
            "iia",
            [1.0],
            {"layers": index},
            identity=metric_record_identity("match"),
        )
    return ShardOutput(
        digests=digests,
        tensor_files={},
        metric_files={"iia.json": table},
        train_evals=[],
        fit_diagnostics=[],
        routing_mismatch=[],
        summaries=[{"point": d} for d in digests],
        cells=[],
        scoring={},
        model={"key": "m"},
        attention_backends=frozenset(),
        forwards=len(shard),
    )


class TestSimulatedScenario:
    pytestmark = pytest.mark.unit

    @_SETTINGS
    @given(schedule=schedules())
    @example(schedule=[])
    def test_dp2_tp2_publishing_set_and_shard_union(self, schedule: Schedule) -> None:
        world = SimulatedWorld(
            groups_for(4, data=2, tensor=2), world=4, schedule=schedule
        )
        results = world.run(_program)
        publishing = [rank for rank, (publish, _, _) in enumerate(results) if publish]
        assert publishing == [0, 2]  # the model-index-0 rank of each replica
        shards = {replica: shard for _, shard, replica in results}
        assert shards == {0: range(0, 4), 1: range(4, 7)}
        assert sorted(i for shard in shards.values() for i in shard) == list(
            range(N_POINTS)
        )
        # the join over the publishing ranks' shards, in replica order
        joined = join_shards(
            tuple(_shard_output(results[rank][1]) for rank in publishing), DIGESTS
        )
        assert joined.digests == DIGESTS
        # a metric row carries its coordinates, not a point digest: the joined
        # table's rows sit in campaign point order
        assert [
            DIGESTS[int(r["layers"])] for r in joined.metric_files["iia.json"].rows
        ] == list(DIGESTS)

    def test_the_join_refuses_a_duplicate_digest_by_name(self) -> None:
        world = SimulatedWorld(groups_for(4, data=2, tensor=2), world=4, schedule=0)
        results = world.run(_program)
        first, second = results[0][1], results[2][1]
        overlapping = range(first.stop - 1, second.stop)  # two replicas share a point
        with pytest.raises(ProtocolError) as err:
            join_shards((_shard_output(first), _shard_output(overlapping)), DIGESTS)
        assert DIGESTS[first.stop - 1] in str(err.value)
        assert "replica 0" in str(err.value) and "replica 1" in str(err.value)

    @_SETTINGS
    @given(schedule=schedules())
    def test_the_scenario_is_schedule_independent(self, schedule: Schedule) -> None:
        def run(schedule: Schedule) -> list[Any]:
            return SimulatedWorld(
                groups_for(4, data=2, tensor=2), world=4, schedule=schedule
            ).run(_program)

        assert run(schedule) == run(())


@pytest.mark.unit
def test_launch_is_frozen_and_typed() -> None:
    launch = Launch("joined", 1, 2, 1)
    with pytest.raises(Exception):
        launch.rank = 0  # type: ignore[misc]
    with pytest.raises(ValueError):
        Launch("torchrun", 0, 1, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Launch("joined", 2, 2, 0)
    with pytest.raises(ValueError):
        Launch("solo", 0, 2, 0)


@pytest.mark.unit
def test_seeds_strategy_is_the_repositorys() -> None:
    assert isinstance(seeds(), st.SearchStrategy)


class TestRankPublisher:
    """The production publisher over the process's one mesh: its replica,
    its verdict and its group are the mesh's, and it creates no process
    group of its own."""

    pytestmark = pytest.mark.unit

    def test_it_takes_the_meshs_data_group_and_carves_none(self) -> None:
        recorder = Recorder()
        mesh = Mesh(FOUR, 2, groups=recorder)  # rank 2 = replica 1, model index 0
        carved = list(recorder.calls)
        publisher = RankPublisher(Launch("spawned", 2, 4, 2), mesh)
        assert recorder.calls == carved, "the publisher carved a group of its own"
        assert publisher.mesh is mesh
        assert publisher.launcher == "spawned"
        assert (publisher.replica, publisher.replicas) == (1, 2)
        assert publisher.publish is True
        other = RankPublisher(
            Launch("spawned", 3, 4, 3), Mesh(FOUR, 3, groups=Recorder())
        )
        assert other.publish is False

    def test_a_launch_disagreeing_with_the_meshs_rank_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            RankPublisher(Launch("joined", 1, 4, 1), Mesh(FOUR, 2, groups=Recorder()))

    def test_one_replica_gathers_to_itself_without_a_group(self) -> None:
        """``data == 1``: the mesh has no data group, and the gather is the
        identity — no collective for a world that has one replica."""
        mesh = Mesh(ParallelGeometry(tensor=2), 0, groups=Recorder())
        assert mesh.group("data") is None
        publisher = RankPublisher(Launch("spawned", 0, 2, 0), mesh)
        assert publisher.gather({"shard": 1}) == ({"shard": 1},)

    def test_the_lockstep_of_a_rank_publisher_runs_over_its_mesh(self) -> None:
        """The joiner's decisions travel on the process's one mesh: the
        ``CollectiveLockstep`` is over a ``TorchCollective`` over the
        publisher's mesh, and any other publisher gets the world-1 identity."""
        publisher = RankPublisher(
            Launch("spawned", 2, 4, 2), Mesh(FOUR, 2, groups=Recorder())
        )
        step = lockstep(publisher)
        assert isinstance(step, CollectiveLockstep)
        assert isinstance(step.collective, TorchCollective)
        assert step.collective.mesh is publisher.mesh
        from causalab.protocol.publish import SOLO

        assert lockstep(SOLO) is SOLO_LOCKSTEP


# --------------------------------------------------------------------------- #
# the gather under the simulator: points and rows over dp=2, tp=2
# --------------------------------------------------------------------------- #

#: The rank programs register the simulator's collective here before the
#: gather; the stand-in below hands it back for the mesh's rank.
_SIMULATED_COLLECTIVES: dict[int, Collective] = {}


class _TorchCollectiveOverTheSimulator:
    """Stands in for ``TorchCollective(mesh)`` inside ``RankPublisher.gather``:
    the simulator's collective for the mesh's rank, its tensors on CPU — so
    the payload's pickling, padding and rank-ordered receipt run under the
    deterministic simulation instead of ``gloo``."""

    def __init__(self, mesh: Mesh) -> None:
        self._collective = _SIMULATED_COLLECTIVES[mesh.rank]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._collective, name)


def _gather_program(geometry: ParallelGeometry) -> Any:
    def program(rank: int, collective: Collective) -> Any:
        _SIMULATED_COLLECTIVES[rank] = collective
        mesh = Mesh(geometry, rank, groups=Recorder())
        publisher = RankPublisher(Launch("spawned", rank, geometry.world, rank), mesh)
        if not publisher.publish:
            with pytest.raises(AssertionError, match=f"rank {rank} does not publish"):
                publisher.gather({"rank": rank})
            return "silent"
        return publisher.gather({"rank": rank})

    return program


class TestRankPublisherGather:
    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def simulated_collective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _SIMULATED_COLLECTIVES.clear()
        monkeypatch.setattr(
            collective_module, "TorchCollective", _TorchCollectiveOverTheSimulator
        )

    @_SETTINGS
    @given(schedule=schedules())
    def test_over_points_the_joiner_receives_every_replicas_payload(
        self, schedule: Schedule
    ) -> None:
        """The publishing ranks are rank 0's data group, (0, 2); each hands
        its payload in, the joiner receives them in replica order, the other
        publishing rank receives nothing, and a rank that does not publish
        has nothing to gather — refused by name."""
        world = SimulatedWorld(
            groups_for(4, data=2, tensor=2), world=4, schedule=schedule
        )
        results = world.run(_gather_program(FOUR))
        assert results == [({"rank": 0}, {"rank": 2}), "silent", None, "silent"]
        assert {event.axis for event in world.transcript} == {"data"}

    def test_over_rows_the_joiner_publishes_alone_with_no_collective(self) -> None:
        """§8.3: every replica holds the whole campaign, so rank 0 of replica
        0 publishes alone and its gather hands its payload straight back —
        no rank waits on another."""
        rows = ParallelGeometry(data=2, tensor=2, data_mode="rows")
        world = SimulatedWorld(groups_for(4, data=2, tensor=2), world=4, schedule=0)
        results = world.run(_gather_program(rows))
        assert results == [({"rank": 0},), "silent", "silent", "silent"]
        assert world.transcript == []


class FakeStore:
    """``torch.distributed.TCPStore``'s constructor and the three verbs the
    heartbeat uses, recording how it was built."""

    made: list[tuple[Any, ...]] = []
    instances: list[FakeStore] = []

    def __init__(
        self,
        host: str,
        port: int,
        world: int,
        is_master: bool,
        timeout: Any = None,
        wait_for_workers: bool = True,
    ) -> None:
        FakeStore.made.append((host, port, world, is_master, timeout, wait_for_workers))
        FakeStore.instances.append(self)
        self.values: dict[str, bytes] = {}

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode()

    def get(self, key: str) -> bytes:
        return self.values[key]

    def check(self, keys: Sequence[str]) -> bool:
        return all(key in self.values for key in keys)


class TestRendezvous:
    """The rendezvous store and the heartbeat over it (§3 "when a rank
    dies"): rank 0 hosts, every rank joins with the collective timeout and
    **no wait for workers** — the heartbeat is the watch over who arrives,
    and names a rank that never does, where the store's barrier named a
    count — and every rank watches through a second client connection with
    the grace as its timeout: a slow arrival at ``new_group`` is bounded by
    the former, a lost peer noticed within the latter."""

    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def fake_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch.distributed as dist

        FakeStore.made.clear()
        FakeStore.instances.clear()
        monkeypatch.setattr(dist, "TCPStore", FakeStore)

    def test_rank_zero_hosts_and_every_rank_watches_with_the_grace(self) -> None:
        config = Settings(timeout=120.0, grace=6.0)
        for rank in (0, 1):
            store, heartbeat = rendezvous(
                Launch("joined", rank, 2, rank), TWO, config, environ=_joined(2, rank)
            )
            assert isinstance(store, FakeStore)
            heartbeat.finish(0)
        assert FakeStore.made == [
            ("127.0.0.1", 29500, 2, True, datetime.timedelta(seconds=120), False),
            ("127.0.0.1", 29500, 2, False, datetime.timedelta(seconds=6), False),
            ("127.0.0.1", 29500, 2, False, datetime.timedelta(seconds=120), False),
            ("127.0.0.1", 29500, 2, False, datetime.timedelta(seconds=6), False),
        ]

    def test_a_store_that_cannot_be_reached_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rank 0 never started hosting: torch's client times out connecting
        (a ``DistNetworkError``, a ``RuntimeError``), and with no store there
        is no heartbeat to name it — the one startup death the watchdog
        cannot name (§11), rendered as the launcher's refusal, never a bare
        backend traceback."""
        import torch.distributed as dist

        from causalab.neural.shared.parallel.launcher import RendezvousFailed

        class Unreachable(RuntimeError):
            pass

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise Unreachable("The client socket has timed out after 600000ms")

        monkeypatch.setattr(dist, "TCPStore", refuse)
        with pytest.raises(RendezvousFailed) as refused:
            rendezvous(
                Launch("joined", 1, 2, 1), TWO, Settings(), environ=_joined(2, 1)
            )
        text = str(refused.value)
        assert text.startswith("[P4] at --parallel ")
        assert (
            "the rendezvous on 127.0.0.1:29500, which rank 0 hosts, failed on rank 1 of 2"
            in text
        )
        assert "Unreachable: The client socket has timed out" in text
        assert COLLECTIVE_TIMEOUT_VARIABLE in text
        assert isinstance(refused.value.__cause__, Unreachable)

    def test_the_heartbeat_beats_before_the_group_exists_and_finishes_with_the_status(
        self,
    ) -> None:
        config = Settings(timeout=120.0, grace=6.0)
        _, heartbeat = rendezvous(
            Launch("joined", 1, 2, 1), TWO, config, environ=_joined(2, 1)
        )
        watch = FakeStore.instances[1]
        assert watch.values[key_for(1)] == b"beat:1", (
            "beating before init_process_group"
        )
        heartbeat.finish(3)
        assert watch.values[key_for(1)] == b"done:3"
        assert FakeStore.instances[0].values == {}, "the rendezvous store is torch's"

    def test_a_missing_or_malformed_address_is_refused_by_name(self) -> None:
        config = Settings()
        launch = Launch("joined", 1, 2, 1)
        for environ, word in (
            ({"WORLD_SIZE": "2", "RANK": "1"}, "MASTER_ADDR"),
            ({**_joined(2, 1), "MASTER_PORT": "port"}, "MASTER_PORT='port'"),
        ):
            with pytest.raises(ProtocolError) as refused:
                rendezvous(launch, TWO, config, environ=environ)
            assert refused.value.code == "P4" and word in str(refused.value)
        assert FakeStore.made == []

    @pytest.mark.parametrize("missing", ["MASTER_ADDR", "MASTER_PORT"])
    def test_either_half_of_the_address_missing_is_refused(self, missing: str) -> None:
        """One of the two set is as useless as neither: refused naming both,
        before any store is built."""
        environ = {k: v for k, v in _joined(2, 1).items() if k != missing}
        with pytest.raises(ProtocolError) as refused:
            rendezvous(Launch("joined", 1, 2, 1), TWO, Settings(), environ=environ)
        assert (refused.value.code, refused.value.path) == ("P4", "--parallel")
        assert "MASTER_ADDR and MASTER_PORT" in str(refused.value)
        assert FakeStore.made == []

    def test_under_torchruns_agent_store_no_rank_hosts_and_the_host_is_named_as_the_agent(
        self,
    ) -> None:
        """``TORCHELASTIC_USE_AGENT_STORE=True``: the store at
        ``MASTER_ADDR:MASTER_PORT`` is rank 0's agent's, so rank 0 connects
        as a client like everyone (torch's own master would fall back to
        one on the failed bind, logging so) and the heartbeat knows whose
        store an unreachable one was."""
        environ = {**_joined(2, 0), "TORCHELASTIC_USE_AGENT_STORE": "True"}
        _, heartbeat = rendezvous(
            Launch("joined", 0, 2, 0), TWO, Settings(), environ=environ
        )
        try:
            assert FakeStore.made[0][3] is False, (
                "rank 0 is a client of the agent's store"
            )
            assert heartbeat.store_host == "agent"
        finally:
            heartbeat.finish(0)
        FakeStore.made.clear()
        _, heartbeat = rendezvous(
            Launch("joined", 0, 2, 0), TWO, Settings(), environ=_joined(2, 0)
        )
        try:
            assert FakeStore.made[0][3] is True, "no agent: rank 0 hosts"
            assert heartbeat.store_host == "rank"
        finally:
            heartbeat.finish(0)

    def test_the_heartbeat_watches_under_the_launch_geometry(self) -> None:
        """A peer's death is named with the geometry the run was launched
        for: the heartbeat carries the launch's, not a default."""
        _, heartbeat = rendezvous(
            Launch("joined", 0, 2, 0), TWO, Settings(), environ=_joined(2, 0)
        )
        try:
            assert heartbeat.geometry is TWO
            assert (heartbeat.rank, heartbeat.world) == (0, 2)
        finally:
            heartbeat.finish(0)


class TestStoreHost:
    """Torch-free: who hosts the rendezvous store, and who builds it."""

    pytestmark = pytest.mark.unit

    def test_rank_zero_hosts_unless_the_agent_does(self) -> None:
        from causalab.neural.shared.parallel.launcher import (
            AGENT_STORE_VARIABLE,
            hosts_store,
            store_host,
        )

        assert (
            store_host({}) == "rank" and hosts_store(0, {}) and not hosts_store(1, {})
        )
        agent = {AGENT_STORE_VARIABLE: "True"}
        assert store_host(agent) == "agent"
        assert not hosts_store(0, agent) and not hosts_store(1, agent)
        # torchrun spells the word exactly; anything else is not the agent
        assert store_host({AGENT_STORE_VARIABLE: "False"}) == "rank"
        assert store_host({AGENT_STORE_VARIABLE: "true"}) == "rank"


class TestNcclEnvironment:
    """Torch-free: what a NCCL rank sets before its first process group
    (§3 "a hang without a death"): the watchdog's post-timeout sleep cut
    from 60 s to 2 s, unless the user spelled the wait; the two settings
    under which a hang is never detected or never ended refused by name."""

    pytestmark = pytest.mark.unit

    def test_the_dump_wait_is_cut_unless_spelled(self) -> None:
        from causalab.neural.shared.parallel.launcher import (
            NCCL_DUMP_WAIT_MS,
            NCCL_DUMP_WAIT_VARIABLE,
            nccl_abort_delay,
            nccl_environment,
        )

        assert NCCL_DUMP_WAIT_VARIABLE == "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"
        assert nccl_environment({}) == {NCCL_DUMP_WAIT_VARIABLE: "500"}
        assert nccl_environment({"OMP_NUM_THREADS": "1"}) == {
            NCCL_DUMP_WAIT_VARIABLE: "500"
        }
        assert nccl_environment({NCCL_DUMP_WAIT_VARIABLE: "15000"}) == {}
        # torch's default error handling (3) and the tear-down mode (1) stand
        for mode in ("1", "3"):
            assert nccl_environment({"TORCH_NCCL_ASYNC_ERROR_HANDLING": mode}) == {
                NCCL_DUMP_WAIT_VARIABLE: "500"
            }
        # the abort lands four waits and a poll after the timeout
        assert nccl_abort_delay({}) == pytest.approx(4 * NCCL_DUMP_WAIT_MS / 1000 + 0.1)
        assert nccl_abort_delay({NCCL_DUMP_WAIT_VARIABLE: "15000"}) == pytest.approx(
            60.1
        )
        assert nccl_abort_delay({NCCL_DUMP_WAIT_VARIABLE: ""}) == pytest.approx(2.1)

    @pytest.mark.parametrize("name", ["TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT"])
    def test_the_blocking_wait_is_refused_by_name(self, name: str) -> None:
        """It creates no watchdog thread and the caller does not block: a
        hang can therefore go undetected."""
        from causalab.neural.shared.parallel.launcher import nccl_environment

        for value in ("1", "true", " YES "):
            with pytest.raises(ProtocolError) as refused:
                nccl_environment({name: value})
            assert refused.value.code == "P4" and refused.value.path == "--parallel"
            assert f"{name}={value!r} makes a hang undetectable" in str(refused.value)
            assert "no watchdog thread" in str(refused.value)
        for value in ("0", "", "false", "off"):
            assert nccl_environment({name: value}) == {
                "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": "500"
            }

    @pytest.mark.parametrize("mode", ["0", "2", " 2 "])
    def test_an_error_handling_that_never_ends_the_rank_is_refused(
        self, mode: str
    ) -> None:
        from causalab.neural.shared.parallel.launcher import nccl_environment

        with pytest.raises(ProtocolError) as refused:
            nccl_environment({"TORCH_NCCL_ASYNC_ERROR_HANDLING": mode})
        text = str(refused.value)
        assert (
            f"TORCH_NCCL_ASYNC_ERROR_HANDLING={mode!r} never ends a hung rank" in text
        )
        assert "tears the process down" in text

    @pytest.mark.parametrize("backend", ["nccl", "gloo"])
    def test_join_group_sets_it_for_a_nccl_rank_only(
        self, monkeypatch: pytest.MonkeyPatch, backend: str
    ) -> None:
        """Before the device pin and the store: ProcessGroupNCCL's monitor
        reads the variable as each group is built."""
        import torch
        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module

        order: list[str] = []
        monkeypatch.delenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", raising=False)
        monkeypatch.delenv("TORCH_NCCL_BLOCKING_WAIT", raising=False)
        monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
        monkeypatch.setattr(
            torch.cuda,
            "set_device",
            lambda index: order.append(
                f"device {index} {os.environ.get('TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC')}"
            ),
        )
        monkeypatch.setattr(dist, "TCPStore", FakeStore)
        monkeypatch.setattr(module, "agree_environment", lambda *a, **k: {})
        monkeypatch.setattr(
            dist, "init_process_group", lambda *a, **k: order.append("group")
        )
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        heartbeat = module.join_group(
            Launch("joined", 1, 2, 1),
            backend,
            geometry=TWO,
            settings=Settings(),  # type: ignore[arg-type]
        )
        assert heartbeat is not None
        heartbeat.finish(0)
        if backend == "nccl":
            assert order == ["device 1 500", "group"], (
                "set before the pin and the store"
            )
            assert os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] == "500"
        else:
            assert order == ["group"]
            assert "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC" not in os.environ

    def test_a_users_choice_stands_in_join_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch
        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module

        monkeypatch.setenv("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC", "2000")
        monkeypatch.delenv("TORCH_NCCL_BLOCKING_WAIT", raising=False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
        monkeypatch.setattr(dist, "TCPStore", FakeStore)
        monkeypatch.setattr(module, "agree_environment", lambda *a, **k: {})
        monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        heartbeat = module.join_group(
            Launch("joined", 0, 2, 0), "nccl", geometry=TWO, settings=Settings()
        )
        assert heartbeat is not None
        heartbeat.finish(0)
        assert os.environ["TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"] == "2000"


class TestDescribeChildExit:
    """Torch-free: the spawn parent's line for a failed child — its
    traceback, its status, its signal; a ``SIGABRT`` under NCCL named as the
    watchdog's teardown of a hung rank (§3)."""

    pytestmark = pytest.mark.unit

    def test_a_sigabrt_under_nccl_is_the_watchdogs_teardown(self) -> None:
        import signal

        from causalab.neural.shared.parallel.launcher import describe_child_exit

        line = describe_child_exit(
            index=0,
            world=2,
            exitcode=-signal.SIGABRT,
            raised=None,
            geometry=TWO,
            backend="nccl",
            settings=Settings(timeout=15.0, grace=3.0),
        )
        assert line == (
            "refused: [P4] at --parallel rank 0 of 2 was ended by NCCL's watchdog "
            "(SIGABRT): a collective timed out after CAUSALAB_COLLECTIVE_TIMEOUT=15 s "
            "while its peers were alive — a hang without a death, which no heartbeat "
            "names; NCCL's own words stand above (docs/model_parallelism.md §3) "
            f"(--parallel {format_geometry(TWO)})"
        )

    def test_every_other_ending_keeps_the_parents_words(self) -> None:
        import signal

        from causalab.neural.shared.parallel.launcher import describe_child_exit

        common = dict(index=1, world=2, geometry=TWO, settings=Settings())
        assert (
            describe_child_exit(
                exitcode=-signal.SIGABRT, raised=None, backend="gloo", **common
            )
            == f"refused: rank 1 of 2 was killed by SIGABRT (--parallel {format_geometry(TWO)})"
        )
        assert (
            describe_child_exit(
                exitcode=-signal.SIGKILL, raised=None, backend="nccl", **common
            )
            == f"refused: rank 1 of 2 was killed by SIGKILL (--parallel {format_geometry(TWO)})"
        )
        assert (
            describe_child_exit(exitcode=3, raised=None, backend="nccl", **common)
            == f"refused: rank 1 of 2 exited with status 3 (--parallel {format_geometry(TWO)})"
        )
        assert describe_child_exit(
            exitcode=1, raised="Traceback …\nValueError: x", backend="nccl", **common
        ).startswith(
            "refused: rank 1 of 2 raised:\nTraceback …\nValueError: x (--parallel"
        )


class TestLeaveIsBounded:
    """Bound process-group teardown by the grace period. A wedged peer can
    prevent ``destroy_process_group`` from returning; after the bound,
    report the hang and arrange process exit."""

    pytestmark = pytest.mark.unit

    @pytest.fixture
    def initialised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)

    def test_a_teardown_that_returns_is_reported_so(
        self, monkeypatch: pytest.MonkeyPatch, initialised: None
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel.launcher import leave_group

        calls: list[str] = []
        monkeypatch.setattr(
            dist, "destroy_process_group", lambda: calls.append("destroy")
        )
        assert leave_group(bound=1.0) is True
        assert leave_group() is True
        assert calls == ["destroy", "destroy"]

    def test_no_group_is_nothing_to_leave(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel.launcher import leave_group

        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        monkeypatch.setattr(
            dist, "destroy_process_group", lambda: pytest.fail("destroyed")
        )
        assert leave_group(bound=0.1) is True

    def test_a_teardown_that_never_returns_is_left_behind_at_the_bound(
        self, monkeypatch: pytest.MonkeyPatch, initialised: None
    ) -> None:
        import threading
        import time

        import torch.distributed as dist

        from causalab.neural.shared.parallel.launcher import leave_group

        release = threading.Event()
        monkeypatch.setattr(dist, "destroy_process_group", release.wait)
        started = time.monotonic()
        try:
            assert leave_group(bound=0.3) is False
            assert 0.3 <= time.monotonic() - started < 3.0
        finally:
            release.set()

    def test_a_teardown_that_raises_still_returns_with_the_words(
        self, monkeypatch: pytest.MonkeyPatch, initialised: None, capsys
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel.launcher import leave_group

        def broken() -> None:
            raise RuntimeError("NCCL communicator was aborted")

        monkeypatch.setattr(dist, "destroy_process_group", broken)
        assert leave_group(bound=2.0) is True
        assert (
            "teardown raised RuntimeError: NCCL communicator was aborted"
            in capsys.readouterr().err
        )

    def _publisher(self, grace: float) -> Any:
        from causalab.neural.shared.parallel import launcher as module

        class Beat:
            settings = Settings(timeout=120.0, grace=grace)
            finished: list[int] = []

            def complete(self) -> None:
                pass

            def finish(self, status: int) -> None:
                self.finished.append(status)

        publisher = module.RankPublisher.__new__(module.RankPublisher)
        publisher.heartbeat = Beat()  # type: ignore[assignment]
        return publisher

    def test_leave_past_the_grace_writes_the_hang_up_and_ends_the_process(
        self, monkeypatch: pytest.MonkeyPatch, initialised: None, capsys
    ) -> None:
        import threading

        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module

        release = threading.Event()
        monkeypatch.setattr(dist, "destroy_process_group", release.wait)
        ended: list[int] = []
        publisher = self._publisher(grace=0.2)
        try:
            module.leave(publisher, 1, hard_exit=ended.append)
        finally:
            release.set()
        assert publisher.heartbeat.finished == [1], "the peers are told first"
        assert ended == [1]
        err = capsys.readouterr().err
        assert "teardown did not return within CAUSALAB_RANK_GRACE=0.2 s" in err
        assert "a hang without a death, which no heartbeat names" in err
        assert "exits with status 1 without waiting" in err

    def test_leave_with_a_prompt_teardown_ends_nothing_early(
        self, monkeypatch: pytest.MonkeyPatch, initialised: None, capsys
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module

        monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
        ended: list[int] = []
        publisher = self._publisher(grace=5.0)
        module.leave(publisher, 0, hard_exit=ended.append)
        assert ended == [] and publisher.heartbeat.finished == [0]
        assert capsys.readouterr().err == ""

    def test_the_default_hard_exit_is_an_atexit_os_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import atexit

        from causalab.neural.shared.parallel import launcher as module

        registered: list[Any] = []
        monkeypatch.setattr(atexit, "register", registered.append)
        module._exit_at_shutdown(7)  # pyright: ignore[reportPrivateUsage]
        assert len(registered) == 1
        exits: list[int] = []
        monkeypatch.setattr(os, "_exit", exits.append)
        registered[0]()
        assert exits == [7]


class TestJoinGroupHoldsForThePeer:
    """A backend error inside ``join_group``'s two steps over the store —
    the environment agreement's read of a peer's key, ``init_process_group``
    — is held for the heartbeat's word first (a peer that never arrived is
    named there, the process ends) and, when nobody is lost, refused as
    ``RendezvousFailed`` naming the step; a rendezvous that succeeds hands
    the heartbeat back."""

    pytestmark = pytest.mark.unit

    @pytest.fixture(autouse=True)
    def fake_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch.distributed as dist

        FakeStore.made.clear()
        FakeStore.instances.clear()
        monkeypatch.setattr(dist, "TCPStore", FakeStore)

    @pytest.fixture
    def held(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
        """Record successful heartbeat holds with the exit seam disarmed.
        Stop every heartbeat at teardown so rank-named daemon threads
        cannot leak into other tests."""
        from causalab.neural.shared.parallel import heartbeat as module

        calls: list[str] = []
        started: list[module.Heartbeat] = []

        class Watch(module.Heartbeat):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["exit"] = lambda status: calls.append(f"exit {status}")
                super().__init__(*args, **kwargs)
                started.append(self)

            def hold(self) -> None:
                calls.append("hold")

        monkeypatch.setattr(module, "Heartbeat", Watch)
        yield calls
        for watch in started:
            watch.finish(0)

    def test_a_failing_step_is_held_then_refused_naming_the_step(
        self, monkeypatch: pytest.MonkeyPatch, held: list[str]
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module
        from causalab.neural.shared.parallel.launcher import RendezvousFailed

        class StoreTimeout(RuntimeError):
            pass

        def waiting(*args: Any, **kwargs: Any) -> None:
            raise StoreTimeout(
                "wait timeout after 600000ms, keys: /causalab/environment/1"
            )

        monkeypatch.setattr(module, "agree_environment", waiting)
        for rank in (0, 1):
            monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
            monkeypatch.setenv("MASTER_PORT", "29500")
            with pytest.raises(RendezvousFailed) as refused:
                module.join_group(
                    Launch("joined", rank, 2, rank),
                    "gloo",
                    geometry=TWO,
                    settings=Settings(),
                )
            assert held == ["hold"], "held for the heartbeat before refusing"
            held.clear()
            text = str(refused.value)
            assert "the environment agreement over the rendezvous store failed" in text
            assert f"on rank {rank} of 2: StoreTimeout: wait timeout" in text
            assert refused.value.code == "P4" and refused.value.step.startswith(
                "the env"
            )
        monkeypatch.setattr(module, "agree_environment", lambda *a, **k: {})

        def no_group(*args: Any, **kwargs: Any) -> None:
            raise StoreTimeout("Timed out")

        monkeypatch.setattr(dist, "init_process_group", no_group)
        with pytest.raises(RendezvousFailed) as refused:
            module.join_group(
                Launch("joined", 0, 2, 0), "gloo", geometry=TWO, settings=Settings()
            )
        assert held == [
            "hold"
        ] and "init_process_group('gloo') failed on rank 0" in str(refused.value)

    def test_a_rendezvous_that_succeeds_hands_the_heartbeat_back(
        self, monkeypatch: pytest.MonkeyPatch, held: list[str]
    ) -> None:
        import torch.distributed as dist

        from causalab.neural.shared.parallel import launcher as module

        monkeypatch.setattr(module, "agree_environment", lambda *a, **k: {})
        monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        heartbeat = module.join_group(
            Launch("joined", 1, 2, 1), "gloo", geometry=TWO, settings=Settings()
        )
        assert heartbeat is not None and held == []
        heartbeat.finish(0)


class TestEnterAndLeave:
    """``enter`` reads the settings once, joins with them, builds the mesh
    with the collective timeout and hands the heartbeat to the publisher;
    ``leave`` finishes the heartbeat with the run's status before leaving
    the group."""

    pytestmark = pytest.mark.unit

    def test_the_mesh_takes_the_timeout_and_leave_reports_the_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.parallel import launcher as module

        calls: list[Any] = []

        class Beat:
            settings = Settings()

            def finish(self, status: int) -> None:
                calls.append(("finish", status))

        heartbeat = Beat()

        def join_group(launch, backend, *, geometry, settings):
            calls.append(("join", launch.rank, backend, settings))
            return heartbeat

        def mesh(geometry, rank, *, timeout=None):
            calls.append(("mesh", rank, timeout))
            return Mesh(geometry, rank, groups=Recorder())

        monkeypatch.setattr(module, "join_group", join_group)
        # `enter` imports the mesh lazily (the spawn parent stays torch-free),
        # so the stand-in goes where the import reads it
        import causalab.neural.shared.parallel.mesh as mesh_module

        monkeypatch.setattr(mesh_module, "Mesh", mesh)
        monkeypatch.setattr(
            module,
            "leave_group",
            lambda bound=None: calls.append("leave_group") or True,
        )
        monkeypatch.setenv(COLLECTIVE_TIMEOUT_VARIABLE, "77")
        monkeypatch.setenv(RANK_GRACE_VARIABLE, "7")
        publisher = module.enter(Launch("joined", 1, 2, 1), TWO, "cpu")
        assert isinstance(publisher, RankPublisher)
        assert publisher.heartbeat is heartbeat
        module.leave(publisher, 3)
        assert calls == [
            ("join", 1, "gloo", Settings(timeout=77.0, grace=7.0)),
            ("mesh", 1, 77.0),
            ("finish", 3),
            "leave_group",
        ]

    def test_enter_joins_under_its_geometry_and_leave_defaults_to_a_clean_finish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The group join and the heartbeat behind it name the geometry the
        rank was launched for; ``leave`` with no status tells the peers a
        clean finish (``done:0``), the default a run that completed takes."""
        from causalab.neural.shared.parallel import launcher as module

        calls: list[Any] = []

        class Beat:
            settings = Settings()

            def complete(self) -> None:
                pass

            def finish(self, status: int) -> None:
                calls.append(("finish", status))

        def join_group(launch, backend, *, geometry, settings):
            calls.append(("join", geometry))
            return Beat()

        monkeypatch.setattr(module, "join_group", join_group)
        import causalab.neural.shared.parallel.mesh as mesh_module

        monkeypatch.setattr(
            mesh_module,
            "Mesh",
            lambda geometry, rank, *, timeout=None: Mesh(
                geometry, rank, groups=Recorder()
            ),
        )
        monkeypatch.setattr(
            module,
            "leave_group",
            lambda bound=None: calls.append("leave_group") or True,
        )
        publisher = module.enter(Launch("joined", 1, 2, 1), TWO, "cpu")
        module.leave(publisher)
        assert calls == [("join", TWO), ("finish", 0), "leave_group"]

    def test_a_malformed_setting_is_refused_before_any_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.parallel import launcher as module

        monkeypatch.setattr(
            module, "join_group", lambda *a, **k: pytest.fail("joined a group")
        )
        monkeypatch.setenv(RANK_GRACE_VARIABLE, "soon")
        with pytest.raises(ProtocolError, match=RANK_GRACE_VARIABLE):
            module.enter(Launch("joined", 1, 2, 1), TWO, "cpu")

    def test_leave_of_a_solo_publisher_only_leaves_the_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from causalab.neural.shared.parallel import launcher as module
        from causalab.protocol.publish import SOLO

        calls: list[str] = []
        monkeypatch.setattr(
            module,
            "leave_group",
            lambda bound=None: calls.append("leave_group") or True,
        )
        module.leave(SOLO, 0)
        assert calls == ["leave_group"]


@pytest.mark.unit
def test_visible_devices_counts_cuda_devices_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Script a nonzero CUDA count so CPU hosts exercise the device filter."""

    from causalab.neural.shared.parallel.launcher import visible_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    nowhere = Path("/nonexistent/nvidia/gpus")
    assert visible_devices("cuda", {}, nowhere) == 3
    assert visible_devices(" cuda:1", {}, nowhere) == 3
    assert visible_devices("cpu", {}, nowhere) is None
    assert visible_devices("mps", {}, nowhere) is None


@pytest.mark.unit
def test_visible_devices_reads_the_environment_and_the_driver_before_torch(
    tmp_path: Path,
) -> None:
    """The spawn parent's count is torch-free where it can be (``spawn.py``):
    ``CUDA_VISIBLE_DEVICES`` first, its entries counted; else the driver's
    listing, one entry per device."""
    from causalab.neural.shared.parallel.launcher import NVIDIA_GPUS, visible_devices

    assert visible_devices("cuda", {"CUDA_VISIBLE_DEVICES": "0,1"}, tmp_path) == 2
    assert visible_devices("cuda", {"CUDA_VISIBLE_DEVICES": "3"}, tmp_path) == 1
    assert visible_devices("cuda", {"CUDA_VISIBLE_DEVICES": ""}, tmp_path) == 0
    for name in ("0000:17:00.0", "0000:65:00.0", "0000:ca:00.0"):
        (tmp_path / name).mkdir()
    assert visible_devices("cuda", {}, tmp_path) == 3
    assert visible_devices("cpu", {"CUDA_VISIBLE_DEVICES": "0,1"}, tmp_path) is None
    assert NVIDIA_GPUS == Path("/proc/driver/nvidia/gpus")
    # the listing bounds the environment's spelling: CUDA stops at the first
    # ordinal the node does not have
    assert visible_devices("cuda", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, tmp_path) == 3
    assert visible_devices("cuda", {"CUDA_VISIBLE_DEVICES": "3"}, tmp_path) == 0


@pytest.mark.unit
class TestCudaVisibleCount:
    """CUDA's enumeration of ``CUDA_VISIBLE_DEVICES``: the entries in order,
    stopping at the first invalid one — so the parent's count never exceeds
    what torch will see, and a world above it is refused by name before any
    child starts rather than by ``invalid device ordinal`` inside one."""

    @pytest.mark.parametrize(
        ("spelling", "listed", "count"),
        [
            ("0,1", None, 2),
            ("0,1,2,3", 2, 2),  # a two-GPU node: enumeration stops at 2
            ("0,1,2,3", None, 4),  # no listing: nothing to stop at
            ("-1", None, 0),  # the documented spelling for no devices
            ("0,-1,1", None, 1),
            ("0,0", None, 1),  # a repeat is not a second device
            ("1,0", 2, 2),
            ("0,x,1", None, 1),
            ("", 4, 0),
            (" 1 , 0 ", 2, 2),
            ("GPU-1a2b,GPU-3c4d", 1, 2),  # UUIDs count; the driver would check them
            ("MIG-abc", None, 1),
        ],
    )
    def test_the_rule(self, spelling: str, listed: int | None, count: int) -> None:
        from causalab.neural.shared.parallel.launcher import cuda_visible_count

        assert cuda_visible_count(spelling, listed) == count

    @given(st.lists(st.integers(0, 7), min_size=0, max_size=8), st.integers(1, 8))
    @settings(max_examples=30, deadline=None)
    def test_the_count_is_the_distinct_valid_prefix_and_never_exceeds_the_listing(
        self, ordinals: list[int], listed: int
    ) -> None:
        from causalab.neural.shared.parallel.launcher import cuda_visible_count

        count = cuda_visible_count(",".join(map(str, ordinals)), listed)
        prefix: list[int] = []
        for ordinal in ordinals:
            if ordinal in prefix or ordinal >= listed:
                break
            prefix.append(ordinal)
        assert count == len(prefix) <= listed


@pytest.mark.unit
class TestHoldForPeer:
    """A backend error escaping the rank's run (transformers' own
    ``dist.all_reduce`` in its styles) consults the heartbeat the way the
    package's collectives do; an out-of-memory error and a process without
    a heartbeat hold nobody."""

    class _Watch:
        def __init__(self) -> None:
            self.held = 0

        def hold(self) -> None:
            self.held += 1

    def _running(self, monkeypatch: pytest.MonkeyPatch, watch: object) -> None:
        from causalab.neural.shared.parallel import heartbeat

        monkeypatch.setattr(heartbeat, "running", lambda: watch)

    def test_a_runtime_error_holds_on_the_running_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        watch = self._Watch()
        self._running(monkeypatch, watch)
        hold_for_peer(RuntimeError("Connection closed by peer"))
        assert watch.held == 1

    def test_no_heartbeat_holds_nobody(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._running(monkeypatch, None)
        hold_for_peer(RuntimeError("x"))  # no error, nothing to wait on

    @pytest.mark.parametrize(
        "error",
        [ValueError("not a backend"), KeyboardInterrupt()],
        ids=["value", "interrupt"],
    )
    def test_only_a_runtime_error_is_a_candidate(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        watch = self._Watch()
        self._running(monkeypatch, watch)
        hold_for_peer(error)
        assert watch.held == 0

    def test_an_out_of_memory_error_is_the_ranks_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class OutOfMemoryError(RuntimeError):  # torch's, by name
            pass

        watch = self._Watch()
        self._running(monkeypatch, watch)
        assert hold_for_peer(OutOfMemoryError("CUDA out of memory")) is None
        assert watch.held == 0

    def test_torchs_distributed_error_with_nobody_lost_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hold returned (every peer alive, or the bound passed): a
        ``DistError`` — NCCL's timeout surfacing at a sync outside our
        collectives, transformers' own all-reduce failing — is rendered as
        ``BackendFailed``, the rank's refusal naming the rank and the
        timeout, for the CLI to raise instead of the traceback; any other
        ``RuntimeError`` stays the caller's, as it was."""
        from causalab.neural.shared.parallel.launcher import BackendFailed

        class DistError(RuntimeError):  # torch.distributed's, by name
            pass

        class DistBackendError(DistError):
            pass

        watch = self._Watch()
        watch.rank, watch.world = 1, 2  # type: ignore[attr-defined]
        watch.settings = Settings(timeout=120.0, grace=3.0)  # type: ignore[attr-defined]
        self._running(monkeypatch, watch)
        cause = DistBackendError(
            "[Rank 1] Watchdog caught collective operation timeout: WorkNCCL(...)\n"
            "ran for 120004 milliseconds before timing out."
        )
        refusal = hold_for_peer(cause)
        assert watch.held == 1
        assert isinstance(refusal, BackendFailed) and refusal.rank == 1
        text = str(refusal)
        assert text.startswith("[P4] at --parallel ")
        assert "a collective outside this package's calls failed on rank 1 of 2" in text
        assert (
            "DistBackendError: [Rank 1] Watchdog caught collective operation timeout"
            in text
        )
        assert "ran for" not in text, "the first line only"
        assert (
            f"{RANK_GRACE_VARIABLE}=3" in text and COLLECTIVE_TIMEOUT_VARIABLE in text
        )
        assert hold_for_peer(RuntimeError("not torch's")) is None
        assert watch.held == 2

    def test_gloos_bare_runtime_error_is_the_backends_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gloo raises plain ``RuntimeError``s stamped with its source path:
        the survivor of a wedge, timing out in transformers' own
        ``all_reduce``, escaped with one and printed a traceback where the
        refusal belonged. Both spellings seen — Linux's tcp pair, macOS's
        uv buffer — are the backend's; a ``RuntimeError`` of ours is not."""
        from causalab.neural.shared.parallel.launcher import (
            BackendFailed,
            backend_error,
        )

        linux = RuntimeError(
            "[../third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection "
            "closed by peer [127.0.0.1]:53412"
        )
        # The macOS torch wheel from PyPI stamps its CI build directory, a
        # home-directory path; it is rewritten to a neutral ``/srv/runner``.
        # `backend_error` keys on the ``gloo`` path segment, not the prefix.
        macos = RuntimeError(
            "[/srv/runner/work/pytorch/pytorch/pytorch/third_party/gloo/gloo/"
            "transport/uv/unbound_buffer.cc:65] Timed out waiting 15000ms for recv "
            "operation to complete"
        )
        assert backend_error(linux) and backend_error(macos)
        assert not backend_error(RuntimeError("shape mismatch in the featurizer"))
        assert not backend_error(ValueError("gloo is not a RuntimeError here"))
        watch = self._Watch()
        watch.rank, watch.world = 0, 2  # type: ignore[attr-defined]
        watch.settings = Settings(timeout=15.0, grace=3.0)  # type: ignore[attr-defined]
        self._running(monkeypatch, watch)
        refusal = hold_for_peer(macos)
        assert isinstance(refusal, BackendFailed)
        assert "RuntimeError: [/srv/runner" in str(refusal)
        assert "Timed out waiting 15000ms for recv operation to complete" in str(
            refusal
        )
