"""The ``Collective`` conformance suite run against every implementation.

``collective_contract.py`` states the protocol's contract lines as programs
and checks; this file runs them against `SOLO` (world 1), the
simulator's ``RankCollective`` (``SimulatedWorld.run``) and the production
[`TorchCollective`][causalab.neural.shared.parallel.collective.TorchCollective] under ``gloo`` in spawned processes (the smoke tier,
``docs/model_parallelism.md`` §10.6). Three implementations agreeing on one
set of functions is the point: a disagreement between the simulator and torch
means the simulator is wrong.

The hand-written mutations close each tier, the repository's convention: an
all-gather that concatenates in arrival order (the simulator, a slow rank
arriving last) or with the rank's own chunk first (torch) fails the rank-order
contract; a torch broadcast that skips the shape exchange fails on a
non-source ``None``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import pytest
import torch
import torch.distributed as dist
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.collective import (
    DTYPES,
    HEADER_WIDTH,
    MAX_DIMS,
    SOLO,
    Collective,
    CollectiveError,
    CollectiveFailed,
    TorchCollective,
    decode_header,
    encode_header,
)
from causalab.neural.shared.parallel.placement import AXES, Axis
from causalab.protocol.parallel import (
    ONE,
    MeshLayout,
    ParallelGeometry,
    format_geometry,
)
from tests._helpers import parallel_strategies as ps
from tests._helpers.geometries import mesh_geometries
from tests._helpers.gloo_world import GlooWorld, RankCrashed, WorldTimedOut
from tests._helpers.simulated_world import Schedule, SimulatedWorld
from tests._helpers.simulated_world import world as world_module
from tests._helpers.simulated_world.rendezvous import Rendezvous
from tests.neural.shared.parallel.collective_contract import (
    BY_NAME,
    CONTRACTS,
    REFUSED,
    Contract,
    every_contract,
    results_of,
    verify_all,
)

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: The geometries the spawned ``gloo`` worlds run (§10.6, ``world ∈ {2, 4}``):
#: one multi-member axis at world 2; two nested ones (``model = 4``, tensor
#: groups of 2 inside) at world 4.
GEOMETRIES: tuple[ParallelGeometry, ...] = (
    ParallelGeometry(tensor=2),
    ParallelGeometry(tensor=2, expert=4),
)

#: The simulator adds an odd group (a trailing member sits out of send/recv)
#: and the two strided outer axes, where groups on one axis are many.
SIMULATED_GEOMETRIES: tuple[ParallelGeometry, ...] = GEOMETRIES + (
    ParallelGeometry(tensor=3),
    ParallelGeometry(data=2, pipeline=2),
)

_ids: Callable[[Any], str] = lambda x: (  # noqa: E731 - pytest id helper
    x.name if isinstance(x, Contract) else format_geometry(x)
)


def _solo_run(program: Callable[[int, Collective], Any]) -> list[Any]:
    return [program(0, SOLO)]


def _simulated(geometry: ParallelGeometry, schedule: Schedule = ()) -> SimulatedWorld:
    """The simulator laid out by ``MeshLayout`` — the same groups the mesh builds."""
    layout = MeshLayout(geometry)
    return SimulatedWorld(
        {axis: layout.groups(axis) for axis in AXES},
        world=geometry.world,
        schedule=schedule,
    )


# --------------------------------------------------------------------------- #
# unit: Solo and the simulator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestSolo:
    def test_solo_is_a_collective(self) -> None:
        assert isinstance(SOLO, Collective)

    def test_every_collective_names_its_device(self) -> None:
        """``Collective.device`` is a protocol member, not a ``getattr``
        fallback: ``Solo`` and the simulator's ranks are CPU-side, and a
        collective without one is not a ``Collective`` at all."""
        assert SOLO.device == torch.device("cpu")
        assert all(
            _simulated(ParallelGeometry(tensor=2)).run(
                lambda rank, c: c.device == torch.device("cpu")
            )
        )

        class _Nameless:
            def rank(self, axis: str) -> int:
                return 0

        assert not isinstance(_Nameless(), Collective)

    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds_at_world_one(self, contract: Contract) -> None:
        contract.check(_solo_run, MeshLayout(ONE))


@pytest.mark.unit
class TestSimulated:
    @pytest.mark.parametrize("geometry", SIMULATED_GEOMETRIES, ids=_ids)
    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds(
        self, contract: Contract, geometry: ParallelGeometry
    ) -> None:
        contract.check(_simulated(geometry).run, MeshLayout(geometry))


@pytest.mark.property
class TestSimulatedProperties:
    @_SETTINGS
    @given(geometry=mesh_geometries(bound=2), schedule=ps.schedules())
    @example(geometry=ParallelGeometry(tensor=2), schedule=[])
    def test_every_contract_holds_on_every_layout_under_every_schedule(
        self, geometry: ParallelGeometry, schedule: Schedule
    ) -> None:
        verify_all(
            _simulated(geometry, schedule).run(every_contract), MeshLayout(geometry)
        )


# --------------------------------------------------------------------------- #
# unit: the header codec
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestHeader:
    @_SETTINGS
    @given(
        shape=st.lists(st.integers(0, 5), min_size=0, max_size=MAX_DIMS).map(tuple),
        dtype=st.sampled_from(DTYPES),
    )
    def test_the_codec_round_trips_shape_and_dtype(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        header = encode_header(torch.empty(shape, dtype=dtype), torch.device("cpu"))
        assert header.shape == (HEADER_WIDTH,) and header.dtype == torch.int64
        assert decode_header(header) == (shape, dtype)

    def test_too_many_dimensions_are_refused_by_count(self) -> None:
        with pytest.raises(CollectiveError, match=f"{MAX_DIMS + 1} dimensions"):
            encode_header(torch.empty((1,) * (MAX_DIMS + 1)), torch.device("cpu"))

    def test_a_dtype_without_a_code_is_refused_by_name(self) -> None:
        with pytest.raises(CollectiveError, match="float8"):
            encode_header(
                torch.empty(2, dtype=torch.float8_e4m3fn), torch.device("cpu")
            )


# --------------------------------------------------------------------------- #
# unit: the simulator mutation
# --------------------------------------------------------------------------- #


class _ArrivalOrderRendezvous(Rendezvous):
    """The mutation: an all-gather concatenated in the order members arrived."""

    def resolve(self) -> None:
        signature = self.first.signature
        if signature.op != "all_gather":
            return super().resolve()
        arrived = list(self.arrivals.values())  # insertion order is arrival order
        gathered = torch.cat([a.payload for a in arrived], dim=signature.detail[0])
        for arrival in arrived:
            arrival.result = gathered.clone()
        return None


@pytest.mark.unit
class TestSimulatedMutations:
    def test_an_all_gather_in_arrival_order_fails_the_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(world_module, "Rendezvous", _ArrivalOrderRendezvous)
        world = _simulated(ParallelGeometry(tensor=2, expert=4))
        world.slow(0)  # rank 0 arrives last, so arrival order is not rank order
        with pytest.raises(AssertionError):
            BY_NAME["all_gather_in_rank_order"].check(world.run, world_layout(world))
        # the mutation touches nothing else: the other lines still hold
        for name in (
            "all_reduce_sum_is_a_new_equal_tensor",
            "broadcast_carries_shape_and_dtype",
        ):
            BY_NAME[name].check(world.run, world_layout(world))


def world_layout(world: SimulatedWorld) -> MeshLayout:
    return (
        MeshLayout(ParallelGeometry(tensor=2, expert=4))
        if world.world == 4
        else MeshLayout(ParallelGeometry(tensor=2))
    )


# --------------------------------------------------------------------------- #
# smoke: TorchCollective under gloo, in spawned processes
# --------------------------------------------------------------------------- #


def _torch_refusals(rank: int, c: Collective) -> dict[str, str]:
    """Refusals only the production implementation can raise, checked by name."""
    out: dict[str, str] = {}
    try:  # members disagreeing on shape: (2, 3) on rank 0, (3, 3) on rank 1
        c.all_gather(torch.zeros((2 + rank, 3)), 0, "tensor")
        out["unequal"] = "accepted"
    except CollectiveError as error:
        out["unequal"] = str(error)
    try:  # a tensor off the collective's device
        c.all_reduce_sum(torch.empty(2, device="meta"), "tensor")
        out["off_device"] = "accepted"
    except CollectiveError as error:
        out["off_device"] = str(error)
    return out


def _own_chunk_first() -> None:
    """The mutation: each rank concatenates its own chunk first — the order
    the chunks 'arrive' from its point of view — instead of rank order."""
    real = TorchCollective.all_gather

    def all_gather(
        self: TorchCollective, tensor: torch.Tensor, dim: int, axis: Axis
    ) -> torch.Tensor:
        gathered = real(self, tensor, dim, axis)
        me, size = self.rank(axis), self.size(axis)
        if size == 1:
            return gathered
        chunks = list(gathered.chunk(size, dim))
        return torch.cat(chunks[me:] + chunks[:me], dim=dim)

    TorchCollective.all_gather = all_gather  # type: ignore[method-assign]


def _broadcast_without_the_shape_exchange() -> None:
    """The mutation: broadcast the tensor straight away — a non-source, holding
    ``None``, has nothing to size a buffer by."""

    def broadcast(
        self: TorchCollective, tensor: torch.Tensor | None, src: int, axis: Axis
    ) -> torch.Tensor:
        group = self.mesh.group(axis)
        buffer = torch.empty(0) if tensor is None else tensor.contiguous()
        if group is None:
            return buffer
        dist.broadcast(buffer, src=self.mesh.global_rank(axis, src), group=group)
        return buffer

    TorchCollective.broadcast = broadcast  # type: ignore[method-assign]


@pytest.fixture(scope="module", params=GEOMETRIES, ids=_ids)
def gloo_results(
    request: pytest.FixtureRequest,
) -> tuple[MeshLayout, Sequence[dict[str, Any]]]:
    """Every contract's program run once per geometry in one spawned world."""
    geometry: ParallelGeometry = request.param
    return MeshLayout(geometry), GlooWorld(geometry).run(every_contract)


@pytest.mark.smoke
class TestTorchCollectiveUnderGloo:
    @pytest.mark.parametrize("contract", CONTRACTS, ids=_ids)
    def test_the_contract_holds(
        self,
        contract: Contract,
        gloo_results: tuple[MeshLayout, Sequence[dict[str, Any]]],
    ) -> None:
        layout, results = gloo_results
        contract.verify(results_of(contract.name, results), layout)

    def test_torch_only_refusals_name_what_did_not_fit(self) -> None:
        results = GlooWorld(ParallelGeometry(tensor=2)).run(_torch_refusals)
        for rank, result in enumerate(results):
            other = 1 - rank
            assert f"rank {other} holds ({2 + other}, 3)" in result["unequal"], rank
            assert f"rank {rank} holds ({2 + rank}, 3)" in result["unequal"], rank
            assert "meta" in result["off_device"] and "cpu" in result["off_device"]

    def test_an_all_gather_with_the_own_chunk_first_fails_the_contract(self) -> None:
        world = GlooWorld(ParallelGeometry(tensor=2), mutation=_own_chunk_first)
        with pytest.raises(AssertionError):
            BY_NAME["all_gather_in_rank_order"].check(
                world.run, MeshLayout(world.geometry)
            )

    def test_a_broadcast_without_the_shape_exchange_fails_on_a_non_source_none(
        self,
    ) -> None:
        world = GlooWorld(
            ParallelGeometry(tensor=2), mutation=_broadcast_without_the_shape_exchange
        )
        # a wrong result, a rank refusing, or a world that never completes:
        # each is the failure the exchange exists to prevent, none a hang
        with pytest.raises((AssertionError, RankCrashed, WorldTimedOut)):
            BY_NAME["broadcast_carries_shape_and_dtype"].check(
                world.run, MeshLayout(world.geometry)
            )

    def test_the_refusal_marker_is_the_shared_one(self) -> None:
        assert REFUSED == "refused"


# --------------------------------------------------------------------------- #
# a backend failure inside a collective (§3 "when a rank dies")
# --------------------------------------------------------------------------- #


class TestCollectiveFailure:
    """``TorchCollective._inside``: a backend ``RuntimeError`` is re-rendered
    as [`CollectiveFailed`][causalab.neural.shared.parallel.collective.CollectiveFailed], after the process's heartbeat — when one
    runs — has had the bound to say whether a peer's death explains it. The
    mesh is a stand-in carrying what the refusal names; no process group is
    touched, as the failure is raised inside the block."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _collective() -> TorchCollective:
        from types import SimpleNamespace

        mesh = SimpleNamespace(
            device_type="cpu",
            rank=0,
            geometry=ParallelGeometry(tensor=2),
            ranks=lambda axis: (0, 1),
        )
        return TorchCollective(mesh)  # type: ignore[arg-type]

    def test_without_a_heartbeat_the_failure_is_the_collectives_own_at_once(
        self,
    ) -> None:
        from causalab.neural.shared.parallel.heartbeat import running

        assert running() is None
        collective = self._collective()
        with pytest.raises(CollectiveFailed) as failed:
            with collective._inside("tensor", "all_gather"):  # pyright: ignore[reportPrivateUsage]
                raise RuntimeError("[gloo] Read error [127.0.0.1]: Connection reset")
        text = str(failed.value)
        assert text.startswith("[P4] at --parallel all_gather on axis 'tensor' failed")
        assert "rank 0 of 2 (group ranks [0, 1])" in text
        assert "RuntimeError: [gloo] Read error" in text
        assert "no rank watchdog runs in this process" in text
        assert failed.value.op == "all_gather" and failed.value.members == (0, 1)
        assert isinstance(failed.value.__cause__, RuntimeError)

    def test_with_live_peers_the_heartbeat_clears_the_failure_as_the_collectives(
        self,
    ) -> None:
        import threading

        from causalab.neural.shared.parallel.heartbeat import Heartbeat, key_for
        from causalab.neural.shared.parallel.watchdog import Beat, Settings, encode
        from tests._helpers.simulated_world.heartbeat import FakeStore

        store, lines, statuses = FakeStore(), [], []
        heartbeat = Heartbeat(
            store,
            rank=0,
            world=2,
            settings=Settings(timeout=10.0, grace=0.1),
            geometry=ParallelGeometry(tensor=2),
            exit=statuses.append,
            write=lines.append,
        )  # the wall clock: the peer's beats land at distinct times
        stop = threading.Event()

        def beat() -> None:
            count = 0
            while not stop.is_set():
                count += 1
                store.set(key_for(1), encode(Beat(count)))
                stop.wait(0.001)

        peer = threading.Thread(target=beat, daemon=True)
        peer.start()
        heartbeat.start()
        try:
            with pytest.raises(CollectiveFailed) as failed:
                with self._collective()._inside("tensor", "all_gather"):  # pyright: ignore[reportPrivateUsage]
                    raise RuntimeError("[gloo] timed out")
        finally:
            stop.set()
            peer.join(5.0)
            heartbeat.finish(1)
        assert statuses == [] and lines == []
        text = str(failed.value)
        assert "no peer went silent for the grace after the failure" in text
        assert "CAUSALAB_RANK_GRACE=0.1" in text

    def test_with_a_dead_peer_the_heartbeat_refuses_by_name_first(self) -> None:
        """The peer beat once and never again, and the clock is past the
        grace: the heartbeat names it (through the exit seam here;
        ``os._exit`` in a rank, so the collective's refusal below is never
        reached there). A peer never seen at all would be the timeout's
        (``absent``, ``test_watchdog.py``), not the grace's."""
        from causalab.neural.shared.parallel.heartbeat import Heartbeat, key_for
        from causalab.neural.shared.parallel.watchdog import (
            LOST_STATUS,
            Beat,
            Settings,
            encode,
        )
        from tests._helpers.simulated_world.heartbeat import FakeStore

        store, lines, statuses, clock = FakeStore(), [], [], [0.0]
        store.set(key_for(1), encode(Beat(1)))
        heartbeat = Heartbeat(
            store,
            rank=0,
            world=2,
            settings=Settings(timeout=10.0, grace=0.1),
            geometry=ParallelGeometry(tensor=2),
            clock=lambda: clock[0],
            exit=statuses.append,
            write=lines.append,
        )
        heartbeat.start()
        try:
            clock[0] = 1.0
            with pytest.raises(CollectiveFailed):
                with self._collective()._inside("tensor", "all_gather"):  # pyright: ignore[reportPrivateUsage]
                    raise RuntimeError("[gloo] Read error: Connection reset by peer")
        finally:
            heartbeat.finish(LOST_STATUS)
        assert statuses == [LOST_STATUS]
        (line,) = lines
        assert line.startswith("refused: [P4] at --parallel rank 1 of 2 has sent no ")
        assert "waiting in all_gather on axis 'tensor'" in line


# --------------------------------------------------------------------------- #
# inside CUDA graph capture (docs/cuda_graphs.md "Multi-rank execution")
# --------------------------------------------------------------------------- #


def _captured_everywhere() -> None:
    """The mutation for the capture smoke: every call sees a capturing stream,
    and a header exchange — the one step capture cannot record — fails the
    rank by name, so a result proves the tensor collectives skipped it."""
    from causalab.neural.shared.parallel import collective

    def no_header(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a header was exchanged inside capture")

    collective.capturing = lambda device: True  # type: ignore[assignment]
    TorchCollective._agree_header = no_header  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]


#: The tensor collectives a captured forward and backward reach under tensor
#: and expert parallelism (the router and KV-head gradient sums).
_CAPTURED = ("all_gather_in_rank_order", "all_reduce_sum_is_a_new_equal_tensor")


def _captured_tensor_collectives(rank: int, c: Collective) -> dict[str, Any]:
    return {name: BY_NAME[name].program(rank, c) for name in _CAPTURED}


@pytest.mark.smoke
class TestTensorCollectivesInsideCapture:
    def test_they_run_without_the_header_exchange_and_keep_their_contract(
        self,
    ) -> None:
        world = GlooWorld(ParallelGeometry(tensor=2), mutation=_captured_everywhere)
        results = world.run(_captured_tensor_collectives)
        layout = MeshLayout(world.geometry)
        for name in _CAPTURED:
            BY_NAME[name].verify([result[name] for result in results], layout)


class TestHostStepsInsideCapture:
    """What capture cannot record is refused by name before
    ``torch.distributed`` is touched — a host scalar (``agree_*``), a barrier,
    a broadcast's shape header — rather than left to invalidate the capture
    with a CUDA error that names no collective. The mesh is a stand-in with a
    two-member group; any backend call fails the test."""

    pytestmark = pytest.mark.unit

    @staticmethod
    def _collective(monkeypatch: pytest.MonkeyPatch) -> TorchCollective:
        from types import SimpleNamespace

        from causalab.neural.shared.parallel import collective

        def touched(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("torch.distributed was called inside capture")

        for name in ("all_reduce", "all_gather", "broadcast", "barrier"):
            monkeypatch.setattr(dist, name, touched)
        monkeypatch.setattr(collective, "capturing", lambda device: True)
        mesh = SimpleNamespace(
            device_type="cpu",
            rank=0,
            geometry=ParallelGeometry(tensor=2),
            ranks=lambda axis: (0, 1),
            size=lambda axis: 2 if axis in ("tensor", "model") else 1,
            local_rank=lambda axis: 0,
            global_rank=lambda axis, index: index,
            group=lambda axis: object() if axis in ("tensor", "model") else None,
        )
        return TorchCollective(mesh)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "op, call",
        [
            ("agree", lambda c: c.agree_min(3, "tensor")),
            ("agree", lambda c: c.agree_any(True, "tensor")),
            ("agree", lambda c: c.agree_sum(3, "tensor")),
            ("barrier", lambda c: c.barrier("tensor")),
            ("broadcast", lambda c: c.broadcast(torch.ones(2), 0, "tensor")),
            ("broadcast", lambda c: c.broadcast(None, 1, "tensor")),
        ],
    )
    def test_a_host_step_is_refused_by_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        op: str,
        call: Callable[[TorchCollective], Any],
    ) -> None:
        from causalab.neural.shared.parallel.collective import CaptureUnsafe

        with pytest.raises(CaptureUnsafe) as refused:
            call(self._collective(monkeypatch))
        assert isinstance(refused.value, CollectiveError)
        assert f"{op} on axis 'tensor'" in str(refused.value)
        assert "CUDA graph capture" in str(refused.value)

    def test_a_group_of_one_stays_the_identity_while_capturing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        collective = self._collective(monkeypatch)
        assert collective.agree_min(3, "data") == 3
        collective.barrier("data")
        value = torch.ones(2)
        assert collective.broadcast(value, 0, "data") is value

    def test_no_stream_is_capturing_off_cuda(self) -> None:
        from causalab.neural.shared.parallel.collective import capturing

        assert not capturing(torch.device("cpu"))
        assert not capturing(torch.device("meta"))
