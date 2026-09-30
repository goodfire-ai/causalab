"""``gather_objects``: Python objects over the tensor collectives
(``docs/model_parallelism.md`` §3, §8.3).

``unit``: world 1 is the identity with no call on the collective; under
the simulator the group's first member — the joiner — receives every
payload in rank order, whatever the payloads' sizes (each member's pickled
bytes travel as one row of its own length), an empty payload included, and
every other member hands its own over and holds ``None``. ``property``:
over mesh geometries and drawn schedules the joiner's tuple equals the
members' payloads and nobody else holds one. ``smoke``: the same program
over ``gloo``. The hand-written mutation — every row received at the
joiner's own length instead of each member's — fails the contract: under
the simulator the pair's signatures disagree, under a real backend the
receive would truncate or hang.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.shared.parallel import objects
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.objects import gather_objects
from causalab.neural.shared.parallel.placement import AXES, Axis
from causalab.protocol.parallel import MeshLayout, ParallelGeometry
from tests._helpers import parallel_strategies as ps
from tests._helpers.geometries import mesh_geometries
from tests._helpers.gloo_world import GlooWorld
from tests._helpers.simulated_world import Schedule, SimulatedWorld

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def payload_of(rank: int) -> Any:
    """A payload whose pickled size grows with the rank — rank 0's is
    empty-ish, rank 3's carries a tensor — so the per-member lengths are
    exercised."""
    if rank == 0:
        return ()
    return {"rank": rank, "rows": list(range(rank * 7)), "tensor": torch.arange(rank)}


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    return bool(a == b)


def _program(rank: int, c: Collective) -> dict[Axis, tuple[Any, ...] | None]:
    return {axis: gather_objects(payload_of(rank), axis, c) for axis in AXES}


def _verify(
    results: list[dict[Axis, tuple[Any, ...] | None]], layout: MeshLayout
) -> None:
    for rank, result in enumerate(results):
        for axis in AXES:
            group = layout.group_of(rank, axis)
            got = result[axis]
            if rank != group[0]:
                assert got is None, (rank, axis, "only the joiner holds the payloads")
                continue
            assert got is not None and len(got) == len(group), (rank, axis)
            for member, value in zip(group, got):
                assert _equal(value, payload_of(member)), (rank, axis, member)


def _simulated(geometry: ParallelGeometry, schedule: Schedule = ()) -> SimulatedWorld:
    layout = MeshLayout(geometry)
    return SimulatedWorld(
        {axis: layout.groups(axis) for axis in AXES},
        world=geometry.world,
        schedule=schedule,
    )


class _Counting:
    """A collective of one that counts its calls."""

    calls = 0
    device = torch.device("cpu")

    def rank(self, axis: Axis) -> int:
        return 0

    def size(self, axis: Axis) -> int:
        return 1

    def all_gather(self, tensor: torch.Tensor, dim: int, axis: Axis) -> torch.Tensor:
        self.calls += 1
        return tensor


@pytest.mark.unit
class TestGatherObjects:
    def test_world_one_is_the_identity_with_no_call(self) -> None:
        counting = _Counting()
        payload = {"a": [1, 2], "t": torch.ones(2)}
        assert gather_objects(payload, "data", counting) == (payload,)  # type: ignore[arg-type]
        assert counting.calls == 0
        assert gather_objects(payload, "data", SOLO) == (payload,)

    @pytest.mark.parametrize(
        "geometry",
        (ParallelGeometry(data=2), ParallelGeometry(data=2, tensor=2)),
        ids=("dp2", "dp2_tp2"),
    )
    def test_the_joiner_receives_every_payload_in_rank_order_and_nobody_else(
        self, geometry: ParallelGeometry
    ) -> None:
        _verify(_simulated(geometry).run(_program), MeshLayout(geometry))

    def test_a_destination_outside_the_group_is_refused(self) -> None:
        def program(rank: int, c: Collective) -> Any:
            return gather_objects(payload_of(rank), "data", c, to=2)

        with pytest.raises(Exception) as err:  # noqa: B017 - the rank's own error
            _simulated(ParallelGeometry(data=2)).run(program)
        assert "to=2 is outside the data group of 2" in str(err.value.__cause__)

    def test_receiving_every_row_at_the_joiners_own_length_fails_the_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mutation: the lengths exchanged and then ignored, every row
        received at the joiner's own length instead of each member's —
        the simulator refuses the pair whose signatures disagree; a real
        backend would truncate the longer pickle or hang on the shorter."""
        real = objects.gather_objects

        def own_length(payload: Any, axis: Axis, c: Collective, *, to: int = 0) -> Any:
            size = c.size(axis)
            if size == 1:
                return real(payload, axis, c, to=to)
            import pickle

            blob = pickle.dumps(payload)
            c.all_gather(torch.tensor([len(blob)]), 0, axis)
            mine = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            me = c.rank(axis)
            if me != to:
                c.send(mine, to, axis)
                return None
            rows = [
                mine
                if m == me
                else c.recv((len(blob),), torch.uint8, c.device, m, axis)
                for m in range(size)
            ]
            return tuple(pickle.loads(r.numpy().tobytes()) for r in rows)

        monkeypatch.setattr(objects, "gather_objects", own_length)

        def program(rank: int, c: Collective) -> dict[Axis, tuple[Any, ...] | None]:
            return {
                axis: objects.gather_objects(payload_of(rank), axis, c) for axis in AXES
            }

        geometry = ParallelGeometry(data=2)
        with pytest.raises(Exception):  # noqa: B017 - the truncation's own error
            _verify(_simulated(geometry).run(program), MeshLayout(geometry))


@pytest.mark.property
class TestGatherObjectsProperties:
    @_SETTINGS
    @given(geometry=mesh_geometries(bound=2), schedule=ps.schedules())
    @example(geometry=ParallelGeometry(data=2), schedule=[])
    def test_the_joiners_tuple_is_the_members_payloads_under_every_schedule(
        self, geometry: ParallelGeometry, schedule: Schedule
    ) -> None:
        _verify(_simulated(geometry, schedule).run(_program), MeshLayout(geometry))


@pytest.mark.smoke
class TestGatherObjectsUnderGloo:
    def test_the_joiner_receives_every_payload(self) -> None:
        geometry = ParallelGeometry(data=2, tensor=2)
        _verify(GlooWorld(geometry).run(_program), MeshLayout(geometry))
