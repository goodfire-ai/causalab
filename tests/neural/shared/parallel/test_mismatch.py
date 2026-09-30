"""The routing-mismatch record shared over the pipeline
(``docs/model_parallelism.md`` §6.5, §8.3; ``parallel/mismatch.py``).

The record is packed as ``[layer, example, mismatched, slots]`` rows and
unpacked under the write's name — a round trip for every record hypothesis
draws. ``shared_mismatch`` is the identity at a group of one without a call
on the collective (the refusing collective proves it); under the simulated
world at ``pp=2`` and ``pp=4`` the owner's record for each write lands on
every stage, whatever the stages held before, in the caller's order; a
rank that skips one share is a typed refusal (``Abandoned``), never a hang.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.mismatch import (
    COLUMNS,
    RoutingMismatch,
    pack,
    shared_mismatch,
    unpack,
)
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Abandoned,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

WRITES = ("mask_routed", "mask_other", "patch")

counts = st.tuples(st.integers(0, 64), st.integers(1, 64))
records = st.dictionaries(
    st.tuples(st.sampled_from(WRITES), st.integers(0, 40), st.integers(0, 32)),
    counts,
    max_size=24,
)


@pytest.mark.property
class TestPackUnpack:
    @_SETTINGS
    @given(records=records, write=st.sampled_from(WRITES))
    def test_pack_then_unpack_is_the_writes_entries(
        self, records: RoutingMismatch, write: str
    ) -> None:
        packed = pack(records, write)
        assert packed.dtype == torch.int64 and packed.shape[1] == COLUMNS
        assert unpack(write, packed) == {
            k: v for k, v in records.items() if k[0] == write
        }

    @_SETTINGS
    @given(records=records, write=st.sampled_from(WRITES))
    def test_the_rows_are_sorted_by_layer_and_example(
        self, records: RoutingMismatch, write: str
    ) -> None:
        rows = pack(records, write)[:, :2].tolist()
        assert rows == sorted(rows)

    def test_a_write_with_nothing_recorded_packs_empty(self) -> None:
        packed = pack({("other", 1, 0): (2, 10)}, "mask_routed")
        assert packed.shape == (0, COLUMNS)
        assert unpack("mask_routed", packed) == {}

    def test_a_tensor_of_the_wrong_width_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=f"\\(n, {COLUMNS}\\)"):
            unpack("mask_routed", torch.zeros((2, 3), dtype=torch.int64))


@pytest.mark.unit
class TestGroupOfOne:
    def test_the_identity_without_touching_the_collective(self) -> None:
        mine: RoutingMismatch = {("mask_routed", 3, 0): (4, 10)}
        out = shared_mismatch(mine, [("mask_routed", 0)], RefusingCollective())
        assert out == mine and out is not mine
        assert shared_mismatch(mine, [("mask_routed", 0)], SOLO) == mine

    def test_no_owners_is_the_identity_at_any_size(self) -> None:
        world = SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=0)
        mine: RoutingMismatch = {("mask_routed", 3, 0): (4, 10)}
        assert world.run(lambda rank, c: shared_mismatch(mine, [], c)) == [mine, mine]
        assert not world.transcript


def _held(rank: int) -> RoutingMismatch:
    """What each stage recorded before the share: the owner of each write
    its real counts, another stage nothing or a stale entry."""
    held: RoutingMismatch = {}
    if rank == 1:
        held[("mask_routed", 3, 0)] = (4, 10)
        held[("mask_routed", 3, 1)] = (0, 10)
    if rank == 0:
        held[("patch", 0, 0)] = (1, 8)
        # stale: an entry for a write another stage owns
        held[("mask_routed", 3, 0)] = (99, 99)
    return held


@pytest.mark.property
class TestSharedUnderSimulation:
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_every_stage_ends_with_the_owners_record_for_each_write(
        self, schedule: Schedule
    ) -> None:
        owners = [("mask_routed", 1), ("patch", 0)]
        world = SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=schedule)
        results = world.run(lambda rank, c: shared_mismatch(_held(rank), owners, c))
        expected = {
            ("mask_routed", 3, 0): (4, 10),
            ("mask_routed", 3, 1): (0, 10),
            ("patch", 0, 0): (1, 8),
        }
        assert results == [expected, expected]
        assert {e.op for e in world.transcript} == {"broadcast"}

    def test_four_stages_with_the_owner_in_the_middle(self) -> None:
        def program(rank: int, c: Collective) -> RoutingMismatch:
            held: RoutingMismatch = (
                {("mask_routed", 2, i): (i, 5) for i in range(3)} if rank == 2 else {}
            )
            return shared_mismatch(held, [("mask_routed", 2)], c)

        world = SimulatedWorld(groups_for(4, pipeline=4), world=4, schedule=3)
        results = world.run(program)
        assert all(
            r == {("mask_routed", 2, i): (i, 5) for i in range(3)} for r in results
        )

    def test_a_rank_that_skips_the_share_is_a_refusal_never_a_hang(self) -> None:
        def program(rank: int, c: Collective) -> RoutingMismatch:
            if rank == 1:
                return {}
            return shared_mismatch(_held(rank), [("mask_routed", 1)], c)

        world = SimulatedWorld(groups_for(2, pipeline=2), world=2, schedule=0)
        with pytest.raises(Abandoned, match="rank 1 finished"):
            world.run(program)
