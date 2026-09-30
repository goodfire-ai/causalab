"""Who publishes, and which points a replica runs (``docs/model_parallelism.md``
§3, §8.3): the torch-free half of the SPMD launcher.

``unit``: the ``Solo`` publisher is world 1 — it publishes, joins, and its
gather is the identity; the launcher vocabulary is the receipt's closed
``LAUNCHERS``; a replica count above the point count is refused naming
``--parallel.data`` beside its twin. ``property``: the point shards of a
selection partition it in order, sizes differing by at most one with the
first shards longer (the workflow's ``fan_out.over.shards`` arithmetic), and
compose with an authored ``--points`` range — the replica shards the
selected range, never the campaign.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.publish import (
    LAUNCHERS,
    SOLO,
    Publisher,
    Solo,
    is_joiner,
    point_shard,
    point_shards,
)
from causalab.protocol.publish import LAUNCHERS as RUN_LAUNCHERS
from causalab.protocol.receipt import parse_points

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestSolo:
    pytestmark = pytest.mark.unit

    def test_solo_is_world_one(self) -> None:
        assert isinstance(SOLO, Solo)
        assert SOLO.launcher == "solo"
        assert SOLO.replica == 0 and SOLO.replicas == 1
        assert SOLO.publish is True
        assert is_joiner(SOLO)
        assert isinstance(SOLO, Publisher)

    def test_the_solo_gather_is_the_identity(self) -> None:
        payload = object()
        gathered = SOLO.gather(payload)
        assert gathered is not None and list(gathered) == [payload]

    def test_the_launcher_vocabulary_is_the_receipts(self) -> None:
        assert LAUNCHERS == ("solo", "spawned", "joined")
        assert RUN_LAUNCHERS is LAUNCHERS

    def test_the_joiner_is_the_publisher_of_replica_zero(self) -> None:
        class _At:
            launcher = "spawned"
            replicas = 2

            def __init__(self, replica: int, publish: bool) -> None:
                self.replica = replica
                self.publish = publish

            def gather(self, payload):  # pragma: no cover - never called here
                return None

        assert is_joiner(_At(0, True))
        assert not is_joiner(_At(1, True))
        assert not is_joiner(_At(0, False))


class TestPointShards:
    pytestmark = pytest.mark.unit

    def test_one_replica_runs_the_whole_selection(self) -> None:
        assert point_shards(range(5), 1) == (range(5),)
        assert point_shard(range(5), 0, 1) == range(5)

    def test_the_remainder_goes_to_the_first_shards(self) -> None:
        # the workflow's arithmetic (fan_out.expand): 7 points over 3 replicas
        # are 3, 2, 2 — never an empty replica
        assert point_shards(range(7), 3) == (range(0, 3), range(3, 5), range(5, 7))

    def test_a_shard_of_an_authored_range_stays_inside_it(self) -> None:
        selected = parse_points("2:6", 10)
        assert point_shards(selected, 2) == (range(2, 4), range(4, 6))
        assert point_shard(selected, 1, 2) == range(4, 6)

    def test_more_replicas_than_points_is_refused_naming_the_axis(self) -> None:
        with pytest.raises(ProtocolError) as err:
            point_shards(range(3), 4)
        assert err.value.code == "P4"
        message = str(err.value)
        assert "--parallel.data" in message
        assert "4" in message and "3" in message
        # the twin: as many replicas as points is one point each
        assert point_shards(range(3), 3) == (range(0, 1), range(1, 2), range(2, 3))

    def test_a_replica_outside_the_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            point_shard(range(4), 2, 2)
        with pytest.raises(ValueError):
            point_shard(range(4), -1, 2)

    def test_the_refusal_reads_in_full_and_counts_its_points(self) -> None:
        with pytest.raises(ProtocolError) as err:
            point_shards(range(1), 2)
        assert err.value.path == "--parallel.data"
        assert err.value.message == (
            "dp=2 replicas over 1 selected point — every replica runs at least one "
            "point, so the data axis is at most the point count (shard a smaller "
            "campaign, or select more points)"
        )
        with pytest.raises(ProtocolError) as err:
            point_shards(range(3), 4)
        assert err.value.message.startswith("dp=4 replicas over 3 selected points — ")

    def test_a_replica_count_below_one_and_a_replica_outside_it_are_named(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="replicas must be at least 1, got 0"):
            point_shards(range(4), 0)
        with pytest.raises(ValueError, match=r"replica 2 is outside range\(2\)"):
            point_shard(range(4), 2, 2)


@st.composite
def _selections(draw: st.DrawFn) -> tuple[range, int]:
    """An authored ``--points``-style selection and a replica count that
    does not exceed it."""
    n_points = draw(st.integers(min_value=1, max_value=40))
    start = draw(st.integers(min_value=0, max_value=n_points - 1))
    stop = draw(st.integers(min_value=start + 1, max_value=n_points))
    selected = parse_points(f"{start}:{stop}", n_points)
    replicas = draw(st.integers(min_value=1, max_value=len(selected)))
    return selected, replicas


class TestShardProperties:
    pytestmark = pytest.mark.property

    @_SETTINGS
    @given(_selections())
    def test_the_shards_partition_the_selection_in_order(
        self, drawn: tuple[range, int]
    ) -> None:
        """§10.3 ``data over points``: the replicas' point sets partition
        the campaign — here the selected range — contiguous, in order,
        non-empty, sizes differing by at most one and the first longer."""
        selected, replicas = drawn
        shards = point_shards(selected, replicas)
        assert len(shards) == replicas
        flat = [index for shard in shards for index in shard]
        assert flat == list(selected)
        sizes = [len(shard) for shard in shards]
        assert min(sizes) >= 1
        assert max(sizes) - min(sizes) <= 1
        assert sizes == sorted(sizes, reverse=True)
        for replica, shard in enumerate(shards):
            assert point_shard(selected, replica, replicas) == shard

    @_SETTINGS
    @given(_selections())
    def test_shards_compose_with_points(self, drawn: tuple[range, int]) -> None:
        """The replica shards the ``--points`` range: every shard is inside
        it and sharding the whole campaign would place the same indices only
        when the range is the whole campaign."""
        selected, replicas = drawn
        for shard in point_shards(selected, replicas):
            assert selected.start <= shard.start and shard.stop <= selected.stop
