"""Who publishes under each mode of the data axis (``docs/model_parallelism.md``
§3, §8.3): ``launcher.publishes`` — ``publish_here`` over points, where every
replica holds a shard to hand the joiner, narrowed to replica 0 over rows,
where every replica holds the whole campaign and nothing is joined.

``unit`` on the two-by-two world, ``property`` over mesh geometries of both
modes: exactly ``data`` ranks publish over points, exactly one over rows, and
that one is the joiner — rank 0.
"""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.launcher import (
    Launch,
    publish_here,
    publishes,
)
from causalab.protocol.parallel import DATA_MODES, MeshLayout, ParallelGeometry

from tests._helpers.geometries import mesh_geometries

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: ranks (d, m): 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
POINTS = ParallelGeometry(data=2, tensor=2)
ROWS = ParallelGeometry(data=2, tensor=2, data_mode="rows")


def _launch(rank: int, geometry: ParallelGeometry) -> Launch:
    return Launch("spawned", rank, geometry.world, rank)


@pytest.mark.unit
class TestPublishes:
    def test_over_points_every_replicas_first_rank_publishes(self) -> None:
        assert [publishes(_launch(r, POINTS), POINTS) for r in range(4)] == [
            True,
            False,
            True,
            False,
        ]

    def test_over_rows_the_joiner_alone_publishes(self) -> None:
        assert [publishes(_launch(r, ROWS), ROWS) for r in range(4)] == [
            True,
            False,
            False,
            False,
        ]

    def test_publish_here_is_unchanged_by_the_mode(self) -> None:
        layout = MeshLayout(ROWS)
        assert [publish_here(_launch(r, ROWS), layout) for r in range(4)] == [
            True,
            False,
            True,
            False,
        ]

    def test_a_pure_rows_world_publishes_from_rank_zero(self) -> None:
        geometry = ParallelGeometry(data=3, data_mode="rows")
        assert [publishes(_launch(r, geometry), geometry) for r in range(3)] == [
            True,
            False,
            False,
        ]

    def test_a_launch_outside_the_world_is_refused(self) -> None:
        with pytest.raises(ValueError):
            publishes(Launch("spawned", 4, 4, 0), POINTS)


@pytest.mark.property
@_SETTINGS
@given(mesh_geometries(), st.sampled_from(DATA_MODES))
def test_points_publishes_one_rank_per_replica_and_rows_publishes_the_joiner(
    geometry: ParallelGeometry, mode: str
) -> None:
    geometry = dataclasses.replace(geometry, data_mode=mode)  # type: ignore[arg-type]
    layout = MeshLayout(geometry)
    publishing = [
        rank
        for rank in range(geometry.world)
        if publishes(Launch("joined", rank, geometry.world, rank), geometry)
    ]
    if mode == "points":
        assert len(publishing) == geometry.data
        assert tuple(publishing) == layout.group_of(0, "data")
    else:
        assert publishing == [0]
    # rows never publishes from a rank points would not
    assert all(
        publish_here(Launch("joined", r, geometry.world, r), layout) for r in publishing
    )
