"""Hypothesis strategies over [`ParallelGeometry`][causalab.protocol.parallel.ParallelGeometry]
(``docs/model_parallelism.md`` §10.3).

``dividing_geometries(info)`` draws geometries every §2 divisibility fact of a
[`ModelInfo`][causalab.protocol.registry.models.ModelInfo] holds for; ``any_geometries``
draws small geometries with no regard for a model, so a share of them is
deliberately non-dividing; ``geometries(info)`` mixes the two. ``mesh_geometries``
draws what [`MeshLayout`][causalab.protocol.parallel.MeshLayout] accepts — the two
model-group facts only — with a bounded world, so a property over every rank
stays cheap.
"""

from __future__ import annotations

from hypothesis import strategies as st

from causalab.protocol.parallel import GEOMETRY_AXES, ParallelGeometry
from causalab.protocol.registry import ModelInfo


def divisors(n: int) -> list[int]:
    """Every positive divisor of ``n``, ascending."""
    return [d for d in range(1, n + 1) if n % d == 0]


def _model_group_divides(geometry: ParallelGeometry) -> bool:
    return (
        geometry.model % geometry.tensor == 0 and geometry.model % geometry.expert == 0
    )


def dividing_geometries(info: ModelInfo) -> st.SearchStrategy[ParallelGeometry]:
    """Geometries that satisfy every divisibility fact of ``info`` — the
    tensor axis a divisor of the heads that shards the KV heads or replicates
    them (§6.6); the context axis has no model fact to satisfy (§8.4) and is
    drawn freely."""
    tensor = st.sampled_from(
        [
            d
            for d in divisors(info.num_heads)
            if info.num_kv_heads % d == 0 or d % info.num_kv_heads == 0
        ]
    )
    expert = (
        st.just(1)
        if info.num_experts is None
        else st.sampled_from(divisors(info.num_experts))
    )
    layers = info.num_layers
    pipeline = st.sampled_from(
        sorted({1, 2, max(1, layers // 2), layers} & set(range(1, layers + 1)))
    )
    return st.builds(
        ParallelGeometry,
        data=st.integers(min_value=1, max_value=3),
        pipeline=pipeline,
        context=st.integers(min_value=1, max_value=3),
        tensor=tensor,
        expert=expert,
    ).filter(_model_group_divides)


def any_geometries(bound: int = 5) -> st.SearchStrategy[ParallelGeometry]:
    """Every axis in ``1..bound``, no model in view — most draws break some fact."""
    axis = st.integers(min_value=1, max_value=bound)
    return st.builds(ParallelGeometry, **{name: axis for name in GEOMETRY_AXES})


def geometries(info: ModelInfo) -> st.SearchStrategy[ParallelGeometry]:
    """The dividing draws and the indifferent ones, mixed."""
    return st.one_of(dividing_geometries(info), any_geometries())


def mesh_geometries(bound: int = 4) -> st.SearchStrategy[ParallelGeometry]:
    """Geometries a mesh can be laid out for: ``tensor | model`` and
    ``expert | model`` hold; the world stays at most ``bound ** 4``."""
    axis = st.integers(min_value=1, max_value=bound)
    return st.builds(
        ParallelGeometry,
        data=axis,
        pipeline=axis,
        context=axis,
        tensor=axis,
        expert=axis,
    ).filter(_model_group_divides)
