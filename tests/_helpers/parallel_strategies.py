"""Hypothesis strategies for the parallelism properties (``docs/model_parallelism.md`` §10.3).

Torch-free at import: a drawn tensor is a `ShardedTensor` or
`TensorSpec` spec whose ``tensor()`` builds the ``torch.Tensor``
inside the property. [`placements`][causalab.neural.shared.parallel.placements] alone reaches the torch side — the
placement table as code, ``placements.placement_for`` — and imports it when
it draws, never when this module loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from hypothesis import strategies as st

from causalab.neural.shared.parallel.placement import (
    AXES,
    Axis,
    ExpertLocal,
    Placement,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from causalab.protocol.parallel import MeshLayout, ParallelGeometry
from causalab.protocol.registry import KV_REPLICATED, STYLES, PlanRow

if TYPE_CHECKING:
    from tests._helpers.simulated_world.schedule import Schedule

Layout = dict[Axis, tuple[tuple[int, ...], ...]]
#: The side of a module a tap sits on — the placement table's spelling.
Side = Literal["input", "output"]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _partition(world: int, size: int, strided: bool) -> tuple[tuple[int, ...], ...]:
    """``world / size`` groups of ``size`` ranks: contiguous blocks, or every
    ``world / size``-th rank (the layout of an outer mesh axis)."""
    if strided:
        stride = world // size
        return tuple(tuple(range(first, world, stride)) for first in range(stride))
    return tuple(tuple(range(first, first + size)) for first in range(0, world, size))


@st.composite
def group_layouts(draw: st.DrawFn, world: int, axes: Sequence[Axis] = AXES) -> Layout:
    """Per axis a partition of ``range(world)`` into equal groups — contiguous
    or strided — with a size dividing ``world``."""
    layout: Layout = {}
    for axis in axes:
        size = draw(st.sampled_from(_divisors(world)))
        strided = draw(st.booleans())
        layout[axis] = _partition(world, size, strided)
    return layout


def layout_of(geometry: ParallelGeometry) -> Layout:
    """The §2 mesh laid out by [`MeshLayout`][causalab.protocol.parallel.MeshLayout] — the groups the production
    mesh builds — as the layout a simulated world takes."""
    layout = MeshLayout(geometry)
    return {axis: layout.groups(axis) for axis in AXES}


@st.composite
def small_mesh_geometries(draw: st.DrawFn, max_world: int = 8) -> ParallelGeometry:
    """A mesh geometry of at most ``max_world`` ranks — a world with a
    divisor between one and itself, so no axis is forced to span it — the
    ``model`` group drawn first (it is the one the placements shard over),
    then the pipeline and context axes out of what it left, the data axis
    the rest. ``model`` is ``max(tensor, expert)``: one of the two spans it
    and the other divides it."""
    # two first (the smallest world with a group above one, the shrink
    # target), then one, then the worlds whose factors give more than one
    # axis something to do; a prime above three would only spend ranks
    worlds = [2, 1] + [
        w for w in range(3, max_world + 1) if w == 3 or len(_divisors(w)) > 2
    ]
    world = draw(st.sampled_from(worlds))
    # the model group spans the world first: the simplest geometry a failure
    # shrinks to is the one where every rank is in the placement's group
    model = draw(st.sampled_from(sorted(_divisors(world), reverse=True)))
    pipeline = draw(st.sampled_from(_divisors(world // model)))
    context = draw(st.sampled_from(_divisors(world // (model * pipeline))))
    data = world // (model * pipeline * context)
    other = draw(st.sampled_from(_divisors(model)))
    spans = draw(st.sampled_from(["tensor", "expert", "both"]))
    tensor = model if spans in ("tensor", "both") else other
    expert = model if spans in ("expert", "both") else other
    return ParallelGeometry(
        data=data, pipeline=pipeline, context=context, tensor=tensor, expert=expert
    )


@dataclass(frozen=True)
class TensorSpec:
    """A small fp32 tensor: its shape and its values, row-major."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def tensor(self) -> Any:
        import torch

        return torch.tensor(self.values, dtype=torch.float32).reshape(self.shape)


#: The experts styles whose interior the routed dispatch taps (§6.3).
EXPERTS_STYLES: frozenset[str] = frozenset({"grouped_gemm", "moe_tp_experts"})
#: The styles a plan puts on the expert axis; every other style is tensor-parallel.
EXPERT_AXIS_STYLES: frozenset[str] = frozenset({"grouped_gemm", "ep_router"})
#: The router's three outputs, and the untyped tap (``None``): what
#: ``placement_for`` takes as ``tuple_index``; the scores are the one
#: output the table places ``ExpertLocal``.
ROUTER_OUTPUTS: tuple[int | None, ...] = (None, 0, 1, 2)
ROUTER_SCORES = 1
#: The tensor-parallel styles and the side of the module the table shards
#: on (``replicated_with_grad_allreduce`` on both).
SHARDED_SIDES: Mapping[str, tuple[Side, ...]] = {
    "colwise": ("output",),
    "packed_colwise": ("output",),
    KV_REPLICATED: ("output",),
    "rowwise": ("input",),
    "replicated_with_grad_allreduce": ("input", "output"),
}


def innermost(placement: Placement) -> Placement:
    """The row's own placement under the stage and sequence wrappers."""
    while isinstance(placement, (StageLocal, SequenceSharded)):
        placement = placement.inner
    return placement


def placement_axes(placement: Placement) -> frozenset[Axis]:
    """Every mesh axis ``placement`` names, wrappers included."""
    axes: set[Axis] = set()
    while True:
        group = getattr(placement, "group", None)
        if group is not None:
            axes.add(group)
        if isinstance(placement, (StageLocal, SequenceSharded)):
            placement = placement.inner
            continue
        return frozenset(axes)


def components(
    layout: Layout, world: int, axes: frozenset[Axis]
) -> tuple[tuple[int, ...], ...]:
    """The ranks a placement ties together — one ``whole`` per component:
    the connected components of "shares a group on one of ``axes``",
    sorted by first rank."""
    parent = list(range(world))

    def root(rank: int) -> int:
        while parent[rank] != rank:
            parent[rank] = parent[parent[rank]]
            rank = parent[rank]
        return rank

    for axis in sorted(axes):
        for group in layout[axis]:
            for rank in group[1:]:
                parent[root(rank)] = root(group[0])
    by_root: dict[int, list[int]] = {}
    for rank in range(world):
        by_root.setdefault(root(rank), []).append(rank)
    return tuple(
        sorted((tuple(ranks) for ranks in by_root.values()), key=lambda g: g[0])
    )


@dataclass(frozen=True)
class PlacementCase:
    """One §4 placement, drawn from the table as code, in the world it lives in.

    ``row`` and ``side`` (with ``interior`` and ``tuple_index`` for the
    experts and the router) are what ``placements.placement_for`` was asked;
    ``placement`` is its answer, wrapped in the sequence chunk under
    ``context > 1`` and the stage under ``pipeline > 1`` (§4 "compose, do
    not replace"). Every component — the ranks the placement ties together —
    has one global tensor, and for an ``ExpertLocal`` one routing table.
    """

    geometry: ParallelGeometry
    layout: Layout
    row: PlanRow | None
    side: Side
    interior: bool
    tuple_index: int | None
    placement: Placement
    remapped_routing: bool
    components: tuple[tuple[int, ...], ...]
    globals_by_component: dict[tuple[int, ...], TensorSpec]
    routing_by_component: dict[tuple[int, ...], tuple[tuple[int, ...], ...]]
    num_experts: int | None

    @property
    def world(self) -> int:
        return self.geometry.world

    def component_of(self, rank: int) -> tuple[int, ...]:
        return next(c for c in self.components if rank in c)

    def global_of(self, rank: int) -> TensorSpec:
        return self.globals_by_component[self.component_of(rank)]

    def routing_of(self, rank: int) -> tuple[tuple[int, ...], ...] | None:
        return self.routing_by_component.get(self.component_of(rank))


def _values(draw: st.DrawFn, shape: Sequence[int]) -> TensorSpec:
    count = math.prod(shape)
    finite = st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, width=32)
    return TensorSpec(
        tuple(shape), tuple(draw(st.lists(finite, min_size=count, max_size=count)))
    )


@st.composite
def placements(
    draw: st.DrawFn,
    styles: Sequence[str | None] = (None, *sorted(STYLES)),
    max_world: int = 8,
) -> PlacementCase:
    """A placement with a tensor whose sharded axis divides the group (§10.3).

    The style is one of the registry's `STYLES` — or ``None``, a
    module no row names — on the axis a plan puts it on (the experts and the
    router on ``expert``, the rest on ``tensor``; ``kv_replicated`` with a
    ``repeat`` dividing the tensor group), asked of ``placement_for`` on a
    drawn side — the experts' output asked for its interior, the router's
    for a drawn output position. ``moe_tp_experts`` is drawn on the tensor
    axis, where the table replicates its boundary, and its interior — the
    one branch of ``experts_placement`` answering ``PartialSum`` and a
    slotted ``Sharded`` — is never asked for: ``fragment(whole(x)) == x`` is
    false for a partial sum (the summands are not recoverable from their
    sum), and ``test_autograd``'s ``PartialSum`` case is where its forward
    and the ``sum_for_edit`` / ``edit_summand`` pairing are held. The tensor
    is ``(batch, position, feature)`` — the feature axis ``size / repeat``
    chunks wide under a shard, times its ``slots`` (``packed_colwise``'s two
    runs), the position axis
    ``context`` chunks long under the sequence wrap — or the experts'
    token-major ``(tokens, top_k · d)`` with a routing table per component
    and ``num_experts`` a multiple of the expert group. A frameless sequence
    chunk cannot unfold a flat token axis, so the experts' interior is not
    wrapped in it (``fragments._frameless``; the framed compositions are
    hand-written tests).
    """
    from causalab.neural.shared.parallel.placements import placement_for, sequenced
    from causalab.protocol.registry.shapes import bsd

    geometry = draw(small_mesh_geometries(max_world))
    layout = layout_of(geometry)
    # most rows of the table are replicated on most sides, so the draw is
    # balanced over what the seam does: any row on any side, a sharding row
    # on its sharded side, the experts' interior
    sharding = [s for s in styles if s is not None and s in SHARDED_SIDES]
    experts = [s for s in styles if s is not None and s in EXPERT_AXIS_STYLES]
    focus = draw(
        st.sampled_from(
            ["any"]
            + (["sharded"] if sharding else [])
            + (["expert"] if experts else [])
        )
    )
    side: Side
    if focus == "sharded":
        style = draw(st.sampled_from(sharding))
        side = draw(st.sampled_from(SHARDED_SIDES[style]))
    elif focus == "expert":
        style = draw(st.sampled_from(experts))
        side = "output"
    else:
        style = draw(st.sampled_from(list(styles)))
        side = draw(st.sampled_from(("input", "output")))
    row: PlanRow | None = None
    axis = "expert" if style in EXPERT_AXIS_STYLES else "tensor"
    if style is not None:
        repeat = (
            draw(st.sampled_from(_divisors(geometry.tensor)))
            if style == KV_REPLICATED
            else 1
        )
        row = PlanRow(style, axis, repeat)
    # the experts' output is asked for the interior (§6.3), the one place the
    # table answers ``ExpertLocal``; its module boundary is replicated like
    # every boundary the table replicates, which the other styles draw
    interior = style in EXPERTS_STYLES and axis == "expert" and side == "output"
    tuple_index: int | None = None
    if style == "ep_router":
        tuple_index = (
            ROUTER_SCORES
            if focus == "expert"
            else draw(st.sampled_from(ROUTER_OUTPUTS))
        )
    placed = placement_for(
        row, side, interior=interior, axis=-1, tuple_index=tuple_index
    )
    inner = placed.placement

    routing_by_component: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    num_experts: int | None = None
    top_k = 0
    if isinstance(inner, ExpertLocal):
        tokens = draw(st.integers(1, 4))
        top_k = draw(st.integers(1, 3))
        width = draw(st.integers(1, 3))
        num_experts = geometry.expert * draw(st.integers(1, 3))
        shape: tuple[int, ...] = (tokens, top_k * width)
        placement: Placement = inner
    else:
        if isinstance(inner, Sharded):
            # ``slots`` runs, each ``size / repeat`` chunks wide
            width = (
                (geometry.tensor // inner.repeat)
                * inner.slots
                * draw(st.integers(1, 2))
            )
        else:
            width = draw(st.integers(1, 4))
        batch = draw(st.integers(1, 2))
        chunk = draw(st.integers(1, 2 if geometry.context < 4 else 1))
        shape = (batch, geometry.context * chunk, width)
        if geometry.context > 1:
            placed = sequenced(placed, bsd(width))
        placement = placed.placement
    if geometry.pipeline > 1:
        stage = draw(st.integers(0, geometry.pipeline - 1))
        placement = StageLocal(stage, inner=placement)  # type: ignore[arg-type]

    parts = components(layout, geometry.world, placement_axes(placement))
    globals_by_component = {part: _values(draw, shape) for part in parts}
    if num_experts is not None:
        for part in parts:
            routing_by_component[part] = draw(
                routing_tables(num_experts, top_k, tokens=shape[0])
            )
    return PlacementCase(
        geometry=geometry,
        layout=layout,
        row=row,
        side=side,
        interior=interior,
        tuple_index=tuple_index,
        placement=placement,
        remapped_routing=placed.remapped_routing,
        components=parts,
        globals_by_component=globals_by_component,
        routing_by_component=routing_by_component,
        num_experts=num_experts,
    )


def seeds() -> st.SearchStrategy[int]:
    """Schedule seeds (§10.5): a failing interleaving shrinks to a small seed."""
    return st.integers(min_value=0, max_value=2**32)


@dataclass(frozen=True)
class ShardedTensor:
    """A small fp32 tensor and the axis a group of ``group_size`` ranks shards it along."""

    shape: tuple[int, ...]
    axis: int
    group_size: int
    values: tuple[float, ...]

    def tensor(self) -> Any:
        import torch

        return torch.tensor(self.values, dtype=torch.float32).reshape(self.shape)


@st.composite
def sharded_tensors(draw: st.DrawFn, group_size: int) -> ShardedTensor:
    """Rank 1–3, sides 1–3, the chosen axis a multiple of ``group_size``."""
    ndim = draw(st.integers(min_value=1, max_value=3))
    sides = list(draw(st.lists(st.integers(1, 3), min_size=ndim, max_size=ndim)))
    axis = draw(st.integers(min_value=0, max_value=ndim - 1))
    sides[axis] = group_size * draw(st.integers(min_value=1, max_value=2))
    shape = tuple(sides)
    count = math.prod(shape)
    finite = st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, width=32)
    values = draw(st.lists(finite, min_size=count, max_size=count))
    return ShardedTensor(shape, axis, group_size, tuple(values))


@st.composite
def routing_tables(
    draw: st.DrawFn,
    num_experts: int,
    top_k: int,
    *,
    tokens: int | None = None,
    sentinel: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """A ``(tokens, top_k)`` table of global expert ids — ``tokens`` rows,
    one to five when unspecified — as the placement seam must take it: any
    expert in any slot, repeats included, and ``sentinel``
    (``fragments.NO_SLOT``, a slot the router filled with nothing) admitted
    where one is given. ``torch.tensor(table, dtype=torch.int64)`` is the
    tensor."""
    rows = tokens if tokens is not None else draw(st.integers(1, 5))
    entry = st.integers(0, num_experts - 1)
    if sentinel is not None:
        entry = st.one_of(entry, st.just(sentinel))
    row = st.lists(entry, min_size=top_k, max_size=top_k).map(tuple)
    return tuple(draw(st.lists(row, min_size=rows, max_size=rows)))


@dataclass(frozen=True)
class FireDeclaration:
    member: str
    count: int
    stage: int


def fire_declarations(
    max_members: int = 6, stages: int = 4
) -> st.SearchStrategy[tuple[FireDeclaration, ...]]:
    """Members with declared fire counts and the pipeline stage that owns each."""
    names = st.from_regex(r"[a-z][a-z0-9_]{0,7}", fullmatch=True)
    one = st.builds(
        FireDeclaration,
        member=names,
        count=st.integers(min_value=1, max_value=8),
        stage=st.integers(min_value=0, max_value=stages - 1),
    )
    return st.lists(
        one, min_size=1, max_size=max_members, unique_by=lambda d: d.member
    ).map(tuple)


def memory_readings(
    world: int, probes: int
) -> st.SearchStrategy[dict[int, tuple[tuple[int, int], ...]]]:
    """Per rank ``probes`` scripted ``(peak, available)`` readings in bytes."""
    reading = st.tuples(
        st.integers(min_value=1, max_value=2**20),
        st.integers(min_value=1, max_value=2**24),
    )
    script = st.lists(reading, min_size=probes, max_size=probes).map(tuple)
    return st.fixed_dictionaries({rank: script for rank in range(world)})


def schedules(max_len: int = 64, max_rank: int = 8) -> st.SearchStrategy[Schedule]:
    """Rank interleavings for ``SimulatedWorld(schedule=…)`` (§10.5): a tape of
    up to ``max_len`` entries below ``max_rank`` — the ``i``-th *choice* (a
    switch with more than one runnable rank) picks the ``tape[i] % len``-th,
    the choices past its end the lowest — or, as often (``st.one_of`` is
    uniform over its branches), a seed, a whole run's worth of randomness the
    tapes' short draws do not reach. A failure shrinks towards the empty tape,
    rank order."""
    tapes = st.lists(st.integers(min_value=0, max_value=max_rank - 1), max_size=max_len)
    return st.one_of(tapes, seeds())
