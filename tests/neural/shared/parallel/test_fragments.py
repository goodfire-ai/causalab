"""The placement seam: ``whole`` / ``fragment`` and the routing reconstruction.

``docs/model_parallelism.md`` §4, §6.1–6.3 and the §10.3 rows ``Fragments``,
``head slices``, ``routing reconstruction`` and ``gaussian.axis``. Every
property is asserted **bit for bit** (``torch.equal``) on CPU fp32: a
placement round trip that is only close is a placement that is wrong.

Three tiers, one class each: the ``property`` tier holds the §10.3 rows, the
``unit`` tier the hand-written mutations (each applies the named change
through a seam or a monkeypatch and asserts the property *fails*, the
repository's convention) and the refusals by name.

The placements come from ``tests/_helpers/parallel_strategies.py:placements``
— a registry style asked of the table as code, ``placement_for``, in a small
mesh geometry — and the expected fragments are spelled by hand here
(``_expected_local``: ``narrow``, a masked ``where``, an empty tensor, over
``MeshLayout``'s coordinates), so the round trips are checked against an
independent statement of §4, not against the implementation's own output.
The ranks run on ``SimulatedWorld`` (§10.2), one thread per rank under a
drawn schedule; hypothesis draws only on the test thread, so a property
draws its tensors and layout in the test body and hands them to the rank
program. `Case` / `placement_cases` is the same strategy seen
over one group axis, the view ``test_autograd``'s world-1 oracle enumerates
ranks by.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping, Sequence, TypeVar

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel import fragments
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import (
    NO_SLOT,
    Fragments,
    PlacementError,
    expert_slot_mask,
    fragment,
    reconstruct_routing,
    remap_routing,
    whole,
)
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    Axis,
    ExpertLocal,
    Placement,
    Replicated,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.parallel import MeshLayout, ParallelGeometry
from causalab.protocol.registry.shapes import bsd
from tests._helpers import parallel_strategies as ps
from tests._helpers.parallel_strategies import PlacementCase, TensorSpec
from tests._helpers.refusing_collective import RefusingCollective
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.simulated_world import collective as simulated_collective
from tests._helpers.simulated_world import world as world_module
from tests._helpers.simulated_world.rendezvous import Rendezvous

# The repository's settings idiom (tests/tasks/graph_walk/test_graph_walk.py).
_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

T = TypeVar("T")
Program = Callable[[int, Collective], T]


# --------------------------------------------------------------------------- #
# worlds: the simulator over a layout, and the §2 mesh by named axes
# --------------------------------------------------------------------------- #


def _world(
    layout: Mapping[Axis, tuple[tuple[int, ...], ...]],
    world: int,
    schedule: Schedule = (),
) -> SimulatedWorld:
    return SimulatedWorld(layout, world=world, schedule=schedule)


def _mesh(world: int, schedule: Schedule = (), **axes: int) -> SimulatedWorld:
    """The §2 mesh over ``world`` ranks, the named axes above one."""
    return _world(groups_for(world, **axes), world, schedule)


def _case_world(case: PlacementCase, schedule: Schedule = ()) -> SimulatedWorld:
    return _world(case.layout, case.world, schedule)


def _on_a_pair(call: Callable[[Collective], T], axis: Axis = "tensor") -> list[T]:
    """``call(collective)`` on both ranks of a two-rank group on ``axis``, the
    results in rank order; a rank's refusal is re-raised as itself."""
    try:
        return _mesh(2, **{axis: 2}).run(lambda rank, c: call(c))
    except RankFailed as failed:
        raise failed.error from failed


# --------------------------------------------------------------------------- #
# §4 spelled by hand: what a rank holds of its component's global tensor
# --------------------------------------------------------------------------- #


def _randn(shape: Sequence[int], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32)


def _spec(shape: Sequence[int], seed: int) -> TensorSpec:
    return TensorSpec(tuple(shape), tuple(_randn(shape, seed).flatten().tolist()))


def _routing_tensor(rows: tuple[tuple[int, ...], ...] | None) -> torch.Tensor | None:
    return None if rows is None else torch.tensor(rows, dtype=torch.int64)


def _expected_expert_local(
    g: torch.Tensor, routing: torch.Tensor, num_experts: int, rank: int, size: int
) -> torch.Tensor:
    """§6.3 spelled by hand: keep the slots whose expert this rank owns."""
    num_local = num_experts // size
    top_k = routing.shape[-1]
    d = g.shape[-1] // top_k
    owned = (routing // num_local) == rank  # (tokens, top_k)
    keep = owned.unsqueeze(-1).expand(*routing.shape, d).reshape(g.shape)
    return torch.where(keep, g, torch.zeros((), dtype=g.dtype))


def _expect(
    placement: Placement,
    g: torch.Tensor,
    routing: torch.Tensor | None,
    num_experts: int | None,
    layout: MeshLayout,
    rank: int,
) -> torch.Tensor:
    """The stage rule, then the position chunk, then the row's own placement —
    each over ``MeshLayout``'s coordinates, the independent statement of who
    is where."""
    if isinstance(placement, Replicated):
        return g
    if isinstance(placement, StageLocal):
        group = layout.group_of(rank, placement.group)
        if len(group) > 1 and layout.rank_in(rank, placement.group) != placement.stage:
            return g.new_empty((0, *g.shape[1:]))
        return _expect(placement.inner, g, routing, num_experts, layout, rank)
    if isinstance(placement, SequenceSharded):
        group = layout.group_of(rank, placement.group)
        if len(group) > 1:
            width = g.shape[placement.axis] // len(group)
            start = layout.rank_in(rank, placement.group) * width
            g = g.narrow(placement.axis, start, width)
        return _expect(placement.inner, g, routing, num_experts, layout, rank)
    group = layout.group_of(rank, placement.group)
    size, local = len(group), layout.rank_in(rank, placement.group)
    if size == 1:
        return g
    if isinstance(placement, ExpertLocal):
        assert routing is not None and num_experts is not None
        return _expected_expert_local(g, routing, num_experts, local, size)
    # the hand oracle models the placements the strategy draws; anything else
    # reads as the oracle's gap rather than the property's failure
    assert isinstance(placement, Sharded), (
        f"the hand oracle models no {placement}: extend _expect before widening the draw"
    )
    axis = placement.axis % g.dim()
    if placement.slots > 1:
        # a slotted shard (``packed_colwise``'s gate-and-up runs): the axis is
        # ``slots`` runs of one width, this rank's chunk of every run in rank
        # order — ``repeat`` plays no part, as in ``fragments._narrow``
        runs = g.unflatten(axis, (placement.slots, -1))
        width = runs.shape[axis + 1] // size
        return runs.narrow(axis + 1, local * width, width).flatten(axis, axis + 1)
    chunks = size // placement.repeat
    width = g.shape[axis] // chunks
    return g.narrow(axis, (local // placement.repeat) * width, width)


def _expected_local(case: PlacementCase, rank: int) -> torch.Tensor:
    return _expect(
        case.placement,
        case.global_of(rank).tensor(),
        _routing_tensor(case.routing_of(rank)),
        case.num_experts,
        MeshLayout(case.geometry),
        rank,
    )


def _hand_case(
    geometry: ParallelGeometry,
    placement: Placement,
    spec: TensorSpec,
    routing: tuple[tuple[int, ...], ...] | None = None,
    num_experts: int | None = None,
) -> PlacementCase:
    """A placement case spelled by hand, every component holding ``spec``."""
    layout = ps.layout_of(geometry)
    parts = ps.components(layout, geometry.world, ps.placement_axes(placement))
    return PlacementCase(
        geometry=geometry,
        layout=layout,
        row=None,
        side="output",
        interior=False,
        tuple_index=None,
        placement=placement,
        remapped_routing=False,
        components=parts,
        globals_by_component={part: spec for part in parts},
        routing_by_component={part: routing for part in parts} if routing else {},
        num_experts=num_experts,
    )


def _expert_placements() -> st.SearchStrategy[PlacementCase]:
    """The experts' interior: ``grouped_gemm`` asked on the output side for
    the interior — the draws that answer ``ExpertLocal``."""
    return ps.placements(styles=("grouped_gemm",)).filter(
        lambda case: isinstance(ps.innermost(case.placement), ExpertLocal)
    )


# --------------------------------------------------------------------------- #
# the single-axis view: what ``test_autograd``'s world-1 oracle enumerates by
# --------------------------------------------------------------------------- #

Groups = tuple[tuple[int, ...], ...]

PLACEMENT_KINDS = (
    "replicated",
    "sharded",
    "expert_local",
    "stage_local",
    "sequence_sharded",
)
_KIND_OF: dict[type, str] = {
    Replicated: "replicated",
    Sharded: "sharded",
    ExpertLocal: "expert_local",
    StageLocal: "stage_local",
    SequenceSharded: "sequence_sharded",
}


@dataclasses.dataclass(frozen=True)
class Case:
    """A drawn placement over **one** group axis — every other axis
    singletons — with the global tensor of each group and the local fragment
    every rank is expected to hold: the view ``test_autograd``'s oracle is
    written for, a `PlacementCase` whose components are the groups of
    its one axis."""

    placement: Placement
    world: int
    groups: Groups
    globals_by_group: dict[tuple[int, ...], torch.Tensor]
    locals_by_rank: dict[int, torch.Tensor]
    routing_by_group: dict[tuple[int, ...], torch.Tensor] = dataclasses.field(
        default_factory=dict
    )
    num_experts: int | None = None

    @property
    def group_axis(self) -> Axis:
        return getattr(self.placement, "group", "tensor")

    def group_of(self, rank: int) -> tuple[int, ...]:
        return next(g for g in self.groups if rank in g)

    def global_of(self, rank: int) -> torch.Tensor:
        return self.globals_by_group[self.group_of(rank)]

    def routing_of(self, rank: int) -> torch.Tensor | None:
        return self.routing_by_group.get(self.group_of(rank))


def _single_axis(case: PlacementCase) -> bool:
    """One group axis at most: a wrapper only around a replicated inner."""
    return len(ps.placement_axes(case.placement)) <= 1


def _as_case(case: PlacementCase) -> Case:
    routing = {
        part: torch.tensor(rows, dtype=torch.int64)
        for part, rows in case.routing_by_component.items()
    }
    return Case(
        placement=case.placement,
        world=case.world,
        groups=case.components,
        globals_by_group={
            part: spec.tensor() for part, spec in case.globals_by_component.items()
        },
        locals_by_rank={
            rank: _expected_local(case, rank) for rank in range(case.world)
        },
        routing_by_group=routing,
        num_experts=case.num_experts,
    )


def placement_cases(kinds: Sequence[str] = PLACEMENT_KINDS) -> st.SearchStrategy[Case]:
    """The single-axis draws of `ps.placements` whose outermost
    placement is one of ``kinds``, as `Case`."""
    # ``.get``: a placement the ``Case`` view has no kind for (``PartialSum``,
    # should the strategy ever draw it) is filtered out, not a ``KeyError``
    # inside generation that hypothesis reports as the drawing test's failure
    return (
        ps.placements()
        .filter(lambda c: _single_axis(c) and _KIND_OF.get(type(c.placement)) in kinds)
        .map(_as_case)
    )


# --------------------------------------------------------------------------- #
# The property checks, as functions so the mutation tests can assert they fail.
# --------------------------------------------------------------------------- #


def _round_trip_program(
    case: PlacementCase,
) -> Program[tuple[torch.Tensor, torch.Tensor]]:
    locals_by_rank = {rank: _expected_local(case, rank) for rank in range(case.world)}
    routing_by_rank = {
        rank: _routing_tensor(case.routing_of(rank)) for rank in range(case.world)
    }

    def program(rank: int, collective: Collective) -> tuple[torch.Tensor, torch.Tensor]:
        made_whole = whole(locals_by_rank[rank], case.placement, collective)
        back = fragment(
            made_whole,
            case.placement,
            collective,
            routing=routing_by_rank[rank],
            num_experts=case.num_experts,
        )
        return made_whole, back

    return program


def check_fragment_of_whole_is_identity(
    case: PlacementCase, world: SimulatedWorld
) -> None:
    """``fragment(whole(x)) == x`` on every rank."""
    for rank, (_, back) in enumerate(world.run(_round_trip_program(case))):
        assert torch.equal(back, _expected_local(case, rank)), (rank, case.placement)


def check_whole_identical_on_every_rank(
    case: PlacementCase, world: SimulatedWorld
) -> None:
    """``whole(x)`` is the component's global tensor on every rank."""
    for rank, (made_whole, _) in enumerate(world.run(_round_trip_program(case))):
        assert torch.equal(made_whole, case.global_of(rank).tensor()), (
            rank,
            case.placement,
        )


def check_whole_of_fragment_is_identity(
    case: PlacementCase, world: SimulatedWorld
) -> None:
    """``whole(fragment(g)) == g`` for a global ``g``."""
    routing_by_rank = {
        rank: _routing_tensor(case.routing_of(rank)) for rank in range(case.world)
    }

    def program(rank: int, collective: Collective) -> torch.Tensor:
        piece = fragment(
            case.global_of(rank).tensor(),
            case.placement,
            collective,
            routing=routing_by_rank[rank],
            num_experts=case.num_experts,
        )
        return whole(piece, case.placement, collective)

    for rank, made_whole in enumerate(world.run(program)):
        assert torch.equal(made_whole, case.global_of(rank).tensor()), (
            rank,
            case.placement,
        )


def check_expert_ownership_partitions_slots(
    routing: torch.Tensor, num_experts: int, size: int
) -> None:
    """The ranks' ownership masks are disjoint and cover every routed slot;
    a ``NO_SLOT`` entry is owned by nobody."""
    # through the module attribute, so the ownership mutation reaches it
    owners = torch.stack(
        [
            fragments.expert_slot_mask(routing, num_experts, rank, size).to(torch.int64)
            for rank in range(size)
        ]
    ).sum(dim=0)
    assert torch.equal(owners, (routing != NO_SLOT).to(torch.int64))


def check_reconstruct_inverts_remap(
    table: torch.Tensor, num_experts: int, world: SimulatedWorld
) -> None:
    """``reconstruct(remap(table)) == table`` on every rank, with ``NO_SLOT``
    exactly where every rank holds the sentinel."""
    groups = world.groups["expert"]
    locals_by_rank = {
        rank: remap_routing(table, num_experts, group.index(rank), len(group))
        for group in groups
        for rank in group
    }
    num_local = num_experts // len(groups[0])
    results = world.run(
        lambda rank, collective: reconstruct_routing(
            locals_by_rank[rank], num_experts, collective
        )
    )
    for rank, reconstructed in enumerate(results):
        assert torch.equal(reconstructed, table), rank
        group = world.group_of("expert", rank)
        all_sentinel = torch.stack([locals_by_rank[r] == num_local for r in group]).all(
            dim=0
        )
        assert torch.equal(reconstructed == NO_SLOT, all_sentinel), rank


def _transformers_remap(
    table: torch.Tensor, num_experts: int, rank: int, size: int
) -> torch.Tensor:
    """``EpRouterParallel.transform_output_post_forward`` on the index tensor,
    transcribed from transformers 5.16 — the oracle ``remap_routing`` mirrors."""
    num_local = num_experts // size
    non_local = (table // num_local) != rank
    indices = table.masked_fill(non_local, -1)
    if num_local > 1:
        indices = torch.fmod(indices, num_local)
    else:
        indices = indices.masked_fill(indices > 0, 0).masked_fill(indices < 0, -1)
    return indices.masked_fill(indices == -1, num_local)


# --------------------------------------------------------------------------- #
# property tier
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestFragmentsProperties:
    @_HYPOTHESIS_SETTINGS
    @given(case=ps.placements(), schedule=ps.schedules())
    def test_fragment_of_whole_is_the_identity_for_every_placement(
        self, case: PlacementCase, schedule: Schedule
    ) -> None:
        check_fragment_of_whole_is_identity(case, _case_world(case, schedule))

    @_HYPOTHESIS_SETTINGS
    @given(case=ps.placements(), schedule=ps.schedules())
    def test_whole_is_identical_on_every_rank(
        self, case: PlacementCase, schedule: Schedule
    ) -> None:
        check_whole_identical_on_every_rank(case, _case_world(case, schedule))

    @_HYPOTHESIS_SETTINGS
    @given(case=ps.placements(), schedule=ps.schedules())
    def test_whole_of_fragment_is_the_identity_for_a_global_tensor(
        self, case: PlacementCase, schedule: Schedule
    ) -> None:
        check_whole_of_fragment_is_identity(case, _case_world(case, schedule))

    @_HYPOTHESIS_SETTINGS
    @given(case=_expert_placements())
    def test_expert_slots_are_owned_by_exactly_one_rank_so_the_sum_is_exact(
        self, case: PlacementCase
    ) -> None:
        assert case.num_experts is not None
        size = case.geometry.expert
        layout = MeshLayout(case.geometry)
        for part, rows in case.routing_by_component.items():
            routing = torch.tensor(rows, dtype=torch.int64)
            check_expert_ownership_partitions_slots(routing, case.num_experts, size)
            # and the fragments carry zeros exactly where the rank owns nothing
            g = case.globals_by_component[part].tensor()
            for rank in part:
                piece = _expected_local(case, rank)
                if piece.shape[0] == 0:
                    continue  # another stage's ranks hold nothing
                local = layout.rank_in(rank, "expert")
                mask = expert_slot_mask(routing, case.num_experts, local, size)
                d = g.shape[-1] // routing.shape[-1]
                keep = mask.unsqueeze(-1).expand(*routing.shape, d).reshape(g.shape)
                assert torch.equal(piece[~keep], torch.zeros_like(piece[~keep]))
                assert torch.equal(piece[keep], g[keep])
        check_whole_of_fragment_is_identity(case, _case_world(case))

    @_HYPOTHESIS_SETTINGS
    @given(
        world=st.integers(1, 8),
        experts_per_rank=st.integers(1, 3),
        top_k=st.integers(1, 3),
        schedule=ps.schedules(),
        data=st.data(),
    )
    def test_reconstruct_inverts_remap_and_only_sentinels_map_to_no_slot(
        self,
        world: int,
        experts_per_rank: int,
        top_k: int,
        schedule: Schedule,
        data: st.DataObject,
    ) -> None:
        layout = data.draw(ps.group_layouts(world))
        ep = len(layout["expert"][0])
        num_experts = ep * experts_per_rank
        table = torch.tensor(
            data.draw(ps.routing_tables(num_experts, top_k, sentinel=NO_SLOT)),
            dtype=torch.int64,
        )
        check_reconstruct_inverts_remap(
            table, num_experts, _world(layout, world, schedule)
        )

    @_HYPOTHESIS_SETTINGS
    @given(
        ep=st.integers(1, 4),
        experts_per_rank=st.integers(1, 3),
        top_k=st.integers(1, 3),
        data=st.data(),
    )
    def test_remap_mirrors_transformers_ep_router_parallel(
        self, ep: int, experts_per_rank: int, top_k: int, data: st.DataObject
    ) -> None:
        num_experts = ep * experts_per_rank
        table = torch.tensor(
            data.draw(ps.routing_tables(num_experts, top_k)), dtype=torch.int64
        )
        for rank in range(ep):
            assert torch.equal(
                remap_routing(table, num_experts, rank, ep),
                _transformers_remap(table, num_experts, rank, ep),
            ), rank

    @_HYPOTHESIS_SETTINGS
    @given(
        heads_per_rank=st.integers(1, 4),
        tp=st.integers(1, 4),
        head_dim=st.integers(1, 3),
        batch=st.integers(1, 2),
        positions=st.integers(1, 3),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_global_head_slice_is_the_rank_order_concatenation_of_local_slices(
        self,
        heads_per_rank: int,
        tp: int,
        head_dim: int,
        batch: int,
        positions: int,
        seed: int,
    ) -> None:
        """§6.1: ``Shard(0)`` of a projection's weight is a contiguous block of
        heads per rank, so the ``head:`` slice of the gathered tensor is the
        slice ``sites._head_slice`` computes in the *global* contract, and no
        per-rank head arithmetic is needed. ``per_head = width // space`` is
        that function's formula."""
        num_heads = heads_per_rank * tp
        width = num_heads * head_dim
        g = _randn((batch, positions, width), seed)
        placement = Sharded(axis=-1, group="tensor")
        local_by_rank = {
            rank: g.narrow(
                -1, rank * heads_per_rank * head_dim, heads_per_rank * head_dim
            )
            for rank in range(tp)
        }
        made_whole = _mesh(tp, tensor=tp).run(
            lambda rank, c: whole(local_by_rank[rank], placement, c)
        )
        per_head = width // num_heads
        for rank in range(tp):
            for head in range(num_heads):
                global_slice = slice(
                    head * per_head, (head + 1) * per_head
                )  # sites._head_slice
                owner, local_head = divmod(head, heads_per_rank)
                local_slice = slice(local_head * per_head, (local_head + 1) * per_head)
                assert torch.equal(
                    made_whole[rank][..., global_slice],
                    local_by_rank[owner][..., local_slice],
                )
            assert torch.equal(
                made_whole[rank],
                torch.cat([local_by_rank[r] for r in range(tp)], dim=-1),
            )

    @_HYPOTHESIS_SETTINGS
    @given(
        tp=st.integers(1, 4),
        per_rank=st.integers(1, 3),
        rows=st.integers(1, 4),
        seed=st.integers(0, 2**31 - 1),
        axis_value=st.sampled_from(["tp_duplicated", "tp_split"]),
    )
    def test_gaussian_draw_fragmented_equals_the_world_one_draw_rank_slice(
        self, tp: int, per_rank: int, rows: int, seed: int, axis_value: str
    ) -> None:
        """§6.7: the draw is made over the global feature axis with the
        document's seed on every rank identically, then fragmented like any
        write — for either ``gaussian.axis`` value, so ``axis_value`` does not
        enter the draw."""
        width = tp * per_rank
        world_one = fragments.Fragments(RefusingCollective())
        reference = world_one.fragment(_randn((rows, width), seed), Sharded(axis=-1))
        drawn = _mesh(tp, tensor=tp).run(
            lambda rank, c: fragment(_randn((rows, width), seed), Sharded(axis=-1), c)
        )
        for rank in range(tp):
            assert torch.equal(
                drawn[rank], reference.narrow(-1, rank * per_rank, per_rank)
            ), (rank, axis_value)

    @_HYPOTHESIS_SETTINGS
    @given(case=ps.placements())
    def test_world_one_is_the_identity_without_touching_the_collective(
        self, case: PlacementCase
    ) -> None:
        pieces = Fragments(RefusingCollective())
        x = case.global_of(0).tensor()
        routing = _routing_tensor(case.routing_of(0))
        assert pieces.whole(x, case.placement) is x
        assert pieces.fragment(x, case.placement) is x
        assert (
            pieces.fragment(
                x, case.placement, routing=routing, num_experts=case.num_experts
            )
            is x
        )
        # and a group of one on a wider world is the identity too
        (identity,) = _mesh(1).run(
            lambda rank, c: (
                whole(x, case.placement, c) is x,
                fragment(x, case.placement, c) is x,
            )
        )
        assert identity == (True, True)


# --------------------------------------------------------------------------- #
# unit tier: the §10.3 mutations and the refusals
# --------------------------------------------------------------------------- #


def _sharded_case(axis: int = -1) -> PlacementCase:
    return _hand_case(
        ParallelGeometry(tensor=2),
        Sharded(axis=axis, group="tensor"),
        _spec((2, 2, 4), 7),
    )


def _expert_case() -> PlacementCase:
    return _hand_case(
        ParallelGeometry(expert=2),
        ExpertLocal(group="expert"),
        _spec((3, 2 * 3), 11),
        routing=((0, 3), (1, 2), (3, 0)),
        num_experts=4,
    )


class _OwnChunkFirstRendezvous(Rendezvous):
    """The mutation: each member concatenates from its own chunk on — the
    rank offset that puts chunk ``r`` at position ``r`` is gone."""

    def resolve(self) -> None:
        signature = self.first.signature
        if signature.op != "all_gather":
            return super().resolve()
        ordered = [self.arrivals[rank] for rank in sorted(self.arrivals)]
        for index, arrival in enumerate(ordered):
            rotated = ordered[index:] + ordered[:index]
            arrival.result = torch.cat(
                [a.payload for a in rotated], dim=signature.detail[0]
            ).detach()
        return None


class _ReversedRendezvous(Rendezvous):
    """The mutation: an all-gather concatenated in descending rank order."""

    def resolve(self) -> None:
        signature = self.first.signature
        if signature.op != "all_gather":
            return super().resolve()
        ordered = [self.arrivals[rank] for rank in sorted(self.arrivals, reverse=True)]
        gathered = torch.cat([a.payload for a in ordered], dim=signature.detail[0])
        for arrival in ordered:
            arrival.result = gathered.detach().clone()
        return None


@pytest.mark.unit
class TestNamedMutations:
    """Each hand-applies one §10.3 mutation and asserts the property fails.
    The gather-order mutations reach the simulator through its rendezvous
    seam, the axis mutation through its ``RankCollective``."""

    def test_the_properties_hold_on_the_unmutated_examples(self) -> None:
        for case in (_sharded_case(), _expert_case()):
            check_fragment_of_whole_is_identity(case, _case_world(case))
            check_whole_identical_on_every_rank(case, _case_world(case))
            check_whole_of_fragment_is_identity(case, _case_world(case))

    def test_dropping_the_rank_offset_from_the_gather_order_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Each rank places its own chunk first — the offset that puts chunk r
        # at position r is gone — so the ranks disagree on the whole.
        monkeypatch.setattr(world_module, "Rendezvous", _OwnChunkFirstRendezvous)
        case = _sharded_case()
        with pytest.raises(AssertionError):
            check_whole_identical_on_every_rank(case, _case_world(case))
        with pytest.raises(AssertionError):
            check_fragment_of_whole_is_identity(case, _case_world(case))

    def test_reversing_the_rank_order_fails_the_head_slice_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(world_module, "Rendezvous", _ReversedRendezvous)
        case = _sharded_case()
        with pytest.raises(AssertionError):
            check_whole_identical_on_every_rank(case, _case_world(case))

    def test_gathering_along_the_wrong_axis_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = simulated_collective.RankCollective.all_gather

        def wrong_axis(self, tensor, dim, axis):
            return real(self, tensor, (dim + 1) % tensor.dim(), axis)

        monkeypatch.setattr(
            simulated_collective.RankCollective, "all_gather", wrong_axis
        )
        case = _sharded_case(axis=-1)
        # the gathered shape is wrong, so the round trip either compares
        # unequal or the re-fragment refuses the axis by name (inside the
        # rank, so reported as that rank's failure)
        with pytest.raises((AssertionError, RankFailed)):
            check_fragment_of_whole_is_identity(case, _case_world(case))
        with pytest.raises(AssertionError):
            check_whole_identical_on_every_rank(case, _case_world(case))

    def test_two_ranks_owning_one_slot_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def greedy_mask(routing, num_experts, rank, size):
            owner = routing // (num_experts // size)
            return (owner == rank) | (owner == (rank + 1) % size)

        monkeypatch.setattr(fragments, "expert_slot_mask", greedy_mask)
        case = _expert_case()
        routing = torch.tensor(case.routing_by_component[(0, 1)], dtype=torch.int64)
        with pytest.raises(AssertionError):
            check_expert_ownership_partitions_slots(routing, 4, 2)
        with pytest.raises(AssertionError):
            check_whole_of_fragment_is_identity(case, _case_world(case))

    def test_dropping_the_plus_one_makes_expert_zero_vanish(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Owned slots contribute the bare global id; 0 is then both "expert 0"
        # and "nobody", and the sum maps expert 0 to NO_SLOT.
        def bare_contribution(local_table, num_local, rank):
            owned = local_table != num_local
            return torch.where(
                owned,
                local_table.to(torch.int64) + rank * num_local,
                torch.zeros_like(local_table, dtype=torch.int64),
            )

        def zero_is_no_slot(summed):
            return torch.where(summed == 0, torch.full_like(summed, NO_SLOT), summed)

        monkeypatch.setattr(fragments, "_routing_contribution", bare_contribution)
        monkeypatch.setattr(fragments, "_routing_from_sum", zero_is_no_slot)
        with pytest.raises(AssertionError):
            check_reconstruct_inverts_remap(
                torch.tensor([[0, 3], [1, 2]]), 4, _mesh(2, expert=2)
            )
        # and it is exactly expert 0 that vanishes: a table without it survives
        check_reconstruct_inverts_remap(
            torch.tensor([[1, 3], [1, 2]]), 4, _mesh(2, expert=2)
        )

    def test_mapping_the_sentinel_to_expert_zero_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def sentinel_is_expert_zero(local_table, num_local, rank):
            as_local = torch.where(
                local_table == num_local, torch.zeros_like(local_table), local_table
            )
            return as_local.to(torch.int64) + rank * num_local + 1

        monkeypatch.setattr(fragments, "_routing_contribution", sentinel_is_expert_zero)
        with pytest.raises(AssertionError):
            check_reconstruct_inverts_remap(
                torch.tensor([[0, 3], [1, 2]]), 4, _mesh(2, expert=2)
            )

    def test_drawing_the_gaussian_per_rank_fails(self) -> None:
        tp, per_rank, rows, seed = 2, 3, 4, 5
        reference = _randn((rows, tp * per_rank), seed)
        per_rank_draw = _mesh(tp, tensor=tp).run(
            lambda rank, c: _randn((rows, per_rank), seed)
        )
        assert not all(
            torch.equal(
                per_rank_draw[rank], reference.narrow(-1, rank * per_rank, per_rank)
            )
            for rank in range(tp)
        )


@pytest.mark.unit
class TestRefusals:
    """Every refusal names what did not fit. A refusal is raised inside the
    rank, before any collective; ``_on_a_pair`` re-raises it as itself."""

    def test_fragment_refuses_an_axis_the_group_does_not_divide(self) -> None:
        with pytest.raises(PlacementError, match=r"axis 2 .*3.*2 ranks") as info:
            _on_a_pair(lambda c: fragment(torch.zeros(2, 2, 3), Sharded(axis=2), c))
        assert info.value.placement == Sharded(axis=2)

    def test_fragment_refuses_an_axis_the_tensor_does_not_have(self) -> None:
        with pytest.raises(PlacementError, match=r"axis 3 .*3-D"):
            _on_a_pair(lambda c: fragment(torch.zeros(2, 2, 4), Sharded(axis=3), c))
        with pytest.raises(PlacementError, match=r"axis 1 .*1-D"):
            _on_a_pair(
                lambda c: whole(
                    torch.zeros(4), SequenceSharded(axis=1, group="context"), c
                ),
                "context",
            )

    def test_expert_local_fragment_needs_the_routing_table_and_expert_count(
        self,
    ) -> None:
        with pytest.raises(PlacementError, match="routing"):
            _on_a_pair(
                lambda c: fragment(torch.zeros(3, 4), ExpertLocal(), c, num_experts=4),
                "expert",
            )
        with pytest.raises(PlacementError, match="num_experts"):
            _on_a_pair(
                lambda c: fragment(
                    torch.zeros(3, 4),
                    ExpertLocal(),
                    c,
                    routing=torch.zeros(3, 2, dtype=torch.int64),
                ),
                "expert",
            )

    def test_expert_local_refuses_a_routing_table_that_does_not_lead_the_tensor(
        self,
    ) -> None:
        with pytest.raises(PlacementError, match=r"\(2, 2\).*\(3, 4\)"):
            _on_a_pair(
                lambda c: fragment(
                    torch.zeros(3, 4),
                    ExpertLocal(),
                    c,
                    routing=torch.zeros(2, 2, dtype=torch.int64),
                    num_experts=4,
                ),
                "expert",
            )

    def test_expert_local_refuses_a_feature_axis_that_is_not_whole_slots(self) -> None:
        with pytest.raises(PlacementError, match=r"feature axis 5 .*top_k=2"):
            _on_a_pair(
                lambda c: fragment(
                    torch.zeros(3, 5),
                    ExpertLocal(),
                    c,
                    routing=torch.zeros(3, 2, dtype=torch.int64),
                    num_experts=4,
                ),
                "expert",
            )

    def test_expert_count_must_divide_by_the_group(self) -> None:
        with pytest.raises(PlacementError, match=r"num_experts=5 .*2"):
            expert_slot_mask(torch.zeros(3, 2, dtype=torch.int64), 5, 0, 2)
        with pytest.raises(PlacementError, match=r"num_experts=5 .*2"):
            remap_routing(torch.zeros(3, 2, dtype=torch.int64), 5, 0, 2)
        with pytest.raises(PlacementError, match=r"num_experts=5 .*2"):
            _on_a_pair(
                lambda c: reconstruct_routing(
                    torch.zeros(3, 2, dtype=torch.int64), 5, c
                ),
                "expert",
            )

    def test_rank_must_be_inside_the_group(self) -> None:
        with pytest.raises(PlacementError, match=r"rank 2 .*2"):
            expert_slot_mask(torch.zeros(3, 2, dtype=torch.int64), 4, 2, 2)

    def test_reconstruct_refuses_a_local_table_outside_its_range(self) -> None:
        for rows in ([[0, 7]], [[0, -1]]):
            table = torch.tensor(rows)
            with pytest.raises(PlacementError, match=r"local .*\[0, 2\]"):
                _on_a_pair(lambda c: reconstruct_routing(table, 4, c), "expert")

    def test_stage_local_refuses_a_stage_outside_the_group(self) -> None:
        with pytest.raises(PlacementError, match=r"stage 2 .*2"):
            _on_a_pair(
                lambda c: fragment(torch.zeros(2, 3), StageLocal(stage=2), c),
                "pipeline",
            )
        with pytest.raises(PlacementError, match=r"stage 2 .*2"):
            _on_a_pair(
                lambda c: whole(torch.zeros(2, 3), StageLocal(stage=2), c), "pipeline"
            )

    def test_a_placement_error_carries_its_placement_and_opens_with_it(self) -> None:
        placement = StageLocal(stage=2)
        with pytest.raises(PlacementError) as info:
            _on_a_pair(lambda c: fragment(torch.zeros(2, 3), placement, c), "pipeline")
        assert info.value.placement == placement
        assert str(info.value) == f"{placement}: stage 2 is outside the group's 2 ranks"
        bare = PlacementError(None, "the routing table has 3 rows")
        assert bare.placement is None
        assert str(bare) == "the routing table has 3 rows"

    def test_stage_local_non_owners_hold_an_empty_tensor_of_the_right_type(
        self,
    ) -> None:
        _owner, piece = _on_a_pair(
            lambda c: fragment(
                torch.zeros(2, 3, dtype=torch.float64), StageLocal(stage=0), c
            ),
            "pipeline",
        )
        assert piece.shape == (0, 3) and piece.dtype == torch.float64

    def test_fragments_honours_the_protocol(self) -> None:
        assert all(_on_a_pair(lambda c: isinstance(c, Collective)))
        assert isinstance(RefusingCollective(), Collective)


# --------------------------------------------------------------------------- #
# the repeated shard (§6.6): one KV head per rank, held by ``repeat`` ranks
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestRepeatedShard:
    @_HYPOTHESIS_SETTINGS
    @given(
        kv_heads=st.integers(1, 4),
        repeat=st.integers(2, 4),
        head_dim=st.integers(1, 3),
        axis=st.sampled_from([1, -1]),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_whole_of_the_repeated_locals_is_the_kv_tensor_and_fragment_its_chunk(
        self, kv_heads: int, repeat: int, head_dim: int, axis: int, seed: int
    ) -> None:
        """Under KV-head replication rank ``r`` of the tensor group holds KV
        head ``r // repeat`` (``tp = kv_heads · repeat``). ``whole`` is the
        model's ``kv_heads``-head tensor on every rank — the copies a gather
        carries are dropped, never averaged — and ``fragment`` of it is the
        rank's head again, bit for bit."""
        tp = kv_heads * repeat
        shape = (
            [2, kv_heads * head_dim, 3] if axis == 1 else [2, 3, kv_heads * head_dim]
        )
        g = _randn(shape, seed)
        placement = Sharded(axis=axis, group="tensor", repeat=repeat)
        local = {
            rank: g.narrow(axis, (rank // repeat) * head_dim, head_dim)
            for rank in range(tp)
        }
        made_whole = _mesh(tp, tensor=tp).run(
            lambda rank, c: whole(local[rank], placement, c)
        )
        fragments_ = _mesh(tp, tensor=tp).run(lambda rank, c: fragment(g, placement, c))
        for rank in range(tp):
            assert torch.equal(made_whole[rank], g), rank
            assert torch.equal(fragments_[rank], local[rank]), rank
            assert fragments_[rank].shape[axis] == head_dim

    @_HYPOTHESIS_SETTINGS
    @given(
        kv_heads=st.integers(1, 3),
        repeat=st.integers(2, 3),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_a_repeat_of_one_is_the_plain_shard(
        self, kv_heads: int, repeat: int, seed: int
    ) -> None:
        tp = kv_heads * repeat
        g = _randn((2, tp * 2), seed)
        plain = _mesh(tp, tensor=tp).run(
            lambda rank, c: fragment(g, Sharded(axis=-1), c)
        )
        spelled = _mesh(tp, tensor=tp).run(
            lambda rank, c: fragment(g, Sharded(axis=-1, repeat=1), c)
        )
        for rank in range(tp):
            assert torch.equal(plain[rank], spelled[rank])
            assert torch.equal(plain[rank], g.narrow(-1, rank * 2, 2))


@pytest.mark.unit
class TestRepeatedShardRefusals:
    def test_a_group_that_is_not_whole_repeats_is_refused(self) -> None:
        placement = Sharded(axis=-1, repeat=2)
        with pytest.raises(RankFailed) as info:
            _mesh(3, tensor=3).run(
                lambda rank, c: fragment(torch.zeros(2, 4), placement, c)
            )
        assert isinstance(info.value.error, PlacementError)
        assert "3 ranks" in str(info.value.error)
        assert "repeats of 2" in str(info.value.error)

    def test_a_slotted_shard_is_never_repeated(self) -> None:
        with pytest.raises(ValueError, match="slots=2 and repeat=2"):
            Sharded(axis=-1, slots=2, repeat=2)
        with pytest.raises(ValueError, match="repeat"):
            Sharded(axis=-1, repeat=0)

    def test_the_mutation_shifting_the_held_chunk_fails_the_round_trip(self) -> None:
        """A rank holding its neighbour's KV head (chunk ``rank // repeat +
        1``, the smoke tier's mutation) is not what ``whole`` gathers back
        into the KV tensor, and ``fragment`` disagrees with it on every
        rank."""
        tp, repeat, head_dim = 4, 2, 3
        g = _randn((2, 2 * head_dim), 7)
        placement = Sharded(axis=-1, repeat=repeat)
        shifted = {
            rank: g.narrow(-1, ((rank // repeat + 1) % 2) * head_dim, head_dim)
            for rank in range(tp)
        }
        made_whole = _mesh(tp, tensor=tp).run(
            lambda rank, c: whole(shifted[rank], placement, c)
        )
        fragments_ = _mesh(tp, tensor=tp).run(lambda rank, c: fragment(g, placement, c))
        for rank in range(tp):
            assert not torch.equal(made_whole[rank], g)
            assert not torch.equal(fragments_[rank], shifted[rank])


@pytest.mark.unit
class TestResolvedSitePlacement:
    """§4: ``ResolvedSite`` carries a placement; a module no plan row names is
    replicated. (Deriving it from a plan row is a later phase, so this sits
    beside the seam's tests rather than in a ``test_sites.py`` of its own.)"""

    def test_the_default_placement_is_replicated(self) -> None:
        site = ResolvedSite(module=None, kind="out", shape=bsd(4))
        assert site.placement == REPLICATED
        assert isinstance(site.placement, Replicated)


@pytest.mark.unit
class TestAxisSpellingsAndCompositions:
    """Axis boundaries and composed placements outside the registry plans."""

    def test_the_most_negative_axis_is_the_first(self) -> None:
        """``-ndim`` spells axis 0, as it does for torch."""
        x = torch.arange(8.0).reshape(4, 2)
        negative, positive = _on_a_pair(
            lambda c: (
                fragment(x, Sharded(axis=-2), c),
                fragment(x, Sharded(axis=0), c),
            )
        )[0]
        assert torch.equal(negative, positive)
        assert negative.shape == (2, 2)

    def test_a_slotted_shard_on_an_inner_axis_round_trips(self) -> None:
        """``slots`` on an axis with dimensions after it: each run of the axis
        is sharded over the group, and the refold touches that axis alone."""
        g = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        placement = Sharded(axis=1, slots=2)  # two runs of four, each halved

        def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
            local = fragment(g, placement, c)
            return local, whole(local, placement, c)

        runs = g.unflatten(1, (2, 4))  # (2, slots, width, 3)
        for rank, (local, back) in enumerate(_mesh(2, tensor=2).run(program)):
            assert torch.equal(local, runs.narrow(2, rank * 2, 2).flatten(1, 2))
            assert torch.equal(back, g)

    def test_a_stage_local_shard_hands_its_inner_the_routing_table(self) -> None:
        """``StageLocal`` over ``ExpertLocal`` (pipeline × expert): the owning
        stage's fragment is the inner's, which needs the routing table and
        ``num_experts``; the other stage holds nothing."""
        x = torch.arange(3 * 8, dtype=torch.float32).reshape(
            3, 8
        )  # (tokens, top_k · d)
        routing = torch.tensor([[0, 3], [1, 2], [3, 0]])
        placement = StageLocal(0, inner=ExpertLocal("expert"))

        def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
            wrapped = fragment(x, placement, c, routing=routing, num_experts=4)
            direct = fragment(
                x, ExpertLocal("expert"), c, routing=routing, num_experts=4
            )
            return wrapped, direct

        results = _mesh(4, pipeline=2, expert=2).run(program)
        for rank in (0, 1):
            assert torch.equal(*results[rank])
        for rank in (2, 3):
            assert results[rank][0].shape == (0, 8)

    def test_a_stage_local_sequence_shard_gathers_by_the_frame(self) -> None:
        """``StageLocal`` over ``SequenceSharded`` (pipeline × context): the
        owning stage gathers its uneven chunks by the frame it was handed,
        then broadcasts the whole to the other stage."""
        from causalab.neural.shared.parallel.context import SequenceFrame

        g = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        mask = torch.ones(1, 5, dtype=torch.long)  # chunks of 2 and 3
        placement = StageLocal(0, inner=SequenceSharded(1, "context"))

        def program(rank: int, c: Collective) -> torch.Tensor:
            frame = SequenceFrame(c, mask)
            chunk = frame.chunk
            local = (
                g[:, chunk.start : chunk.stop] if c.rank("pipeline") == 0 else g[:, :0]
            )
            return whole(local, placement, c, frame=frame)

        for got in _mesh(4, pipeline=2, context=2).run(program):
            assert torch.equal(got, g)

    def test_a_sequence_shard_differentiates_through_the_frame(self) -> None:
        """The frame carries gradients through gather and fragment: each
        chunk receives its gradient, and a replicated fragment receives
        the whole gradient on every rank."""
        from causalab.neural.shared.parallel.context import SequenceFrame
        from causalab.protocol.parallel import sequence_chunks

        g = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
        mask = torch.ones(1, 5, dtype=torch.long)  # chunks of 2 and 3
        placement = SequenceSharded(1, "context")
        weights = torch.arange(1.0, 6.0)[None, :, None]

        def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
            frame = SequenceFrame(c, mask)
            chunk = frame.chunk
            local = g[:, chunk.start : chunk.stop].clone().requires_grad_(True)
            (whole(local, placement, c, frame=frame) * weights).sum().backward()
            replicated = g.clone().requires_grad_(True)
            fragment(replicated, placement, c, frame=frame).sum().backward()
            assert local.grad is not None and replicated.grad is not None
            return local.grad, replicated.grad

        for rank, (chunk_grad, whole_grad) in enumerate(
            _mesh(2, context=2).run(program)
        ):
            chunk = sequence_chunks(5, 2)[rank]
            expected = weights[:, chunk.start : chunk.stop].expand(1, len(chunk), 2)
            assert torch.equal(chunk_grad, expected)
            assert torch.equal(whole_grad, torch.ones_like(g))
