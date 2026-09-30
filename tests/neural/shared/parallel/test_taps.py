"""``TapFragments`` on the routing table itself (``docs/model_parallelism.md`` §6.3).

A module-boundary tap on the router's indices (``expert_idx``) under expert
parallelism holds *this rank's remapped table* — local ids on the slots it
owns, the sentinel elsewhere (``EpRouterParallel``). The tap flagged
``routing_table`` makes it ``whole`` by the §6.3 reconstruction (an integer
all-reduce, exact) and ``fragment``\\ s a global table back to the local one
(``remap_routing``), so a read saves the world-1 table and a ``swap`` lands
the rank's remapped view of the swapped one. Deterministic simulation over
drawn tables and schedules; at world 1 both are the identity and the
collective is never called.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import example, given, settings

from causalab.neural.shared.parallel.fragments import (
    Fragments,
    PlacementError,
    remap_routing,
)
from causalab.neural.shared.parallel.placement import REPLICATED
from causalab.neural.shared.parallel.taps import TapFragments
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import Schedule, SimulatedWorld, groups_for
from tests._helpers.refusing_collective import RefusingCollective

pytestmark = pytest.mark.unit

_SETTINGS = settings(deadline=None, max_examples=30)

EXPERTS = 16
EP = 4


def _table(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, EXPERTS, (2, 3, 4), generator=generator)


@_SETTINGS
@given(seed=ps.seeds(), schedule=ps.schedules())
@example(seed=0, schedule=[])
def test_whole_reconstructs_the_global_table_and_fragment_remaps_it(
    seed: int, schedule: Schedule
) -> None:
    table = _table(seed)
    swapped = _table(seed + 100)
    world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=schedule)

    def program(rank: int, c: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tap = TapFragments(
            Fragments(c),
            REPLICATED,
            remapped_routing=True,
            num_experts=EXPERTS,
            routing_table=True,
        )
        local = remap_routing(table, EXPERTS, c.rank("expert"), EP)
        return tap.whole(local), tap.fragment(swapped)

    for rank, (made_whole, back) in enumerate(world.run(program)):
        assert torch.equal(made_whole, table), rank
        assert torch.equal(back, remap_routing(swapped, EXPERTS, rank, EP)), rank


@_SETTINGS
@given(seed=ps.seeds(), schedule=ps.schedules())
def test_rescore_follows_the_edited_table(seed: int, schedule: Schedule) -> None:
    """A write to the table moves slots between experts; the expert-local
    scores are made whole and re-masked to the new ownership, so the experts
    module weights exactly the slots the world-1 module would."""
    table, edited = _table(seed), _table(seed + 50)
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(table.shape, generator=generator)
    world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=schedule)

    def owned(rank: int, routing: torch.Tensor) -> torch.Tensor:
        return torch.div(routing, EXPERTS // EP, rounding_mode="floor") == rank

    def program(rank: int, c: Any) -> torch.Tensor:
        tap = TapFragments(
            Fragments(c),
            REPLICATED,
            remapped_routing=True,
            num_experts=EXPERTS,
            routing_table=True,
        )
        me = c.rank("expert")
        local_scores = scores.masked_fill(~owned(me, table), 0.0)
        return tap.rescore(local_scores, edited)

    for rank, rescored in enumerate(world.run(program)):
        assert torch.equal(rescored, scores.masked_fill(~owned(rank, edited), 0.0)), (
            rank
        )


def test_the_flag_off_leaves_the_table_as_handed() -> None:
    """Without the flag a replicated integral tensor is the identity on both
    sides — the world-1 path, and every integral tensor that is not a
    routing table (``input_ids``)."""
    table = _table(7)
    world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=7)

    def program(rank: int, c: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tap = TapFragments(Fragments(c), REPLICATED, num_experts=EXPERTS)
        return tap.whole(table), tap.fragment(table)

    for made_whole, back in world.run(program):
        assert torch.equal(made_whole, table) and torch.equal(back, table)


def test_at_world_one_the_table_tap_calls_nothing() -> None:
    table = _table(3)
    tap = TapFragments(
        Fragments(RefusingCollective()),
        REPLICATED,
        remapped_routing=True,
        num_experts=EXPERTS,
        routing_table=True,
    )
    assert torch.equal(tap.whole(table), table)
    assert torch.equal(tap.fragment(table), table)


def test_a_table_tap_without_the_expert_count_is_refused_by_name() -> None:
    world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=0)

    def program(rank: int, c: Any) -> str:
        tap = TapFragments(
            Fragments(c), REPLICATED, remapped_routing=True, routing_table=True
        )
        try:
            tap.whole(_table(0))
        except PlacementError as err:
            return str(err)
        return "no refusal"

    for message in world.run(program):
        assert "num_experts" in message
