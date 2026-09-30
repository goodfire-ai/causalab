"""The executor's hook bodies over fragments (``docs/model_parallelism.md`` §4, §6.1, §6.3, §10.4).

Deterministic simulation with a **fake sharded module**: a tiny module whose
forward returns this rank's part of a known global tensor — spelled by hand
per placement kind — hooked by the *real* hook bodies (``_capturing``,
``_installed``, ``_experts_capture``, ``_experts_edit``) with the real
[`Fragments`][causalab.neural.shared.parallel.fragments.Fragments] over the simulated collective. A read equals the global
tensor on every rank and a ``swap`` write lands the rank's part of the
globally edited tensor, bit for bit, across drawn schedules; the
world-1 twin runs over the refusing collective, so the hook bodies are shown
to call nothing there. Two mutations of the hook body close the file: the
``whole`` dropped fails the read property, the ``fragment`` dropped fails the
write property.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings

from causalab.neural.engines.pytorch_hooks import executor as executor_module
from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    _capturing,
    _experts_capture,
    _experts_edit,
    _installed,
)
from causalab.neural.shared.parallel import taps as taps_module
from causalab.neural.shared.parallel.collective import Collective
from causalab.neural.shared.parallel.fragments import Fragments, remap_routing
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    ExpertLocal,
    Placement,
    Sharded,
    StageLocal,
)
from causalab.neural.shared.parallel.taps import TapFragments
from causalab.neural.shared.sites import ResolvedSite
from causalab.protocol.registry.shapes import bsd, flat_td
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

# pyright: reportPrivateUsage=false

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

BATCH, POSITIONS, FEATURE = 2, 3, 8
TP = 2


def _seeded(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


class _Emit(torch.nn.Module):
    """Returns what it was built with: this rank's part of the global tensor."""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self, *args: Any) -> torch.Tensor:
        return self.value


class _Through(torch.nn.Module):
    """Returns its first argument — what a pre-hook rewrote."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


def _chunk(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    """§4 spelled by hand: this rank's contiguous chunk of the feature axis."""
    width = tensor.shape[-1] // size
    return tensor.narrow(-1, rank * width, width)


def _module_boundary(
    kind: str, placement: Placement
) -> Callable[[int, Collective], tuple[torch.Tensor, torch.Tensor]]:
    global_tensor = _seeded((BATCH, POSITIONS, FEATURE), 11)
    replacement = _seeded((BATCH, POSITIONS, FEATURE), 12)

    def program(rank: int, c: Collective) -> tuple[torch.Tensor, torch.Tensor]:
        tap = TapFragments(Fragments(c), placement)
        size = c.size("tensor")
        local = (
            _chunk(global_tensor, c.rank("tensor"), size) if size > 1 else global_tensor
        )
        module: torch.nn.Module = _Emit(local) if kind == "out" else _Through()
        sink: dict[Any, torch.Tensor] = {}
        with _capturing(
            module, kind, sink, "k", shape=bsd(FEATURE), batch_size=BATCH, tap=tap
        ):
            module(local)
        captured = sink["k"]

        def swap(contract: torch.Tensor) -> None:
            contract.copy_(replacement)

        with _installed(
            module, kind, swap, shape=bsd(FEATURE), batch_size=BATCH, tap=tap
        ):
            edited = module(local)
        return captured, edited

    program.global_tensor = global_tensor  # type: ignore[attr-defined]
    program.replacement = replacement  # type: ignore[attr-defined]
    return program


def _assert_module_boundary(
    results: list[tuple[torch.Tensor, torch.Tensor]], program: Any, size: int
) -> None:
    for rank, (captured, edited) in enumerate(results):
        assert torch.equal(captured, program.global_tensor), f"rank {rank}: read"
        expected = (
            _chunk(program.replacement, rank % size, size)
            if size > 1
            else program.replacement
        )
        assert torch.equal(edited, expected), f"rank {rank}: write"


@pytest.mark.property
class TestModuleBoundaryScenarios:
    @pytest.mark.parametrize("kind", ["out", "in"])
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_a_colwise_output_reads_whole_and_writes_its_chunk_under_every_schedule(
        self, kind: str, schedule: Schedule
    ) -> None:
        program = _module_boundary(kind, Sharded(-1, "tensor"))
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=schedule)
        _assert_module_boundary(world.run(program), program, TP)

    @_SETTINGS
    @given(schedule=ps.schedules())
    def test_the_result_does_not_depend_on_the_schedule(
        self, schedule: Schedule
    ) -> None:
        program = _module_boundary("out", Sharded(-1, "tensor"))
        # two tensor groups: every group reads the same global and writes its own chunks
        world = SimulatedWorld(
            groups_for(4, data=2, tensor=2), world=4, schedule=schedule
        )
        _assert_module_boundary(world.run(program), program, 2)

    def test_a_stage_local_output_is_broadcast_to_every_stage(self) -> None:
        placement = StageLocal(1, inner=Sharded(-1, "tensor"))
        global_tensor = _seeded((BATCH, POSITIONS, FEATURE), 21)

        def program(rank: int, c: Collective) -> torch.Tensor:
            tap = TapFragments(Fragments(c), placement)
            owner = c.rank("pipeline") == 1
            local = (
                _chunk(global_tensor, c.rank("tensor"), 2)
                if owner
                else global_tensor.new_empty((0, POSITIONS, FEATURE))
            )
            module = _Emit(local)
            sink: dict[Any, torch.Tensor] = {}
            with _capturing(
                module, "out", sink, "k", shape=bsd(FEATURE), batch_size=BATCH, tap=tap
            ):
                module()
            return sink["k"]

        world = SimulatedWorld(groups_for(4, pipeline=2, tensor=2), world=4, schedule=5)
        for captured in world.run(program):
            assert torch.equal(captured, global_tensor)


# --------------------------------------------------------------------------- #
# the experts interior: ExpertLocal and the remapped routing table (§6.3)
# --------------------------------------------------------------------------- #

TOKENS, TOP_K, D_EXPERT, EXPERTS, EP = 6, 2, 4, 8, 4


def _experts_site() -> ResolvedSite:
    return ResolvedSite(
        module=None,
        kind="experts",
        shape=flat_td(TOP_K * D_EXPERT),
        component="expert_activation",
        interface_slot="activation",
    )


def _experts_program(
    seed: int,
) -> Callable[[int, Collective], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    global_view = torch.randn((TOKENS, TOP_K * D_EXPERT), generator=generator)
    replacement = torch.randn((TOKENS, TOP_K * D_EXPERT), generator=generator)
    # distinct experts per token, as a router draws them
    table = torch.stack(
        [torch.randperm(EXPERTS, generator=generator)[:TOP_K] for _ in range(TOKENS)]
    )
    batch = 2

    def program(
        rank: int, c: Collective
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        site = _experts_site()
        tap = TapFragments(
            Fragments(c),
            ExpertLocal("expert"),
            remapped_routing=True,
            num_experts=EXPERTS,
        )
        me, size = c.rank("expert"), c.size("expert")
        # this rank's view, spelled by hand: its experts' slots carry values,
        # every other slot is zero, and its routing table is the remapped one
        owned = torch.div(table, EXPERTS // size, rounding_mode="floor") == me
        keep = (
            owned.unsqueeze(-1)
            .expand(TOKENS, TOP_K, D_EXPERT)
            .reshape(global_view.shape)
        )
        local = torch.where(keep, global_view, torch.zeros(()))
        local_table = remap_routing(table, EXPERTS, me, size)
        sink: dict[Any, torch.Tensor] = {}
        idx_sink: dict[Any, torch.Tensor] = {}
        _experts_capture(sink, idx_sink, "k", site, batch, tap)(local, local_table)

        def swap(contract: torch.Tensor, routing: torch.Tensor | None = None) -> None:
            # the write math sees the *global* routing table, in contract form
            assert routing is not None
            assert torch.equal(routing, table.reshape(batch, -1, TOP_K))
            contract.copy_(replacement.reshape(batch, -1, TOP_K * D_EXPERT))

        # the experts interface hands the edit a clone of the view it captured
        edited = _experts_edit(site, swap, batch, tap)(local.clone(), local_table)
        return sink["k"], idx_sink["k"], edited

    program.global_view = global_view  # type: ignore[attr-defined]
    program.replacement = replacement  # type: ignore[attr-defined]
    program.routing = table  # type: ignore[attr-defined]
    return program


@pytest.mark.property
class TestExpertsInteriorScenarios:
    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_expert_local_reads_the_global_view_and_writes_its_own_slots(
        self, schedule: Schedule, seed: int
    ) -> None:
        program = _experts_program(seed)
        world = SimulatedWorld(groups_for(EP, expert=EP), world=EP, schedule=schedule)
        for rank, (captured, idx, edited) in enumerate(world.run(program)):
            assert torch.equal(
                captured, program.global_view.reshape(2, -1, TOP_K * D_EXPERT)
            )
            # the routing table the read rides with is the reconstructed global one
            assert torch.equal(idx, program.routing.reshape(2, -1, TOP_K))
            owned = (
                torch.div(program.routing, EXPERTS // EP, rounding_mode="floor") == rank
            )
            keep = (
                owned.unsqueeze(-1)
                .expand(TOKENS, TOP_K, D_EXPERT)
                .reshape(edited.shape)
            )
            expected = torch.where(keep, program.replacement, torch.zeros(()))
            assert torch.equal(edited, expected), (
                f"rank {rank}: the write kept only its slots"
            )


# --------------------------------------------------------------------------- #
# world 1: the identity, and no collective
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestWorldOne:
    def test_the_hook_bodies_call_no_collective_at_world_one(self) -> None:
        refusing = RefusingCollective()
        for placement in (REPLICATED, Sharded(-1, "tensor"), StageLocal(0)):
            program = _module_boundary("out", placement)
            captured, edited = program(0, refusing)
            assert torch.equal(captured, program.global_tensor)
            assert torch.equal(edited, program.replacement)
        program = _experts_program(3)
        captured, idx, edited = program(0, refusing)
        assert torch.equal(
            captured, program.global_view.reshape(2, -1, TOP_K * D_EXPERT)
        )
        assert torch.equal(idx, program.routing.reshape(2, -1, TOP_K))
        assert torch.equal(edited, program.replacement)

    def test_the_executor_holds_solo_fragments_by_default(self) -> None:
        assert isinstance(PointExecutor.fragments, Fragments)
        assert PointExecutor.fragments.whole(torch.ones(2), Sharded(-1)) is not None


# --------------------------------------------------------------------------- #
# mutations of the hook body
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestHookBodyMutations:
    def test_dropping_whole_fails_the_read_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = _module_boundary("out", Sharded(-1, "tensor"))
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        _assert_module_boundary(world.run(program), program, TP)
        monkeypatch.setattr(
            taps_module.TapFragments, "whole", lambda self, native: native
        )
        # the local chunk is not the contract's width: the read is refused on
        # the rank, or where widths happen to fit, unequal to the global
        with pytest.raises((AssertionError, RankFailed)):
            _assert_module_boundary(world.run(program), program, TP)

    def test_dropping_fragment_fails_the_write_property(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        program = _module_boundary("out", Sharded(-1, "tensor"))
        world = SimulatedWorld(groups_for(TP, tensor=TP), world=TP, schedule=0)
        monkeypatch.setattr(
            taps_module.TapFragments,
            "fragment",
            lambda self, edited, routing=None: edited,
        )
        with pytest.raises(AssertionError, match="write"):
            _assert_module_boundary(world.run(program), program, TP)

    def test_the_executor_module_reaches_the_taps_seam(self) -> None:
        # the hook bodies import the seam the mutations above patch
        assert executor_module.TapFragments is taps_module.TapFragments
