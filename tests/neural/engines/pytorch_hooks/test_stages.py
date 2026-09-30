"""Pipeline stages: the stage forward, stage-local hooks, broadcast captures
and resume across stages (``docs/model_parallelism.md`` §6.5, §8.3, §10.3
"pipeline ranges", §10.4 ``pp∈{2,4}``).

Three tiers. **Property**: the two spellings of stage ownership —
``protocol.parallel.stage_layers`` (what the loader keeps) and
``placements.stage_of`` (what a site's placement names) — agree for every
``(num_layers, pp)``, and the order the captures are broadcast in is a pure
function of the group's sites. **Simulation with a fake model**: a tiny
causal LM whose blocks are real ``nn.Module``\\ s, placed by hand the way
``place_stage`` places a family, runs the *real* [`StageForward`][causalab.neural.engines.pytorch_hooks.stages.StageForward], the
real hook bodies and the real broadcast step on every rank of a
``SimulatedWorld``; a read on a stage-1 block, a swap on stage 0 with a read
on stage 1, and a resume from the cached residual at the stage boundary all
equal the whole forward **bit for bit** — pipeline placement moves no
reduction — across drawn schedules. **Simulation with the real
engine**: the tiny Llama and the tiny MoE, each rank holding a placed copy,
run the real [`PointExecutor`][causalab.neural.engines.pytorch_hooks.executor.PointExecutor] over interchange documents with a write
on one stage and a read on the other; the reads, the ``fires`` record and
the resume bookkeeping equal the world-1 run's, a stage that never fires a
write is refused on **every** rank (the non-owner declared the member, so
the summed count is compared against the declaration), and a rank that
skips one broadcast is caught as a `Divergence`, never a hang.
"""

# the executor's private steps are the seam under test
# pyright: reportPrivateUsage=false

from __future__ import annotations

import contextlib
import copy
import dataclasses
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from torch import nn

from causalab.neural.engines.pytorch_hooks import executor as executor_module
from causalab.neural.engines.pytorch_hooks import stages as stages_module
from causalab.neural.engines.pytorch_hooks.executor import (
    PointExecutor,
    _capturing,
    _installed,
    _resumed,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.engines.pytorch_hooks.sharding import (
    Sharding,
    StageStandIn,
    place_stage,
)
from causalab.neural.engines.pytorch_hooks.stages import (
    StageForward,
    broadcast_order,
    hook_placement,
)
from causalab.neural.shared import model_tree
from causalab.neural.shared.devices import DeviceMap
from causalab.neural.shared.executor import tap_key
from causalab.neural.shared.parallel.collective import SOLO, Collective
from causalab.neural.shared.parallel.fragments import Fragments
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    Sharded,
    StageLocal,
)
from causalab.neural.shared.parallel.placements import stage_of
from causalab.neural.shared.parallel import serving
from causalab.neural.shared.parallel.taps import TapFragments
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import MeshLayout, ParallelGeometry, stage_layers
from causalab.protocol.registry import family_for, walk
from causalab.protocol.schema import SiteSpec
from causalab.protocol.registry.shapes import bsd
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    Divergence,
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)

from ._drive import executor_for
from .conftest import TINY_LLAMA, TINY_QWEN35_MOE
from .test_prefix_resume import _campaign, swap_doc
from .test_train import ANSWERS, BASES, COUNTERFACTUALS

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# property: ownership and the broadcast order
# --------------------------------------------------------------------------- #


@st.composite
def towers(draw: st.DrawFn) -> tuple[int, int]:
    """``(num_layers, pp)`` with ``pp <= num_layers``."""
    num_layers = draw(st.integers(min_value=1, max_value=12))
    pipeline = draw(st.integers(min_value=1, max_value=num_layers))
    return num_layers, pipeline


def _site(
    component: str,
    layer: int,
    *,
    kind: str = "out",
    module: Any = None,
    placement: Any = REPLICATED,
    slot: str | None = None,
    tuple_index: int | None = None,
    expert: int | None = None,
) -> ResolvedSite:
    return ResolvedSite(
        module=module if module is not None else object(),
        kind=kind,
        shape=bsd(8),
        layer=layer,
        component=component,
        placement=placement,
        interface_slot=slot,
        tuple_index=tuple_index,
        expert=expert,
    )


@pytest.mark.property
class TestOwnership:
    @_SETTINGS
    @given(tower=towers(), data=st.integers(1, 2), tensor=st.sampled_from([1, 2]))
    def test_stage_layers_and_stage_of_agree_on_every_rank(
        self, tower: tuple[int, int], data: int, tensor: int
    ) -> None:
        """The loader's ranges and the placement's owner are one rule: for
        every rank, ``stage_layers`` is exactly the layers ``stage_of`` gives
        the rank's pipeline coordinate, and over the stages the ranges
        partition the tower."""
        num_layers, pipeline = tower
        geometry = ParallelGeometry(data=data, pipeline=pipeline, tensor=tensor)
        layout = MeshLayout(geometry)
        covered: list[int] = []
        for rank in range(geometry.world):
            stage = layout.rank_in(rank, "pipeline")
            kept = stage_layers(geometry, rank, num_layers)
            assert len(kept) >= 1
            assert set(kept) == {
                layer
                for layer in range(num_layers)
                if stage_of(layer, num_layers=num_layers, stages=pipeline) == stage
            }
            if layout.rank_in(rank, "data") == 0 and layout.rank_in(rank, "model") == 0:
                covered.extend(kept)
        assert sorted(covered) == list(range(num_layers))

    def test_the_last_stage_takes_the_remainder(self) -> None:
        geometry = ParallelGeometry(pipeline=3)
        assert [tuple(stage_layers(geometry, r, 8)) for r in range(3)] == [
            (0, 1),
            (2, 3),
            (4, 5, 6, 7),
        ]
        assert [stage_of(i, num_layers=8, stages=3) for i in range(8)] == [
            0,
            0,
            1,
            1,
            2,
            2,
            2,
            2,
        ]

    @_SETTINGS
    @given(seed=ps.seeds(), n=st.integers(1, 6))
    def test_the_broadcast_order_is_a_pure_function_of_the_sites(
        self, seed: int, n: int
    ) -> None:
        """Every rank builds its own site objects (its own modules), so the
        order cannot read the modules: a permutation of the sites — and
        different module objects behind them — gives the same order."""
        generator = torch.Generator().manual_seed(seed)
        sites = [
            _site(
                component,
                layer,
                slot=slot,
                placement=StageLocal(layer % 2),
            )
            for layer in range(n)
            for component, slot in (
                ("block_output", None),
                ("attention_output", None),
                ("attention_query", "query"),
            )
        ]
        order = torch.randperm(len(sites), generator=generator).tolist()
        shuffled = [dataclasses.replace(sites[i], module=object()) for i in order]
        spell = lambda s: (s.layer, s.component, s.interface_slot)  # noqa: E731
        assert [spell(s) for s in broadcast_order(sites)] == [
            spell(s) for s in broadcast_order(shuffled)
        ]
        # two sites of one tap are broadcast once
        doubled = [*sites, *sites]
        assert len(broadcast_order(doubled)) == len(broadcast_order(sites))

    def test_hook_placement_strips_the_stage_and_keeps_the_interior(self) -> None:
        """The owner's hook runs the stage-*interior* placement: the broadcast
        to the other stages is the post-forward step's, never the hook's — a
        hook that broadcast mid-forward would wait on a stage that is itself
        waiting to receive this stage's residual."""
        assert hook_placement(StageLocal(1)) == REPLICATED
        assert hook_placement(StageLocal(1, inner=Sharded(-1, "tensor"))) == Sharded(
            -1, "tensor"
        )
        assert hook_placement(Sharded(-1, "tensor")) == Sharded(-1, "tensor")
        assert hook_placement(REPLICATED) == REPLICATED


@pytest.mark.unit
class TestStageForwardFacts:
    def test_owner_and_installs_read_the_placement(self) -> None:
        stages = StageForward(SOLO, stages=2, stage=1, num_layers=2)
        mine = _site("block_output", 1, placement=StageLocal(1))
        theirs = _site("block_output", 0, placement=StageLocal(0))
        everyone = _site("input_ids", 0)
        assert stages.owner(mine) == 1 and stages.installs(mine)
        assert stages.owner(theirs) == 0 and not stages.installs(theirs)
        assert stages.owner(everyone) is None and stages.installs(everyone)
        assert stages.owns_block(1) and not stages.owns_block(0)
        # a resume swaps blocks on the stage owning the block it starts at
        assert stages.swaps(1) and not stages.swaps(0)
        assert not StageForward(SOLO, stages=2, stage=0, num_layers=2).swaps(1)
        # world 1: everything is this rank's, and a resume above 0 swaps
        solo = StageForward(SOLO, stages=1, stage=0, num_layers=2)
        assert solo.installs(theirs) and solo.swaps(1) and not solo.swaps(0)

    def test_a_derived_component_is_refused_by_name_under_pipeline(self) -> None:
        stages = StageForward(SOLO, stages=2, stage=0, num_layers=2)
        derived = dataclasses.replace(
            _site("attention_result", 1, placement=StageLocal(1)),
            derivation="attention_result",
        )
        with pytest.raises(ProtocolError) as err:
            stages.refuse_unserved("read 'r'", derived)
        assert err.value.code == "P4"
        assert "--parallel.pipeline" in str(err.value)
        assert "attention_result" in str(err.value)
        # world 1 refuses nothing
        StageForward(SOLO, stages=1, stage=0, num_layers=2).refuse_unserved(
            "read 'r'", derived
        )

    def test_the_serving_table_refuses_no_axis(self) -> None:
        """Every axis is served (§6.5 pipeline, §8.4 context): the serving
        module keeps the two collective checks and no per-axis refusal."""
        assert not hasattr(serving, "refuse_unserved")
        assert sorted(serving.__all__) == ["check_collective", "process_mesh"]


# --------------------------------------------------------------------------- #
# the fake model: real nn.Modules split across simulated ranks
# --------------------------------------------------------------------------- #

HIDDEN, VOCAB, BATCH, POSITIONS = 8, 11, 2, 5
IDS = torch.tensor([[1, 4, 7, 2, 9], [3, 3, 8, 10, 5]])
MASK = torch.ones_like(IDS)
POS = torch.arange(POSITIONS).expand(BATCH, POSITIONS)


class _Block(nn.Module):
    """One decoder block: a residual update the loop calls with keywords, as
    a transformers block is called."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(HIDDEN, HIDDEN)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(HIDDEN, HIDDEN, generator=generator))
            self.proj.bias.copy_(torch.randn(HIDDEN, generator=generator))

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return hidden_states + torch.tanh(self.proj(hidden_states))


class _Base(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(100)
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.randn(VOCAB, HIDDEN, generator=generator)
            )
        self.layers = nn.ModuleList(_Block(seed) for seed in range(num_layers))
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> Any:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds")
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.layers:
            hidden = layer(
                hidden, attention_mask=attention_mask, position_ids=position_ids
            )
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden), past_key_values=None
        )


class _FakeLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN, tie_word_embeddings=False)
        self.model = _Base(num_layers)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(7))
            )

    def forward(self, **kwargs: Any) -> Any:
        base = self.model(**kwargs)
        return SimpleNamespace(
            logits=self.lm_head(base.last_hidden_state), past_key_values=None
        )


def _placed_fake(whole: _FakeLM, stage: int, stages: int) -> _FakeLM:
    """This stage's copy of ``whole``, placed as ``place_stage`` places a
    registered family: the stage's layers kept, the rest stand-ins carrying
    the replaced module as their shadow."""
    model = copy.deepcopy(whole)
    num_layers = len(model.model.layers)
    keep = stage_layers(ParallelGeometry(pipeline=stages), stage, num_layers)
    for index in range(num_layers):
        if index not in keep:
            model.model.layers[index] = StageStandIn(model.model.layers[index])
    if stage != 0:
        model.model.embed_tokens = StageStandIn(model.model.embed_tokens)
    if stage != stages - 1:
        model.model.norm = StageStandIn(model.model.norm)
        model.lm_head = StageStandIn(model.lm_head)
    return model


def _forward(stages: StageForward, model: _FakeLM, *, start: int = 0) -> Any:
    return stages.forward(
        model,
        hidden_size=HIDDEN,
        device=CPU,
        input_ids=IDS,
        attention_mask=MASK,
        position_ids=POS,
        use_cache=False,
        start=start,
    )


def _whole_logits(whole: _FakeLM) -> torch.Tensor:
    with torch.no_grad():
        return whole(
            input_ids=IDS, attention_mask=MASK, position_ids=POS, use_cache=False
        ).logits


def _block_site(
    model: _FakeLM, layer: int, stages: int, num_layers: int
) -> ResolvedSite:
    return _site(
        "block_output",
        layer,
        module=model.model.layers[layer],
        placement=StageLocal(stage_of(layer, num_layers=num_layers, stages=stages)),
    )


def _stage_program(
    whole: _FakeLM,
    stages: int,
    body: Callable[[StageForward, _FakeLM, Collective], Any],
) -> Callable[[int, Collective], Any]:
    num_layers = len(whole.model.layers)

    def program(rank: int, c: Collective) -> Any:
        stage = c.rank("pipeline")
        forward = StageForward(c, stages=stages, stage=stage, num_layers=num_layers)
        model = _placed_fake(whole, stage, stages)
        with torch.no_grad():
            return body(forward, model, c)

    return program


def _world(stages: int, schedule: Schedule, *, tensor: int = 1) -> SimulatedWorld:
    world = stages * tensor
    return SimulatedWorld(
        groups_for(world, pipeline=stages, tensor=tensor),
        world=world,
        schedule=schedule,
    )


@pytest.mark.property
class TestFakeStageScenarios:
    @pytest.mark.parametrize(("num_layers", "stages"), [(2, 2), (4, 2), (4, 4), (5, 2)])
    @given(schedule=ps.schedules())
    @example(schedule=[])
    @_SETTINGS
    def test_the_stage_forward_equals_the_whole_forward_on_every_rank(
        self, num_layers: int, stages: int, schedule: Schedule
    ) -> None:
        whole = _FakeLM(num_layers)
        program = _stage_program(
            whole, stages, lambda forward, model, c: _forward(forward, model).logits
        )
        expected = _whole_logits(whole)
        for rank, logits in enumerate(_world(stages, schedule).run(program)):
            assert torch.equal(logits, expected), f"rank {rank}"

    def test_every_stage_but_the_last_sends_once_and_only_the_last_owns_the_logits(
        self,
    ) -> None:
        whole = _FakeLM(4)
        world = _world(4, 3)
        world.run(
            _stage_program(
                whole, 4, lambda forward, model, c: _forward(forward, model).logits
            )
        )
        sends = [e for e in world.transcript if e.op == "send"]
        recvs = [e for e in world.transcript if e.op == "recv"]
        broadcasts = [e for e in world.transcript if e.op == "broadcast"]
        assert sorted(e.rank for e in sends) == [0, 1, 2]
        assert sorted(e.rank for e in recvs) == [1, 2, 3]
        # one logits broadcast, every rank in it
        assert sorted(e.rank for e in broadcasts) == [0, 1, 2, 3]

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_read_on_stage_one_is_the_whole_value_on_every_rank(
        self, schedule: Schedule
    ) -> None:
        whole = _FakeLM(2)
        reference_site = _block_site(whole, 1, 2, 2)
        reference: dict[Any, torch.Tensor] = {}
        with _capturing(
            reference_site.module,
            "out",
            reference,
            "k",
            shape=bsd(HIDDEN),
            batch_size=BATCH,
        ):
            _whole_logits(whole)

        def body(forward: StageForward, model: _FakeLM, c: Collective) -> torch.Tensor:
            site = _block_site(model, 1, 2, 2)
            capture: dict[Any, torch.Tensor] = {tap_key(site): torch.empty(0)}
            fragments = Fragments(c)
            with contextlib.ExitStack() as hooks:
                if forward.installs(site):
                    hooks.enter_context(
                        _capturing(
                            site.module,
                            "out",
                            capture,
                            tap_key(site),
                            shape=bsd(HIDDEN),
                            batch_size=BATCH,
                            tap=TapFragments(fragments, hook_placement(site.placement)),
                        )
                    )
                _forward(forward, model)
            forward.broadcast_captures([site], capture, {})
            return capture[tap_key(site)]

        for rank, captured in enumerate(
            _world(2, schedule).run(_stage_program(whole, 2, body))
        ):
            assert torch.equal(captured, reference["k"]), f"rank {rank}"

    @given(schedule=ps.schedules(), seed=ps.seeds())
    @_SETTINGS
    def test_a_swap_on_stage_zero_and_a_read_on_stage_one_equal_the_whole_run(
        self, schedule: Schedule, seed: int
    ) -> None:
        whole = _FakeLM(2)
        replacement = torch.randn(
            BATCH, POSITIONS, HIDDEN, generator=torch.Generator().manual_seed(seed)
        )

        def swap(contract: torch.Tensor) -> None:
            contract.copy_(replacement)

        reference: dict[Any, torch.Tensor] = {}
        with (
            _installed(
                whole.model.layers[0], "out", swap, shape=bsd(HIDDEN), batch_size=BATCH
            ),
            _capturing(
                whole.model.layers[1],
                "out",
                reference,
                "k",
                shape=bsd(HIDDEN),
                batch_size=BATCH,
            ),
        ):
            expected_logits = _whole_logits(whole)

        def body(
            forward: StageForward, model: _FakeLM, c: Collective
        ) -> tuple[torch.Tensor, torch.Tensor]:
            fragments = Fragments(c)
            write = _block_site(model, 0, 2, 2)
            read = _block_site(model, 1, 2, 2)
            capture: dict[Any, torch.Tensor] = {tap_key(read): torch.empty(0)}
            with contextlib.ExitStack() as hooks:
                if forward.installs(write):
                    hooks.enter_context(
                        _installed(
                            write.module,
                            "out",
                            swap,
                            shape=bsd(HIDDEN),
                            batch_size=BATCH,
                            tap=TapFragments(
                                fragments, hook_placement(write.placement)
                            ),
                        )
                    )
                if forward.installs(read):
                    hooks.enter_context(
                        _capturing(
                            read.module,
                            "out",
                            capture,
                            tap_key(read),
                            shape=bsd(HIDDEN),
                            batch_size=BATCH,
                            tap=TapFragments(fragments, hook_placement(read.placement)),
                        )
                    )
                logits = _forward(forward, model).logits
            forward.broadcast_captures([read], capture, {})
            return capture[tap_key(read)], logits

        for rank, (captured, logits) in enumerate(
            _world(2, schedule).run(_stage_program(whole, 2, body))
        ):
            assert torch.equal(captured, reference["k"]), f"rank {rank}: read"
            assert torch.equal(logits, expected_logits), f"rank {rank}: logits"

    @pytest.mark.parametrize(
        ("num_layers", "stages", "start"), [(2, 2, 1), (4, 2, 2), (4, 2, 1), (4, 4, 3)]
    )
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_resume_from_the_cached_residual_equals_the_whole_run(
        self, num_layers: int, stages: int, start: int, schedule: Schedule
    ) -> None:
        """The residual entering block ``start`` is cached on the stage
        that owns the block; that stage starts from it with the blocks below
        swapped out, every stage below it runs nothing and sends nothing,
        every stage above runs after receiving."""
        whole = _FakeLM(num_layers)
        residual: list[torch.Tensor] = []

        def keep(_m: Any, args: tuple[Any, ...]) -> None:
            residual.append(args[0].detach().clone())

        handle = whole.model.layers[start].register_forward_pre_hook(keep)
        try:
            expected = _whole_logits(whole)
        finally:
            handle.remove()
        (cached,) = residual

        def body(forward: StageForward, model: _FakeLM, c: Collective) -> torch.Tensor:
            with contextlib.ExitStack() as swap:
                if forward.swaps(start):
                    swap.enter_context(_resumed(model.model.layers, start, cached))
                return _forward(forward, model, start=start).logits

        world = _world(stages, schedule)
        for rank, logits in enumerate(world.run(_stage_program(whole, stages, body))):
            assert torch.equal(logits, expected), f"rank {rank}"
        owner = stage_of(start, num_layers=num_layers, stages=stages)
        # the stages below the owner neither send nor receive
        for event in world.transcript:
            if event.op in ("send", "recv"):
                assert event.rank >= owner, event
        # the owner receives nothing: it starts from the residual it holds
        assert not [e for e in world.transcript if e.op == "recv" and e.rank == owner]

    def test_a_rank_that_skips_one_broadcast_is_caught_as_a_divergence(self) -> None:
        whole = _FakeLM(2)

        def body(forward: StageForward, model: _FakeLM, c: Collective) -> int:
            site = _block_site(model, 1, 2, 2)
            capture: dict[Any, torch.Tensor] = {tap_key(site): torch.zeros(1)}
            _forward(forward, model)
            if forward.stage != 1:  # rank 1 skips the broadcast it owes
                forward.broadcast_captures([site], capture, {})
            return c.agree_sum(1, "pipeline")  # the fire sum every rank reaches

        with pytest.raises(Divergence) as err:
            _world(2, 0).run(_stage_program(whole, 2, body))
        assert {err.value.op, err.value.first_op} == {"broadcast", "agree_sum"}

    def test_decode_under_a_pipeline_is_refused_by_name(self) -> None:
        whole = _FakeLM(2)

        def body(forward: StageForward, model: _FakeLM, c: Collective) -> str:
            try:
                forward.forward(
                    model,
                    hidden_size=HIDDEN,
                    device=CPU,
                    input_ids=IDS,
                    attention_mask=MASK,
                    position_ids=POS,
                    use_cache=True,
                )
            except ProtocolError as error:
                assert error.code == "P4"
                return str(error)
            return ""

        for message in _world(2, 1).run(_stage_program(whole, 2, body)):
            assert "--parallel.pipeline" in message and "decode" in message

    def test_world_one_calls_the_model_as_before_and_no_collective(self) -> None:
        from tests._helpers.refusing_collective import RefusingCollective

        whole = _FakeLM(2)
        forward = StageForward(RefusingCollective(), stages=1, stage=0, num_layers=2)
        with torch.no_grad():
            assert torch.equal(_forward(forward, whole).logits, _whole_logits(whole))
            # use_cache passes through at world 1
            forward.forward(
                whole,
                hidden_size=HIDDEN,
                device=CPU,
                input_ids=IDS,
                attention_mask=MASK,
                position_ids=POS,
                use_cache=True,
            )
        forward.broadcast_captures(
            [_block_site(whole, 1, 1, 2)],
            {tap_key(_block_site(whole, 1, 1, 2)): torch.zeros(1)},
            {},
        )


# --------------------------------------------------------------------------- #
# the stand-in and the resolver on a block this rank does not hold
# --------------------------------------------------------------------------- #


def _staged(bundle: ModelBundle, geometry: ParallelGeometry, rank: int) -> ModelBundle:
    """This rank's placed copy of ``bundle``: the real ``place_stage`` over a
    deep copy, the device map read off the copy (the stand-ins on the CPU).
    The placement is by the rank's pipeline coordinate alone — the model
    group's tensors stay whole here, the simulation cannot run the styles."""
    model = copy.deepcopy(bundle.model)
    stage = MeshLayout(geometry).rank_in(rank, "pipeline")
    pipeline = ParallelGeometry(pipeline=geometry.pipeline)
    place_stage(model, Sharding(geometry=pipeline, rank=stage, meshes={}))
    adapter = family_for(model)
    devices = DeviceMap.of_modules(
        walk(model, adapter.tree.embedding),
        list(adapter.blocks_of(model)),
        walk(model, adapter.tree.lm_head),
        empty=CPU,
    )
    return dataclasses.replace(bundle, model=model, geometry=geometry, devices=devices)


@pytest.fixture(scope="module")
def llama() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


@pytest.mark.unit
class TestAbsentBlocks:
    def test_a_stand_in_is_an_identity_that_shows_the_shadowed_tree(self) -> None:
        block = _Block(0)
        stand_in = StageStandIn(block)
        hidden = torch.randn(2, 3, HIDDEN)
        assert torch.equal(stand_in(hidden, attention_mask=None), hidden)
        assert stand_in.shadow is block
        assert stand_in.proj is block.proj  # the fall-through
        assert list(stand_in.parameters()) == [] and stand_in.state_dict() == {}
        assert list(stand_in.children()) == []
        with pytest.raises(AttributeError):
            stand_in.no_such_attribute

    def test_a_site_on_an_absent_block_resolves_with_its_owner_placement(
        self, llama: ModelBundle
    ) -> None:
        geometry = ParallelGeometry(pipeline=2)
        for rank in (0, 1):
            staged = _staged(llama, geometry, rank)
            for layer in (0, 1):
                site = resolve_site(
                    staged, SiteSpec(component="attention_output", layers=(layer,))
                )
                assert site.placement == StageLocal(layer)
                assert site.layer == layer and site.kind == "out"
                held = staged.holds(layer)
                assert held == (layer == rank)
                # the module is the held one or a shadow of it, never None
                assert site.module is not None
                if not held:
                    assert isinstance(staged.blocks[layer], StageStandIn)
                    assert site.module is staged.blocks[layer].shadow.self_attn
            # a head slice and the stream check read the registry either way
            head = resolve_site(
                staged,
                SiteSpec(component="attention_query_pre_rope", layers=(1,), head=1),
            )
            assert head.feature_slice is not None
            assert staged.stream_at(0) == staged.stream_at(1) == "full_attention"
            # the model boundary: the ids and the embedding on the first stage,
            # the norm and the head on the last
            assert resolve_site(staged, SiteSpec(component="input_ids")).placement == (
                StageLocal(0)
            )
            assert resolve_site(staged, SiteSpec(component="lm_head")).placement == (
                StageLocal(1)
            )

    def test_two_absent_sites_have_two_tap_keys(self, moe: ModelBundle) -> None:
        staged = _staged(moe, ParallelGeometry(pipeline=2), 1)
        a = resolve_site(staged, SiteSpec(component="block_output", layers=(0,)))
        b = resolve_site(staged, SiteSpec(component="block_output", layers=(1,)))
        assert not staged.holds(0) and not staged.holds(1)
        assert tap_key(a) != tap_key(b)
        # resolving twice names the same module — a capture keyed by the
        # first resolution is found by the second
        assert tap_key(a) == tap_key(
            resolve_site(staged, SiteSpec(component="block_output", layers=(0,)))
        )

    def test_the_hybrid_stream_of_an_absent_layer_is_the_registry_answer(
        self, moe: ModelBundle
    ) -> None:
        assert moe.info.layer_types is not None
        for rank in (0, 1):
            staged = _staged(moe, ParallelGeometry(pipeline=2), rank)
            assert staged.streams == moe.streams == tuple(moe.info.layer_types)

    def test_a_bare_identity_block_answers_from_layer_types_or_refuses(self) -> None:
        blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        mixers = {"self_attn": "full_attention", "linear_attn": "linear_attention"}
        assert (
            model_tree.stream_at(
                blocks,
                1,
                key="k",
                mixers=mixers,
                layer_types=("linear_attention", "full_attention"),
            )
            == "full_attention"
        )
        with pytest.raises(ProtocolError, match="no recognised mixer child"):
            model_tree.stream_at(blocks, 1, key="k", mixers=mixers)
        with pytest.raises(ProtocolError, match="does not hold"):
            model_tree.mixer_at(
                blocks,
                1,
                key="k",
                mixers=mixers,
                layer_types=("linear_attention", "full_attention"),
            )


# --------------------------------------------------------------------------- #
# the real engine under simulation
# --------------------------------------------------------------------------- #


def _interchange(key: str, write_layer: int, read_layer: int) -> dict[str, Any]:
    """A swap at ``write_layer`` scored at the head, with a read of the
    patched residual at ``read_layer`` beside it."""
    raw = swap_doc(key=key, layer=write_layer)
    raw["method"]["sites"]["probe_site"] = {
        "component": "block_output",
        "layers": [read_layer],
    }
    raw["method"]["reads"]["probe"] = {"site": "probe_site", "pos": {"index": -1}}
    raw["method"]["intervened_models"]["patched"]["reads"].append("probe")
    raw["method"]["save"].append(
        {"read": "probe", "model": "patched", "file_path": "probe.safetensors"}
    )
    return raw


Reads = dict[str, torch.Tensor]
Outcome = tuple[list[tuple[Reads, dict[Any, dict[str, int]]]], list[int] | None]


def _executor(
    raw: dict[str, Any], bundle: ModelBundle, interning: Any
) -> PointExecutor:
    return executor_for(
        raw,
        bundle,
        base_texts=BASES,
        counterfactual_texts=COUNTERFACTUALS,
        extra_columns={"label": ANSWERS},
        interning=interning,
    )


def _outcome(executor: PointExecutor) -> tuple[Reads, dict[Any, dict[str, int]]]:
    executor.run_all()
    reads = {name: executor.dense_value(name) for name in executor.doc.reads}
    return reads, {k: dict(v) for k, v in executor.fires.items()}


def _reference(
    raws: Sequence[dict[str, Any]], bundle: ModelBundle, *, campaign: bool
) -> Outcome:
    handles: list[Any]
    cache = None
    if campaign:
        _docs, handles, cache = _campaign(raws)
    else:
        handles = [None] * len(raws)
    outcomes = [_outcome(_executor(raw, bundle, h)) for raw, h in zip(raws, handles)]
    return outcomes, (list(cache.resumed) if cache is not None else None)


def _engine_program(
    raws: Sequence[dict[str, Any]],
    bundle: ModelBundle,
    geometry: ParallelGeometry,
    *,
    campaign: bool = False,
    block_fires: dict[int, list[list[int]]] | None = None,
) -> Callable[[int, Collective], Outcome | str]:
    """The rank program: the real executor over every document in order,
    the fragments bound to this rank's collective, on this rank's placed
    copy. A refusal is returned as its message so the caller can assert
    every rank made it."""

    def program(rank: int, c: Collective) -> Outcome | str:
        staged = _staged(bundle, geometry, rank)
        handles: list[Any]
        cache = None
        if campaign:
            _docs, handles, cache = _campaign(raws)
        else:
            handles = [None] * len(raws)
        outcomes = []
        with contextlib.ExitStack() as hooks:
            if block_fires is not None:
                block_fires[rank] = _count_block_fires(hooks, staged)
            for raw, handle in zip(raws, handles):
                executor = _executor(raw, staged, handle)
                executor.fragments = Fragments(c)
                try:
                    outcomes.append(_outcome(executor))
                except ProtocolError as error:
                    return str(error)
        return outcomes, (list(cache.resumed) if cache is not None else None)

    return program


def _count_block_fires(
    hooks: contextlib.ExitStack, bundle: ModelBundle
) -> list[list[int]]:
    """Per block, the batch size of every forward that reached it — a
    pre-hook on the block itself, which a stand-in or a skipped stage
    never runs a real block for."""
    fires: list[list[int]] = [[] for _ in bundle.blocks]
    for index, block in enumerate(bundle.blocks):
        if isinstance(block, StageStandIn):
            continue

        def hook(_m: Any, args: tuple[Any, ...], _kw: Any, *, i: int = index) -> None:
            hidden = args[0] if args else _kw["hidden_states"]
            fires[i].append(int(hidden.shape[0]))

        handle = block.register_forward_pre_hook(hook, with_kwargs=True)
        hooks.callback(handle.remove)
    return fires


def _assert_equal_outcomes(actual: Outcome | str, expected: Outcome, rank: int) -> None:
    assert not isinstance(actual, str), f"rank {rank} refused: {actual}"
    assert len(actual[0]) == len(expected[0])
    for (reads, fires), (want_reads, want_fires) in zip(actual[0], expected[0]):
        assert set(reads) == set(want_reads)
        for name in want_reads:
            assert torch.equal(reads[name], want_reads[name]), f"rank {rank}: {name}"
        assert fires == want_fires, f"rank {rank}: fires"
    assert actual[1] == expected[1], f"rank {rank}: resumed"


PP2 = ParallelGeometry(pipeline=2)


#: the sites a read of the head may capture: the head itself, or its input
#: when the read projects the head over its gathered rows (shared/head.py)
HEAD_CAPTURES = frozenset({"lm_head", "ln_final"})


@pytest.mark.property
class TestEngineScenarios:
    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_write_on_stage_zero_and_a_read_on_stage_one_equal_world_one(
        self, llama: ModelBundle, schedule: Schedule
    ) -> None:
        raws = [_interchange(TINY_LLAMA, write_layer=0, read_layer=1)]
        expected = _reference(raws, llama, campaign=False)
        assert expected[0][0][1] == {("patched", "base"): {"patch": 1}}
        world = _world(2, schedule)
        for rank, outcome in enumerate(world.run(_engine_program(raws, llama, PP2))):
            _assert_equal_outcomes(outcome, expected, rank)

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_a_write_on_stage_one_and_a_read_on_stage_zero_equal_world_one(
        self, llama: ModelBundle, schedule: Schedule
    ) -> None:
        raws = [_interchange(TINY_LLAMA, write_layer=1, read_layer=0)]
        expected = _reference(raws, llama, campaign=False)
        for rank, outcome in enumerate(
            _world(2, schedule).run(_engine_program(raws, llama, PP2))
        ):
            _assert_equal_outcomes(outcome, expected, rank)

    def test_two_stages_beside_two_tensor_ranks_equal_world_one(
        self, llama: ModelBundle
    ) -> None:
        """``pp=2`` over a model group of two: the stage's ranks hold the
        whole (replicated) tensors here, so the stage interior is the
        identity and the broadcast crosses the pipeline groups alone."""
        raws = [_interchange(TINY_LLAMA, write_layer=0, read_layer=1)]
        expected = _reference(raws, llama, campaign=False)
        geometry = ParallelGeometry(pipeline=2, tensor=2)
        world = _world(2, 4, tensor=2)
        for rank, outcome in enumerate(
            world.run(_engine_program(raws, llama, geometry))
        ):
            _assert_equal_outcomes(outcome, expected, rank)

    @given(schedule=ps.schedules())
    @_SETTINGS
    def test_the_layer_scan_resumes_across_the_stage_boundary(
        self, moe: ModelBundle, schedule: Schedule
    ) -> None:
        """One point per layer of the four-layer MoE at ``pp=2``: the
        layer-2 point resumes at block 1 (stage 0's), the layer-3 point at
        block 2 — stage 1's, so stage 0 runs no block at all for it and
        sends nothing. Every read, the fires record and the resume
        bookkeeping equal the world-1 scan."""
        n = len(moe.blocks)
        raws = [swap_doc(key=TINY_QWEN35_MOE, layer=layer) for layer in range(n)]
        expected = _reference(raws, moe, campaign=True)
        assert expected[1] == [1, 2]
        fires: dict[int, list[list[int]]] = {}
        world = _world(2, schedule)
        outcomes = world.run(
            _engine_program(raws, moe, PP2, campaign=True, block_fires=fires)
        )
        for rank, outcome in enumerate(outcomes):
            _assert_equal_outcomes(outcome, expected, rank)
        # a stage's real blocks fire exactly as the world-1 scan's do
        # (test_prefix_resume): the layer-0 point runs the harvest and a whole
        # patched forward through every block (2), the layer-1 point one whole
        # forward (1), and each later point reaches block ``b`` iff
        # ``b >= L - 1`` — so blocks 0..3 fire 3, 4, 5, 5 times, each on the
        # one stage that holds them, and the layer-3 point (resumed at block 2,
        # stage 1's) reaches no block of stage 0 at all
        assert [len(f) for f in fires[0]] == [3, 4, 0, 0]
        assert [len(f) for f in fires[1]] == [0, 0, 5, 5]
        assert all(size == 4 for f in (*fires[0], *fires[1]) for size in f)

    def test_a_write_that_never_fires_is_refused_on_every_rank(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The owner's hook is suppressed: its count is 0 and so is every
        other stage's, so the pipeline's sum is 0 against the declared 1 —
        and the refusal is made on every rank, the non-owner included,
        because the non-owner declared the member too."""
        raws = [_interchange(TINY_LLAMA, write_layer=0, read_layer=1)]

        @contextlib.contextmanager
        def never_installed(*args: Any, **kwargs: Any) -> Any:
            yield

        monkeypatch.setattr(executor_module, "_installed", never_installed)
        for message in _world(2, 2).run(_engine_program(raws, llama, PP2)):
            assert isinstance(message, str), message
            assert "'patch'" in message and "fired 0 times" in message

    def test_a_rank_that_skips_one_broadcast_is_caught_as_a_divergence(
        self, llama: ModelBundle, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raws = [_interchange(TINY_LLAMA, write_layer=0, read_layer=1)]
        real = stages_module.StageForward.broadcast_captures

        def skipping(self: StageForward, sites: Any, capture: Any, idx: Any) -> None:
            # stage 1 owes the head's capture to stage 0 — `ln_final`, the
            # site a positional `lm_head` read taps (shared/head.py) — and
            # skips it; the next collective it reaches — the fire sum — is
            # not the broadcast stage 0 is waiting in
            if self.stage == 1:
                sites = [site for site in sites if site.component not in HEAD_CAPTURES]
            real(self, sites, capture, idx)

        monkeypatch.setattr(stages_module.StageForward, "broadcast_captures", skipping)
        with pytest.raises(Divergence) as err:
            _world(2, 0).run(_engine_program(raws, llama, PP2))
        assert err.value.rank != err.value.first_rank
        assert {err.value.op, err.value.first_op} == {"broadcast", "agree_sum"}

    def test_the_executor_module_reaches_the_stages_seam(self) -> None:
        assert executor_module.StageForward is stages_module.StageForward


@pytest.mark.unit
class TestWorldOne:
    def test_the_solo_executor_builds_a_one_stage_forward(
        self, llama: ModelBundle
    ) -> None:
        executor = _executor(_interchange(TINY_LLAMA, 0, 1), llama, None)
        stages = executor._stages
        assert stages.stages == 1 and stages.stage == 0
        assert stages.num_layers == len(llama.blocks)

    def test_a_bundle_and_a_collective_that_disagree_are_refused(
        self, llama: ModelBundle
    ) -> None:
        class _Two:
            def rank(self, axis: str) -> int:
                return 0

            def size(self, axis: str) -> int:
                return 2 if axis == "pipeline" else 1

        with pytest.raises(ProtocolError) as err:
            StageForward.of(llama, _Two())  # type: ignore[arg-type]
        assert "--parallel.pipeline" in str(err.value)
        staged = _staged(llama, PP2, 1)
        # the copy placed for stage 1 handed a collective saying stage 0
        stages = StageForward.of(staged, _Two())  # type: ignore[arg-type]
        with pytest.raises(ProtocolError, match="stage 0 of 2"):
            stages.check_placement(staged)
        placed = _staged(llama, PP2, 0)
        StageForward.of(placed, _Two()).check_placement(placed)  # type: ignore[arg-type]

    def test_a_rank_failure_inside_the_world_names_the_rank(
        self, llama: ModelBundle
    ) -> None:
        def program(rank: int, c: Collective) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RankFailed):
            _world(2, 0).run(program)
