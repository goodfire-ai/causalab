"""The §4 style → placement table as code, and its derivation for a site.

``docs/model_parallelism.md`` §4 (the table), §6.1 (module-boundary
components), §6.3 (the experts interior and the remapped routing table), §6.5
(stage ownership under pipeline placement). The ``property`` tier holds the
closed style set: every style maps on both sides and every unknown one is
refused by name; the ``unit`` tier spells the table row by row, the stage
composition, the derivation through ``resolve_site`` from the **registry's
own plans** (``ModelInfo.parallel_plan``) and the bundle's geometry, and the
census — every component of the A3B sweep on both tiny fixtures at ``tp=2``,
``ep=4`` and ``tp=2,ep=4``, table-driven: component → the placement the
hooks must see.

Two rows the design's table had wrong, corrected here from transformers'
own code: ``replicated_with_grad_allreduce`` (Qwen's ``q_norm`` / ``k_norm``)
sits between a colwise projection and the attention function and *its
activation is this rank's heads* — the parameter is replicated, the tensor is
not — so both sides are sharded on the head axis; and ``ep_router``'s second
output, the routing scores, is zeroed on the slots this rank does not own
(``EpRouterParallel``), which is exactly [`ExpertLocal`][causalab.neural.shared.parallel.placement.ExpertLocal].
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.parallel import placements
from causalab.neural.shared.parallel.fragments import Fragments, fragment, whole
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    ExpertLocal,
    Placement,
    Replicated,
    SequenceSharded,
    Sharded,
    StageLocal,
)
from causalab.neural.shared.parallel.placements import (
    INHERITED,
    STYLES,
    StyleError,
    TapPlacement,
    child_rows,
    experts_placement,
    interior_placement,
    placement_for,
    sequenced,
    shard_axis,
    stage_of,
)
from causalab.neural.shared.sites import ResolvedSite, resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import ONE, ParallelGeometry
from causalab.protocol.registry import (
    ATTENTION_FUNCTION_SLOTS,
    LLAMA_PLAN,
    QWEN35_MOE_PLAN,
    ParallelPlan,
    PlanRow,
)
from causalab.protocol.schema import LAYERLESS_COMPONENTS, SiteSpec
from causalab.protocol.registry.shapes import (
    Axis,
    FeatureShape,
    attention_pattern,
    bsd,
    bshd,
    flat_td,
)
from tests._helpers import a3b_sweep
from tests._helpers.simulated_world import SimulatedWorld, groups_for
from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

TP = Sharded(-1, "tensor")
EL = ExpertLocal("expert")
#: The two styles a routed-experts module or its parameters carry.
_EXPERTS_STYLES = frozenset({"moe_tp_experts", "grouped_gemm"})


def _row(style: str, axis: str = "tensor") -> PlanRow:
    return PlanRow(style, axis)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# property tier: the closed style set
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestStyleTableProperties:
    @_SETTINGS
    @given(
        style=st.sampled_from(sorted(STYLES)),
        axis=st.sampled_from(["tensor", "expert"]),
        side=st.sampled_from(["input", "output"]),
        interior=st.booleans(),
    )
    def test_every_style_maps_on_both_sides(
        self, style: str, axis: str, side: str, interior: bool
    ) -> None:
        # the one cell the table refuses by name: the experts interior under
        # tensor-parallel experts (§6.3, second half) — its own unit test below
        assume(not (style in _EXPERTS_STYLES and axis == "tensor" and interior))
        placed = placement_for(_row(style, axis), side, interior=interior)
        assert isinstance(placed, TapPlacement)
        assert isinstance(placed.placement, (Replicated, Sharded, ExpertLocal))
        # a group is always the row's own axis
        group = getattr(placed.placement, "group", axis)
        assert group == axis

    @_SETTINGS
    @given(
        style=st.text(min_size=1, max_size=24).filter(lambda s: s not in STYLES),
        side=st.sampled_from(["input", "output"]),
    )
    def test_every_unknown_style_is_refused_by_name(
        self, style: str, side: str
    ) -> None:
        with pytest.raises(StyleError) as err:
            placement_for(_row(style), side)
        assert repr(style) in str(err.value)
        assert err.value.style == style

    @_SETTINGS
    @given(
        num_layers=st.integers(1, 40),
        stages=st.integers(1, 8),
    )
    def test_stage_ranges_partition_the_layers_contiguously(
        self, num_layers: int, stages: int
    ) -> None:
        if stages > num_layers:
            with pytest.raises(StyleError):
                stage_of(0, num_layers=num_layers, stages=stages)
            return
        owners = [
            stage_of(layer, num_layers=num_layers, stages=stages)
            for layer in range(num_layers)
        ]
        assert owners == sorted(owners), "stages own contiguous ranges in order"
        assert set(owners) == set(range(stages)), "every stage owns a layer"
        # even ranges, the remainder to the last stage (the DeviceMap rule)
        base = num_layers // stages
        for stage in range(stages - 1):
            assert owners.count(stage) == base


# --------------------------------------------------------------------------- #
# unit tier: the table, row by row
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestStyleTable:
    def test_colwise_input_replicated_output_sharded_on_the_feature_axis(self) -> None:
        assert placement_for(_row("colwise"), "input").placement == REPLICATED
        assert placement_for(_row("colwise"), "output").placement == TP

    def test_packed_colwise_output_is_the_gate_and_up_runs_each_sharded(self) -> None:
        """The fused ``[gate | up]`` projection's output is two contiguous
        runs, each sharded across the group: ``slots=2`` — what the styles
        contract's ``_PACKED_OUT`` states and ``FragmentStyles`` installs. 📐 A
        plain ``Sharded`` here would interleave the ranks' gate slices ahead
        of every up slice on the gathered whole."""
        assert placement_for(_row("packed_colwise"), "input").placement == REPLICATED
        packed = placement_for(_row("packed_colwise"), "output").placement
        assert packed == Sharded(-1, "tensor", slots=2)
        assert packed != TP

    def test_rowwise_input_sharded_output_replicated_after_its_all_reduce(self) -> None:
        assert placement_for(_row("rowwise"), "input").placement == TP
        assert placement_for(_row("rowwise"), "output").placement == REPLICATED

    @pytest.mark.parametrize("style", ["colwise_gather_output", "moe_tp_experts"])
    def test_gathered_styles_are_replicated_on_both_sides(self, style: str) -> None:
        # A replicated norm weight does not imply replicated activations:
        # the norm receives and returns this rank's head shard. See
        # test_interior_placements.py.
        for side in ("input", "output"):
            assert placement_for(_row(style), side).placement == REPLICATED

    def test_the_replicated_weight_style_passes_this_ranks_heads_through(self) -> None:
        """``replicated_with_grad_allreduce`` replicates the *parameter*; the
        activation through it is the colwise producer's local shard
        (transformers: "the activation is already local"), so both sides are
        sharded — on the head axis the tap keeps, or the flat feature axis."""
        row = _row("replicated_with_grad_allreduce")
        assert placement_for(row, "input").placement == TP
        assert placement_for(row, "output").placement == TP
        assert placement_for(row, "output", axis=2).placement == Sharded(2, "tensor")

    def test_the_shard_axis_is_the_kept_head_axis_or_the_last(self) -> None:
        assert shard_axis(bshd(8, 32)) == 2  # (batch, position, head, feature)
        assert shard_axis(attention_pattern(8)) == 1  # (batch, head, q, k)
        assert shard_axis(bsd(64)) == -1  # the flat feature axis

    def test_grouped_gemm_module_output_is_replicated_and_the_interior_expert_local(
        self,
    ) -> None:
        """§6.3: the experts module's own output is complete after its
        all-reduce; only the token-major interior the experts interface taps
        is this rank's slots."""
        row = _row("grouped_gemm", "expert")
        assert placement_for(row, "output").placement == REPLICATED
        assert placement_for(row, "input").placement == REPLICATED
        interior = placement_for(row, "output", interior=True)
        assert interior.placement == EL
        assert interior.remapped_routing, "the interior rides the remapped table"

    def test_expert_parallel_experts_interior_is_expert_local_too(self) -> None:
        """The experts *module* carries ``moe_tp_experts`` on the expert axis
        under an EP plan (the parameters carry ``grouped_gemm``), and the
        interior tapped through it is the same rank-owned slots."""
        row = _row("moe_tp_experts", "expert")
        interior = placement_for(row, "output", interior=True)
        assert interior.placement == EL and interior.remapped_routing
        assert placement_for(row, "output").placement == REPLICATED

    def test_the_experts_interior_under_tensor_parallel_experts_is_refused(
        self,
    ) -> None:
        """§6.3's second half — per-slot ``Sharded(d_expert, tp)`` — is not
        served: refused by name rather than read as replicated."""
        for style in sorted(_EXPERTS_STYLES):
            with pytest.raises(ProtocolError, match=style) as err:
                placement_for(_row(style, "tensor"), "output", interior=True)
            assert err.value.code == "P4"

    def test_ep_router_output_is_replicated_and_flagged_remapped(self) -> None:
        placed = placement_for(_row("ep_router", "expert"), "output")
        assert placed.placement == REPLICATED
        assert placed.remapped_routing
        assert not placement_for(_row("ep_router", "expert"), "input").remapped_routing

    def test_ep_routers_three_outputs_are_three_placements(self) -> None:
        """``(router_logits, router_scores, router_indices)``: the logits pass
        through untouched, the scores are zeroed off this rank's experts
        (owned slots — [`ExpertLocal`][causalab.neural.shared.parallel.placement.ExpertLocal]), the indices are the remapped
        table."""
        row = _row("ep_router", "expert")
        logits = placement_for(row, "output", tuple_index=0)
        assert logits.placement == REPLICATED and not logits.remapped_routing
        scores = placement_for(row, "output", tuple_index=1)
        assert scores.placement == EL and not scores.remapped_routing
        indices = placement_for(row, "output", tuple_index=2)
        assert indices.placement == REPLICATED and indices.remapped_routing

    def test_no_row_is_replicated_and_unflagged(self) -> None:
        for side in ("input", "output"):
            placed = placement_for(None, side)
            assert placed.placement == REPLICATED and not placed.remapped_routing

    def test_only_the_flagged_rows_carry_the_routing_flag(self) -> None:
        flagged = {
            style
            for style in STYLES
            if placement_for(_row(style, "expert"), "output").remapped_routing
        }
        assert flagged == {"ep_router"}

    def test_the_stage_wraps_a_block_scoped_placement_and_keeps_its_interior(
        self,
    ) -> None:
        """Compose, do not replace: under ``pipeline > 1`` a colwise output
        is ``StageLocal(stage, inner=Sharded(-1))`` — the owner's ranks still
        hold shards of it."""
        placed = placement_for(_row("colwise"), "output", stage=1)
        assert placed.placement == StageLocal(1, inner=TP)
        assert placement_for(None, "output", stage=0).placement == StageLocal(0)
        assert placement_for(_row("colwise"), "output", stage=None).placement == TP

    def test_a_side_outside_the_two_is_refused(self) -> None:
        with pytest.raises(StyleError):
            placement_for(_row("colwise"), "sideways")  # type: ignore[arg-type]

    def test_stage_of_spells_the_device_map_rule(self) -> None:
        # 5 layers on 2 stages: [0, 1] and [2, 3, 4] — the remainder to the last
        assert [stage_of(i, num_layers=5, stages=2) for i in range(5)] == [
            0,
            0,
            1,
            1,
            1,
        ]
        assert stage_of(3, num_layers=4, stages=1) == 0
        with pytest.raises(StyleError):
            stage_of(5, num_layers=5, stages=2)

    def test_the_inherited_components_are_the_two_between_projections_and_the_normed_key(
        self,
    ) -> None:
        """A rowless module between two planned projections carries its
        neighbour's tensor: the dense activation (before the rowwise
        ``down_proj``) and the attention pattern (the colwise ``q_proj``'s
        heads); and the key before RoPE carries ``k_proj``'s output — on a
        family with a ``k_norm`` the norm's own row replicates its weight
        and says nothing about the projection, which is sharded or
        replicated (§6.6) by the geometry."""
        assert dict(INHERITED) == {
            "mlp_activation": ("down_proj", "input"),
            "attention_probs": ("q_proj", "output"),
            "attention_key_pre_rope": ("k_proj", "output"),
        }

    def test_the_replicated_kv_projection_is_whole_in_and_repeated_out(self) -> None:
        """§6.6: ``kv_replicated`` holds the projection's weight whole on
        every rank; its output is the one KV head this rank's query heads
        read, held by ``repeat`` consecutive ranks — a repeated shard, whose
        ``whole`` is the model's KV heads."""
        row = PlanRow("kv_replicated", "tensor", repeat=4)
        assert placement_for(row, "input").placement == REPLICATED
        assert placement_for(row, "output").placement == Sharded(-1, "tensor", repeat=4)
        assert placement_for(row, "output", axis=2).placement == Sharded(
            2, "tensor", repeat=4
        )
        # the default row (repeat 1) is the plain shard the property tier draws
        assert placement_for(_row("kv_replicated"), "output").placement == TP


@pytest.mark.unit
class TestStageComposition:
    """The composed placement through the real ``whole`` / ``fragment``: the
    owner stage's ranks gather their shards, then the stage broadcasts."""

    def test_whole_and_fragment_round_trip_a_stage_local_sharded_tensor(self) -> None:
        world = SimulatedWorld(groups_for(4, pipeline=2, tensor=2), world=4, schedule=1)
        placement = StageLocal(1, inner=TP)
        global_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

        def program(rank: int, c: Any) -> tuple[torch.Tensor, torch.Tensor]:
            stage, shard = c.rank("pipeline"), c.rank("tensor")
            # what this rank holds, spelled by hand: owners a 2-wide chunk, others nothing
            local = (
                global_tensor.narrow(-1, shard * 2, 2)
                if stage == 1
                else global_tensor.new_empty((0, 3, 4))
            )
            made_whole = whole(local, placement, c)
            back = fragment(made_whole, placement, c)
            return made_whole, back

        results = world.run(program)
        for rank, (made_whole, back) in enumerate(results):
            assert torch.equal(made_whole, global_tensor), f"rank {rank}"
            stage, shard = divmod(rank, 2)
            if stage == 1:
                assert torch.equal(back, global_tensor.narrow(-1, shard * 2, 2))
            else:
                assert back.shape == (0, 3, 4)

    def test_a_stage_group_of_one_still_gathers_its_interior(self) -> None:
        """``pipeline=1, tensor=2``: the stage wrapper is the identity but the
        inner shard is not — the fast path must look inside."""
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)
        placement = StageLocal(0, inner=TP)
        global_tensor = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)

        def program(rank: int, c: Any) -> torch.Tensor:
            return Fragments(c).whole(global_tensor.narrow(-1, rank * 2, 2), placement)

        for made_whole in world.run(program):
            assert torch.equal(made_whole, global_tensor)


# --------------------------------------------------------------------------- #
# unit tier: the derivation through resolve_site, from the registry's plan
# --------------------------------------------------------------------------- #

_ENTRYS_PLAN = object()


def _under(
    bundle: ModelBundle, geometry: ParallelGeometry, plan: Any = _ENTRYS_PLAN
) -> ModelBundle:
    """The world-1 bundle as a rank of ``geometry`` would hold it: the same
    model, the entry's plan (or the one given), the geometry recorded."""
    info = bundle.info
    if plan is not _ENTRYS_PLAN:
        info = dataclasses.replace(info, parallel_plan=plan)
    return dataclasses.replace(bundle, info=info, geometry=geometry)


class _Never(ParallelPlan):
    def style_for(self, path: str, *, prefix: str | None = None) -> PlanRow | None:
        raise AssertionError("the plan was read at world 1")


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return load_model(TINY_LLAMA)


@pytest.fixture(scope="module")
def moe() -> ModelBundle:
    return load_model(TINY_QWEN35_MOE)


def _resolved(
    bundle: Any, component: str, layer: int = 1, **extra: Any
) -> ResolvedSite:
    layers = None if component in LAYERLESS_COMPONENTS else (layer,)
    return resolve_site(bundle, SiteSpec(component=component, layers=layers, **extra))


def _site(bundle: Any, component: str, layer: int = 1) -> Placement:
    return _resolved(bundle, component, layer).placement


def _tensor_rows_on(axis: str) -> ParallelPlan:
    return ParallelPlan(
        {pattern: _row(row.style, axis) for pattern, row in LLAMA_PLAN.rows.items()}
    )


@pytest.mark.unit
class TestResolveSitePlacement:
    def test_at_world_one_every_site_is_replicated_and_the_plan_is_never_read(
        self, bundle: ModelBundle
    ) -> None:
        placed = _under(bundle, ONE, _Never({}))
        for component in (
            "block_output",
            "attention_query_pre_rope",
            "attention_output",
            "lm_head",
        ):
            assert _site(placed, component) == REPLICATED

    def test_the_loaded_entry_carries_the_registrys_plan(
        self, bundle: ModelBundle, moe: ModelBundle
    ) -> None:
        assert bundle.info.parallel_plan == LLAMA_PLAN
        assert moe.info.parallel_plan == QWEN35_MOE_PLAN
        assert bundle.geometry == ONE and moe.geometry == ONE

    def test_the_tensor_plan_places_the_projections_and_leaves_the_stream_whole(
        self, bundle: ModelBundle
    ) -> None:
        placed = _under(bundle, ParallelGeometry(tensor=2))
        # q_proj's output is a colwise shard (§4 row 1)
        assert _site(placed, "attention_query_pre_rope") == TP
        assert _site(placed, "attention_value_states") == TP
        # the attention-function slots (§6.2) are local head shards: the
        # mixer's projections are colwise, and the slot's head axis is dim 1
        assert _site(placed, "attention_query") == Sharded(1, "tensor")
        # o_proj's *input* is the sharded premix; its output has been all-reduced
        assert _site(placed, "attention_premix") == TP
        assert _site(placed, "attention_result") == TP
        assert _site(placed, "attention_output") == REPLICATED
        # the residual stream and the model boundary are in no row
        assert _site(placed, "block_output") == REPLICATED
        assert _site(placed, "mlp_output") == REPLICATED
        assert _site(placed, "lm_head") == REPLICATED
        assert _site(placed, "embeddings") == REPLICATED
        # the two inherited placements: the activation before down_proj, the
        # pattern the mixer returns (its head axis is native axis 1)
        assert _site(placed, "mlp_activation") == TP
        assert _site(placed, "attention_probs") == Sharded(1, "tensor")

    def test_above_the_kv_heads_the_key_and_value_are_repeated_shards(
        self, moe: ModelBundle
    ) -> None:
        """§6.6 on the MoE fixture (8 heads, 4 KV heads) at ``tp=8``: the
        plan the sites are placed from is the registry's rewritten for the
        geometry — ``k_proj`` / ``v_proj`` replicated, each rank holding KV
        head ``rank // 2`` — so ``k_norm``'s output (the key before RoPE,
        inherited from ``k_proj``), the value states and the ``key`` slot
        inside the attention function are repeated shards whose ``whole``
        is the four KV heads, while the query-space slots are the plain
        head shards they are at ``tp=2``."""
        placed = _under(moe, ParallelGeometry(tensor=8))
        layer = moe.streams.index("full_attention")
        assert _site(placed, "attention_key_pre_rope", layer) == Sharded(
            2, "tensor", repeat=2
        )
        assert _site(placed, "attention_value_states", layer) == Sharded(
            -1, "tensor", repeat=2
        )
        assert _site(placed, "attention_key", layer) == Sharded(1, "tensor", repeat=2)
        assert _site(placed, "attention_query", layer) == Sharded(1, "tensor")
        assert _site(placed, "attention_query_pre_rope", layer) == Sharded(2, "tensor")
        assert _site(placed, "attention_scores", layer) == Sharded(1, "tensor")
        assert _site(placed, "attention_probs", layer) == Sharded(1, "tensor")
        assert _site(placed, "attention_z", layer) == Sharded(2, "tensor")
        assert _site(placed, "attention_premix", layer) == TP
        assert _site(placed, "attention_output", layer) == REPLICATED
        # at tp=4 the KV heads are sharded, and nothing is rewritten
        divided = _under(moe, ParallelGeometry(tensor=4))
        assert _site(divided, "attention_key_pre_rope", layer) == Sharded(2, "tensor")
        assert _site(divided, "attention_key", layer) == Sharded(1, "tensor")
        assert _site(divided, "attention_value_states", layer) == TP

    def test_a_row_on_an_axis_of_size_one_is_not_read(
        self, bundle: ModelBundle
    ) -> None:
        """``apply_plan`` applies no row on an axis of size one; the
        placement table reads the same rows, so a tensor row moved onto the
        expert axis leaves the projection whole under ``tp=2`` alone."""
        placed = _under(bundle, ParallelGeometry(tensor=2), _tensor_rows_on("expert"))
        assert _site(placed, "attention_query_pre_rope") == REPLICATED

    def test_under_pipeline_every_block_scoped_site_is_wrapped_in_its_stage(
        self, bundle: ModelBundle
    ) -> None:
        placed = _under(bundle, ParallelGeometry(pipeline=2, tensor=2))
        # two stages own the first and the last block respectively
        last = len(bundle.blocks) - 1
        assert last >= 1
        assert _site(placed, "block_output", layer=0) == StageLocal(0)
        assert _site(placed, "block_output", layer=last) == StageLocal(1)
        assert _site(placed, "attention_query_pre_rope", layer=last) == StageLocal(
            1, inner=TP
        )
        # the model boundary: the embedding on the first stage, the head on the last
        assert _site(placed, "embeddings") == StageLocal(0)
        assert _site(placed, "lm_head") == StageLocal(1)
        assert _site(placed, "ln_final") == StageLocal(1)
        # the ids: every rank encodes them, but the embedding whose input
        # they are runs on the first stage, and a tap is where its hook fires
        assert _site(placed, "input_ids") == StageLocal(0)

    def test_a_placed_site_is_a_resolved_site_with_the_rest_untouched(
        self, bundle: ModelBundle
    ) -> None:
        placed = _under(bundle, ParallelGeometry(tensor=2))
        spec = SiteSpec(component="attention_query_pre_rope", layers=(1,))
        with_plan = resolve_site(placed, spec)
        without = resolve_site(bundle, spec)
        assert dataclasses.replace(with_plan, placement=REPLICATED) == without
        assert not with_plan.remapped_routing

    def test_a_row_naming_an_unknown_style_is_refused_by_name_at_resolution(
        self, bundle: ModelBundle
    ) -> None:
        plan = ParallelPlan({"layers.*.self_attn.q_proj": _row("diagonal")})
        placed = _under(bundle, ParallelGeometry(tensor=2), plan)
        with pytest.raises(StyleError, match="'diagonal'"):
            _site(placed, "attention_query_pre_rope")

    def test_the_module_path_lookup_names_the_tapped_module(
        self, bundle: ModelBundle
    ) -> None:
        module = bundle.model.model.layers[1].self_attn.q_proj
        assert (
            placements.module_path(bundle.model, module)
            == "model.layers.1.self_attn.q_proj"
        )
        with pytest.raises(StyleError):
            placements.module_path(bundle.model, torch.nn.Linear(2, 2))

    def test_the_prefix_rule_is_the_models_base_model_prefix(
        self, moe: ModelBundle
    ) -> None:
        """The plan's patterns are spelled without the base-model prefix
        (``layers.*.mlp.gate``); the tapped module's path carries it
        (``model.layers.3.mlp.gate``). The prefix stripped is the model's
        own, the way ``apply_plan`` matches."""
        assert moe.model.base_model_prefix == "model"
        placed = _under(moe, ParallelGeometry(expert=4))
        assert _resolved(placed, "expert_idx", 3).remapped_routing


# --------------------------------------------------------------------------- #
# unit tier: the census — every sweep component, both fixtures, three geometries
# --------------------------------------------------------------------------- #

#: Where a tap's tensor lives under tensor parallelism, per fixture: the
#: colwise outputs and rowwise inputs of §4, the head-kept norms of Qwen
#: (``Sharded`` on the native head axis), the two inherited components, and
#: the attention-function slots (§6.2: local heads on the slot's own head
#: axis — dim 1 for ``(b, H, s, d)`` and the pattern, dim 2 for ``z``).
#: Everything else the sweep reaches is replicated.
TENSOR_PLACED: dict[str, dict[str, Placement]] = {
    "llama": {
        "attention_query_pre_rope": TP,
        "attention_key_pre_rope": TP,
        "attention_value_states": TP,
        "attention_premix": TP,
        "attention_result": TP,
        "attention_probs": Sharded(1, "tensor"),
        "attention_query": Sharded(1, "tensor"),
        "attention_key": Sharded(1, "tensor"),
        "attention_scores": Sharded(1, "tensor"),
        "attention_z": Sharded(2, "tensor"),
        "mlp_activation": TP,
    },
    "moe": {
        # q_norm / k_norm keep the head axis: (batch, position, head, feature)
        "attention_query_pre_rope": Sharded(2, "tensor"),
        "attention_key_pre_rope": Sharded(2, "tensor"),
        "attention_value_states": TP,
        "attention_gate": TP,
        "attention_premix": TP,
        "attention_result": TP,
        "attention_probs": Sharded(1, "tensor"),
        "attention_query": Sharded(1, "tensor"),
        "attention_key": Sharded(1, "tensor"),
        "attention_scores": Sharded(1, "tensor"),
        "attention_z": Sharded(2, "tensor"),
        "shared_expert_gate_proj": TP,
        "shared_expert_up_proj": TP,
        "shared_expert_activation": TP,
    },
}

#: Under expert parallelism: the experts interior (§6.3) and the router's
#: scores are this rank's slots; the routing table itself is replicated and
#: flagged for reconstruction. ``(placement, remapped_routing)``.
EXPERT_PLACED: dict[str, tuple[Placement, bool]] = {
    "expert_gate_proj": (EL, True),
    "expert_up_proj": (EL, True),
    "expert_activation": (EL, True),
    "expert_neuron_output": (EL, True),
    "expert_output": (EL, True),
    "router_scores": (EL, False),
    "expert_idx": (REPLICATED, True),
}

SWEEP = (
    a3b_sweep.SHARED_LAYERLESS
    + a3b_sweep.SHARED_ANY_STREAM
    + a3b_sweep.SHARED_FULL_ONLY
    + a3b_sweep.SHARED_LINEAR_ONLY
    + a3b_sweep.HOOKS_ONLY
    + ("mlp_activation",)
)

GEOMETRIES = (
    ParallelGeometry(tensor=2),
    ParallelGeometry(expert=4),
    ParallelGeometry(tensor=2, expert=4),
)


def _expected(
    fixture: str, component: str, geometry: ParallelGeometry
) -> tuple[Placement, bool]:
    if geometry.tensor > 1 and component in TENSOR_PLACED[fixture]:
        return TENSOR_PLACED[fixture][component], False
    if geometry.expert > 1 and component in EXPERT_PLACED:
        return EXPERT_PLACED[component]
    return REPLICATED, False


def _layers(bundle: ModelBundle, component: str) -> tuple[int, ...]:
    """The layers the census resolves ``component`` at: both ends of the
    tower for a component every block carries, the one block of its stream
    otherwise."""
    if component in LAYERLESS_COMPONENTS:
        return (0,)
    streams = bundle.streams
    full = streams.index("full_attention")
    linear = (
        streams.index("linear_attention") if "linear_attention" in streams else full
    )
    if component in a3b_sweep.SHARED_FULL_ONLY:
        return (full,)
    if component in a3b_sweep.SHARED_LINEAR_ONLY or component in a3b_sweep.HOOKS_ONLY:
        return (linear,)
    return tuple(sorted({0, len(bundle.blocks) - 1}))


def _present(bundle: ModelBundle, component: str, layer: int) -> bool:
    """Whether the fixture has the component at ``layer`` at world 1 — a
    dense fixture has no experts, a DeltaNet layer no attention."""
    try:
        _resolved(bundle, component, layer)
    except ProtocolError:
        return False
    return True


@pytest.mark.unit
class TestPlacedFromTheRegistryPlan:
    @pytest.mark.parametrize(
        "geometry", GEOMETRIES, ids=lambda g: f"tp{g.tensor}ep{g.expert}"
    )
    @pytest.mark.parametrize("fixture", ["llama", "moe"])
    def test_every_sweep_component_is_placed_as_the_table_says(
        self,
        fixture: str,
        geometry: ParallelGeometry,
        bundle: ModelBundle,
        moe: ModelBundle,
    ) -> None:
        world_one = bundle if fixture == "llama" else moe
        if fixture == "llama" and geometry.expert > 1:
            pytest.skip("a dense model has no expert axis (check refuses ep > 1)")
        placed = _under(world_one, geometry)
        seen: set[str] = set()
        for component in SWEEP:
            for layer in _layers(world_one, component):
                if not _present(world_one, component, layer):
                    continue
                site = _resolved(placed, component, layer)
                expected = _expected(fixture, component, geometry)
                assert (site.placement, site.remapped_routing) == expected, (
                    fixture,
                    component,
                    layer,
                    geometry,
                )
                seen.add(component)
        # the census covered every row the table names for this fixture
        assert set(TENSOR_PLACED[fixture]) <= seen
        if fixture == "moe":
            assert set(EXPERT_PLACED) <= seen

    def test_the_stream_at_world_one_is_the_stream_at_any_geometry(
        self, moe: ModelBundle
    ) -> None:
        placed = _under(moe, ParallelGeometry(tensor=2, expert=4))
        for layer in range(len(moe.blocks)):
            assert placed.stream_at(layer) == moe.stream_at(layer)


# --------------------------------------------------------------------------- #
# Refusals and spellings outside the built-in registry plans
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestPlanRowRefusals:
    def test_projections_sharded_over_two_axes_are_refused(self) -> None:
        plan = ParallelPlan(
            {
                "layers.*.self_attn.q_proj": _row("colwise", "tensor"),
                "layers.*.self_attn.k_proj": _row("colwise", "expert"),
            }
        )
        with pytest.raises(StyleError, match="two axes"):
            interior_placement(plan, "model.layers.0.self_attn", head_axis=1)

    def test_kv_rows_disagreeing_on_their_repeat_are_refused(self) -> None:
        key = ATTENTION_FUNCTION_SLOTS["attention_key"]
        rows = {
            "layers.*.self_attn.q_proj": _row("colwise"),
            "layers.*.self_attn.k_proj": PlanRow("kv_replicated", "tensor", repeat=2),
        }
        agreed = ParallelPlan(
            {**rows, "layers.*.self_attn.v_proj": rows["layers.*.self_attn.k_proj"]}
        )
        placed = interior_placement(
            agreed, "model.layers.0.self_attn", head_axis=1, slot=key
        )
        assert placed.placement == Sharded(1, "tensor", repeat=2)
        disagreeing = ParallelPlan(
            {
                **rows,
                "layers.*.self_attn.v_proj": PlanRow(
                    "kv_replicated", "tensor", repeat=4
                ),
            }
        )
        with pytest.raises(StyleError, match="disagree"):
            interior_placement(
                disagreeing, "model.layers.0.self_attn", head_axis=1, slot=key
            )

    def test_expert_parameters_sharded_over_two_axes_are_refused(self) -> None:
        plan = ParallelPlan(
            {
                "layers.*.mlp.experts.gate_up_proj": _row("packed_colwise", "tensor"),
                "layers.*.mlp.experts.down_proj": _row("rowwise", "expert"),
            }
        )
        with pytest.raises(StyleError, match="two axes"):
            experts_placement(
                plan, "model.layers.0.mlp.experts", slot="gate_up", slots=2
            )


@pytest.mark.unit
class TestRowLookupSpellings:
    def test_child_rows_strip_the_base_model_prefix_by_default(self) -> None:
        """A plan spelled without the prefix (every transformers config)
        matches a module path spelled with it, with or without the prefix
        named."""
        plan = ParallelPlan(
            {
                "layers.*.self_attn.q_proj": _row("colwise"),
                "layers.*.mlp.up_proj": _row("colwise"),
            }
        )
        rows = child_rows(plan, "model.layers.3.self_attn")
        assert set(rows) == {"layers.*.self_attn.q_proj"}
        assert child_rows(plan, "model.layers.3.self_attn", prefix="model") == rows
        assert child_rows(plan, "model.layers.3.mlp.gate") == {}

    def test_the_sequence_wrapper_keeps_the_inner_and_reads_the_flat_token_axis(
        self,
    ) -> None:
        placed = TapPlacement(Sharded(-1, "tensor"))
        flat = sequenced(placed, flat_td(16))
        assert flat.placement == SequenceSharded(
            0, "context", inner=Sharded(-1, "tensor"), flat=True
        )
        folded = sequenced(placed, bsd(16))
        assert folded.placement == SequenceSharded(
            1, "context", inner=Sharded(-1, "tensor"), flat=False
        )

    def test_the_shard_axis_counts_a_flat_batch_pair_as_one_dimension(self) -> None:
        flat_heads = FeatureShape(
            axes=(Axis("batch"), Axis("position"), Axis("head", 4), Axis("feature", 8)),
            flat_batch=True,
            flat_inner=False,
        )
        assert shard_axis(flat_heads) == 1  # (batch·position, head, feature)
        headless = FeatureShape(
            axes=(Axis("batch"), Axis("position"), Axis("feature", 8)), flat_inner=False
        )
        assert shard_axis(headless) == -1
