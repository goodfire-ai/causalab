"""The function-interior placements: attention slots, experts slots, DeltaNet
(``docs/model_parallelism.md`` §6.2–6.4), and the two placement facts they
need of the seam — a shard of *slots* and a partial sum.

The ``unit`` tier is the derivation table: every attention-function, experts
and delta slot on both fixtures' registry plans at ``tp=2``, ``ep=2`` and
``tp=2,ep=4``, through the same ``site_placement`` the resolver calls, with
the expected placement spelled by hand from the transformers style each
child row carries. The ``property`` tier holds the round trips bit for bit
under the ``SimulatedWorld``'s drawn schedules: a slotted shard's ``fragment(whole(x))``
is ``x`` and its ``whole`` is the hand-interleaved gather; a partial sum's
``whole`` is the sum and its fragments sum back to the edited tensor exactly.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from causalab.neural.shared.parallel.fragments import (
    Fragments,
    PlacementError,
    fragment,
    whole,
)
from causalab.neural.shared.parallel.placement import (
    REPLICATED,
    ExpertLocal,
    PartialSum,
    Placement,
    Sharded,
)
from causalab.neural.shared.parallel.placements import (
    StyleError,
    TapPlacement,
    experts_placement,
    interior_placement,
    placement_for,
    site_placement,
    slot_count,
)
from causalab.protocol.parallel import ParallelGeometry
from causalab.protocol.registry import (
    ATTENTION_FUNCTION_SLOTS,
    DELTA_KERNEL_SLOTS,
    EXPERTS_FUNCTION_SLOTS,
    LLAMA_PLAN,
    QWEN35_MOE_PLAN,
    ModelInfo,
    ParallelPlan,
    PlanRow,
    component_shape,
)
from causalab.protocol.registry.shapes import FeatureShape, bs_fused_blocks
from tests._helpers import parallel_strategies as ps
from tests._helpers.simulated_world import (
    RankFailed,
    Schedule,
    SimulatedWorld,
    groups_for,
)
from tests._helpers.refusing_collective import RefusingCollective

_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# --------------------------------------------------------------------------- #
# the two fixtures, as the registry sizes them (the tiny checkpoints' configs)
# --------------------------------------------------------------------------- #

TINY_LLAMA_INFO = ModelInfo(
    key="hf-internal-testing/tiny-random-LlamaForCausalLM",
    hidden_size=16,
    num_layers=2,
    num_heads=4,
    num_kv_heads=4,
    head_dim=4,
    intermediate_size=64,
    vocab_size=32000,
    native_dtype="fp32",
    family="llama",
    parallel_plan=LLAMA_PLAN,
)

TINY_MOE_INFO = ModelInfo(
    key="tiny-random/qwen3.5-moe",
    hidden_size=8,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=32,
    intermediate_size=None,
    vocab_size=2048,
    native_dtype="fp32",
    family="qwen3_5_moe_text",
    num_experts=128,
    num_experts_per_tok=10,
    shared_expert_intermediate_size=32,
    moe_intermediate_size=32,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    layer_types=("linear_attention",) * 3 + ("full_attention",),
    parallel_plan=QWEN35_MOE_PLAN,
)

#: The experts on the tensor axis: what a config with a ``base_model_tp_plan``
#: and no ``base_model_ep_plan`` derives to (Qwen3.5-MoE ships both, and the
#: EP rows replace these).
EXPERTS_TP_PLAN = ParallelPlan(
    {
        "layers.*.mlp.experts.gate_up_proj": PlanRow("packed_colwise", "tensor"),
        "layers.*.mlp.experts.down_proj": PlanRow("rowwise", "tensor"),
        "layers.*.mlp.experts": PlanRow("moe_tp_experts", "tensor"),
    }
)
TINY_MOE_TP_EXPERTS = dataclasses.replace(TINY_MOE_INFO, parallel_plan=EXPERTS_TP_PLAN)

TP2 = ParallelGeometry(tensor=2)
EP2 = ParallelGeometry(expert=2)
TP2_EP4 = ParallelGeometry(tensor=2, expert=4)

ATTN = "model.layers.3.self_attn"
DELTA = "model.layers.0.linear_attn"
EXPERTS = "model.layers.0.mlp.experts"
ROUTER = "model.layers.0.mlp.gate"


def _place(
    info: ModelInfo,
    geometry: ParallelGeometry,
    *,
    path: str,
    kind: str,
    component: str,
    slot: str | None = None,
    tuple_index: int | None = None,
    layer: int = 0,
    shape: FeatureShape | None = None,
) -> TapPlacement:
    if shape is None:
        shape = component_shape(info, component)
    assert info.parallel_plan is not None
    return site_placement(
        plan=info.parallel_plan,
        geometry=geometry,
        path=path,
        kind=kind,
        component=component,
        layer=layer,
        num_layers=info.num_layers,
        layerless=False,
        shape=shape,
        slot=slot,
        tuple_index=tuple_index,
        prefix="model",
    )


def _attention_cases(
    info: ModelInfo, geometry: ParallelGeometry
) -> list[tuple[str, TapPlacement]]:
    """Every attention-function slot and the mixer-returned pattern, with the
    §6.2 answer: the head axis of the slot's own shape, over the tensor group,
    when the mixer's projections are colwise on an active tensor axis."""
    sharded = geometry.tensor > 1
    cases = []
    for component in ATTENTION_FUNCTION_SLOTS:
        head_axis = component_shape(info, component).head_axis_index
        assert head_axis is not None
        expected = Sharded(head_axis, "tensor") if sharded else REPLICATED
        cases.append((component, TapPlacement(expected)))
    # the pattern the mixer returns: a module tap (element 1) computed with local heads
    cases.append(
        (
            "attention_probs",
            TapPlacement(Sharded(1, "tensor") if sharded else REPLICATED),
        )
    )
    return cases


@pytest.mark.unit
class TestAttentionInteriorTable:
    @pytest.mark.parametrize(
        "info, geometry",
        [
            (TINY_LLAMA_INFO, TP2),
            (TINY_LLAMA_INFO, EP2),
            (TINY_MOE_INFO, TP2),
            (TINY_MOE_INFO, EP2),
            (TINY_MOE_INFO, TP2_EP4),
        ],
        ids=["llama tp=2", "llama ep=2", "moe tp=2", "moe ep=2", "moe tp=2,ep=4"],
    )
    def test_every_attention_slot_is_a_local_head_shard_under_tp_and_whole_otherwise(
        self, info: ModelInfo, geometry: ParallelGeometry
    ) -> None:
        for component, expected in _attention_cases(info, geometry):
            slot = ATTENTION_FUNCTION_SLOTS.get(component, "probs")
            kind = "interface" if component in ATTENTION_FUNCTION_SLOTS else "out"
            placed = _place(
                info,
                geometry,
                path=ATTN,
                kind=kind,
                component=component,
                slot=slot,
                tuple_index=1 if kind == "out" else None,
                layer=3 if info is TINY_MOE_INFO else 1,
            )
            assert placed == expected, (component, geometry)

    def test_the_slots_shard_on_their_own_head_axis_not_a_fixed_one(self) -> None:
        """§6.2's four shapes put the head axis in two places: dim 1 for
        q / k / scores (``(b, H, s, d)``, ``(b, H, q, k)``), dim 2 for ``z``
        (``(b, s, H, d)``)."""
        query = _place(
            TINY_MOE_INFO,
            TP2,
            path=ATTN,
            kind="interface",
            component="attention_query",
            slot="query",
        )
        z = _place(
            TINY_MOE_INFO,
            TP2,
            path=ATTN,
            kind="interface",
            component="attention_z",
            slot="z",
        )
        assert query.placement == Sharded(1, "tensor")
        assert z.placement == Sharded(2, "tensor")

    def test_a_head_sharded_mixer_with_a_slot_lacking_a_head_axis_is_refused_by_name(
        self,
    ) -> None:
        with pytest.raises(StyleError, match="head axis"):
            interior_placement(QWEN35_MOE_PLAN, ATTN, head_axis=None, prefix="model")

    def test_a_norm_between_colwise_and_rowwise_emits_the_local_heads(self) -> None:
        """``replicated_with_grad_allreduce`` replicates the weight. Its
        activation is the colwise output's per-head view, so input and
        output both contain this rank's head shard."""
        row = PlanRow("replicated_with_grad_allreduce", "tensor")
        for side in ("input", "output"):
            assert placement_for(row, side, axis=2).placement == Sharded(2, "tensor")
        # a shape that keeps no head axis packs its heads head-major into the
        # flat feature axis, so the shard is that axis (``shard_axis``)
        assert placement_for(row, "output").placement == Sharded(-1, "tensor")

    def test_the_q_norm_module_boundary_agrees_with_the_function_interior(
        self,
    ) -> None:
        """``attention_query_pre_rope`` on the MoE taps ``q_norm`` — its
        ``(b, s, H, d)`` output is the same local-heads tensor the interface's
        ``query`` is a transpose of, so the two derivations shard the same axis
        by different routes (the norm's row; the mixer's projections)."""
        # the site carries the module's native packing (``registry.native_shape``):
        # Qwen's ``q_norm`` keeps the head axis, so the shard axis is dim 2 —
        # the flat spelling of the same component shards its last axis
        kept = dataclasses.replace(
            component_shape(TINY_MOE_INFO, "attention_query_pre_rope"), flat_inner=False
        )
        placed = _place(
            TINY_MOE_INFO,
            TP2,
            path=f"{ATTN}.q_norm",
            kind="out",
            component="attention_query_pre_rope",
            layer=3,
            shape=kept,
        )
        assert placed.placement == Sharded(2, "tensor")
        # and llama's bare projection stays the flat colwise shard of §4 row 1
        placed = _place(
            TINY_LLAMA_INFO,
            TP2,
            path="model.layers.1.self_attn.q_proj",
            kind="out",
            component="attention_query_pre_rope",
            layer=1,
        )
        assert placed.placement == Sharded(-1, "tensor")

    def test_the_mixer_itself_carries_no_row_so_its_output_is_whole(self) -> None:
        placed = _place(
            TINY_MOE_INFO,
            TP2,
            path=ATTN,
            kind="out",
            component="attention_output",
            layer=3,
        )
        assert placed == TapPlacement(REPLICATED)


# --------------------------------------------------------------------------- #
# the experts interior (§6.3), under EP and under TP of the experts
# --------------------------------------------------------------------------- #

TOP_K = 10


@pytest.mark.unit
class TestExpertsInteriorTable:
    @pytest.mark.parametrize("geometry", [EP2, TP2_EP4], ids=["ep=2", "tp=2,ep=4"])
    def test_under_expert_parallelism_every_slot_is_expert_local_with_the_remapped_table(
        self, geometry: ParallelGeometry
    ) -> None:
        for component, slot in EXPERTS_FUNCTION_SLOTS.items():
            placed = _place(
                TINY_MOE_INFO,
                geometry,
                path=EXPERTS,
                kind="experts",
                component=component,
                slot=slot,
            )
            assert placed == TapPlacement(
                ExpertLocal("expert"), remapped_routing=True
            ), (component, geometry)

    def test_under_tensor_parallelism_the_neuron_axis_is_sharded_per_slot_and_down_is_partial(
        self,
    ) -> None:
        """``packed_colwise`` on ``gate_up_proj`` leaves each rank
        ``[gate_r | up_r]`` per (token, slot) pair — ``top_k · 2`` runs of
        ``d_e / tp``; the activation ``top_k`` runs; ``rowwise`` on
        ``down_proj`` leaves each rank a partial sum the module's all-reduce
        completes. A plan with the experts on the tensor axis — a config that
        ships a ``base_model_tp_plan`` and no ``base_model_ep_plan``."""
        expected = {
            "expert_gate_proj": Sharded(-1, "tensor", slots=TOP_K * 2),
            "expert_up_proj": Sharded(-1, "tensor", slots=TOP_K * 2),
            "expert_activation": Sharded(-1, "tensor", slots=TOP_K),
            "expert_neuron_output": Sharded(-1, "tensor", slots=TOP_K),
            "expert_output": PartialSum("tensor"),
        }
        for component, slot in EXPERTS_FUNCTION_SLOTS.items():
            placed = _place(
                TINY_MOE_TP_EXPERTS,
                TP2,
                path=EXPERTS,
                kind="experts",
                component=component,
                slot=slot,
            )
            assert placed == TapPlacement(expected[component]), component

    def test_the_table_names_every_slot_the_dispatch_has(self) -> None:
        """The placement table and the experts dispatch agree on the slots —
        the slot the dispatch gained (``neuron_output``, the down-projection's
        input ``act(gate) * up``) is placed like the activation it is made
        from, under tensor and under expert parallelism alike."""
        from causalab.neural.engines.pytorch_hooks.experts_interface import (
            EXPERTS_SLOTS,
        )
        from causalab.neural.shared.parallel import placements

        assert placements._EXPERTS_SLOTS == EXPERTS_SLOTS  # pyright: ignore[reportPrivateUsage]
        for plan, geometry in ((TINY_MOE_TP_EXPERTS, TP2), (TINY_MOE_INFO, EP2)):
            kwargs = dict(path=EXPERTS, kind="experts")
            neuron = _place(
                plan,
                geometry,
                component="expert_neuron_output",
                slot="neuron_output",
                **kwargs,
            )
            activation = _place(
                plan,
                geometry,
                component="expert_activation",
                slot="activation",
                **kwargs,
            )
            assert neuron == activation, geometry

    def test_the_registry_plan_keeps_the_experts_whole_under_tensor_parallelism_alone(
        self,
    ) -> None:
        """Qwen3.5-MoE's ``base_model_ep_plan`` *replaces* the tensor rows
        under ``experts`` (``parallel_plan_from_hf_config``), so at ``tp=2``
        alone no expert row is active: every rank holds every expert and the
        interior is whole."""
        for component, slot in EXPERTS_FUNCTION_SLOTS.items():
            placed = _place(
                TINY_MOE_INFO,
                TP2,
                path=EXPERTS,
                kind="experts",
                component=component,
                slot=slot,
            )
            assert placed == TapPlacement(REPLICATED), component

    def test_the_experts_module_output_is_whole_under_both(self) -> None:
        for info, geometry in (
            (TINY_MOE_TP_EXPERTS, TP2),
            (TINY_MOE_INFO, EP2),
            (TINY_MOE_INFO, TP2_EP4),
        ):
            placed = _place(
                info, geometry, path=EXPERTS, kind="out", component="routed_output"
            )
            assert placed == TapPlacement(REPLICATED), geometry

    def test_the_router_payload_under_ep_logits_whole_scores_owned_indices_remapped(
        self,
    ) -> None:
        """``EpRouterParallel`` passes the logits through, zeroes the scores
        of the slots this rank does not own, and remaps the indices."""
        logits = _place(
            TINY_MOE_INFO,
            EP2,
            path=ROUTER,
            kind="out",
            component="router_logits",
            tuple_index=0,
        )
        scores = _place(
            TINY_MOE_INFO,
            EP2,
            path=ROUTER,
            kind="out",
            component="router_scores",
            tuple_index=1,
        )
        indices = _place(
            TINY_MOE_INFO,
            EP2,
            path=ROUTER,
            kind="out",
            component="expert_idx",
            tuple_index=2,
        )
        assert logits == TapPlacement(REPLICATED)
        assert scores == TapPlacement(ExpertLocal("expert"))
        assert indices == TapPlacement(REPLICATED, remapped_routing=True)
        # under tensor parallelism alone the router has no row
        assert _place(
            TINY_MOE_INFO,
            TP2,
            path=ROUTER,
            kind="out",
            component="router_scores",
            tuple_index=1,
        ) == TapPlacement(REPLICATED)

    def test_slot_count_is_the_product_of_the_inner_axes_before_the_feature(
        self,
    ) -> None:
        assert slot_count(component_shape(TINY_MOE_INFO, "expert_gate_proj")) == (
            TOP_K * 2
        )
        assert slot_count(component_shape(TINY_MOE_INFO, "expert_activation")) == TOP_K
        assert slot_count(component_shape(TINY_MOE_INFO, "block_output")) == 1

    def test_mixed_expert_rows_are_refused_by_name(self) -> None:
        plan = ParallelPlan(
            {
                "layers.*.mlp.experts.gate_up_proj": PlanRow("grouped_gemm", "expert"),
                "layers.*.mlp.experts.down_proj": PlanRow("rowwise", "tensor"),
            }
        )
        with pytest.raises(StyleError):
            experts_placement(plan, EXPERTS, slot="down", slots=1, prefix="model")
        with pytest.raises(StyleError, match="'diagonal'"):
            experts_placement(
                ParallelPlan(
                    {"layers.*.mlp.experts.down_proj": PlanRow("diagonal", "tensor")}
                ),
                EXPERTS,
                slot="down",
                slots=1,
                prefix="model",
            )


# --------------------------------------------------------------------------- #
# DeltaNet (§6.4): every slot replicated, because every projection is gathered
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestDeltaNetTable:
    @pytest.mark.parametrize(
        "geometry", [TP2, EP2, TP2_EP4], ids=["tp=2", "ep=2", "tp=2,ep=4"]
    )
    def test_all_ten_delta_slots_are_replicated(
        self, geometry: ParallelGeometry
    ) -> None:
        assert len(DELTA_KERNEL_SLOTS) == 10
        for component, slot in DELTA_KERNEL_SLOTS.items():
            placed = _place(
                TINY_MOE_INFO,
                geometry,
                path=DELTA,
                kind="delta",
                component=component,
                slot=slot,
            )
            assert placed == TapPlacement(REPLICATED), (component, geometry)

    def test_the_deltanet_projections_are_all_gathered_so_their_boundaries_are_whole(
        self,
    ) -> None:
        for component, path, kind in (
            ("delta_qkv", f"{DELTA}.in_proj_qkv", "out"),
            ("delta_gate", f"{DELTA}.in_proj_z", "out"),
            ("delta_premix", f"{DELTA}.out_proj", "in"),
        ):
            placed = _place(
                TINY_MOE_INFO, TP2, path=path, kind=kind, component=component
            )
            assert placed == TapPlacement(REPLICATED), component

    def test_a_row_on_an_inactive_axis_is_not_applied(self) -> None:
        """``apply_plan`` skips rows on an axis of size one; the placement
        table answers the same way, so a tensor-row model at ``ep=2`` alone
        is whole everywhere."""
        placed = _place(
            TINY_MOE_INFO,
            EP2,
            path=ATTN,
            kind="interface",
            component="attention_query",
            slot="query",
            layer=3,
        )
        assert placed == TapPlacement(REPLICATED)


# --------------------------------------------------------------------------- #
# property tier: the slotted shard and the partial sum, bit for bit
# --------------------------------------------------------------------------- #


def _interleaved_whole(chunks: list[torch.Tensor], slots: int) -> torch.Tensor:
    """The hand statement of a slotted gather: within every slot, the ranks'
    chunks in rank order."""
    per_rank = [c.unflatten(-1, (slots, -1)) for c in chunks]
    return torch.cat(per_rank, dim=-1).flatten(-2)


@pytest.mark.property
class TestSlottedShard:
    @_SETTINGS
    @given(
        seed=ps.seeds(),
        schedule=ps.schedules(),
        tp=st.sampled_from([2, 4]),
        slots=st.integers(1, 4),
        per_rank=st.integers(1, 3),
        lead=st.integers(1, 3),
    )
    @example(seed=0, schedule=[], tp=2, slots=1, per_rank=1, lead=1)
    def test_fragment_of_whole_is_the_identity_and_whole_is_the_interleaved_gather(
        self,
        seed: int,
        schedule: Schedule,
        tp: int,
        slots: int,
        per_rank: int,
        lead: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        locals_ = [
            torch.randn((lead, slots * per_rank), generator=generator)
            for _ in range(tp)
        ]
        placement = Sharded(-1, "tensor", slots=slots)
        expected = _interleaved_whole(locals_, slots)

        def program(rank: int, c: Any) -> tuple[torch.Tensor, torch.Tensor]:
            mine = locals_[c.rank("tensor")]
            made_whole = whole(mine, placement, c)
            return made_whole, fragment(made_whole, placement, c)

        world = SimulatedWorld(groups_for(tp, tensor=tp), world=tp, schedule=schedule)
        for rank, (made_whole, back) in enumerate(world.run(program)):
            assert torch.equal(made_whole, expected), f"rank {rank}: whole"
            assert torch.equal(back, locals_[rank]), f"rank {rank}: round trip"

    def test_a_slotted_shard_of_one_slot_is_the_plain_shard(self) -> None:
        assert Sharded(-1, "tensor") == Sharded(-1, "tensor", slots=1)

    def test_an_axis_the_slots_do_not_divide_is_refused(self) -> None:
        world = SimulatedWorld(groups_for(2, tensor=2), world=2, schedule=0)

        def program(rank: int, c: Any) -> None:
            whole(torch.zeros(2, 5), Sharded(-1, "tensor", slots=3), c)

        with pytest.raises(RankFailed) as err:
            world.run(program)
        assert isinstance(err.value.__cause__, PlacementError)

    def test_slots_below_one_are_refused(self) -> None:
        with pytest.raises(ValueError):
            Sharded(-1, "tensor", slots=0)


@pytest.mark.property
class TestPartialSum:
    @_SETTINGS
    @given(
        seed=ps.seeds(),
        schedule=ps.schedules(),
        tp=st.sampled_from([2, 3, 4]),
        shape=st.lists(st.integers(1, 3), min_size=1, max_size=3),
    )
    def test_whole_is_the_sum_and_the_fragments_sum_back_exactly(
        self, seed: int, schedule: Schedule, tp: int, shape: list[int]
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        partials = [torch.randn(tuple(shape), generator=generator) for _ in range(tp)]
        edited = torch.randn(tuple(shape), generator=generator)
        placement = PartialSum("tensor")

        def program(rank: int, c: Any) -> tuple[torch.Tensor, torch.Tensor]:
            made_whole = whole(partials[c.rank("tensor")], placement, c)
            return made_whole, fragment(edited, placement, c)

        world = SimulatedWorld(groups_for(tp, tensor=tp), world=tp, schedule=schedule)
        results = world.run(program)
        total = partials[0].clone()
        for p in partials[1:]:
            total = total + p
        for rank, (made_whole, _) in enumerate(results):
            assert torch.equal(made_whole, total), f"rank {rank}"
        fragments_ = [back for _, back in results]
        assert torch.equal(fragments_[0], edited), "the group's first rank keeps it"
        for back in fragments_[1:]:
            assert torch.equal(back, torch.zeros_like(edited))
        summed = fragments_[0].clone()
        for back in fragments_[1:]:
            summed = summed + back
        assert torch.equal(summed, edited), "x + 0 == x bit for bit"

    def test_a_group_of_one_is_the_identity_without_a_collective(self) -> None:
        f = Fragments(RefusingCollective())
        x = torch.arange(6.0).reshape(2, 3)
        assert f.whole(x, PartialSum("tensor")) is x
        assert f.fragment(x, Sharded(-1, "tensor", slots=3)) is x


@pytest.mark.unit
class TestPlacementTypes:
    def test_the_new_placements_are_values(self) -> None:
        assert PartialSum("tensor") == PartialSum("tensor")
        assert PartialSum("tensor") != PartialSum("expert")
        assert hash(Sharded(-1, "tensor", slots=4)) == hash(
            Sharded(-1, "tensor", slots=4)
        )
        placement: Placement = PartialSum()
        assert dataclasses.is_dataclass(placement)

    def test_head_axis_index_is_the_leading_head_dimension(self) -> None:
        moe = TINY_MOE_INFO
        assert component_shape(moe, "attention_query").head_axis_index == 1
        assert component_shape(moe, "attention_z").head_axis_index == 2
        assert component_shape(moe, "attention_probs").head_axis_index == 1
        assert component_shape(moe, "attention_query_pre_rope").head_axis_index == 2
        # head-major and packed with the feature: the packed dimension
        assert component_shape(moe, "attention_premix").head_axis_index == 2
        assert component_shape(moe, "block_output").head_axis_index is None

    def test_a_fused_axis_outside_the_heads_has_no_shardable_head_dimension(
        self,
    ) -> None:
        shape: FeatureShape = bs_fused_blocks(3, 0, 4, 8)
        assert shape.head_axis_index is None
