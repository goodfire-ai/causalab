"""The memory pre-flight's torch-free half (``docs/model_parallelism.md``
§2, §3, §5.3, §11; ``protocol/parallel_memory.py``): the placement table,
the per-rank resident estimate, the measured rule, the fit search and the
refusal.

Three tiers. ``unit``: the placement's validation, a hand table under each
axis, the refusal text naming rank, device, bytes and fits, the fit order.
``property``: over hypothesis tables and geometries — the stages of a model
group sum to the whole, a replicated parameter is whole on every rank, a
sharded one is ``1 / size`` and sums to the whole over its group, the
estimate is monotone in the pipeline and expert axes, copies scale the
sharded bytes alone, and a geometry is refused exactly when its resident weights
exceeds the free bytes. And the **pins** against the record: the estimate
reproduces the loader's ``bytes_requested`` per rank on every recorded
geometry of ``tests/golden/parallel_goldens.json`` to the byte, the
footprint never sits below a recorded peak, and it lands within the stated
band of every reference card peak (``CARD_PEAKS_GIB``) — so a change to the rule or to the loader's
accounting shows here, on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.checkpoint_census import checkpoint_targets, tree_of
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    ONE,
    ParallelGeometry,
    parse_geometry,
)
from causalab.protocol.parallel_memory import (
    DTYPE_ITEMSIZES,
    RULE,
    EstimateRule,
    Fit,
    Placement,
    estimate_resident,
    fitting_geometries,
    format_bytes,
    memory_check,
    memory_refusal,
    placement_table,
    whole_bytes,
)
from causalab.protocol.registry import LLAMA_TREE, get_model_info

from tests._helpers.header_census import load_census

_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "tests" / "golden" / "parallel_goldens.json"
A3B = "Qwen/Qwen3.6-35B-A3B"
GIB = 1 << 30

#: Reference peaks in GiB for a six-step workflow on the public
#: ``Qwen/Qwen3.6-35B-A3B`` checkpoint in bf16, one row per geometry, measured
#: on H100 80 GB cards (``H100_TOTAL`` below) by ``nvidia-smi memory.used``
#: unless a row notes another source. The values are hand-recorded references
#: for the band test below; no script in this repository regenerates them.
CARD_PEAKS_GIB: dict[str, float] = {
    "": 72.9,
    "pp=2": 42.0,  # maximum across the pipeline cards
    "pp=4": 24.0,
    "ep=2": 45.9,
    # These three references include a second allocator segment and are
    # conservative bounds for the one-copy loader.
    "ep=4": 38.6,
    "ep=8": 24.9,
    "pp=2,ep=2": 29.6,
    "tp=2": 74.3,
    "tp=4": 71.8,
    "tp=8": 74.1,
    # Inference peak from torch.max_memory_allocated, one rank per machine;
    # unlike the other rows, this is not a workflow card-memory peak.
    "tp=2,ep=2": 35.0,
    "cp=2": 77.0,
}
#: An H100 80 GB as ``mem_get_info`` reports it before a load (81559 MiB
#: total; ~0.5 GiB to the context).
H100_TOTAL = 81559 * (1 << 20)
H100_FREE = H100_TOTAL - (1 << 29)


# --------------------------------------------------------------------------- #
# hand tables
# --------------------------------------------------------------------------- #


def _tower(
    layers: int, *, q: int = 64, expert: int = 128, norm: int = 8
) -> dict[str, Placement]:
    """A small model: the embedding first, ``layers`` blocks each with a
    replicated norm, a tensor-sharded projection, a K/V projection sharded
    up to two ranks, and an expert-sharded fused parameter; the final norm
    and the head last."""
    table = {"model.embed_tokens.weight": Placement(256, home="first")}
    for layer in range(layers):
        table[f"model.layers.{layer}.norm.weight"] = Placement(
            norm, home="layer", layer=layer
        )
        table[f"model.layers.{layer}.q_proj.weight"] = Placement(
            q, axis="tensor", home="layer", layer=layer
        )
        table[f"model.layers.{layer}.k_proj.weight"] = Placement(
            q // 2, axis="tensor", home="layer", layer=layer, shard_limit=2
        )
        table[f"model.layers.{layer}.experts.gate_up_proj"] = Placement(
            expert, axis="expert", home="layer", layer=layer
        )
    table["model.norm.weight"] = Placement(norm, home="last")
    table["lm_head.weight"] = Placement(256, home="last")
    return table


@pytest.mark.unit
class TestPlacement:
    def test_a_layer_home_names_its_index_and_no_other_home_does(self) -> None:
        Placement(4, home="layer", layer=0)
        with pytest.raises(ValueError, match="layer home names its index"):
            Placement(4, home="layer")
        with pytest.raises(ValueError, match="layer home names its index"):
            Placement(4, home="first", layer=1)

    @pytest.mark.parametrize("bad", (-1, True, 2.0))
    def test_elements_is_a_non_negative_int(self, bad) -> None:
        with pytest.raises(ValueError):
            Placement(bad)  # type: ignore[arg-type]

    def test_an_axis_outside_the_plan_axes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            Placement(4, axis="pipeline")  # type: ignore[arg-type]

    def test_shards_is_the_axis_size_up_to_the_limit(self) -> None:
        kv = Placement(8, axis="tensor", shard_limit=2)
        assert kv.shards(ParallelGeometry(tensor=2)) == 2
        assert kv.shards(ParallelGeometry(tensor=4)) == 1  # replicated whole (§6.6)
        assert Placement(8).shards(ParallelGeometry(tensor=4)) == 1
        assert Placement(8, axis="expert").shards(ParallelGeometry(expert=4)) == 4


@pytest.mark.unit
class TestEstimateOnAHandTable:
    def test_world_one_holds_the_whole_model(self) -> None:
        table = _tower(4)
        assert estimate_resident(table, ParallelGeometry(), 2) == (
            whole_bytes(table, 2),
        )

    def test_two_stages_split_the_layers_and_place_embedding_and_head_apart(
        self,
    ) -> None:
        table = _tower(4)
        first, last = estimate_resident(table, ParallelGeometry(pipeline=2), 1)
        per_layer = 8 + 64 + 32 + 128
        assert first == 256 + 2 * per_layer
        assert last == 2 * per_layer + 8 + 256
        assert first + last == whole_bytes(table, 1)

    def test_tensor_shards_the_projections_and_replicates_kv_above_its_limit(
        self,
    ) -> None:
        table = _tower(1)
        (per_rank,) = set(estimate_resident(table, ParallelGeometry(tensor=2), 1))
        assert per_rank == 256 + 8 + 32 + 16 + 128 + 8 + 256
        (per_rank,) = set(estimate_resident(table, ParallelGeometry(tensor=4), 1))
        assert per_rank == 256 + 8 + 16 + 32 + 128 + 8 + 256  # k_proj whole again

    def test_expert_shards_the_experts_and_copies_scale_them_alone(self) -> None:
        table = _tower(1)
        (one,) = set(estimate_resident(table, ParallelGeometry(expert=4), 1))
        assert one == 256 + 8 + 64 + 32 + 32 + 8 + 256
        (doubled,) = set(
            estimate_resident(
                table, ParallelGeometry(expert=4), 1, copies={"expert": 1.8}
            )
        )
        assert doubled - one == round(32 * 1.8) - 32

    def test_the_itemsize_scales_everything(self) -> None:
        table = _tower(2)
        assert estimate_resident(table, ParallelGeometry(), 4) == (
            2 * estimate_resident(table, ParallelGeometry(), 2)[0],
        )
        with pytest.raises(ProtocolError, match="itemsize"):
            estimate_resident(table, ParallelGeometry(), 0)

    def test_more_stages_than_layers_is_the_pipeline_refusal(self) -> None:
        with pytest.raises(ProtocolError, match="--parallel.pipeline"):
            estimate_resident(_tower(2), ParallelGeometry(pipeline=3), 1)

    def test_a_slice_of_a_table_splits_the_stages_by_the_models_layers(self) -> None:
        """A table missing its last layers (a per-dtype slice) would split
        the layers it has; given the model's count it splits as the loader
        does — the 160 bytes the gloo pin found on the tiny MoE."""
        table = _tower(4)
        first_three = {k: p for k, p in table.items() if p.layer is None or p.layer < 3}
        by_itself = estimate_resident(first_three, ParallelGeometry(pipeline=2), 1)
        as_model = estimate_resident(
            first_three, ParallelGeometry(pipeline=2), 1, num_layers=4
        )
        per_layer = 8 + 64 + 32 + 128
        assert by_itself[0] == 256 + per_layer  # three layers split [0], [1, 2]
        assert as_model[0] == 256 + 2 * per_layer  # four layers split [0, 1], [2, 3]
        assert as_model[1] == per_layer + 8 + 256

    def test_a_table_without_layer_homes_takes_the_models_layer_count(self) -> None:
        """A table with no layer home (the embedding and the head alone)
        leaves the stage split to the model's count: the pipeline is neither
        refused nor confined to one stage, and the search still offers it."""
        table = {
            "model.embed_tokens.weight": Placement(256, home="first"),
            "lm_head.weight": Placement(256, home="last"),
        }
        info = get_model_info(A3B)
        assert estimate_resident(table, ParallelGeometry(pipeline=2), 1) == (256, 256)
        for rank in (0, 1):
            assert (
                memory_check(
                    geometry=ParallelGeometry(pipeline=2),
                    rank=rank,
                    device="cuda:0",
                    dtype="bf16",
                    table=table,
                    info=info,
                    free=1 << 40,
                    total=1 << 40,
                )
                is None
            )
        fits = fitting_geometries(info, table, 2, capacity=1 << 40, worlds=(2,))
        assert ParallelGeometry(pipeline=2) in {fit.geometry for fit in fits}

    @pytest.mark.parametrize("bad", (0, -1, True, 2.0))
    def test_the_itemsize_is_a_positive_int_refused_as_p4_on_the_dtype(
        self, bad
    ) -> None:
        with pytest.raises(ProtocolError) as info:
            estimate_resident(_tower(1), ParallelGeometry(), bad)  # type: ignore[arg-type]
        assert info.value.code == "P4" and info.value.path == "--dtype"
        # an internal guard: pinned to the facts it names, not the character
        assert "positive int" in info.value.message
        assert f"got {bad!r}" in info.value.message


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #

_SIZES = st.sampled_from((1, 2, 4))


@st.composite
def _tables(draw: st.DrawFn) -> dict[str, Placement]:
    layers = draw(st.sampled_from((1, 2, 4, 8)))
    unit = 8  # every count a multiple of the largest axis size, so shards divide
    return _tower(
        layers,
        q=unit * draw(st.integers(min_value=1, max_value=16)),
        expert=unit * draw(st.integers(min_value=1, max_value=16)),
        norm=unit * draw(st.integers(min_value=1, max_value=4)),
    )


def _layers_of(table: dict[str, Placement]) -> int:
    return 1 + max(p.layer for p in table.values() if p.layer is not None)


@st.composite
def _geometries(draw: st.DrawFn, table: dict[str, Placement]) -> ParallelGeometry:
    layers = _layers_of(table)
    pipeline = draw(st.sampled_from([d for d in (1, 2, 4, 8) if layers % d == 0]))
    tensor, expert = draw(_SIZES), draw(_SIZES)
    model = max(tensor, expert)
    if model % tensor or model % expert:
        tensor = expert = model
    return ParallelGeometry(
        data=draw(st.sampled_from((1, 2))),
        pipeline=pipeline,
        context=draw(st.sampled_from((1, 2))),
        tensor=tensor,
        expert=expert,
    )


@pytest.mark.property
class TestProperties:
    @_HYPOTHESIS_SETTINGS
    @given(table=_tables(), data=st.data())
    def test_the_stages_of_a_model_group_sum_to_the_whole(self, table, data) -> None:
        """Under the pipeline alone (any ``dp`` / ``cp``), one rank per stage
        holds a partition of the model: the stages sum to the whole."""
        geometry = data.draw(_geometries(table))
        geometry = ParallelGeometry(
            data=geometry.data,
            pipeline=geometry.pipeline,
            context=geometry.context,
        )
        per_rank = estimate_resident(table, geometry, 2)
        stages = [
            per_rank[stage * geometry.context] for stage in range(geometry.pipeline)
        ]
        assert sum(stages) == whole_bytes(table, 2)

    @_HYPOTHESIS_SETTINGS
    @given(
        elements=st.integers(min_value=0, max_value=4096),
        geometry=st.builds(
            ParallelGeometry, pipeline=st.just(1), tensor=_SIZES, expert=_SIZES
        ).filter(lambda g: g.model % g.tensor == 0 and g.model % g.expert == 0),
    )
    def test_a_replicated_parameter_is_whole_on_every_rank(
        self, elements, geometry
    ) -> None:
        table = {
            "norm": Placement(elements),
            "embed": Placement(elements, home="first"),
        }
        assert set(estimate_resident(table, geometry, 2)) == {2 * 2 * elements}

    @_HYPOTHESIS_SETTINGS
    @given(
        chunks=st.integers(min_value=1, max_value=64),
        tensor=_SIZES,
        expert=_SIZES,
    )
    def test_a_sharded_parameter_is_one_over_its_axis_and_sums_to_the_whole(
        self, chunks, tensor, expert
    ) -> None:
        model = max(tensor, expert)
        if model % tensor or model % expert:
            tensor = expert = model
        geometry = ParallelGeometry(tensor=tensor, expert=expert)
        table = {
            "q": Placement(chunks * 4, axis="tensor"),
            "experts": Placement(chunks * 4, axis="expert"),
        }
        per_rank = estimate_resident(table, geometry, 1)
        assert set(per_rank) == {chunks * 4 // tensor + chunks * 4 // expert}
        # over one tensor group (contiguous ranks) the q shards sum to the whole
        q_only = estimate_resident({"q": table["q"]}, geometry, 1)
        assert sum(q_only[:tensor]) == chunks * 4
        e_only = estimate_resident({"experts": table["experts"]}, geometry, 1)
        assert sum(e_only[:expert]) == chunks * 4

    @_HYPOTHESIS_SETTINGS
    @given(table=_tables(), data=st.data())
    def test_monotone_in_the_pipeline_and_expert_axes(self, table, data) -> None:
        """Doubling ``pp`` (dividing the layers) or ``ep`` never raises the
        largest rank's bytes — the two axes that buy memory (§11)."""
        geometry = data.draw(_geometries(table))
        worst = max(estimate_resident(table, geometry, 2))
        if _layers_of(table) % (2 * geometry.pipeline) == 0:
            deeper = ParallelGeometry(
                data=geometry.data,
                pipeline=2 * geometry.pipeline,
                context=geometry.context,
                tensor=geometry.tensor,
                expert=geometry.expert,
            )
            assert max(estimate_resident(table, deeper, 2)) <= worst
        if geometry.expert < 4:
            narrow = ParallelGeometry(
                data=geometry.data,
                pipeline=geometry.pipeline,
                context=geometry.context,
                expert=geometry.expert,
            )
            wider = ParallelGeometry(
                data=geometry.data,
                pipeline=geometry.pipeline,
                context=geometry.context,
                expert=2 * geometry.expert,
            )
            assert max(estimate_resident(table, wider, 2)) <= max(
                estimate_resident(table, narrow, 2)
            )

    @_HYPOTHESIS_SETTINGS
    @given(
        table=_tables(), data=st.data(), factor=st.floats(min_value=1.0, max_value=3.0)
    )
    def test_copies_scale_the_sharded_bytes_alone(self, table, data, factor) -> None:
        geometry = data.draw(_geometries(table))
        one = estimate_resident(table, geometry, 2)
        scaled = estimate_resident(table, geometry, 2, copies={"expert": factor})
        experts_only = {k: p for k, p in table.items() if p.axis == "expert"}
        experts = estimate_resident(experts_only, geometry, 2)
        for rank in range(geometry.world):
            if geometry.expert == 1:
                assert scaled[rank] == one[rank]
            else:
                # each expert-sharded parameter is rounded on its own: the
                # difference is within half a byte per such parameter
                grown = scaled[rank] - one[rank]
                assert abs(grown - experts[rank] * (factor - 1)) <= len(experts_only)

    @_HYPOTHESIS_SETTINGS
    @given(
        table=_tables(),
        data=st.data(),
        slack=st.integers(min_value=-(1 << 12), max_value=1 << 12),
    )
    def test_refused_exactly_when_resident_weights_exceed_the_free_bytes(
        self, table, data, slack
    ) -> None:
        geometry = data.draw(_geometries(table))
        rank = data.draw(st.integers(min_value=0, max_value=geometry.world - 1))
        resident = estimate_resident(table, geometry, 2)[rank]
        free = max(0, resident + slack)
        text = memory_check(
            geometry=geometry,
            rank=rank,
            device="cuda:0",
            dtype="bf16",
            table=table,
            info=get_model_info(A3B),
            free=free,
            total=free + (1 << 20),
        )
        assert (text is None) == (resident <= free)
        if text is not None:
            assert text.startswith("--parallel:") and f"rank {rank} (cuda:0)" in text

    @_HYPOTHESIS_SETTINGS
    @given(
        elements=st.integers(min_value=0, max_value=1 << 10),
        itemsize=st.sampled_from((1, 2, 4)),
        tensor=_SIZES,
    )
    def test_a_share_is_the_bytes_over_the_shards_rounded_up_to_a_whole_byte(
        self, elements, itemsize, tensor
    ) -> None:
        """The last chunk of an uneven split is shorter, so a rank's share is
        the ceiling: an integer bound per rank, never a fraction of a byte."""
        table = {"q": Placement(elements, axis="tensor")}
        per_rank = estimate_resident(table, ParallelGeometry(tensor=tensor), itemsize)
        expected = -(-(elements * itemsize) // tensor)
        assert per_rank == (expected,) * tensor
        assert all(type(share) is int for share in per_rank)


# --------------------------------------------------------------------------- #
# the rule, the fits, the refusal
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestRuleAndRefusal:
    def test_the_rule_is_the_measured_one(self) -> None:
        assert RULE.copies == {"tensor": 1.0, "expert": 1.0}
        assert RULE.base == 0.15 and RULE.context == 0.10
        assert RULE.headroom(100, ParallelGeometry()) == 15
        assert RULE.headroom(100, ParallelGeometry(context=2)) == 25
        text = RULE.describe(ParallelGeometry(context=2, expert=2))
        assert "15%" in text and "+10% under cp" in text and "1.8x" not in text
        assert "§11" in text
        assert "1.8x" not in RULE.describe(ParallelGeometry(tensor=2))

    def test_a_rule_below_one_copy_or_a_negative_fraction_is_refused(self) -> None:
        with pytest.raises(ValueError):
            EstimateRule(copies={"expert": 0.5})
        with pytest.raises(ValueError):
            EstimateRule(base=-0.1)

    def test_format_bytes_speaks_gib(self) -> None:
        assert format_bytes(69321221376) == "64.56 GiB"
        assert format_bytes(0) == "0.00 GiB"

    def test_the_dtype_words_are_the_documents(self) -> None:
        assert DTYPE_ITEMSIZES == {"fp32": 4, "bf16": 2, "fp16": 2}
        with pytest.raises(ProtocolError, match="dtype"):
            memory_check(
                geometry=ParallelGeometry(tensor=2),
                rank=0,
                device="cuda:0",
                dtype="int8",
                table=_tower(1),
                info=get_model_info(A3B),
                free=1,
                total=1,
            )

    def test_fits_are_ordered_by_world_then_tensor_then_expert(self) -> None:
        table = _tower(8)
        info = get_model_info(A3B)  # 16 heads, 2 kv, 256 experts, 40 layers
        fits = fitting_geometries(
            info, table, 2, capacity=1 << 40, worlds=(2, 4), rule=EstimateRule()
        )
        spelled = [
            (
                f.geometry.world,
                f.geometry.tensor,
                f.geometry.expert,
                f.geometry.pipeline,
            )
            for f in fits
        ]
        assert spelled[:3] == [(2, 1, 1, 2), (2, 1, 2, 1), (2, 2, 1, 1)]
        assert all(a <= b for a, b in zip(spelled, spelled[1:]))
        assert all(f.geometry.world in (2, 4) for f in fits)

    def test_the_search_stays_inside_check_and_the_capacity(self) -> None:
        table = _tower(8)
        info = get_model_info("meta-llama/Llama-3.1-8B")  # dense: no expert axis
        fits = fitting_geometries(info, table, 2, capacity=1 << 40, worlds=(2,))
        assert all(f.geometry.expert == 1 for f in fits)
        tight = fitting_geometries(info, table, 2, capacity=0, worlds=(2, 4))
        assert tight == ()

    def test_the_refusal_names_rank_device_bytes_and_the_first_two_fits(self) -> None:
        text = memory_refusal(
            geometry=ParallelGeometry(context=2),
            rank=1,
            device="cuda:1",
            dtype="bf16",
            resident=64 * GIB,
            footprint=80 * GIB,
            free=79 * GIB,
            total=80 * GIB,
            fits=(),
        )
        assert text.startswith("--parallel: cp=2 would place 64.00 GiB of bf16 weights")
        assert "rank 1 (cuda:1)" in text and "80.00 GiB" in text
        assert "79.00 GiB available of 80.00 GiB" in text
        assert "no pp/ep/tp geometry" in text and "--device cuda:1" in text
        # a cp (or dp) run is answered in the placement axes, and the text
        # says which axes the remedy searched and which it held
        assert "; searched pp/ep/tp with dp and cp held at 1; " in text

    def test_one_layer_per_stage_is_a_candidate(self) -> None:
        """``pp`` equal to the layer count is the deepest pipeline the table
        admits, and the search offers it."""
        info = get_model_info(A3B)
        fits = fitting_geometries(info, _tower(2), 2, capacity=1 << 40, worlds=(2,))
        assert ParallelGeometry(pipeline=2) in {fit.geometry for fit in fits}

    def test_a_refused_candidate_does_not_end_the_worlds_search(self) -> None:
        """The dense Llama refuses ``ep=2``, which the search meets before
        ``tp=2`` of the same world; the latter still fits."""
        info = get_model_info("meta-llama/Llama-3.1-8B")
        fits = fitting_geometries(info, _tower(8), 2, capacity=1 << 40, worlds=(2,))
        assert ParallelGeometry(tensor=2) in {fit.geometry for fit in fits}

    def test_a_footprint_equal_to_the_capacity_fits_under_the_tables_split(
        self,
    ) -> None:
        """The fit is ``<=``, and its footprint is the table's own stage
        split (eight layers over the stages, not the model's forty)."""
        table = _tower(8)
        info = get_model_info(A3B)
        geometry = ParallelGeometry(pipeline=2)
        worst = max(RULE.footprint(table, geometry, 2, num_layers=8))
        fits = {
            fit.geometry: fit.footprint
            for fit in fitting_geometries(info, table, 2, capacity=worst, worlds=(2,))
        }
        assert fits[geometry] == worst
        short = fitting_geometries(info, table, 2, capacity=worst - 1, worlds=(2,))
        assert geometry not in {fit.geometry for fit in short}

    @staticmethod
    def _refusal(geometry: ParallelGeometry, fits: tuple[Fit, ...] = ()) -> str:
        return memory_refusal(
            geometry=geometry,
            rank=0,
            device="cuda:0",
            dtype="bf16",
            resident=GIB,
            footprint=2 * GIB,
            free=GIB,
            total=2 * GIB,
            fits=fits,
        )

    def test_the_refusal_spells_a_geometry_as_the_flag_does(self) -> None:
        """Axes at one are left out, several axes are comma-joined the way
        ``--parallel`` reads them, and world 1 is spelled as such."""
        text = self._refusal(ParallelGeometry(pipeline=2, expert=2))
        assert text.startswith("--parallel: pp=2,ep=2 would place")
        assert self._refusal(ONE).startswith("--parallel: world 1 would place")
        fits = (Fit(ParallelGeometry(pipeline=2, tensor=2), GIB),)
        assert (
            "geometries estimated to fit here: pp=2,tp=2 (world 4, about 1.00 GiB per rank)"
            in self._refusal(ONE, fits)
        )

    def test_the_refusal_names_two_fits_and_says_when_there_are_none(self) -> None:
        fits = tuple(Fit(ParallelGeometry(pipeline=p), p * GIB) for p in (2, 4, 8))
        text = self._refusal(ONE, fits)
        assert text.endswith(
            "geometries estimated to fit here: pp=2 (world 2, about 2.00 GiB per rank); "
            "pp=4 (world 4, about 4.00 GiB per rank). Refused before any weight "
            "is read (--device cuda:0)."
        )
        assert "pp=8" not in text
        assert self._refusal(ONE).endswith(
            "; no pp/ep/tp geometry of this world or twice its size satisfies "
            "the headroom estimate. Refused before any weight is read (--device cuda:0)."
        )

    def test_the_dtype_refusal_carries_its_code_path_and_words(self) -> None:
        with pytest.raises(ProtocolError) as info:
            memory_check(
                geometry=ParallelGeometry(),
                rank=0,
                device="cuda:0",
                dtype="int8",
                table=_tower(1),
                info=get_model_info(A3B),
                free=1,
                total=1,
            )
        assert info.value.code == "P4" and info.value.path == "model.dtype"
        assert (
            info.value.message == "dtype 'int8' is not one of ['bf16', 'fp16', 'fp32']"
        )

    @pytest.mark.parametrize(
        ("rank", "fact"),
        (
            (2, "rank 2 is outside range(2)"),
            (-1, "rank -1 is outside range(2)"),
            # a bool is not a rank, though ``True in range(2)``: the type
            # refusal, not a false sentence about the range
            (True, "rank must be an int, got True"),
            (1.0, "rank must be an int, got 1.0"),
        ),
    )
    def test_a_rank_outside_the_world_or_not_an_int_is_refused_before_any_estimate(
        self, rank: object, fact: str
    ) -> None:
        geometry = ParallelGeometry(tensor=2)
        with pytest.raises(ProtocolError) as info:
            memory_check(
                geometry=geometry,
                rank=rank,
                device="cuda:0",
                dtype="bf16",
                table=_tower(1),
                info=get_model_info(A3B),
                free=1 << 40,
                total=1 << 40,
            )
        assert info.value.code == "P4" and info.value.path == "--parallel"
        # an internal guard (the launcher hands the mesh's rank): pinned to
        # the fact it names, not the character
        assert fact in info.value.message

    def test_the_rule_handed_in_governs_the_search_and_the_words(self) -> None:
        """Headroom affects suggestions for a resident-weight refusal."""
        table = _tower(8)
        info = get_model_info(A3B)
        whole = whole_bytes(table, 2)
        greedy = EstimateRule(base=10.0)
        common = dict(
            geometry=ONE,
            rank=0,
            device="cuda:0",
            dtype="bf16",
            table=table,
            info=info,
            free=whole - 1,
            total=whole,
        )
        text = memory_check(**common, rule=greedy)
        assert text is not None
        assert "1000% of the model's bytes" in text and "15%" not in text
        assert "no pp/ep/tp geometry of this world or twice its size" in text
        default = memory_check(**common)
        assert default is not None and "pp=2 (world 2" in default


# --------------------------------------------------------------------------- #
# the pins: the record and the reference card peaks
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def a3b_table() -> dict[str, Placement]:
    info = get_model_info(A3B)
    census = load_census()
    elements = {key: _product(shape) for key, (_, shape) in census.items()}
    tree = tree_of(elements, info.num_layers)
    assert tree == LLAMA_TREE.tree
    targets = checkpoint_targets(elements, tree, info.num_layers)
    assert targets is not None and info.parallel_plan is not None
    return placement_table(elements, targets, info.parallel_plan, tree, info)


def _product(shape: tuple[int, ...]) -> int:
    count = 1
    for n in shape:
        count *= n
    return count


def _realization_model(block: dict, record: dict) -> str:
    """The model a document block was captured on: its own ``realization``
    when it names one (the fp32 dense fit), else the record's."""
    return (block.get("realization") or {}).get("model", record["model"])


@pytest.mark.unit
class TestPinnedAgainstTheRecord:
    def test_the_estimate_reproduces_the_loaders_bytes_per_rank_to_the_byte(
        self, a3b_table
    ) -> None:
        """Every ``load`` block of the parallel golden — three documents,
        ``dp=2``, ``dp=2:rows``, ``pp=2``, ``tp=2``, ``ep=2`` — per rank."""
        record = json.loads(RECORD.read_text())
        checked = 0
        for block in record["documents"].values():
            if _realization_model(block, record) != record["model"]:
                continue  # a document on another model: not this header table's
            for spelled, ranks in block["load"].items():
                estimate = estimate_resident(a3b_table, parse_geometry(spelled), 2)
                for name, entry in ranks.items():
                    assert estimate[int(name[4:])] == entry["bytes_requested"], (
                        spelled,
                        name,
                    )
                    checked += 1
        assert checked >= 18
        assert whole_bytes(a3b_table, 2) == 69321221376  # 64.56 GiB

    def test_the_footprint_never_sits_below_a_recorded_peak(self, a3b_table) -> None:
        record = json.loads(RECORD.read_text())
        seen = 0
        for block in record["documents"].values():
            if _realization_model(block, record) != record["model"]:
                continue
            for spelled, ranks in block.get("memory", {}).items():
                geometry = ONE if spelled == "solo" else parse_geometry(spelled)
                footprint = RULE.footprint(a3b_table, geometry, 2)
                for name, entry in ranks.items():
                    assert footprint[int(name[4:])] >= entry["peak_bytes_allocated"]
                    seen += 1
        assert seen >= 8

    @pytest.mark.parametrize("spelled", sorted(CARD_PEAKS_GIB))
    def test_the_footprint_lands_within_the_band_of_the_card_peak(
        self, a3b_table, spelled
    ) -> None:
        """Allow 5% error above 60 GiB and one third below that. The wider
        band covers conservative allocator references and the inference
        peak compared against workflow headroom."""
        card = CARD_PEAKS_GIB[spelled] * GIB
        estimate = max(RULE.footprint(a3b_table, parse_geometry(spelled), 2))
        band = 0.05 if card > 60 * GIB else 1 / 3
        assert abs(estimate - card) <= band * card, (spelled, estimate / GIB)

    def test_cp2_headroom_is_advisory_and_smaller_estimates_remain_available(
        self, a3b_table
    ) -> None:
        info = get_model_info(A3B)
        text = memory_check(
            geometry=parse_geometry("cp=2"),
            rank=0,
            device="cuda:0",
            dtype="bf16",
            table=a3b_table,
            info=info,
            free=H100_FREE,
            total=H100_TOTAL,
        )
        assert text is None
        assert max(RULE.footprint(a3b_table, parse_geometry("cp=2"), 2)) > H100_FREE
        fits = fitting_geometries(info, a3b_table, 2, H100_FREE, worlds=(2, 4))
        assert [fit.geometry for fit in fits[:2]] == [
            parse_geometry("pp=2"),
            parse_geometry("ep=2"),
        ]
        assert [format_bytes(fit.footprint) for fit in fits[:2]] == [
            "41.96 GiB",
            "44.24 GiB",
        ]
        for spelled in ("pp=2", "ep=2", "tp=2", "pp=2,ep=2", "tp=8"):
            assert (
                memory_check(
                    geometry=parse_geometry(spelled),
                    rank=0,
                    device="cuda:0",
                    dtype="bf16",
                    table=a3b_table,
                    info=info,
                    free=H100_FREE,
                    total=H100_TOTAL,
                )
                is None
            ), spelled

    def test_fp32_doubles_the_bytes_and_two_h100s_only_miss_estimated_headroom(
        self, a3b_table
    ) -> None:
        info = get_model_info(A3B)
        assert whole_bytes(a3b_table, 4) == 2 * whole_bytes(a3b_table, 2)
        text = memory_check(
            geometry=parse_geometry("ep=2"),
            rank=1,
            device="cuda:1",
            dtype="fp32",
            table=a3b_table,
            info=info,
            free=H100_FREE,
            total=H100_TOTAL,
        )
        assert text is None
        geometry = parse_geometry("ep=2")
        assert max(estimate_resident(a3b_table, geometry, 4)) < H100_FREE
        assert max(RULE.footprint(a3b_table, geometry, 4)) > H100_FREE
        fits = fitting_geometries(info, a3b_table, 4, H100_FREE, worlds=(2, 4))
        assert fits and all(fit.geometry.world == 4 for fit in fits)
