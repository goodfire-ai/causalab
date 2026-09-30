"""KV-head replication, the torch-free half (``docs/model_parallelism.md``
§5.2, §6.6): the head arithmetic of ``protocol/kv_replication.py``, the
plan rewrite ``ParallelPlan.for_geometry`` and the relaxed KV rule of
``check``.

``property``: over every GQA shape a tensor axis can divide, the map
``(rank, local query head) → KV head`` is total and — the invariant the
engine rests on — selecting the rank's KV heads and repeating them
``local_groups`` times, as the library's ``repeat_kv`` does on the rank,
reproduces the rank's slice of the global ``repeat_kv``; the rewrite is the
identity when the tensor axis divides the KV heads and replaces exactly the
colwise ``k_proj`` / ``v_proj`` rows when it exceeds them; the rewrite of
every built-in entry at every divisor of its head count. ``unit``: the
straddle rule pinned on the built-in entries by name — the A3B (16 heads,
2 KV heads) accepts ``tp ∈ {1, 2, 4, 8, 16}``, Qwen2.5-0.5B (14 heads, 2 KV
heads) accepts ``{1, 2, 14}`` and refuses ``7``, where a rank's two query
heads would straddle two KV heads — and the row validation.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from causalab.protocol import parallel
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.kv_replication import KvHeads, kv_refusal
from causalab.protocol.parallel import ParallelGeometry, check
from causalab.protocol.registry.models import _REGISTRY  # pyright: ignore[reportPrivateUsage]
from causalab.protocol.registry import (
    KV_PROJECTIONS,
    LLAMA_PLAN,
    NO_PLAN,
    QWEN3_PLAN,
    QWEN35_MOE_PLAN,
    STYLES,
    ModelInfo,
    ParallelPlan,
    PlanRow,
    get_model_info,
)

from tests._helpers.geometries import divisors

_SETTINGS = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

A3B = "Qwen/Qwen3.6-35B-A3B"
LLAMA_8B = "meta-llama/Llama-3.1-8B"
QWEN25 = "Qwen/Qwen2.5-0.5B"
GEMMA2 = "google/gemma-2-2b"

#: The tensor axes each built-in entry's KV fact admits: every divisor of
#: its head count that divides the KV heads (a colwise shard of them) or is
#: a multiple of them (one KV head replicated per rank). gpt2's plan-less
#: family and an unserved style row refuses elsewhere;
#: this table is the KV fact alone. Its keys are the registry's own entries
#: (``registry.py``'s ``register_model`` calls) — a test that registers a
#: fixture entry in the same process is not a built-in.
ACCEPTED_TENSOR: dict[str, set[int]] = {
    A3B: {1, 2, 4, 8, 16},
    LLAMA_8B: {1, 2, 4, 8, 16, 32},
    "meta-llama/Llama-3.1-8B-Instruct": {1, 2, 4, 8, 16, 32},
    "meta-llama/Llama-3.2-1B": {1, 2, 4, 8, 16, 32},
    "meta-llama/Llama-3.2-1B-Instruct": {1, 2, 4, 8, 16, 32},
    "Qwen/Qwen3-4B-Instruct-2507": {1, 2, 4, 8, 16, 32},
    QWEN25: {1, 2, 14},
    GEMMA2: {1, 2, 4, 8},
    "google/gemma-2-2b-it": {1, 2, 4, 8},
    "gpt2": {1, 2, 3, 4, 6, 12},
    "gpt2-xl": {1, 5, 25},
}
BUILT_IN = tuple(sorted(ACCEPTED_TENSOR))
assert set(BUILT_IN) <= set(_REGISTRY), sorted(set(BUILT_IN) - set(_REGISTRY))


def _info(num_heads: int, num_kv_heads: int, plan: ParallelPlan | None) -> ModelInfo:
    return ModelInfo(
        key="test/gqa",
        hidden_size=num_heads * 8,
        num_layers=4,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=8,
        intermediate_size=64,
        vocab_size=64,
        family="llama",
        parallel_plan=plan,
    )


@st.composite
def gqa_shapes(draw: Any) -> tuple[int, int, int]:
    """``(num_heads, num_kv_heads, tensor)`` with ``num_kv_heads | num_heads``
    (every GQA model) and ``tensor | num_heads`` (the head fact); the KV
    fact is left free, so a share of the draws straddle."""
    num_kv_heads = draw(st.integers(1, 8))
    groups = draw(st.integers(1, 8))
    num_heads = num_kv_heads * groups
    tensor = draw(st.sampled_from(divisors(num_heads)))
    return num_heads, num_kv_heads, tensor


def _global_repeat(num_heads: int, num_kv_heads: int) -> list[int]:
    """``repeat_kv``'s contiguous repeat, spelled by hand: query head ``i``
    reads KV head ``i // (num_heads / num_kv_heads)``."""
    groups = num_heads // num_kv_heads
    return [i // groups for i in range(num_heads)]


# --------------------------------------------------------------------------- #
# the head arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.property
class TestKvHeads:
    @_SETTINGS
    @given(shape=gqa_shapes())
    def test_the_head_map_is_total_and_is_the_library_repeat(
        self, shape: tuple[int, int, int]
    ) -> None:
        num_heads, num_kv_heads, tensor = shape
        heads = KvHeads(num_heads=num_heads, num_kv_heads=num_kv_heads, tensor=tensor)
        assert heads.fits == (num_kv_heads % tensor == 0 or tensor % num_kv_heads == 0)
        if not heads.fits:
            return
        expected = _global_repeat(num_heads, num_kv_heads)
        local = heads.local_heads
        assert local * tensor == num_heads
        for rank in range(tensor):
            selected = heads.kv_heads(rank)
            # the rank's query heads name a contiguous run of KV heads …
            assert list(selected) == list(range(selected.start, selected.stop))
            # … each of which serves exactly ``local_groups`` of them, so the
            # library's repeat on the rank is the global repeat's slice
            assert local == len(selected) * heads.local_groups
            repeated = [selected[j // heads.local_groups] for j in range(local)]
            assert repeated == expected[rank * local : (rank + 1) * local]
            for h in range(local):
                assert heads.kv_head_of(rank, h) == expected[rank * local + h]

    @_SETTINGS
    @given(shape=gqa_shapes())
    def test_replication_is_exactly_the_axis_exceeding_the_kv_heads(
        self, shape: tuple[int, int, int]
    ) -> None:
        num_heads, num_kv_heads, tensor = shape
        heads = KvHeads(num_heads=num_heads, num_kv_heads=num_kv_heads, tensor=tensor)
        assert heads.replicates == (tensor > num_kv_heads)
        if heads.replicates and heads.fits:
            # one KV head per rank, held by ``repeat`` consecutive ranks
            assert heads.repeat * num_kv_heads == tensor
            assert all(len(heads.kv_heads(r)) == 1 for r in range(tensor))
            assert [heads.kv_heads(r).start for r in range(tensor)] == [
                r // heads.repeat for r in range(tensor)
            ]
            assert heads.local_groups == heads.local_heads
        if not heads.replicates and heads.fits:
            assert heads.repeat == 1
            assert heads.local_groups == num_heads // num_kv_heads
            assert all(
                len(heads.kv_heads(r)) == num_kv_heads // tensor for r in range(tensor)
            )

    @_SETTINGS
    @given(shape=gqa_shapes())
    def test_the_refusal_is_present_exactly_when_the_shape_straddles(
        self, shape: tuple[int, int, int]
    ) -> None:
        num_heads, num_kv_heads, tensor = shape
        heads = KvHeads(num_heads=num_heads, num_kv_heads=num_kv_heads, tensor=tensor)
        text = kv_refusal(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tensor=tensor,
            key="test/gqa",
        )
        assert (text is None) == heads.fits
        if text is not None:
            # the sentence ``check`` appends after ``--parallel.tensor: tp=N``
            assert f"num_kv_heads={num_kv_heads}" in text
            assert f"num_heads={num_heads}" in text


@pytest.mark.unit
class TestStraddle:
    def test_the_qwen25_at_tp7_straddles_and_is_named(self) -> None:
        """14 heads, 2 KV heads: 7 query heads per KV head, 2 per rank at
        ``tp=7`` — rank 3 holds query heads 6 and 7, which read KV heads 0
        and 1. Neither a shard of the KV heads nor a replication fits."""
        heads = KvHeads(num_heads=14, num_kv_heads=2, tensor=7)
        assert not heads.fits
        text = kv_refusal(num_heads=14, num_kv_heads=2, tensor=7, key=QWEN25)
        assert text is not None
        assert "num_kv_heads=2" in text and "num_heads=14" in text
        assert "straddle" in text
        with pytest.raises(ProtocolError) as err:
            heads.kv_heads(3)
        assert err.value.code == "P4"

    def test_a_head_count_the_kv_heads_do_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_kv_heads"):
            KvHeads(num_heads=6, num_kv_heads=4, tensor=1)

    def test_an_axis_the_heads_do_not_divide_is_refused(self) -> None:
        with pytest.raises(ValueError, match="num_heads"):
            KvHeads(num_heads=8, num_kv_heads=4, tensor=3)


# --------------------------------------------------------------------------- #
# the plan rewrite
# --------------------------------------------------------------------------- #


def _kv_rows(plan: ParallelPlan) -> dict[str, PlanRow]:
    return {
        pattern: row
        for pattern, row in plan.rows.items()
        if pattern.rsplit(".", 1)[-1] in KV_PROJECTIONS
    }


@pytest.mark.property
class TestForGeometry:
    @_SETTINGS
    @given(
        plan=st.sampled_from([LLAMA_PLAN, QWEN3_PLAN, QWEN35_MOE_PLAN]),
        shape=gqa_shapes(),
    )
    def test_rows_are_rewritten_iff_the_axis_exceeds_the_kv_heads(
        self, plan: ParallelPlan, shape: tuple[int, int, int]
    ) -> None:
        num_heads, num_kv_heads, tensor = shape
        heads = KvHeads(num_heads=num_heads, num_kv_heads=num_kv_heads, tensor=tensor)
        info = _info(num_heads, num_kv_heads, plan)
        geometry = ParallelGeometry(tensor=tensor)
        if not heads.fits:
            with pytest.raises(ProtocolError) as err:
                plan.for_geometry(geometry, info)
            assert err.value.code == "P4"
            assert "--parallel.tensor" in str(err.value)
            return
        placed = plan.for_geometry(geometry, info)
        if not heads.replicates:
            assert placed is plan
            return
        kv = _kv_rows(plan)
        assert kv, "every served dense plan carries k_proj and v_proj rows"
        for pattern, row in plan.rows.items():
            if pattern in kv:
                assert placed.rows[pattern] == PlanRow(
                    "kv_replicated", "tensor", repeat=heads.repeat
                )
            else:
                assert placed.rows[pattern] == row
        assert set(placed.rows) == set(plan.rows)
        # the registry's plan is a fact of the model type: untouched
        assert _kv_rows(plan) == kv
        assert all(row.style == "colwise" for row in kv.values())

    @pytest.mark.parametrize("key", BUILT_IN)
    def test_every_built_in_entry_rewrites_at_every_divisor_of_its_heads(
        self, key: str
    ) -> None:
        info = get_model_info(key)
        plan = info.parallel_plan
        assert plan is not None
        accepted: set[int] = set()
        for tensor in divisors(info.num_heads):
            geometry = ParallelGeometry(tensor=tensor)
            heads = KvHeads(
                num_heads=info.num_heads, num_kv_heads=info.num_kv_heads, tensor=tensor
            )
            if not heads.fits:
                with pytest.raises(ProtocolError):
                    plan.for_geometry(geometry, info)
                continue
            accepted.add(tensor)
            placed = plan.for_geometry(geometry, info)
            if tensor <= info.num_kv_heads or plan.empty:
                assert placed is plan
                continue
            rewritten = {
                pattern for pattern, row in placed.rows.items() if row.repeat > 1
            }
            assert rewritten == set(_kv_rows(plan))
            for pattern in rewritten:
                assert placed.rows[pattern].style == "kv_replicated"
                assert placed.rows[pattern].repeat == tensor // info.num_kv_heads
        assert accepted == ACCEPTED_TENSOR[key], key


@pytest.mark.unit
class TestForGeometryByName:
    def test_the_a3b_replicates_its_two_kv_heads_from_tp4(self) -> None:
        info = get_model_info(A3B)
        plan = info.parallel_plan
        assert plan is not None
        assert plan.for_geometry(ParallelGeometry(tensor=2), info) is plan
        for tensor, repeat in ((4, 2), (8, 4), (16, 8)):
            placed = plan.for_geometry(ParallelGeometry(tensor=tensor), info)
            for leaf in ("k_proj", "v_proj"):
                row = placed.rows[f"layers.*.self_attn.{leaf}"]
                assert row == PlanRow("kv_replicated", "tensor", repeat=repeat)
            # the norm's row stays: its weight is replicated either way
            assert placed.rows["layers.*.self_attn.k_norm"] == PlanRow(
                "replicated_with_grad_allreduce", "tensor"
            )
            assert placed.rows["layers.*.self_attn.q_proj"] == PlanRow(
                "colwise", "tensor"
            )
            # the expert rows are the expert axis's and untouched
            assert (
                placed.rows["layers.*.mlp.experts"] == plan.rows["layers.*.mlp.experts"]
            )

    def test_the_straddle_is_refused_naming_both_facts(self) -> None:
        info = get_model_info(QWEN25)
        plan = info.parallel_plan
        assert plan is not None
        with pytest.raises(ProtocolError) as err:
            plan.for_geometry(ParallelGeometry(tensor=7), info)
        text = str(err.value)
        assert "--parallel.tensor: tp=7" in text
        assert "num_kv_heads=2" in text and "num_heads=14" in text
        assert QWEN25 in text

    def test_a_plan_with_tensor_rows_but_no_kv_projection_is_refused(self) -> None:
        plan = ParallelPlan(
            {
                "layers.*.mlp.gate_proj": PlanRow("colwise", "tensor"),
                "layers.*.mlp.down_proj": PlanRow("rowwise", "tensor"),
            }
        )
        with pytest.raises(ProtocolError) as err:
            plan.for_geometry(ParallelGeometry(tensor=8), _info(8, 4, plan))
        assert err.value.code == "P4"
        assert "k_proj" in str(err.value) and "v_proj" in str(err.value)

    def test_a_kv_row_that_is_not_colwise_is_refused_by_name(self) -> None:
        plan = ParallelPlan(
            {
                "layers.*.self_attn.q_proj": PlanRow("colwise", "tensor"),
                "layers.*.self_attn.k_proj": PlanRow("colwise_gather_output", "tensor"),
                "layers.*.self_attn.v_proj": PlanRow("colwise", "tensor"),
            }
        )
        with pytest.raises(ProtocolError) as err:
            plan.for_geometry(ParallelGeometry(tensor=8), _info(8, 4, plan))
        assert "colwise_gather_output" in str(err.value)
        assert "k_proj" in str(err.value)

    def test_a_plan_with_no_tensor_rows_is_left_to_the_rows_rule(self) -> None:
        """No row means nothing to rewrite; ``tensor_has_plan_rows`` is the
        refusal, not this one."""
        info = _info(8, 4, NO_PLAN)
        assert NO_PLAN.for_geometry(ParallelGeometry(tensor=8), info) is NO_PLAN

    def test_the_kv_projections_are_the_two_transformers_leaves(self) -> None:
        assert KV_PROJECTIONS == frozenset({"k_proj", "v_proj"})
        assert "kv_replicated" in STYLES


# --------------------------------------------------------------------------- #
# the row
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestPlanRowRepeat:
    def test_a_repeat_above_one_belongs_to_the_replicated_style_alone(self) -> None:
        assert PlanRow("kv_replicated", "tensor", repeat=4).repeat == 4
        assert PlanRow("kv_replicated", "tensor").repeat == 1
        with pytest.raises(ValueError, match="repeat"):
            PlanRow("colwise", "tensor", repeat=2)

    def test_a_repeat_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repeat"):
            PlanRow("kv_replicated", "tensor", repeat=0)

    def test_the_repeat_is_part_of_the_rows_identity(self) -> None:
        assert PlanRow("kv_replicated", "tensor", repeat=2) != PlanRow(
            "kv_replicated", "tensor", repeat=4
        )
        assert PlanRow("colwise", "tensor") == PlanRow("colwise", "tensor", repeat=1)


# --------------------------------------------------------------------------- #
# check — the relaxed KV rule
# --------------------------------------------------------------------------- #


def _named(refusals: tuple[str, ...]) -> set[str]:
    return {text.partition(":")[0][len("--parallel.") :] for text in refusals}


@pytest.mark.unit
class TestCheck:
    @pytest.mark.parametrize("tensor", (1, 2, 4, 8, 16))
    def test_the_a3b_accepts_every_divisor_of_its_heads(self, tensor: int) -> None:
        assert check(ParallelGeometry(tensor=tensor), get_model_info(A3B)) == ()

    @pytest.mark.parametrize("tensor", (1, 2, 14))
    def test_the_qwen25_accepts_a_shard_or_a_replication_of_its_kv_heads(
        self, tensor: int
    ) -> None:
        assert check(ParallelGeometry(tensor=tensor), get_model_info(QWEN25)) == ()

    def test_the_qwen25_refuses_tp7_naming_both_facts(self) -> None:
        refusals = check(ParallelGeometry(tensor=7), get_model_info(QWEN25))
        assert _named(refusals) == {"tensor"}
        (only,) = refusals
        assert only.startswith("--parallel.tensor: tp=7 ")
        assert "num_kv_heads=2" in only and "num_heads=14" in only
        assert "straddle" in only

    def test_the_rule_sits_where_the_divisibility_rule_did(self) -> None:
        names = [rule.__name__ for rule in parallel.CHECKS]
        assert "tensor_divides_kv_heads" not in names
        assert names.index("tensor_fits_kv_heads") < names.index("tensor_has_plan_rows")
        assert names.index("tensor_divides_heads") < names.index("tensor_fits_kv_heads")

    def test_the_rule_says_nothing_where_the_head_fact_already_fails(self) -> None:
        """``tp=3`` on 8 heads fails the head fact; the KV rule is silent so
        the refusal stays one line naming the model fact."""
        info = _info(8, 4, LLAMA_PLAN)
        assert parallel.tensor_fits_kv_heads(ParallelGeometry(tensor=3), info) is None
        refusals = check(ParallelGeometry(tensor=3), info)
        assert len(refusals) == 1 and "num_heads=8" in refusals[0]
