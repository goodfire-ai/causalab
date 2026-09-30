"""The registry's parallel plan (``docs/model_parallelism.md`` §5.2) and the
geometry checks that read it (§2, §6.6).

The plan is a registry table — module pattern → ``(style, axis)`` — declared
on the built-in entries and derived from a loaded config's
``base_model_tp_plan`` / ``base_model_ep_plan`` for an adapted one. Held
here: the derivation on both tiny fixture configs (the EP rows *replace* the
MoE rows of the TP plan; the attention rows stay on the tensor axis); the
closed set of styles this phase serves, a style outside it recorded on its
row and refused **by name** by ``check`` before any weights; ``style_for``
matching a concrete module path with and without the base-model prefix;
every built-in entry declaring exactly the plan its transformers config class
ships (so ``dry-run`` reads it torch-free, offline); the two new ``check``
rules beside their twins; and the pipeline stage ranges as a property.

The derivation is torch-free and reads attributes only, so it runs in a
subprocess over a plain object with the two plan attributes; the comparisons
against transformers' own config classes run with torch imported, in the
``unit`` tier, offline.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.parallel import (
    CHECKS,
    PLANLESS_FAMILIES,
    MeshLayout,
    ParallelGeometry,
    check,
    expert_has_plan_rows,
    stage_layers,
    tensor_has_plan_rows,
)
from causalab.protocol.registry.models import _REGISTRY  # pyright: ignore[reportPrivateUsage]
from causalab.protocol.registry import (
    GEMMA2_PLAN,
    VOCABULARY_STYLES,
    LLAMA_PLAN,
    NO_PLAN,
    PLAN_AXES,
    QWEN3_PLAN,
    QWEN35_MOE_PLAN,
    QWEN35_PLAN,
    STYLES,
    ModelInfo,
    ParallelPlan,
    PlanRow,
    get_model_info,
    parallel_plan_from_hf_config,
    wildcard_layers,
)

from tests.neural.engines.pytorch_hooks.conftest import TINY_LLAMA, TINY_QWEN35_MOE

REPO = Path(__file__).resolve().parents[2]

_HYPOTHESIS_SETTINGS = settings(
    deadline=None,
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: The built-in entries, by key — spelled out because other suites register
#: their own entries into the same registry (``snapshot/moe``,
#: ``synthetic/third-tree``), which declare no plan and are not built in.
BUILT_IN = (
    "EleutherAI/gpt-j-6b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "gpt2",
    "gpt2-xl",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
)

#: What the tiny Llama's config class ships (transformers 5.16.1,
#: ``configuration_llama.py``), spelled out so the derivation is held to an
#: independent statement rather than to itself.
_LLAMA_TP = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}

#: The Qwen3.5-MoE text config's two plans (``configuration_qwen3_5_moe.py``).
_MOE_TP = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
    "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
    "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
    "layers.*.mlp.experts.down_proj": "rowwise",
    "layers.*.mlp.experts": "moe_tp_experts",
    "layers.*.mlp.shared_expert.gate_proj": "colwise",
    "layers.*.mlp.shared_expert.up_proj": "colwise",
    "layers.*.mlp.shared_expert.down_proj": "rowwise",
    "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
    "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
    "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
    "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
    "layers.*.linear_attn.out_proj": "colwise_gather_output",
}
_MOE_EP = {
    "layers.*.mlp.gate": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
    "layers.*.mlp.experts": "moe_tp_experts",
}


def _stub(tp: dict[str, str] | None, ep: dict[str, str] | None = None) -> Any:
    """A config-shaped object carrying only the two plan attributes."""
    return SimpleNamespace(base_model_tp_plan=tp, base_model_ep_plan=ep)


def _info(plan: ParallelPlan | None, **overrides: Any) -> ModelInfo:
    """A dense, plan-carrying entry every divisibility fact of ``tp ∈ {1, 2,
    4}`` holds for; ``num_experts`` makes it a MoE entry."""
    fields: dict[str, Any] = dict(
        key="test/planned",
        hidden_size=64,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        head_dim=8,
        intermediate_size=128,
        vocab_size=64,
        family="llama",
        parallel_plan=plan,
    )
    fields.update(overrides)
    return ModelInfo(**fields)


def _named(refusals: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for text in refusals:
        prefix, _, _ = text.partition(":")
        assert prefix.startswith("--parallel."), text
        out.add(prefix[len("--parallel.") :])
    return out


# --------------------------------------------------------------------------- #
# the table's types
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_styles_are_the_closed_set_this_phase_serves() -> None:
    """The eight transformers styles the placement table (§4) has a row for,
    and the repository's own ``kv_replicated`` (§6.6, written by
    ``ParallelPlan.for_geometry``, never by a config); anything else a
    config's plan names is unserved, and refused by name."""
    assert STYLES == frozenset(
        {
            "colwise",
            "rowwise",
            "packed_colwise",
            "colwise_gather_output",
            "replicated_with_grad_allreduce",
            "moe_tp_experts",
            "grouped_gemm",
            "ep_router",
            "kv_replicated",
        }
    )
    assert PLAN_AXES == ("tensor", "expert")


@pytest.mark.unit
def test_a_row_names_a_style_and_one_of_the_two_plan_axes() -> None:
    assert PlanRow("colwise", "tensor").axis == "tensor"
    with pytest.raises(ValueError, match="axis"):
        PlanRow("colwise", "pipeline")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="style"):
        PlanRow("", "tensor")


@pytest.mark.unit
def test_an_empty_plan_is_what_a_family_without_a_transformers_plan_derives_to() -> (
    None
):
    plan = parallel_plan_from_hf_config(_stub(None))
    assert plan == NO_PLAN and plan.empty and not plan.rows
    assert not parallel_plan_from_hf_config(_stub(_LLAMA_TP)).empty


@pytest.mark.unit
def test_the_derivation_is_torch_free() -> None:
    """``dry-run`` reaches the plan from the entry alone, with torch never
    imported (docs/TESTS.md, "The dry run"): the derivation reads two
    attributes off any object and imports nothing above the registry."""
    program = (
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from causalab.protocol.registry import parallel_plan_from_hf_config\n"
        "plan = parallel_plan_from_hf_config(SimpleNamespace("
        "base_model_tp_plan={'layers.*.self_attn.q_proj': 'colwise'},"
        " base_model_ep_plan={'layers.*.mlp.gate': 'ep_router'}))\n"
        "print(sorted((k, r.style, r.axis) for k, r in plan.rows.items()))\n"
        "print('torch' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, cwd=str(REPO)
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines[0] == str(
        [
            ("layers.*.mlp.gate", "ep_router", "expert"),
            ("layers.*.self_attn.q_proj", "colwise", "tensor"),
        ]
    )
    assert lines[1] == "False"


# --------------------------------------------------------------------------- #
# derivation from the two fixture configs
# --------------------------------------------------------------------------- #


def _config(key: str) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(key)
    return getattr(config, "text_config", None) or config


@pytest.mark.unit
def test_the_tiny_llama_derives_to_tensor_rows_only() -> None:
    plan = parallel_plan_from_hf_config(_config(TINY_LLAMA))
    assert {k: r.style for k, r in plan.rows.items()} == _LLAMA_TP
    assert {r.axis for r in plan.rows.values()} == {"tensor"}
    assert not plan.unserved
    assert plan == LLAMA_PLAN


@pytest.mark.unit
def test_the_tiny_moe_derives_with_the_ep_rows_replacing_the_moe_rows() -> None:
    """§5.2: under the plan the experts and the router are the EP plan's
    rows on the expert axis — ``grouped_gemm`` where the TP plan said
    ``packed_colwise`` / ``rowwise`` — and every other row keeps the TP
    plan's style on the tensor axis."""
    plan = parallel_plan_from_hf_config(_config(TINY_QWEN35_MOE))
    experts = {
        k: r for k, r in plan.rows.items() if k.startswith("layers.*.mlp.experts")
    }
    assert {k: (r.style, r.axis) for k, r in experts.items()} == {
        "layers.*.mlp.experts.gate_up_proj": ("grouped_gemm", "expert"),
        "layers.*.mlp.experts.down_proj": ("grouped_gemm", "expert"),
        "layers.*.mlp.experts": ("moe_tp_experts", "expert"),
    }
    assert plan.rows["layers.*.mlp.gate"] == PlanRow("ep_router", "expert")
    tensor_rows = {k: r.style for k, r in plan.rows.items() if r.axis == "tensor"}
    assert tensor_rows == {
        k: v for k, v in _MOE_TP.items() if not k.startswith("layers.*.mlp.experts")
    }
    assert set(plan.rows) == (set(_MOE_TP) | set(_MOE_EP))
    assert not plan.unserved
    assert plan == QWEN35_MOE_PLAN


@pytest.mark.unit
def test_replacement_is_by_pattern_and_its_children_not_by_order() -> None:
    """The EP plan is a mapping: whatever order it lists ``experts`` and
    ``experts.down_proj`` in, both end up on the expert axis with the EP
    style — the mutation (dropping the child rows when the parent is seen)
    left the experts unsharded and the router remapping into nothing."""
    reversed_ep = dict(reversed(list(_MOE_EP.items())))
    a = parallel_plan_from_hf_config(_stub(_MOE_TP, _MOE_EP))
    b = parallel_plan_from_hf_config(_stub(_MOE_TP, reversed_ep))
    assert a == b == QWEN35_MOE_PLAN
    assert a.rows["layers.*.mlp.experts.down_proj"].style == "grouped_gemm"


@pytest.mark.unit
def test_rows_on_an_axis_are_that_axis_alone() -> None:
    plan = QWEN35_MOE_PLAN
    assert set(plan.rows_on("expert")) == set(_MOE_EP)
    assert set(plan.rows_on("tensor")) == set(_MOE_TP) - {
        k for k in _MOE_TP if k.startswith("layers.*.mlp.experts")
    }
    assert set(LLAMA_PLAN.rows_on("expert")) == set()


# --------------------------------------------------------------------------- #
# a style outside the table
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_a_style_outside_the_table_is_kept_on_its_row_and_marked_unserved() -> None:
    """The derivation is total — a world-1 load of a family whose plan names
    a style this phase does not serve must still load — so the row is kept
    and marked, and the refusal is ``check``'s (next test), by name."""
    plan = parallel_plan_from_hf_config(
        _stub({**_LLAMA_TP, "layers.*.self_attn.kv_a_proj": "mla_kv_a_proj"})
    )
    assert plan.unserved == {
        "layers.*.self_attn.kv_a_proj": PlanRow("mla_kv_a_proj", "tensor")
    }
    assert set(plan.rows) == set(_LLAMA_TP) | {"layers.*.self_attn.kv_a_proj"}


@pytest.mark.unit
def test_check_refuses_an_unserved_style_by_name_on_its_axis() -> None:
    """``tp=2`` on a plan with an unserved tensor row is refused naming the
    pattern and the style — before a wrong number, the way nnsight's
    ``plan.py`` refuses a style outside its table. The dividing twin with
    the served rows alone is accepted; the unserved row is on the tensor
    axis, so ``tp=1`` never meets it."""
    plan = parallel_plan_from_hf_config(
        _stub({**_LLAMA_TP, "layers.*.self_attn.kv_a_proj": "mla_kv_a_proj"})
    )
    refusals = check(ParallelGeometry(tensor=2), _info(plan))
    assert _named(refusals) == {"tensor"}
    (only,) = refusals
    assert "kv_a_proj" in only and "mla_kv_a_proj" in only
    assert check(ParallelGeometry(tensor=2), _info(LLAMA_PLAN)) == ()
    assert check(ParallelGeometry(), _info(plan)) == ()


@pytest.mark.unit
def test_a_vocabulary_row_is_dropped_the_embedding_and_head_whole_on_every_rank() -> (
    None
):
    """§6.1's full vocabulary: a config's ``embedding_rowwise`` row is not
    applied and not refused — the embedding (and a head tied to it) is a
    module in no row, replicated. 📐 Qwen3-4B-Instruct-2507 and gemma2 ship
    the row; ``tp=2`` on either was refused by name until this rule."""
    plan = parallel_plan_from_hf_config(
        _stub({**_LLAMA_TP, "embed_tokens": "embedding_rowwise"})
    )
    assert plan.rows == LLAMA_PLAN.rows and plan.unserved == {}
    # the declined row is kept as provenance — dry-run names it — but it is
    # not the plan's value: two plans with the same rows are one plan
    assert dict(plan.unapplied) == {"embed_tokens": "embedding_rowwise"}
    assert plan == LLAMA_PLAN and hash(plan) == hash(LLAMA_PLAN)
    assert dict(LLAMA_PLAN.unapplied) == {}
    assert check(ParallelGeometry(tensor=2), _info(plan)) == ()
    assert GEMMA2_PLAN.rows == LLAMA_PLAN.rows
    # gemma2's class ships the row, so its hand-spelled plan carries the
    # provenance the derivation would have (the conformance test holds it)
    assert dict(GEMMA2_PLAN.unapplied) == {"embed_tokens": "embedding_rowwise"}
    # the K/V-replication rewrite changes rows, never provenance
    replicated = plan.for_geometry(ParallelGeometry(tensor=8), _info(plan))
    assert replicated.rows["layers.*.self_attn.k_proj"].style == "kv_replicated"
    assert dict(replicated.unapplied) == {"embed_tokens": "embedding_rowwise"}
    assert plan.for_geometry(ParallelGeometry(tensor=2), _info(plan)) is plan
    assert VOCABULARY_STYLES == frozenset({"embedding_rowwise"})
    for key in ("google/gemma-2-2b-it", "google/gemma-2-2b"):
        assert check(ParallelGeometry(tensor=2), get_model_info(key)) == ()


# --------------------------------------------------------------------------- #
# style_for — a concrete path against the patterns
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_wildcard_layers_replaces_dotted_numbers_only() -> None:
    """transformers' ``replace_layer_number_by_wildcard``, transcribed: a
    number between dots (or a dot and the end) is a ModuleList index; a
    digit inside a name (``w1``) is not."""
    assert wildcard_layers("model.layers.3.self_attn.q_proj") == (
        "model.layers.*.self_attn.q_proj"
    )
    assert wildcard_layers("layers.10") == "layers.*"
    assert wildcard_layers("mlp.experts.7.w1.weight") == "mlp.experts.*.w1.weight"
    assert wildcard_layers("lm_head") == "lm_head"


@pytest.mark.unit
def test_style_for_matches_with_and_without_the_base_prefix() -> None:
    row = PlanRow("colwise", "tensor")
    assert LLAMA_PLAN.style_for("model.layers.3.self_attn.q_proj") == row
    assert LLAMA_PLAN.style_for("layers.3.self_attn.q_proj") == row
    assert (
        LLAMA_PLAN.style_for("model.layers.3.self_attn.q_proj", prefix="model") == row
    )
    assert LLAMA_PLAN.style_for("layers.11.mlp.down_proj") == PlanRow(
        "rowwise", "tensor"
    )


@pytest.mark.unit
def test_style_for_is_none_off_the_table() -> None:
    for path in (
        "model.layers.3.input_layernorm",
        "model.norm",
        "model.embed_tokens",
        "lm_head",
        "model.layers.3.self_attn",
        "model.layers.3.self_attn.q_proj.weight",
    ):
        assert LLAMA_PLAN.style_for(path) is None, path


@pytest.mark.unit
def test_style_for_names_a_parameter_row_as_well_as_a_module_row() -> None:
    """The MoE plan's ``grouped_gemm`` rows name parameters of the experts
    module (3-D tensors, no per-expert child), its ``moe_tp_experts`` row
    the module — both are looked up the same way, the caller deciding which
    path (parameter or module) to ask about, as ``apply_tensor_parallelism``
    does."""
    assert QWEN35_MOE_PLAN.style_for("model.layers.0.mlp.experts.gate_up_proj") == (
        PlanRow("grouped_gemm", "expert")
    )
    assert QWEN35_MOE_PLAN.style_for("model.layers.0.mlp.experts") == PlanRow(
        "moe_tp_experts", "expert"
    )
    assert QWEN35_MOE_PLAN.style_for("model.layers.2.linear_attn.in_proj_qkv") == (
        PlanRow("colwise_gather_output", "tensor")
    )
    assert QWEN35_MOE_PLAN.style_for("model.layers.0.mlp.gate") == PlanRow(
        "ep_router", "expert"
    )


@pytest.mark.unit
def test_an_explicit_prefix_that_is_not_there_is_not_stripped() -> None:
    """With ``prefix`` given, exactly that prefix is stripped — a path under
    another root is off the table, not guessed at by dropping a component."""
    assert LLAMA_PLAN.style_for("other.layers.3.self_attn.q_proj", prefix="model") is (
        None
    )
    assert LLAMA_PLAN.style_for("other.layers.3.self_attn.q_proj") == PlanRow(
        "colwise", "tensor"
    )


# --------------------------------------------------------------------------- #
# the built-in entries
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("key", BUILT_IN)
def test_every_built_in_entry_declares_the_plan_its_config_class_ships(
    key: str,
) -> None:
    """The declared table equals the derivation from transformers' own config
    class for the entry's family (``AutoConfig.for_model`` — the class's
    defaults, no download), so a transformers upgrade that moves a plan is a
    red test, not a silent drift between ``dry-run`` and the load."""
    from transformers import AutoConfig

    info = get_model_info(key)
    assert info.family is not None
    config = AutoConfig.for_model(info.family.removesuffix("_text"))
    text = getattr(config, "text_config", None) or config
    assert text.model_type == info.family
    assert info.parallel_plan is not None, key
    derived = parallel_plan_from_hf_config(text)
    assert info.parallel_plan == derived, key
    # ``==`` reads the rows alone (a plan is its rows), so the provenance
    # ``dry-run`` prints is held separately: the declined vocabulary row of
    # a hand-spelled plan is the one the class's config ships
    assert dict(info.parallel_plan.unapplied) == dict(derived.unapplied), key


@pytest.mark.unit
def test_the_built_in_census_is_registered() -> None:
    """Every key above is a registered entry with a family. The census cannot
    be closed the other way — the registry is shared, and a suite that loads
    a fixture registers its adapted entry, plan included — so a new built-in
    joins the plan tests by being listed here."""
    for key in BUILT_IN:
        assert key in _REGISTRY and get_model_info(key).family is not None, key


@pytest.mark.unit
def test_a_dense_entry_says_nothing_on_the_expert_axis_beyond_its_fact() -> None:
    """``ep=2`` on the dense Llama is one refusal — the dense fact — not two:
    a dense entry has no experts to want a plan row for."""
    refusals = check(ParallelGeometry(expert=2), _info(LLAMA_PLAN))
    assert len(refusals) == 1 and "no routed experts" in refusals[0]


@pytest.mark.unit
def test_the_planless_families_are_exactly_the_entries_with_an_empty_plan() -> None:
    """One fact, two spellings kept consistent: ``PLANLESS_FAMILIES`` names
    the families whose transformers config ships no plan; their entries
    declare the empty plan, every other entry a plan with rows."""
    for key in BUILT_IN:
        info = get_model_info(key)
        assert info.parallel_plan is not None
        assert (info.family in PLANLESS_FAMILIES) == info.parallel_plan.empty, key
    assert NO_PLAN.empty
    for plan in (LLAMA_PLAN, QWEN3_PLAN, GEMMA2_PLAN, QWEN35_MOE_PLAN, QWEN35_PLAN):
        assert not plan.empty


@pytest.mark.unit
def test_the_named_plans_are_what_the_built_in_entries_declare() -> None:
    by_family = {
        get_model_info(k).family: get_model_info(k).parallel_plan for k in BUILT_IN
    }
    assert by_family["llama"] == LLAMA_PLAN
    assert by_family["qwen2"] == LLAMA_PLAN
    assert by_family["qwen3"] == QWEN3_PLAN
    assert by_family["gemma2"] == GEMMA2_PLAN
    assert by_family["gpt2"] == NO_PLAN
    assert by_family["gptj"] == NO_PLAN
    assert by_family["qwen3_5_moe_text"] == QWEN35_MOE_PLAN
    assert by_family["qwen3_5_text"] == QWEN35_PLAN
    assert QWEN3_PLAN.rows["layers.*.self_attn.q_norm"] == PlanRow(
        "replicated_with_grad_allreduce", "tensor"
    )


@pytest.mark.unit
def test_an_adapted_entry_carries_the_derived_plan() -> None:
    from transformers import AutoConfig

    from causalab.protocol.registry import model_info_from_hf_config

    info = model_info_from_hf_config(
        TINY_QWEN35_MOE, AutoConfig.from_pretrained(TINY_QWEN35_MOE)
    )
    assert info.parallel_plan == QWEN35_MOE_PLAN
    llama = model_info_from_hf_config(
        TINY_LLAMA, AutoConfig.from_pretrained(TINY_LLAMA)
    )
    assert llama.parallel_plan == LLAMA_PLAN


# --------------------------------------------------------------------------- #
# check — the plan rules beside their twins
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_plan_rules_are_in_the_check_table_after_the_divisibility_rules() -> None:
    """Order is report order: the first refusal on an axis stays the
    divisibility fact (the existing tests read ``refusals[0]``)."""
    names = [rule.__name__ for rule in CHECKS]
    assert names.index("expert_divides_experts") < names.index("expert_has_plan_rows")
    assert names.index("tensor_fits_kv_heads") < names.index("tensor_has_plan_rows")
    assert tensor_has_plan_rows in CHECKS and expert_has_plan_rows in CHECKS


@pytest.mark.unit
def test_tensor_above_one_needs_a_tensor_row_beside_its_twin() -> None:
    refusals = check(ParallelGeometry(tensor=2), _info(NO_PLAN))
    assert _named(refusals) == {"tensor"}
    (only,) = refusals
    assert "tp=2" in only and "plan" in only and "tensor" in only
    assert check(ParallelGeometry(tensor=2), _info(LLAMA_PLAN)) == ()


@pytest.mark.unit
def test_expert_above_one_needs_an_expert_row_beside_its_twin() -> None:
    """A MoE entry (the divisibility fact holds) whose plan has no expert
    row — a dense family's plan on a MoE config — is refused naming the
    expert axis; the MoE plan is accepted."""
    moe = dict(num_experts=128, num_experts_per_tok=10, moe_intermediate_size=32)
    refusals = check(ParallelGeometry(expert=2), _info(LLAMA_PLAN, **moe))
    assert _named(refusals) == {"expert"}
    (only,) = refusals
    assert "ep=2" in only and "plan" in only and "expert" in only
    assert check(ParallelGeometry(expert=2), _info(QWEN35_MOE_PLAN, **moe)) == ()
    assert (
        check(ParallelGeometry(tensor=2, expert=4), _info(QWEN35_MOE_PLAN, **moe)) == ()
    )


@pytest.mark.unit
def test_an_entry_declaring_no_plan_is_checked_on_its_facts_alone() -> None:
    """``parallel_plan=None`` is an entry the table has not met (a
    hand-declared one); the plan rules say nothing, the divisibility facts
    still do, and the load derives the plan from the config."""
    assert check(ParallelGeometry(tensor=2), _info(None)) == ()
    assert _named(check(ParallelGeometry(tensor=3), _info(None))) == {"tensor"}


@pytest.mark.unit
def test_the_plan_rules_say_nothing_at_one() -> None:
    for plan in (None, NO_PLAN, LLAMA_PLAN):
        assert tensor_has_plan_rows(ParallelGeometry(), _info(plan)) is None
        assert expert_has_plan_rows(ParallelGeometry(), _info(plan)) is None


@pytest.mark.unit
def test_gpt2_is_refused_twice_on_tensor_once_by_name_once_by_its_empty_plan() -> None:
    """Both spellings of the one fact fire; both name the tensor axis."""
    refusals = check(ParallelGeometry(tensor=2), get_model_info("gpt2"))
    assert _named(refusals) == {"tensor"} and len(refusals) == 2


# --------------------------------------------------------------------------- #
# stage_layers — pipeline placement of the layers
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_stage_layers_is_transformers_even_split_with_the_remainder_last() -> None:
    """``PipelineStage.layer_range_for_rank`` (transformers 5.16.1): 32 layers
    over 3 stages are ``[0, 10) [10, 20) [20, 32)``."""
    geometry = ParallelGeometry(pipeline=3)
    assert stage_layers(geometry, 0, 32) == range(0, 10)
    assert stage_layers(geometry, 1, 32) == range(10, 20)
    assert stage_layers(geometry, 2, 32) == range(20, 32)
    assert stage_layers(ParallelGeometry(), 0, 5) == range(0, 5)


@pytest.mark.unit
def test_stage_layers_is_read_off_the_pipeline_coordinate_of_the_rank() -> None:
    """Ranks of one pipeline stage — whatever their data, context or model
    coordinates — hold the same layers."""
    geometry = ParallelGeometry(data=2, pipeline=2, tensor=2)
    layout = MeshLayout(geometry)
    for rank in range(geometry.world):
        stage = layout.rank_in(rank, "pipeline")
        assert stage_layers(geometry, rank, 6) == (range(0, 3), range(3, 6))[stage]


@pytest.mark.unit
def test_stage_layers_refuses_more_stages_than_layers() -> None:
    with pytest.raises(ProtocolError) as err:
        stage_layers(ParallelGeometry(pipeline=3), 0, 2)
    assert err.value.code == "P4" and "--parallel.pipeline" in str(err.value)


@st.composite
def _towers(draw: st.DrawFn) -> tuple[ParallelGeometry, int]:
    num_layers = draw(st.integers(min_value=1, max_value=64))
    pipeline = draw(st.integers(min_value=1, max_value=num_layers))
    geometry = ParallelGeometry(
        data=draw(st.integers(min_value=1, max_value=2)),
        pipeline=pipeline,
        tensor=draw(st.sampled_from((1, 2))),
    )
    return geometry, num_layers


@pytest.mark.property
@_HYPOTHESIS_SETTINGS
@given(tower=_towers())
def test_stage_ranges_partition_the_layers_contiguously(
    tower: tuple[ParallelGeometry, int],
) -> None:
    """§10.3 row "pipeline ranges": the stages' ranges partition the layers,
    are contiguous, non-empty and ascending; every stage but the last holds
    ``num_layers // pipeline``, the last the rest."""
    geometry, num_layers = tower
    layout = MeshLayout(geometry)
    ranges = [
        stage_layers(geometry, rank, num_layers)
        for rank in range(geometry.world)
        if layout.rank_in(rank, "data") == 0 and layout.rank_in(rank, "model") == 0
    ]
    assert len(ranges) == geometry.pipeline
    flat = [layer for r in ranges for layer in r]
    assert flat == list(range(num_layers))
    assert all(len(r) >= 1 for r in ranges)
    per = num_layers // geometry.pipeline
    assert [len(r) for r in ranges[:-1]] == [per] * (geometry.pipeline - 1)
    assert len(ranges[-1]) == num_layers - per * (geometry.pipeline - 1)


@pytest.mark.unit
def test_mutation_last_stage_losing_the_remainder_fails_the_property() -> None:
    """The named mutation of the row: an even split that drops the remainder
    leaves layers no stage owns."""

    def even(geometry: ParallelGeometry, rank: int, num_layers: int) -> range:
        per = num_layers // geometry.pipeline
        stage = MeshLayout(geometry).rank_in(rank, "pipeline")
        return range(stage * per, (stage + 1) * per)

    geometry, num_layers = ParallelGeometry(pipeline=3), 32
    flat = [layer for rank in range(3) for layer in even(geometry, rank, num_layers)]
    assert flat != list(range(num_layers))
    flat = [
        layer for rank in range(3) for layer in stage_layers(geometry, rank, num_layers)
    ]
    assert flat == list(range(num_layers))


@pytest.mark.unit
def test_model_info_is_still_a_value() -> None:
    """The plan is part of the entry's equality and hashing — two entries
    differing only in their plan are two entries."""
    a, b = _info(LLAMA_PLAN), _info(QWEN3_PLAN)
    assert a != b and hash(a) != hash(b)
    assert dataclasses.replace(a, parallel_plan=QWEN3_PLAN) == b
