"""The component shape table, and the three defects it was built to close.

[`component_shape`][causalab.protocol.registry.components.component_shape] is the one description
everything downstream derives from — the feature width, whether a featurizer may
attach, how many heads ``head`` selects among, and the native↔contract
conversion. Before it, those four answers lived in four places and had drifted:

* the head bound read ``info.num_heads`` no matter the component (§2.2);
* ``stream`` parsed as an integer while the only code that reads it wants a
  string, so no document could use the field (§2.1);
* ``router_scores`` accepted a basis-fitting featurizer over an axis that is a
  per-token ranking;
* the MoE branch outputs were declared hidden-wide in one table and flat in
  another, and ``mlp_activation``'s width on the GPT-2 family came from a config
  key the modeling code ignores.

The last two were found by the width check itself, which is the argument for
having one.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize
from causalab.protocol.rules.errors import ParseError, ValidationError
from causalab.protocol.pipeline import compile_protocol
from causalab.protocol.rules.document import validate_document
from causalab.protocol.registry import (
    COMPONENT_STREAMS,
    ModelInfo,
    component_shape,
    component_width,
    get_model_info,
    model_info_from_hf_config,
    register_model,
)
from causalab.protocol.positions.alignment import COMPONENT_RANK
from causalab.protocol.schema import (
    COMPONENTS,
    DEPRECATED_COMPONENTS,
    STREAMS,
    parse_document,
)

from tests.protocol._docs import base_doc, in_order

pytestmark = pytest.mark.unit


def _validate_sites(raw: dict[str, Any], env: Any) -> None:
    """Rule 4's address half — a site's layer, stream, component and head
    against the model's static metadata — is the checklist's since the rules
    package: ``validate_document`` decides it
    with the environment's ``model_info``, and ``canonicalize`` no longer
    refuses an illegal site (the compiler's ``validate`` stage runs first)."""
    validate_document(parse_document(in_order(raw)), model_info=env.model_info)


#: A GQA model with every optional field populated, so one table can exercise
#: the whole vocabulary. ``num_kv_heads`` is deliberately half ``num_heads`` and
#: ``head_dim`` is deliberately *not* ``hidden_size // num_heads``: both are true
#: of the Qwen3.6 target, and both are the kind of coupling a table that assumed
#: them would silently get wrong.
GQA = ModelInfo(
    key="test/gqa",
    hidden_size=64,
    num_layers=4,
    num_heads=8,
    num_kv_heads=4,
    head_dim=16,
    intermediate_size=128,
    vocab_size=1000,
    num_experts=32,
    num_experts_per_tok=4,
    # deliberately three DIFFERENT inner widths (dense 128, shared 48, routed
    # 24): the tiny fixture checkpoint carries all three equal, so this table
    # is the one place a wrong-spelling pick is distinguishable at all
    shared_expert_intermediate_size=48,
    # ⚠️ deliberately different from shared_expert_intermediate_size (48) and
    # intermediate_size (128): on the fixture all three are 32, so only a table
    # like this one can catch a wrong-field read.
    moe_intermediate_size=24,
    # ⚠️ the DeltaNet mixer's four dimensions, deliberately uncoupled: v-heads
    # is not 2·k-heads and the two head dims differ — the fixture's couplings
    # (2× GVA tiling, equal dims) are exactly what a table must not assume.
    # q/k live in key-head space (3 heads × 10), v/gate/state in value-head
    # space (6 × 12).
    linear_num_value_heads=6,
    linear_num_key_heads=3,
    linear_key_head_dim=10,
    linear_value_head_dim=12,
)


@pytest.fixture(autouse=True)
def _registered() -> None:
    """The document-level tests below name these keys, and the registry is the
    protocol layer's only source of static config (§6: never the network)."""
    register_model(GQA)
    register_model(dataclasses.replace(GQA, key="test/moe"))


#: ``component -> (width, head_space, is_feature_space)``. One row per name in
#: the vocabulary; the completeness test below fails if a component is added
#: without one.
EXPECTED: dict[str, tuple[int | None, int | None, bool]] = {
    "input_ids": (None, None, False),
    "embeddings": (64, None, True),
    "block_input": (64, None, True),
    "attention_input_norm": (64, None, True),
    # the DeltaNet interior's module boundaries: qkv is 2·(3·10) + 6·12 = 132
    # wide with NO head axis (unequal fused widths); the gate and the premix
    # are value-head space, 6 heads of 12
    "delta_qkv": (132, None, True),
    "delta_gate": (72, 6, True),
    "delta_premix": (72, 6, True),
    # the kernel boundary: conv is the fused width again (no head axis); q/k
    # are TILED to the 6 v-heads of the 10-wide key head dim; the gates' one
    # scalar per head makes the feature axis the head axis
    "delta_conv": (132, None, True),
    "delta_query": (60, 6, True),
    "delta_key": (60, 6, True),
    "delta_value": (72, 6, True),
    "delta_beta": (6, 6, True),
    "delta_decay": (6, 6, True),
    # the per-step interior: the derived faces are ordinary v-head spaces; the
    # state's trailing axes form a d_k × d_v matrix per head — a head axis to
    # select on, but no feature vector, so no width and no featurizer
    "delta_kv_mem": (72, 6, True),
    "delta_state_update": (72, 6, True),
    "delta_state": (None, 6, False),
    "delta_kernel_output": (72, 6, True),
    "attention_probs": (None, 8, False),
    "attention_query_pre_rope": (128, 8, True),
    "attention_key_pre_rope": (64, 4, True),
    "attention_value_states": (64, 4, True),
    "attention_gate": (128, 8, True),
    "attention_query": (128, 8, True),
    "attention_key": (64, 4, True),
    "attention_scores": (None, 8, False),
    "attention_z": (128, 8, True),
    # the DeltaNet interior: key_dim = 3·10 = 30, value_dim = 6·12 = 72,
    # so the fused q|k|v projection is 132 wide; q/k are key-head space (3
    # heads), everything value-shaped is value-head space (6)
    "deltanet_query": (30, 3, True),
    "deltanet_key": (30, 3, True),
    # per chunk: a (k_dim x v_dim) matrix per value head — 10·12 per head
    "deltanet_state": (6 * 10 * 12, 6, True),
    "attention_premix": (128, 8, True),
    # ⚠️ the *value's* shape: this component is derived, and hidden-wide per
    # head rather than head_dim-wide
    "attention_result": (8 * 64, 8, True),
    "attention_output": (64, None, True),
    "block_mid": (64, None, True),
    "mlp_input_norm": (64, None, True),
    "mlp_input": (64, None, True),
    "mlp_activation": (128, None, True),
    "mlp_neuron_output": (128, None, True),
    "router_logits": (32, None, True),
    "router_scores": (4, None, True),
    "expert_idx": (4, None, False),
    # token-major routed interior: top_k (4) slots, on a ranking axis. The two
    # projection halves and the activation are moe_intermediate_size (24) wide
    # per slot; the down-projection's output is hidden (64) wide per slot.
    "expert_gate_proj": (4 * 24, None, True),
    "expert_up_proj": (4 * 24, None, True),
    "expert_activation": (4 * 24, None, True),
    "expert_neuron_output": (4 * 24, None, True),
    "expert_permutation": (4, None, False),
    "expert_output": (4 * 64, None, True),
    "routed_output": (64, None, True),
    "shared_expert_gate_proj": (48, None, True),
    "shared_expert_up_proj": (48, None, True),
    "shared_expert_activation": (48, None, True),
    "shared_expert_output": (64, None, True),
    "shared_expert_gate": (1, None, True),
    "mlp_output": (64, None, True),
    "block_output": (64, None, True),
    "ln_final": (64, None, True),
    "lm_head": (1000, None, True),
}


def test_every_component_in_the_vocabulary_has_a_shape() -> None:
    """A new component must declare its axes, not inherit a default.

    ``(batch, position, hidden)`` is the right answer often enough that a
    default would be silently right most of the time and silently wrong for
    exactly the attention-interior taps."""
    assert set(EXPECTED) == set(COMPONENTS)


@pytest.mark.parametrize("component", sorted(EXPECTED))
def test_the_shape_table(component: str) -> None:
    width, head_space, is_feature_space = EXPECTED[component]
    shape = component_shape(GQA, component)
    assert shape.width == width
    assert shape.head_space == head_space
    assert shape.is_feature_space is is_feature_space


@pytest.mark.parametrize("component", sorted(EXPECTED))
def test_width_and_the_shape_agree(component: str) -> None:
    """``component_width`` is a reading of the shape, so it must refuse exactly
    when the shape says there is nothing to measure."""
    shape = component_shape(GQA, component)
    if shape.is_feature_space:
        assert component_width(GQA, component) == shape.width
    else:
        with pytest.raises(ValidationError):
            component_width(GQA, component)


def test_only_the_pattern_and_the_scores_have_two_position_axes() -> None:
    """Which is what makes every one of the executor's refusals about them
    derivable rather than written by hand — and they are the same shape one step
    apart, which is the whole point of `attention_scores`: everything true of
    the pattern's axes is true of the scores', and nothing true of its
    *normalization* is."""
    without_contract = [
        c for c in COMPONENTS if not component_shape(GQA, c).has_contract_form
    ]
    assert sorted(without_contract) == ["attention_probs", "attention_scores"]
    probs = component_shape(GQA, "attention_probs")
    scores = component_shape(GQA, "attention_scores")
    assert probs.axes == scores.axes


# --------------------------------------------------------------------------- #
# §2.2 — the head bound is the component's, not the model's
# --------------------------------------------------------------------------- #


def test_head_on_a_component_with_no_head_axis_is_refused(env) -> None:
    """🐞 This used to validate and then be silently dropped by the backend —
    the same class as the ``expert`` sub-axis ``_moe_site`` refuses by name."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"]["head"] = 2  # block_output has no head axis
    with pytest.raises(ValidationError, match="has no head axis"):
        _validate_sites(raw, env)


def test_head_is_bounded_by_the_components_own_head_space(env) -> None:
    """And the bound is quoted with the shape it came from."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_premix",
        "layers": [3],
        "head": 8,
    }
    with pytest.raises(ValidationError, match="out of range"):
        _validate_sites(raw, env)


def test_a_head_inside_the_components_head_space_is_accepted(env) -> None:
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_premix",
        "layers": [3],
        "head": 7,
    }
    assert canonicalize(raw, env)["method"]["sites"]["tgt"]["head"] == 7


def test_a_kv_space_head_bound_is_narrower_than_the_query_space_one() -> None:
    """The latent half of the defect, pinned before the interior taps walk into it.

    📐 ``head_space`` is the *component's*, so a KV-space component under GQA
    admits half as many heads as a query-space one. Bounding the first by the
    second does not raise — python slices past the end silently — it yields an
    empty slice: a read of ``(b, n_pos, 0)`` and a write that changes nothing.
    """
    query_space = component_shape(GQA, "attention_premix").head_space
    assert query_space == GQA.num_heads == 8
    assert GQA.num_kv_heads == 4  # what the interior `v`, `k` and `k_pre_rope` use
    # the bound that would have been applied to them, and the one that will be
    assert query_space != GQA.num_kv_heads


# --------------------------------------------------------------------------- #
# §2.1 — `stream` is a field a document can finally use
# --------------------------------------------------------------------------- #


def _doc_with_stream(value: object) -> dict[str, object]:
    raw = base_doc()
    raw["method"]["sites"]["tgt"]["stream"] = value
    return raw


@pytest.mark.parametrize("stream", STREAMS)
def test_a_document_may_name_the_stream_it_means(stream: str) -> None:
    """🐞 Every one of these used to be rejected at parse: the field
    parsed with ``_scalar_int``, while ``model_tree._check_stream`` reads it only
    when it is a *string* (``ModelBundle.stream_at`` returns one). The two
    halves of the feature each rejected what the other accepted, so no document
    could reach the check at all."""
    doc = parse_document(_doc_with_stream(stream))
    assert doc.sites["tgt"].stream == stream


def test_an_integer_stream_is_refused_rather_than_ignored() -> None:
    """The failure the ``expert`` refusal's comment already names: ``stream: 0``
    parsed, was stored, and was then silently skipped by the only code that
    reads the field."""
    with pytest.raises(ParseError, match="sites.tgt.stream"):
        parse_document(_doc_with_stream(0))


def test_a_stream_outside_the_vocabulary_is_refused() -> None:
    with pytest.raises(ParseError, match="sites.tgt.stream"):
        parse_document(_doc_with_stream("attention"))


# --------------------------------------------------------------------------- #
# a ranking axis is not a basis
# --------------------------------------------------------------------------- #


def _moe_doc_with_featurizer(kind: str) -> dict[str, object]:
    raw = base_doc()
    raw["model"]["key"] = "test/moe"
    raw["method"]["sites"]["tgt"] = {"component": "router_scores", "layers": [3]}
    raw["method"]["featurizers"] = {"f": {"kind": kind, "k": 2}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "f"
    raw["method"]["writes"]["patch"]["featurizer"] = "f"
    return in_order(raw)


@pytest.mark.parametrize("kind", ["subspace", "pca"])
def test_a_basis_featurizer_on_a_ranking_axis_is_refused(env, kind: str) -> None:
    """📐 Dimensionally the axis has a width, so this used to be accepted — but
    column *k* is the *k*-th ranked expert, a different expert for different
    tokens. A subspace fitted across positions is fitted across a basis that is
    itself shuffled per position."""
    with pytest.raises(ValidationError, match="per-token ranking"):
        canonicalize(_moe_doc_with_featurizer(kind), env)


def test_a_per_column_featurizer_on_a_ranking_axis_still_works(env) -> None:
    """Only the kinds that *fit a basis* are refused. ``standardize`` computes a
    mean and a scale per column, which is meaningful on a ranking: 'how large is
    the top-ranked expert's score, typically'."""
    raw = _moe_doc_with_featurizer("standardize")
    raw["method"]["featurizers"]["f"] = {"kind": "standardize"}
    raw = in_order(raw)
    assert canonicalize(raw, env)["method"]["featurizers"]["f"]["width"] == 4


def test_a_plain_read_of_a_ranking_axis_is_untouched(env) -> None:
    """The refusal is about fitting a basis, not about reading the tensor."""
    raw = base_doc()
    raw["model"]["key"] = "test/moe"
    raw["method"]["sites"]["tgt"] = {"component": "router_scores", "layers": [3]}
    assert (
        canonicalize(raw, env)["method"]["sites"]["tgt"]["component"] == "router_scores"
    )


def test_the_routed_interior_is_a_ranking_axis_too(env) -> None:
    """The token-major representation puts slot *k* of `expert_activation`
    on the *k*-th ranked expert — `router_scores`' situation exactly, so the
    same basis-fitting refusal applies and the same per-column reads do not."""
    shape = component_shape(GQA, "expert_activation")
    assert shape.ranking is True
    raw = _moe_doc_with_featurizer("subspace")
    raw["method"]["sites"]["tgt"] = {"component": "expert_activation", "layers": [3]}
    with pytest.raises(ValidationError, match="per-token ranking"):
        canonicalize(in_order(raw), env)


# --------------------------------------------------------------------------- #
# the two defects the width check found on its own
# --------------------------------------------------------------------------- #


def test_the_moe_branch_outputs_are_hidden_wide_and_flat() -> None:
    """🐞 The width table said hidden-wide and the layout table said flat, in
    two different files, and the tensors are both — but nothing compared a
    declared width against a real tensor, so a shape that was only half right
    would have gone on being half right."""
    for component in ("routed_output", "shared_expert_output"):
        shape = component_shape(GQA, component)
        assert shape.width == GQA.hidden_size
        assert shape.flat_batch
        assert shape.native_rank == 2


def test_gpt2s_mlp_width_comes_from_n_inner() -> None:
    """🐞 ``GPT2Config`` spells the MLP's inner width ``n_inner`` and the block
    computes ``n_inner if n_inner is not None else 4 * hidden``
    (transformers ``models/gpt2/modeling_gpt2.py:250``). Reading
    ``intermediate_size`` instead reported 37 for the 128-wide MLP of
    ``hf-internal-testing/tiny-random-gpt2``, whose config carries that key even
    though nothing in the model reads it."""
    assert get_model_info("gpt2").intermediate_size == 4 * 768


# --------------------------------------------------------------------------- #
# the rename, and the rank table it renumbered
# --------------------------------------------------------------------------- #


def test_the_retired_spelling_still_loads() -> None:
    """The one-release alias: a document written against the old vocabulary is
    not a hard error."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "attention_value", "layers": [3]}
    assert parse_document(raw).sites["tgt"].component == "attention_premix"


def test_the_alias_folds_at_parse_so_nothing_downstream_sees_two_names(
    env,
) -> None:
    """Both spellings canonicalize identically, and therefore digest
    identically — the alias is a courtesy at the door, not a second vocabulary
    the tables have to know about."""
    old, new = base_doc(), base_doc()
    old["method"]["sites"]["tgt"] = {"component": "attention_value", "layers": [3]}
    new["method"]["sites"]["tgt"] = {"component": "attention_premix", "layers": [3]}
    assert canonicalize(old, env) == canonicalize(new, env)


def test_the_retired_name_is_not_reused_by_anything() -> None:
    """The rule that makes the alias safe.

    An alias that *redirects* is fine; one that *rebinds* would let a document
    written against the old vocabulary load and silently mean a different
    tensor. The real value vectors have their own name for
    exactly this reason (nnterp#51 is the same mistake, made after the fact).
    """
    for retired in DEPRECATED_COMPONENTS:
        assert retired not in COMPONENTS


def test_every_component_has_a_rank() -> None:
    """``COMPONENT_RANK`` drives group elision, and a component missing from it
    would silently sort as the deepest tap."""
    assert set(COMPONENT_RANK) == set(COMPONENTS)


def test_the_rank_table_is_written_in_forward_order() -> None:
    """It is read as a story about a block, so its source order and its values
    must agree — a table that sorts differently than it reads is one nobody
    checks against the architecture."""
    ranks = list(COMPONENT_RANK.values())
    assert ranks == sorted(ranks)


def test_every_slot_plan_py_reserved_is_claimed() -> None:
    """Renumber once, then never again.

    The renumbering reserved nine numbers in ``plan.py``, and all nine are
    claimed — each new component an insertion rather than a re-pin, which is
    the whole return on renumbering the band in one go. The gaps that remain
    are for the MoE and DeltaNet interiors.
    """
    # the band still reads in forward order, with room left between its members
    band = [
        COMPONENT_RANK[c]
        for c in COMPONENT_RANK
        if COMPONENT_RANK["attention_input_norm"]
        <= COMPONENT_RANK[c]
        <= COMPONENT_RANK["attention_output"]
    ]
    assert band == sorted(band)
    assert min(band) == 150 and max(band) == 400
    # rounds 2.2, 2.3 and 2.4 claimed nine, at the numbers plan.py reserved
    assert COMPONENT_RANK["attention_query_pre_rope"] == 160
    assert COMPONENT_RANK["attention_key_pre_rope"] == 170
    assert COMPONENT_RANK["attention_value_states"] == 180
    assert COMPONENT_RANK["attention_gate"] == 190
    assert COMPONENT_RANK["attention_query"] == 200
    assert COMPONENT_RANK["attention_key"] == 210
    assert COMPONENT_RANK["attention_scores"] == 220
    assert COMPONENT_RANK["attention_z"] == 240
    assert COMPONENT_RANK["attention_result"] == 350


def test_a_kv_space_head_is_refused_at_load_not_just_at_the_tap(env) -> None:
    """The §2.2 defect in the form the interior taps walk into.

    ``attention_value_states`` is KV-head space, so on this GQA model head 5 is
    valid in query space (8 heads) and not here (4). The bound has to come from
    the *component*: python does not raise on an over-wide slice, it returns an
    empty one, so the read would have saved ``(b, n_pos, 0)`` and the write
    would have changed nothing.
    """
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_value_states",
        "layers": [3],
        "head": 5,
    }
    with pytest.raises(ValidationError, match="4 heads"):
        _validate_sites(raw, env)


def test_the_query_space_twin_accepts_the_same_head(env) -> None:
    """Which is what makes the refusal above about the component and not about
    the number."""
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_query_pre_rope",
        "layers": [3],
        "head": 5,
    }
    assert canonicalize(raw, env)["method"]["sites"]["tgt"]["head"] == 5


# --------------------------------------------------------------------------- #
# the Qwen3.6-35B-A3B entry, and the hybrid layer pattern it declares
# --------------------------------------------------------------------------- #

A3B = "Qwen/Qwen3.6-35B-A3B"
#: One layer of each stream on the A3B's 3+1 schedule: layer 3 is the first
#: gated full-attention layer, layer 0 a Gated DeltaNet one.
FULL_LAYER, DELTA_LAYER = 3, 0


def _a3b_doc(component: str, layer: int, **site: object) -> dict[str, Any]:
    raw = base_doc()
    raw["model"]["key"] = A3B
    raw["method"]["sites"]["tgt"] = {"component": component, "layers": [layer], **site}
    return raw


def _a3b_doc_with_width(component: str, layer: int) -> dict[str, Any]:
    """The site under a ``standardize`` featurizer, whose derived ``width`` is
    the feature width the canonicalizer assigned to (model, site) — the
    per-column kind, so it attaches to ranking axes too."""
    raw = _a3b_doc(component, layer)
    raw["method"]["featurizers"] = {"f": {"kind": "standardize"}}
    raw["method"]["reads"]["v_cf"]["featurizer"] = "f"
    raw["method"]["writes"]["patch"]["featurizer"] = "f"
    return in_order(raw)


def test_the_a3b_entry_is_the_checkpoints_config() -> None:
    """📐 Every value is the checkpoint's config.json (revision 995ad96e) as
    ``model_info_from_hf_config`` reads it, cross-checked against
    ``docs/qwen36-35b-a3b-architecture.html``. Equality with the adapter's
    reading is the property that matters: the run re-registers the key from
    the loaded config, so a static value that differed would make ``validate``
    and ``run`` size the same document differently."""
    info = get_model_info(A3B)
    assert (info.hidden_size, info.num_layers, info.vocab_size) == (2048, 40, 248320)
    assert (info.num_heads, info.num_kv_heads, info.head_dim) == (16, 2, 256)
    assert (info.num_experts, info.num_experts_per_tok) == (256, 8)
    assert info.moe_intermediate_size == info.shared_expert_intermediate_size == 512
    assert (info.linear_num_key_heads, info.linear_key_head_dim) == (16, 128)
    assert (info.linear_num_value_heads, info.linear_value_head_dim) == (32, 128)
    assert info.native_dtype == "bf16"
    # the text config has no dense MLP width at all (every block is MoE), and
    # the adapter now reads that as None rather than falling back to 4·hidden
    # — so `mlp_activation` refuses at load exactly as the run refuses it
    assert info.intermediate_size is None
    # the *text* config's model_type, which is what the adapter reads (the
    # wrapper config says "qwen3_5_moe"; the tiny fixture loads with this one)
    assert info.family == "qwen3_5_moe_text"


#: The dense Qwen3.5 tower of ``demos/papers/addition_heads_dbm.md``.
QWEN35_2B = "Qwen/Qwen3.5-2B"


def test_the_qwen35_2b_entry_is_a_hybrid_with_six_attention_layers() -> None:
    """Layers 3, 7, ..., 23 are full attention, so ``attention_premix`` (8
    query heads of 256) exists at six layers and is refused at the other 18,
    offline, before any weights load."""
    info = get_model_info(QWEN35_2B)
    assert info.layer_types is not None
    full = [i for i, kind in enumerate(info.layer_types) if kind == "full_attention"]
    assert full == [3, 7, 11, 15, 19, 23]
    assert (info.num_heads, info.num_kv_heads, info.head_dim) == (8, 2, 256)
    assert (info.family, info.intermediate_size) == ("qwen3_5_text", 6144)
    assert component_width(info, "attention_premix") == 8 * 256


def test_the_qwen35_2b_entry_matches_its_cached_hf_config() -> None:
    """📐 The static row equals the adapter's reading of the checkpoint's
    config.json, layer pattern and parallel plan included. Offline: skipped,
    never fetched, when the config is not in the local HF cache."""
    import os

    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig

    folder = "models--" + QWEN35_2B.replace("/", "--")
    if not os.path.isdir(os.path.join(HF_HUB_CACHE, folder)):
        pytest.skip(f"{QWEN35_2B} is not in the local HF cache")
    config = AutoConfig.from_pretrained(QWEN35_2B, local_files_only=True)
    read = model_info_from_hf_config(QWEN35_2B, config)
    assert dataclasses.replace(read, parallel_plan=None) == dataclasses.replace(
        get_model_info(QWEN35_2B), parallel_plan=None
    )
    assert read.parallel_plan == get_model_info(QWEN35_2B).parallel_plan


#: The two MIB circuit-track checkpoints registered for the Figure-1
#: replication (``registry.py``, beside the gemma-2-2b-it row).
MIB_ENTRIES = ("Qwen/Qwen2.5-0.5B", "google/gemma-2-2b")


@pytest.mark.parametrize("key", MIB_ENTRIES)
def test_a_mib_entry_matches_its_cached_hf_config(key: str) -> None:
    """📐 Each static row equals what ``model_info_from_hf_config`` reads from
    the checkpoint's own config.json — the property that keeps ``validate`` and
    ``run`` (which re-registers from the loaded config) sizing one document
    identically. Offline: skipped, never fetched, when the config is not in
    the local HF cache."""
    import os

    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig

    folder = "models--" + key.replace("/", "--")
    if not os.path.isdir(os.path.join(HF_HUB_CACHE, folder)):
        pytest.skip(f"{key} is not in the local HF cache")
    config = AutoConfig.from_pretrained(key, local_files_only=True)
    read = model_info_from_hf_config(key, config)
    static = get_model_info(key)
    for field in (
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "family",
    ):
        assert getattr(read, field) == getattr(static, field), field


#: The model that fits no single card (``docs/model_parallelism.md`` §10.6
#: "The large model"), registered beside the 8B so the CLI sizes a document
#: naming it offline and the memory pre-flight has its entry.
LARGE = "meta-llama/Llama-3.1-70B"


def test_the_llama_70b_entry_is_the_checkpoints_config() -> None:
    """📐 Every value is the checkpoint's config.json (revision 349b2ddb) as
    ``model_info_from_hf_config`` reads it; the header census
    ``tests/golden/parallel_headers_llama70b.json`` carries the shapes those
    values imply (``tests/protocol/test_parallel_memory_large.py`` holds the
    two to each other). The plan is the 8B's — one family, one plan."""
    info = get_model_info(LARGE)
    assert (info.hidden_size, info.num_layers, info.vocab_size) == (8192, 80, 128256)
    assert (info.num_heads, info.num_kv_heads, info.head_dim) == (64, 8, 128)
    assert info.intermediate_size == 28672
    assert info.native_dtype == "bf16" and info.family == "llama"
    assert info.num_experts is None and info.layer_types is None
    assert info.parallel_plan is not None
    assert info.parallel_plan == get_model_info("meta-llama/Llama-3.1-8B").parallel_plan


def test_the_llama_70b_entry_matches_its_cached_hf_config() -> None:
    """The same property as the MIB rows': the static entry equals the
    adapter's reading of the cached config.json. Offline: skipped, never
    fetched, when the checkpoint's config is not in the local HF cache
    (``HF_HUB_CACHE``)."""
    import os

    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoConfig

    folder = "models--" + LARGE.replace("/", "--")
    if not os.path.isdir(os.path.join(HF_HUB_CACHE, folder)):
        pytest.skip(f"{LARGE} is not in the local HF cache")
    config = AutoConfig.from_pretrained(LARGE, local_files_only=True)
    read = model_info_from_hf_config(LARGE, config)
    static = get_model_info(LARGE)
    for field in (
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "family",
    ):
        assert getattr(read, field) == getattr(static, field), field
    assert read.parallel_plan == static.parallel_plan


def test_the_mib_entries_have_the_papers_node_counts() -> None:
    """MIB's node set is ``L·H + L + 1`` (heads, MLP blocks, the embedding):
    the counts the Figure-1 replication sizes its `top_k` grid by."""
    qwen, gemma = (get_model_info(k) for k in MIB_ENTRIES)
    assert qwen.num_layers * qwen.num_heads + qwen.num_layers + 1 == 24 * 14 + 25
    assert gemma.num_layers * gemma.num_heads + gemma.num_layers + 1 == 26 * 8 + 27
    assert (qwen.family, gemma.family) == ("qwen2", "gemma2")


def test_the_a3b_layer_pattern_is_three_deltanet_then_one_attention() -> None:
    info = get_model_info(A3B)
    assert info.layer_types is not None
    assert info.layer_types == (("linear_attention",) * 3 + ("full_attention",)) * 10
    full = [i for i, kind in enumerate(info.layer_types) if kind == "full_attention"]
    assert full == list(range(3, 40, 4))


@pytest.mark.parametrize(
    ("component", "layer", "width"),
    [
        # the attention-like head unit: 16 query heads × 256 at a full-attention
        # layer, 32 value heads × 128 at a DeltaNet one — the same 4096, which
        # is why the head map and not the width is what tells them apart
        ("attention_premix", FULL_LAYER, 16 * 256),
        ("delta_premix", DELTA_LAYER, 32 * 128),
        # the MoE block is in every layer, so its widths hold on both streams
        ("expert_activation", FULL_LAYER, 8 * 512),
        ("expert_activation", DELTA_LAYER, 8 * 512),
        ("shared_expert_activation", FULL_LAYER, 512),
        ("shared_expert_activation", DELTA_LAYER, 512),
        ("router_scores", FULL_LAYER, 8),
        ("router_scores", DELTA_LAYER, 8),
    ],
)
def test_a3b_widths_at_a_site(env, component: str, layer: int, width: int) -> None:
    canonical = canonicalize(_a3b_doc_with_width(component, layer), env)
    assert canonical["method"]["featurizers"]["f"]["width"] == width


@pytest.mark.parametrize("layer", [FULL_LAYER, DELTA_LAYER])
def test_a3b_expert_idx_is_an_eight_wide_routing_table(env, layer: int) -> None:
    """Integral, so no featurizer and no ``component_width`` — but eight ids
    per token, readable and writable at every layer."""
    shape = component_shape(get_model_info(A3B), "expert_idx")
    assert (shape.width, shape.integral, shape.is_feature_space) == (8, True, False)
    with pytest.raises(ValidationError):
        component_width(get_model_info(A3B), "expert_idx")
    canonical = canonicalize(_a3b_doc("expert_idx", layer), env)
    assert canonical["method"]["sites"]["tgt"]["component"] == "expert_idx"


def test_a3b_head_spaces_differ_between_the_two_mixers(env) -> None:
    """Head 31 is a value head at a DeltaNet layer and nothing at a
    full-attention one: the bound is the component's, and the two premixes
    are different components with the same width."""
    assert canonicalize(_a3b_doc("attention_premix", FULL_LAYER, head=15), env)
    assert canonicalize(_a3b_doc("delta_premix", DELTA_LAYER, head=31), env)
    with pytest.raises(ValidationError, match="16 heads"):
        _validate_sites(_a3b_doc("attention_premix", FULL_LAYER, head=16), env)


def test_a_full_attention_component_at_a_deltanet_layer_is_refused_at_load(
    env,
) -> None:
    """The refusal the entry exists to make offline. Before it, the pure verbs
    accepted this document and the run refused it against the loaded modules;
    now ``validate`` names the layers that do carry the mixer."""
    with pytest.raises(ValidationError, match="carries 'linear_attention'") as err:
        _validate_sites(_a3b_doc("attention_premix", DELTA_LAYER), env)
    assert "sites.tgt.component" in str(err.value)
    assert "[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]" in str(err.value)


def test_a_deltanet_component_at_a_full_attention_layer_is_refused_at_load(
    env,
) -> None:
    with pytest.raises(ValidationError, match="carries 'full_attention'"):
        _validate_sites(_a3b_doc("delta_premix", FULL_LAYER), env)


def test_a_declared_stream_is_checked_against_the_layer(env) -> None:
    """On a stream-agnostic component the declaration is the only thing to
    check, and it is checked; on a stream-bound one it is checked first, so the
    message names the per-layer fact rather than the component's need — the
    order the run-time check uses."""
    raw = _a3b_doc("block_output", DELTA_LAYER, stream="linear_attention")
    assert (
        canonicalize(raw, env)["method"]["sites"]["tgt"]["stream"] == "linear_attention"
    )
    with pytest.raises(ValidationError, match="per-layer fact") as err:
        _validate_sites(
            _a3b_doc("block_output", DELTA_LAYER, stream="full_attention"), env
        )
    assert "sites.tgt.stream" in str(err.value)
    with pytest.raises(ValidationError, match="per-layer fact"):
        _validate_sites(
            _a3b_doc("delta_premix", DELTA_LAYER, stream="full_attention"), env
        )


def test_a_model_with_no_declared_pattern_defers_the_stream_check_to_the_run(
    env,
) -> None:
    """``layer_types`` is optional. The registered dense entries and the GQA
    table above declare none, so a DeltaNet component on them canonicalizes and
    is refused where it always was — by the site resolver, against the module
    the layer actually carries."""
    assert GQA.layer_types is None
    raw = base_doc()
    raw["model"]["key"] = GQA.key
    raw["method"]["sites"]["tgt"] = {"component": "delta_premix", "layers": [3]}
    assert (
        canonicalize(raw, env)["method"]["sites"]["tgt"]["component"] == "delta_premix"
    )


def test_a_document_naming_the_a3b_loads_and_digests_offline(env) -> None:
    """The contract, end to end: the full loader pipeline on a minimal tree
    against the committed fixture table, with the registry as the only source
    of model facts. Two loads digest identically."""
    loaded = compile_protocol(_a3b_doc("attention_premix", FULL_LAYER, head=4), env=env)
    assert loaded.canonical["model"]["key"] == A3B
    assert loaded.canonical["method"]["sites"]["tgt"]["head"] == 4
    assert re.fullmatch(r"[0-9a-f]{64}", loaded.digests.document)
    again = compile_protocol(_a3b_doc("attention_premix", FULL_LAYER, head=4), env=env)
    assert again.digests.document == loaded.digests.document


class _StubDenseConfig:
    """Just the attributes [`model_info_from_hf_config`][causalab.protocol.registry.models.model_info_from_hf_config] reads, on a
    two-layer tower that declares no layer pattern."""

    num_attention_heads = 8
    hidden_size = 64
    num_hidden_layers = 2
    num_key_value_heads = 8
    head_dim = 8
    intermediate_size = 128
    vocab_size = 512
    dtype = "bfloat16"


class _StubHybridConfig(_StubDenseConfig):
    layer_types = ["linear_attention", "full_attention"]


class _StubSlidingConfig(_StubDenseConfig):
    """The Gemma2/Gemma3 pattern: HF's ``layer_types`` names an attention
    *variant*, not a mixer."""

    layer_types = ["sliding_attention", "full_attention"]


class _StubUnmappedConfig(_StubDenseConfig):
    """A family whose ``layer_types`` vocabulary the adapter has not met."""

    layer_types = ["mamba", "full_attention"]


def test_the_adapter_reads_layer_types_when_the_config_has_them() -> None:
    info = model_info_from_hf_config("test/hybrid", _StubHybridConfig())
    assert info.layer_types == ("linear_attention", "full_attention")


def test_a_sliding_window_layer_is_a_full_attention_layer() -> None:
    """🐞 The adapter copied HF's spellings straight into the
    ``STREAMS``-validated field, so loading ``google/gemma-2-2b-it`` — a
    built-in entry and a golden-protocol model — raised a bare ValueError on
    its ``sliding_attention`` layers. A sliding window is still a ``self_attn``
    child computing an attention matrix, which is what the run-time probe
    answers for it, so the two halves of the stream check must agree."""
    info = model_info_from_hf_config("test/gemma", _StubSlidingConfig())
    assert info.layer_types == ("full_attention", "full_attention")


def test_an_unmapped_layer_kind_defers_the_pattern_to_the_run() -> None:
    """HF's vocabulary is wide (``mamba``, ``chunked_attention``, several
    sparse-attention kinds). One the adapter cannot place leaves the whole
    pattern unset — the run-time check is the documented fallback — rather
    than raising on a model that loads fine, or guessing a stream for the
    layers it does recognise."""
    info = model_info_from_hf_config("test/unmapped", _StubUnmappedConfig())
    assert info.layer_types is None


def test_the_hf_spelling_table_lands_inside_the_stream_vocabulary() -> None:
    """Census guard for ``_HF_LAYER_STREAMS``: every HF spelling the adapter
    places lands on a protocol stream, and ``__post_init__`` therefore never
    sees a mapped pattern it would refuse. A stream added to ``STREAMS`` with
    no HF spelling is fine (the run-time probe still answers for it); a
    spelling mapped onto a name outside ``STREAMS`` is not."""
    from causalab.protocol.registry import _HF_LAYER_STREAMS

    assert set(_HF_LAYER_STREAMS.values()) <= set(STREAMS)
    assert set(_HF_LAYER_STREAMS) >= {"full_attention", "linear_attention"}


def test_the_adapter_leaves_layer_types_unset_when_the_config_has_none() -> None:
    """The pinned transformers' llama and gpt2 configs carry no such field; an
    invented all-full-attention pattern would be right for them and wrong for
    the next family that declares its streams some other way."""
    assert (
        model_info_from_hf_config("test/dense", _StubDenseConfig()).layer_types is None
    )


def test_a_layer_pattern_must_match_the_depth_and_the_stream_vocabulary() -> None:
    """A static entry with a pattern of the wrong length would index past the
    tower or leave layers unchecked; a misspelt stream would never equal the
    declared one. Both are programming errors in the entry, caught when it is
    built."""
    with pytest.raises(ValueError, match="2 entries for a 4-layer"):
        dataclasses.replace(GQA, layer_types=("full_attention", "linear_attention"))
    with pytest.raises(ValueError, match="attention"):
        dataclasses.replace(GQA, layer_types=("attention",) * 4)


def test_the_stream_table_is_the_vocabularys_prefix_rule() -> None:
    """``COMPONENT_STREAMS`` — the ``stream`` cell of the capability rows — is
    the one table both the canonicalizer and the engines' site resolver read.
    Its rule is stated in the spec (§2.4) as a prefix rule with exactly two
    exceptions; the rows declare each component's stream, and this spells the
    rule out independently so a row cannot declare a stream the spec's rule
    does not give it, and a third exception cannot slip in without being
    named here and in the spec."""
    for component in COMPONENTS:
        if component.startswith("attention_") and component not in (
            "attention_input_norm",
            "attention_output",
        ):
            expected: str | None = "full_attention"
        elif component.startswith(("delta_", "deltanet_")):
            expected = "linear_attention"
        else:
            expected = None
        assert COMPONENT_STREAMS.get(component) == expected, component
    assert set(COMPONENT_STREAMS) <= set(COMPONENTS)
    assert set(COMPONENT_STREAMS.values()) <= set(STREAMS)


#: GPT-J 6B, the function-vector replication's model. Its config.json at
#: revision ``float16`` (b71ae8bc86cac13154e03e92b5855203086b722e), spelled
#: here so the adapter is held to the checkpoint's own values offline.
GPTJ_6B = "EleutherAI/gpt-j-6b"
GPTJ_6B_FLOAT16_CONFIG = {
    "n_embd": 4096,
    "n_layer": 28,
    "n_head": 16,
    "n_positions": 2048,
    "rotary_dim": 64,
    "vocab_size": 50400,
    "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-05,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
}
#: The static row as declared, read at collection: a later load of the
#: checkpoint in the same process re-registers the key from the loaded config
#: (``load_model``), which adds the load-time ``experts_implementation`` knob
#: and, for the fp32 ``main`` revision, ``native_dtype`` fp32.
GPTJ_6B_STATIC = get_model_info(GPTJ_6B)


def test_the_gpt_j_6b_entry_is_what_the_adapter_reads_off_its_config() -> None:
    """📐 ``GPTJConfig`` spells the widths ``n_embd`` / ``n_head`` /
    ``n_inner`` and carries no ``head_dim`` and no KV-head count: the adapter
    derives head_dim 4096 / 16 = 256, falls back to one KV head per query
    head, and reads the absent ``n_inner`` (the class default ``None``) as
    ``4 · n_embd``, the width ``GPTJBlock`` builds. Every field the row
    declares equals that reading, so ``validate`` and ``run`` (which
    re-registers from the loaded config) size one document identically.
    Every field, including the empty parallel plan (transformers ships no
    ``base_model_tp_plan`` for GPT-J)."""
    from transformers import GPTJConfig

    read = model_info_from_hf_config(GPTJ_6B, GPTJConfig(**GPTJ_6B_FLOAT16_CONFIG))
    assert read == GPTJ_6B_STATIC
    assert (read.head_dim, read.num_kv_heads, read.intermediate_size) == (
        256,
        16,
        16384,
    )


def test_the_gpt_j_6b_entry_matches_its_cached_hf_config() -> None:
    """The same property against the cached config.json itself, at the
    revision the runs load. Offline: skipped, never fetched, when that
    revision's config.json is not in the local HF cache. The check is on the file
    the test reads, not the repo folder: a tokenizer download creates the
    folder without it."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoConfig

    # a str is a cached path; None and _CACHED_NO_EXIST both mean not cached
    # (the docstring of huggingface_hub.file_download.try_to_load_from_cache)
    cached = try_to_load_from_cache(GPTJ_6B, "config.json", revision="float16")
    if not isinstance(cached, str):
        pytest.skip(f"{GPTJ_6B} float16 config.json is not in the local HF cache")
    config = AutoConfig.from_pretrained(
        GPTJ_6B, revision="float16", local_files_only=True
    )
    read = model_info_from_hf_config(GPTJ_6B, config)
    assert read == GPTJ_6B_STATIC


def _gptj_hub_cache(root: Any, files: dict[str, str]) -> None:
    """A hub cache under ``root`` whose GPT-J ``float16`` snapshot holds only
    ``files``, laid out as ``huggingface_hub`` writes it (refs name the
    commit, the snapshot folder holds the files)."""
    commit = "0" * 40
    repo = root / ("models--" + GPTJ_6B.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "float16").write_text(commit)
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name, text in files.items():
        (snapshot / name).write_text(text)


def test_the_gpt_j_cache_check_skips_when_only_the_tokenizer_is_cached(
    tmp_path, monkeypatch
) -> None:
    """A cache that holds the repo folder but not the ``float16`` config.json
    (a tokenizer download leaves exactly that, as on a CI runner) skips the
    config check instead of failing it with the not-cached ``OSError``."""
    import huggingface_hub.constants

    _gptj_hub_cache(tmp_path, {"tokenizer.json": "{}"})
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="float16"):
        test_the_gpt_j_6b_entry_matches_its_cached_hf_config()


def test_the_gpt_j_cache_check_runs_when_the_float16_config_is_cached(
    tmp_path, monkeypatch
) -> None:
    """The control: with the ``float16`` config.json in the cache the check
    runs, and the declared row equals its reading."""
    import json

    import huggingface_hub.constants

    config = {"model_type": "gptj", **GPTJ_6B_FLOAT16_CONFIG}
    _gptj_hub_cache(tmp_path, {"config.json": json.dumps(config)})
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    test_the_gpt_j_6b_entry_matches_its_cached_hf_config()
