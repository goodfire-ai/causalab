"""The capability registry at load: the two ``entity`` mismatches, the write
policy, the load-decidable predicates, and the all-MoE tower's missing dense
MLP.

Every refusal here has its valid-passes twin, and
the valid twins pin their **digest** to the value the base computed for the
same document — the proof that consolidating the tables changed no canonical
byte (reading (a): zero pins move).
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causalab.protocol.schema.explicit import canonicalize, digest
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.registry import (
    CAPABILITIES,
    INTERIOR_ROWS,
    OVERRIDE_KEYS,
    PACKINGS,
    Capability,
    ModelInfo,
    component_shape,
    component_width,
    expert_axis_refusal,
    families_in_table,
    family_in_table,
    get_model_info,
    model_info_from_hf_config,
    native_shape,
    predicate_holds,
    unavailable_at_load,
    write_policy_refusal,
)
from causalab.protocol.registry.shapes import bs_flat_heads
from causalab.protocol.schema import MECHANISMS, parse_document
from causalab.protocol.rules.document import validate_document

from tests._helpers.refusal_snapshot import A3B, MOE
from tests.protocol._docs import base_doc, in_order
from tests.protocol._env import FIXTURES, build_env


pytestmark = pytest.mark.unit

ENV = build_env(FIXTURES / "artifacts")
DEMO_HEAD_SCAN = (
    Path(__file__).resolve().parents[2]
    / "demos"
    / "onboarding_tutorial"
    / "protocols"
    / "mcqa_head_scan.json"
)

#: Digests of the three valid documents below, pinned so that `validate`
#: refusing more at load never moves a byte of what it accepts. First computed on
#: the base (before the registry); re-pinned once for the protocol v2
#: canonical form (the four groups, §1), which moved every digest by design. Re-pinned again
#: for protocol v3: `layers`, a band, moved every layered site's bytes.
#: And for protocol v4 (reads-first): the reads-on-models block moved every
#: document's bytes — re-pinned once more when it landed on the retired
#: ``token_form``, which these literals no longer spell.
BASE_DIGEST_EXPERT_FACE = (
    "a84edd907544bbc1bb88321b6d42e5a1406ec2456957b4ebfb5436f92e17d013"
)
BASE_DIGEST_ROUTED_OUTPUT = (
    "4b291579c2ee0101260f116e9f502a91b8bef0facb6050756621d82ab0c48906"
)
BASE_DIGEST_SCORES_DELTA = (
    "984158c0f5e6985f11ffd1912b052e160f96b71a85595f78837970dd892870b2"
)
#: Digests for the interior q/k/v on the `gpt2` entry — documents the base
#: before the per-family tap table accepted at load and the run then refused
#: (``fixtures/refusal_snapshot.json`` entry 23); the
#: per-family tap table accepts them at both and moves no byte. First computed
#: on that base with its own code (a `git archive` export); re-pinned once under
#: protocol v2, whose four-group canonical form moved every digest —
#: the one kind of change spec §7 lets move a pin; and again under protocol v3
#: (`layer` → `layers`, a band), the same kind of change.
BASE_DIGEST_GPT2_INTERIOR = {
    "attention_query_pre_rope": (
        "0880fed905697b6ea028ae25d350028ddda68a8a651b549fa8eb58ff78fafb21"
    ),
    "attention_key_pre_rope": (
        "d583532a035b5a8026db52f46a5bf3a403fc7307f6b945bdbe2ec39ac70927bc"
    ),
    "attention_value_states": (
        "c108d3b42f22e2ca2c00595f166f3fec58b5dd92bfba80763bfa9fd40c8652a1"
    ),
}
BASE_DIGEST_GPT2_QUERY_HEAD_2 = (
    "9ca6abe2cb4ce98fc7d21906287509201266a2bb3bb784a7e13a01ee029490aa"
)


def _validate(raw: dict[str, Any]) -> None:
    validate_document(parse_document(in_order(raw)))


def _refusal(raw: dict[str, Any]) -> ValidationError:
    with pytest.raises(ValidationError) as err:
        _validate(raw)
    return err.value


def _digest(raw: dict[str, Any]) -> str:
    _validate(raw)
    return digest(canonicalize(in_order(raw), ENV))


# --------------------------------------------------------------------------- #
# entity mismatch 1 — `expert` looks generic; only the routed interior has it
# --------------------------------------------------------------------------- #


def test_expert_on_a_component_without_the_axis_is_refused_at_load() -> None:
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {
        "component": "router_scores",
        "layers": [3],
        "expert": 3,
    }
    err = _refusal(raw)
    assert err.rule == 4 and err.path == "sites.tgt.expert"
    assert err.reason == "component_unavailable"
    assert "no per-expert axis" in str(err)
    assert "'expert_activation', 'expert_neuron_output' and" in str(err)


def test_expert_on_a_non_moe_component_is_refused_at_load_too() -> None:
    """The run-time twin only ever fired inside the MoE resolver; a generic
    field on `block_output` was silently dropped. Not any more."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {
        "component": "block_output",
        "layers": [3],
        "expert": 0,
    }
    assert _refusal(raw).path == "sites.tgt.expert"


def test_expert_on_the_routed_interior_loads_with_the_bases_digest() -> None:
    """Valid work still passes, byte for byte: the site's canonical form and
    the document digest equal what the base computed."""
    raw = base_doc()
    raw["model"]["key"] = MOE.key
    raw["method"]["sites"]["tgt"] = {
        "component": "expert_activation",
        "layers": [3],
        "expert": 3,
    }
    _validate(raw)
    canonical = canonicalize(in_order(raw), ENV)
    assert canonical["method"]["sites"]["tgt"] == {
        "component": "expert_activation",
        "layers": [3],
        "expert": 3,
    }
    assert digest(canonical) == BASE_DIGEST_EXPERT_FACE


def test_the_expert_axis_refusal_is_one_text() -> None:
    """The validator's and the resolver's refusal are the same function — the
    snapshot pins the resolver's text, so this pins the function to it."""
    faces = [c for c, row in CAPABILITIES.items() if row.expert_selection]
    assert faces == [
        "expert_gate_proj",
        "expert_up_proj",
        "expert_activation",
        "expert_neuron_output",
        "expert_output",
    ]
    assert expert_axis_refusal("router_scores", 3) == (
        "site names expert 3 on component 'router_scores', which has no "
        "per-expert axis: the router's axes are all-experts or top-k, and the "
        "shared expert is not one of the routed experts. The per-expert interior "
        "components are 'expert_gate_proj', 'expert_up_proj', "
        "'expert_activation', 'expert_neuron_output' and 'expert_output'."
    )


# --------------------------------------------------------------------------- #
# entity mismatch 2 — `attention_probs` appears writable; only `swap` is
# --------------------------------------------------------------------------- #


def _pattern_write(do: dict[str, Any]) -> dict[str, Any]:
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "attention_probs", "layers": [3]}
    raw["method"]["reads"]["v_cf"]["pos"] = "all"
    raw["method"]["writes"]["patch"] = {"site": "tgt", "pos": "all", "do": do}
    return raw


def test_a_non_swap_write_to_the_pattern_is_refused_at_load() -> None:
    err = _refusal(_pattern_write({"add_scaled": {"op": "v_cf", "alpha": 1.0}}))
    assert err.rule == 4 and err.path == "writes.patch.do"
    assert err.reason == "unsupported_mechanism"
    assert "sum to 1" in str(err) and "attention_scores" in str(err)
    assert "'add_scaled' to 'attention_probs'" in str(err)


def test_a_swap_on_the_pattern_still_loads() -> None:
    """The onboarding demo's head scan is a `swap` on `attention_probs`; it
    loads, and its four quoted point digests are held by
    tests/demos/test_demos.py::test_quoted_digests_are_current."""
    _validate(_pattern_write({"swap": "v_cf"}))
    demo = json.loads(DEMO_HEAD_SCAN.read_text())
    site = (
        demo["method"]["sites"]["pattern"]
        if "pattern" in demo["method"]["sites"]
        else None
    )
    assert any(
        s["component"] == "attention_probs" for s in demo["method"]["sites"].values()
    )
    assert all("swap" in w["do"] for w in demo["method"]["writes"].values())
    del site


def test_arithmetic_on_the_scores_still_loads_with_the_bases_digest() -> None:
    """Anti-vacuity: the same mechanism one tap earlier is legal, and the
    document's digest is the base's."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "attention_scores", "layers": [3]}
    raw["method"]["reads"]["v_cf"]["pos"] = "all"
    raw["method"]["writes"]["patch"] = {
        "site": "tgt",
        "pos": "all",
        "do": {"add_scaled": {"op": "v_cf", "alpha": 1.0}},
    }
    assert _digest(raw) == BASE_DIGEST_SCORES_DELTA


# --------------------------------------------------------------------------- #
# the write policy is one function, applied at load and at the plan
# --------------------------------------------------------------------------- #


def test_a_write_to_a_read_only_component_is_refused_at_load() -> None:
    raw = base_doc()
    raw["model"]["key"] = MOE.key
    raw["method"]["sites"]["tgt"] = {"component": "router_logits", "layers": [3]}
    err = _refusal(raw)
    assert err.path == "writes.patch.do" and err.reason == "unsupported_mechanism"
    assert "no write may change" in str(err)
    assert "Write 'router_scores' to reweight" in str(err)


def test_a_swap_of_the_routing_table_still_loads() -> None:
    raw = base_doc()
    raw["model"]["key"] = MOE.key
    raw["method"]["sites"]["tgt"] = {"component": "expert_idx", "layers": [3]}
    _validate(raw)


@pytest.mark.parametrize("component", sorted(CAPABILITIES))
def test_write_policy_refusal_follows_the_row(component: str) -> None:
    row = CAPABILITIES[component]
    for mechanism in MECHANISMS:
        refusal = write_policy_refusal("w", component, mechanism)
        if row.writes is None:
            assert refusal is not None and row.why in refusal
        elif mechanism in row.writes:
            assert refusal is None
        else:
            assert refusal is not None and row.why in refusal and mechanism in refusal


# --------------------------------------------------------------------------- #
# the predicates the registry entry can decide
# --------------------------------------------------------------------------- #


def test_a_moe_component_on_a_dense_model_is_refused_at_load() -> None:
    """`routed_output` is hidden-wide, so no width refused it before; the row's
    `moe` predicate does, from the entry's `num_experts`."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "routed_output", "layers": [3]}
    with pytest.raises(ValidationError) as err:
        _validate(raw)  # the checklist reads the registry's static entry
    assert err.value.path == "sites.tgt.component"
    assert err.value.reason == "component_unavailable"
    assert "needs a sparse-MoE block" in str(err.value)


def test_the_same_component_on_a_moe_model_loads_with_the_bases_digest() -> None:
    raw = base_doc()
    raw["model"]["key"] = MOE.key
    raw["method"]["sites"]["tgt"] = {"component": "routed_output", "layers": [3]}
    assert _digest(raw) == BASE_DIGEST_ROUTED_OUTPUT


def test_unavailable_at_load_decides_only_what_the_entry_knows() -> None:
    dense = get_model_info("gpt2")
    assert unavailable_at_load(dense, "block_output") is None
    assert unavailable_at_load(dense, "shared_expert_gate") is not None
    # the gate is a module-tree fact for a family the tap table has not met…
    unmet = dataclasses.replace(dense, key="test/unmet", family=None)
    assert unavailable_at_load(unmet, "attention_gate") is None
    # …and a row fact for one it has: the gpt2 entry refuses it offline
    assert "needs an output gate" in (
        unavailable_at_load(dense, "attention_gate") or ""
    )
    no_shared = dataclasses.replace(
        MOE, key="test/no-shared", shared_expert_intermediate_size=None
    )
    assert unavailable_at_load(no_shared, "routed_output") is None
    assert "needs a shared expert" in (
        unavailable_at_load(no_shared, "shared_expert_output") or ""
    )


# --------------------------------------------------------------------------- #
# an all-MoE tower has no dense MLP
# --------------------------------------------------------------------------- #


def test_mlp_activation_is_refused_on_a_tower_with_no_dense_inner_width() -> None:
    info = dataclasses.replace(MOE, intermediate_size=None)
    with pytest.raises(ValidationError) as err:
        component_shape(info, "mlp_activation")
    assert err.value.rule == 4 and err.value.reason == "component_unavailable"
    assert "expert_activation" in str(err.value)
    # the field is only mlp_activation's: every other row sizes (or refuses)
    # exactly as it does with the width present
    for component in CAPABILITIES:
        if component in {"mlp_activation", "mlp_neuron_output"}:
            continue
        try:
            expected: Any = component_shape(MOE, component)
        except ValidationError as before:
            with pytest.raises(ValidationError, match=re.escape(before.message)):
                component_shape(info, component)
        else:
            assert component_shape(info, component) == expected


def test_mlp_activation_still_sizes_where_a_dense_mlp_exists() -> None:
    assert component_shape(MOE, "mlp_activation").width == 128
    assert component_shape(get_model_info("gpt2"), "mlp_activation").width == 3072


def test_the_a3b_entry_declares_no_dense_inner_width() -> None:
    """The row now says what the checkpoint says (its text config has no
    `intermediate_size`), so `validate` refuses `mlp_activation` on the A3B
    the way the run does — instead of sizing a featurizer at 8192."""
    info = get_model_info(A3B)
    assert info.intermediate_size is None
    assert info.family == "qwen3_5_moe_text"
    with pytest.raises(ValidationError, match="no dense MLP inner width"):
        component_shape(info, "mlp_activation")


def test_the_adapter_reads_none_when_the_config_declares_no_inner_width() -> None:
    """Validate and run must agree: the adapter yields the same `None` the
    static row carries, and an `int` where a config declares one."""
    base = dict(
        num_attention_heads=4,
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=100,
        model_type="qwen3_5_moe",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
    )
    all_moe = model_info_from_hf_config("test/all-moe", SimpleNamespace(**base))
    assert all_moe.intermediate_size is None and all_moe.family == "qwen3_5_moe"
    dense = model_info_from_hf_config(
        "test/dense", SimpleNamespace(**base, intermediate_size=64)
    )
    assert dense.intermediate_size == 64
    gpt2 = model_info_from_hf_config(
        "test/gpt2",
        SimpleNamespace(
            num_attention_heads=4,
            hidden_size=32,
            num_hidden_layers=2,
            vocab_size=100,
            model_type="gpt2",
            n_inner=None,
        ),
    )
    assert gpt2.intermediate_size == 128  # the GPT-2 family's 4·hidden rule
    unnamed = model_info_from_hf_config(
        "test/unnamed",
        SimpleNamespace(
            num_attention_heads=4, hidden_size=32, num_hidden_layers=2, vocab_size=100
        ),
    )
    assert unnamed.family is None


def test_an_entry_without_a_family_sizes_like_one_with() -> None:
    """`family` decides *where* a tap lands and what refuses at load, never a
    width: `component_shape` is family-blind."""
    with_family = dataclasses.replace(MOE, family="qwen3_5_moe_text")
    sized = 0
    for component in CAPABILITIES:
        try:
            expected: Any = component_shape(MOE, component)
        except ValidationError as before:
            with pytest.raises(ValidationError, match=re.escape(before.message)):
                component_shape(with_family, component)
        else:
            assert component_shape(with_family, component) == expected
            sized += 1
    assert sized >= 38  # the stub has no DeltaNet mixer: the 24 delta rows refuse


# --------------------------------------------------------------------------- #
# the per-family tap table — the rows' `overrides`
# --------------------------------------------------------------------------- #

#: 📐 The three fixtures' `config.model_type`, as loaded:
#: `GPT2Config` → gpt2, `LlamaConfig` → llama, `Qwen3_5MoeTextConfig` →
#: qwen3_5_moe_text. The keys of the table are exactly these.
MEASURED_MODEL_TYPES = frozenset({"gpt2", "llama", "qwen3_5_moe_text"})

#: The fixtures' widths (same load), for the shape agreement below.
FIXTURE_STUBS = {
    "gpt2": dict(num_heads=4, num_kv_heads=4, head_dim=8),
    "llama": dict(num_heads=4, num_kv_heads=4, head_dim=4),
    "qwen3_5_moe_text": dict(num_heads=8, num_kv_heads=4, head_dim=32),
}


def _stub(family: str) -> ModelInfo:
    return dataclasses.replace(
        MOE, key=f"stub/{family}", family=family, **FIXTURE_STUBS[family]
    )


def test_every_override_is_a_well_formed_address_over_closed_keys() -> None:
    """The census guard for the two new closed vocabularies: every override
    key is one of `OVERRIDE_KEYS`, every packing one of `PACKINGS`, and a
    fused packing carries its split arithmetic."""
    assert OVERRIDE_KEYS == ("module", "packing", "splits", "split")
    assert PACKINGS == ("flat", "head_axis", "fused_heads", "fused_blocks")
    seen_packings: set[str] = set()
    for component, row in CAPABILITIES.items():
        for family, address in row.overrides.items():
            assert set(address) <= set(OVERRIDE_KEYS), (component, family)
            assert address["module"].isidentifier()
            assert address["packing"] in PACKINGS
            seen_packings.add(address["packing"])
            if address["packing"].startswith("fused"):
                assert 0 <= address["split"] < address["splits"] >= 2
            else:
                assert "splits" not in address and "split" not in address
    assert seen_packings == set(PACKINGS)  # every packing is measured on a row


def test_a_malformed_override_cannot_build_a_row() -> None:
    row = CAPABILITIES["attention_query_pre_rope"]

    def build(address: dict[str, Any]) -> Capability:
        return dataclasses.replace(row, overrides={"gpt2": address})

    with pytest.raises(ValueError, match="outside"):
        build({"module": "c_attn", "packing": "flat", "modul": "x"})
    with pytest.raises(ValueError, match="packing"):
        build({"module": "c_attn", "packing": "fused"})
    with pytest.raises(ValueError, match="splits >= 2"):
        build({"module": "c_attn", "packing": "fused_blocks", "split": 0})
    with pytest.raises(ValueError, match="not in range"):
        build({"module": "c_attn", "packing": "fused_blocks", "splits": 3, "split": 3})
    with pytest.raises(ValueError, match="takes no splits"):
        build({"module": "q_proj", "packing": "flat", "splits": 1})
    with pytest.raises(ValueError, match="module name"):
        build({"module": "c-attn", "packing": "flat"})
    with pytest.raises(ValueError, match="two different packings"):
        dataclasses.replace(
            row,
            overrides={
                "a": {"module": "q_proj", "packing": "flat"},
                "b": {"module": "q_proj", "packing": "head_axis"},
            },
        )


def test_only_the_attention_interior_carries_overrides() -> None:
    """Census: the rows with per-family addresses are exactly the rows that
    require `split_qkv` — the four module-boundary components of the mixer —
    and each of them has one."""
    with_overrides = {c for c, row in CAPABILITIES.items() if row.overrides}
    assert with_overrides == set(INTERIOR_ROWS)
    assert with_overrides == {
        c for c, row in CAPABILITIES.items() if "split_qkv" in row.requires
    }
    assert set(INTERIOR_ROWS) == {
        "attention_query_pre_rope",
        "attention_key_pre_rope",
        "attention_value_states",
        "attention_gate",
    }


def test_the_gpt2_xl_alias_row_is_the_same_model() -> None:
    """`openai-community/gpt2-xl` is an alias row for `gpt2-xl` — the same
    checkpoint under its organization-prefixed Hub id, registered twice
    because a key is looked up as spelled. Field for field identical, so a
    correction to one row cannot be silently absent from the other."""
    canonical = dataclasses.asdict(get_model_info("gpt2-xl"))
    alias = dataclasses.asdict(get_model_info("openai-community/gpt2-xl"))
    assert alias.pop("key") == "openai-community/gpt2-xl"
    assert canonical.pop("key") == "gpt2-xl"
    assert alias == canonical


def test_every_family_in_the_table_is_a_measured_model_type() -> None:
    """Census: a family named in an override is a `model_type` the registry
    meets — carried by a registered entry, and one of the three fixtures'."""
    families = set(families_in_table())
    assert families == MEASURED_MODEL_TYPES
    registered = {
        info.family for info in (get_model_info(k) for k in ("gpt2", "gpt2-xl", A3B))
    } | {get_model_info("meta-llama/Llama-3.1-8B").family}
    assert families <= registered
    # and every built-in entry names its family, so validate keys the table
    # exactly as the run does
    for key in (
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.2-1B",
        "gpt2",
        "gpt2-xl",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-8B",
        "google/gemma-2-2b-it",
        A3B,
    ):
        assert get_model_info(key).family is not None, key


def test_the_rows_agree_on_the_logical_site_across_families() -> None:
    """On every family the same interior component is a
    module-output tap whose native packing flattens to the same logical value
    — same width (in the component's head space), same head count, same axis
    kinds once the fused axis is set aside."""
    for component in INTERIOR_ROWS:
        row = CAPABILITIES[component]
        assert row.tap == "module output"
        for family, address in row.overrides.items():
            info = _stub(family)
            value = component_shape(info, component)
            native = native_shape(address, value)
            assert native.width == value.width == component_width(info, component)
            assert native.head_space == value.head_space
            assert [a.kind for a in native.axes if a.kind != "fused"] == [
                "batch",
                "position",
                "head",
                "feature",
            ]
            assert (native.fused_index is not None) == address["packing"].startswith(
                "fused"
            )
            assert native.flat_inner is (address["packing"] != "head_axis")


def test_native_shape_packs_the_measured_widths() -> None:
    """📐 The projection widths the fixtures emit, from the rows' packings."""
    query = bs_flat_heads(4, 8)  # tiny-gpt2's query space
    gpt2 = native_shape(
        CAPABILITIES["attention_query_pre_rope"].overrides["gpt2"], query
    )
    assert gpt2.describe() == "(batch, position, fused·head·feature)"
    assert [a.width for a in gpt2.inner_axes] == [3, 4, 8]  # 3·4·8 = 96 = c_attn.nf
    gate = native_shape(
        CAPABILITIES["attention_gate"].overrides["qwen3_5_moe_text"],
        bs_flat_heads(8, 32),
    )
    assert [a.width for a in gate.inner_axes] == [8, 2, 32]  # 8·2·32 = 512
    norm = native_shape(
        CAPABILITIES["attention_key_pre_rope"].overrides["qwen3_5_moe_text"],
        bs_flat_heads(4, 32),
    )
    assert (
        norm.native_rank == 4 and norm.describe() == "(batch, position, head, feature)"
    )


def test_the_a3b_family_is_what_the_adapter_reads_off_the_text_config() -> None:
    """🐞 The A3B row said `qwen3_5_moe` (the wrapper config's spelling) while
    the adapter reads the *text* config's `qwen3_5_moe_text` — harmless while
    nothing keyed on `family`, wrong once the tap table does: validate would
    have refused nothing the run refuses. The row now equals the adapter."""
    text = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_attention_heads=16,
        hidden_size=2048,
        head_dim=256,
        num_hidden_layers=40,
        num_key_value_heads=2,
        vocab_size=248320,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
    )
    wrapper = SimpleNamespace(model_type="qwen3_5_moe", text_config=text)
    assert model_info_from_hf_config(A3B, wrapper).family == get_model_info(A3B).family
    assert model_info_from_hf_config(A3B, text).family == get_model_info(A3B).family
    assert get_model_info(A3B).family == "qwen3_5_moe_text"


def test_predicates_decide_offline_for_a_family_in_the_table() -> None:
    """`split_qkv` and `gated_attention` get a definite answer for
    a family the table has met, `None` otherwise (the run measures)."""
    for key, gate in (("gpt2", False), ("meta-llama/Llama-3.1-8B", False), (A3B, True)):
        info = get_model_info(key)
        assert family_in_table(info)
        assert predicate_holds(info, "split_qkv") is True
        assert predicate_holds(info, "gated_attention") is gate
        # a load-time knob: the hand-declared entry does not carry it
        assert predicate_holds(info, "grouped_mm") is None
    for key in ("Qwen/Qwen3-4B-Instruct-2507", "google/gemma-2-2b-it"):
        info = get_model_info(key)  # a family with an entry but no row
        assert not family_in_table(info)
        assert predicate_holds(info, "split_qkv") is None
        assert predicate_holds(info, "gated_attention") is None
    assert not family_in_table(MOE)  # no family at all
    assert predicate_holds(MOE, "gated_attention") is None


def test_the_gate_is_refused_at_load_on_a_family_without_one() -> None:
    """The run refused `attention_gate` on GPT-2 and llama by name
    (``fixtures/refusal_snapshot.json`` entry 24); `validate` now refuses it offline, V4, no new rule."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": "attention_gate", "layers": [3]}
    with pytest.raises(ValidationError) as err:
        _validate(raw)  # the checklist reads the registry's static entry
    assert err.value.rule == 4 and err.value.path == "sites.tgt.component"
    assert err.value.reason == "component_unavailable"
    assert "needs an output gate" in str(err.value)
    assert unavailable_at_load(
        get_model_info("meta-llama/Llama-3.1-8B"), "attention_gate"
    )


def test_the_gate_loads_on_the_family_that_has_one() -> None:
    raw = base_doc()
    raw["model"]["key"] = A3B
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_gate",
        "layers": [3],
    }  # full attention
    _validate(raw)
    canonicalize(in_order(raw), ENV)


@pytest.mark.parametrize("component", sorted(BASE_DIGEST_GPT2_INTERIOR))
def test_the_interior_on_gpt2_loads_with_the_bases_digest(component: str) -> None:
    """Valid work still passes, byte for byte: the base accepted these at load
    (and refused them at run); the table accepts them at both, and the digest
    is the base's — the per-family address enters no canonical form."""
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {"component": component, "layers": [3]}
    assert _digest(raw) == BASE_DIGEST_GPT2_INTERIOR[component]


def test_a_head_on_the_fused_interior_loads_with_the_bases_digest() -> None:
    raw = base_doc()
    raw["method"]["sites"]["tgt"] = {
        "component": "attention_query_pre_rope",
        "layers": [3],
        "head": 2,
    }
    assert _digest(raw) == BASE_DIGEST_GPT2_QUERY_HEAD_2


# --------------------------------------------------------------------------- #
# the rows themselves
# --------------------------------------------------------------------------- #


def test_a_row_cannot_be_inconsistent() -> None:
    from causalab.protocol.registry import Capability

    good = CAPABILITIES["expert_idx"]
    with pytest.raises(ValueError, match="does not match the write policy"):
        dataclasses.replace(good, reason=None)
    with_out_why = CAPABILITIES["router_logits"]
    with pytest.raises(ValueError, match="come together"):
        dataclasses.replace(with_out_why, why="")
    with pytest.raises(ValueError, match="unknown engine"):
        dataclasses.replace(good, reads=frozenset({"megatron"}))
    hooks_only = CAPABILITIES["delta_state"]  # the per-step face: reference engine only
    with pytest.raises(ValueError, match="does not serve"):
        dataclasses.replace(hooks_only, expert_selection=frozenset({"nnsight"}))
    assert isinstance(good, Capability)


def test_the_retired_spellings_are_the_rows_aliases() -> None:
    """Every alias of the vocabulary is exactly one row's `aliases` cell, with
    the deprecation version the alias table records (`schema.DEPRECATED_IN`):
    the earlier rename and the eight DeltaNet folds."""
    from causalab.protocol.schema import DEPRECATED_COMPONENTS, DEPRECATED_IN

    row = CAPABILITIES["attention_premix"]
    assert row.aliases == ("attention_value",) and row.deprecated_in == "1"
    assert CAPABILITIES["delta_kernel_output"].aliases == ("deltanet_core_out",)
    by_row = {c: r.aliases for c, r in CAPABILITIES.items() if r.aliases}
    assert {a for aliases in by_row.values() for a in aliases} == set(
        DEPRECATED_COMPONENTS
    )
    for component, aliases in by_row.items():
        assert all(DEPRECATED_COMPONENTS[a] == component for a in aliases)
        assert CAPABILITIES[component].deprecated_in == DEPRECATED_IN[aliases[0]]
    assert len(by_row) == 9


def test_a_protocol_error_carries_its_reason() -> None:
    err = ProtocolError("P4", "x", reason="unsupported_mechanism")
    assert err.reason == "unsupported_mechanism"
    assert ProtocolError("P4", "x").reason is None


def test_moe_is_a_registered_test_model() -> None:
    assert isinstance(get_model_info(MOE.key), ModelInfo)
