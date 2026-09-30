"""MoE per-expert interior: taps inside the grouped experts dispatch.

The per-expert interior is not a set of module boundaries. 📐
``Qwen3_5MoeExperts`` stores its weights as 3-D parameters and its only child
is one shared ``act_fn`` — there is no per-expert module for a hook to attach
to, on any path. The grouped function (the default dispatch, **including on
CPU**) materializes every diagram box densely in one call, expert-sorted; the
``experts_interface_taps`` wrapper recomputes its sort inside that call's
dynamic extent and hands taps the **token-major** form, ``(tokens, top_k · d)``,
with slot *k* the *k*-th ranked expert.

Why the grouped path is tapped rather than the naive path forced, measured on
``tiny-random/qwen3.5-moe`` (transformers 5.16):

    pass-through wrapper on ALL_EXPERTS_FUNCTIONS["grouped_mm"]:  0.0
    experts_implementation="eager" (the per-expert loop):         logits 4.2e-7 off

The identity-pin bar ("the no-op case is exactly equal") is met by tapping what
actually runs and missed by path-forcing — so a non-grouped implementation is
refused by name (the *dispatch pin*), naming the knob that selects it.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.experts_interface import (
    ExpertsTap,
    _check_call_counts,
    experts_interface_taps,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import component_shape
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .conftest import TINY_LLAMA, TINY_QWEN35_MOE

pytestmark = pytest.mark.smoke

MOE_LAYER = 0
TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

#: 📐 fixture numbers: 128 experts, top-10, moe_intermediate_size 32 — so the
#: token-major contract width of `expert_activation` is 10 · 32 = 320.
TOP_K = 10
D_EXPERT = 32


def _read_doc(component: str = "expert_activation", layer: int = MOE_LAYER) -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": {"component": component, "layers": [layer]}},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _also_read(doc: dict, name: str, component: str, site: str | None = None) -> None:
    """Add a second whole-sequence read of ``component`` at the MoE layer to
    a `_read_doc`, on the same un-intervened forward, and save it."""
    site = site or f"{name}_site"
    doc["method"]["sites"][site] = {"component": component, "layers": [MOE_LAYER]}
    doc["method"]["reads"][name] = {"site": site, "pos": "all"}
    doc["method"]["intervened_models"]["original"]["reads"].append(name)
    doc["method"]["save"].append(saved(name, "original", f"{site}.safetensors"))


def _write_doc(component: str, do: dict, *, layer: int = MOE_LAYER) -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "original_base": {"input": "base", "reads": ["clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": component, "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": "all"},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {"patch": {"site": "tap", "pos": "all", "do": do}},
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


def _own_value(doc: dict) -> dict:
    """Take ``v_cf`` on the un-intervened *base* forward instead of the
    counterfactual one: the write's operand is the tap's own value."""
    models = doc["method"]["intervened_models"]
    del models["original_counterfactual"]
    models["original_base"]["reads"].append("v_cf")
    return doc


def _moved(bundle: ModelBundle, doc: dict, **kw) -> float:
    executor = executor_for(
        doc, bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT], **kw
    )
    after, clean = executor.read_value("after"), executor.read_value("clean")
    return float((after - clean).abs().max())


def _eager_bundle(bundle: ModelBundle) -> ModelBundle:
    """The same checkpoint dispatched on the per-expert loop instead.

    ``load_model`` deliberately has no experts knob (the executor pins the
    default), so the non-grouped realization is built directly — the same way
    an exotic environment would produce one.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        TINY_QWEN35_MOE,
        dtype=torch.float32,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    model.eval()
    model.requires_grad_(False)
    return dataclasses.replace(bundle, model=model)


# --------------------------------------------------------------------------- #
# the tap resolves, and reads the token-major shape
# --------------------------------------------------------------------------- #


def test_the_tap_is_an_experts_slot_not_a_module_side(qwen35moe_bundle):
    """There is no per-expert module boundary, and the site says so."""
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component="expert_activation", layers=(MOE_LAYER,))
    )
    assert site.kind == "experts"
    assert site.interface_slot == "activation"
    # the experts module is carried: it is what says *which* call to tap
    experts = qwen35moe_bundle.model.model.layers[MOE_LAYER].mlp.experts
    assert site.module is experts


def test_the_read_is_token_major_with_the_ranked_slot_axis(qwen35moe_bundle):
    """📐 (1, 5, top_k · d_e) — one row per token, slot k the k-th ranked
    expert's activation. The expert-sorted order the grouped kernel computes in
    never crosses the interface."""
    executor = executor_for(_read_doc(), qwen35moe_bundle, base_texts=[TEXT])
    assert tuple(executor.read_value("r").shape) == (1, 5, TOP_K * D_EXPERT)


def test_the_unsort_matches_a_manual_reference(qwen35moe_bundle):
    """The pin on the permutation plumbing: un-sorting the raw ``act_fn`` rows
    with the router's own indices reproduces the read exactly.

    ⚠️ This is also the tie-order guard, since the read must keep each row
    with its own token under ties: the wrapper
    *recomputes* ``torch.sort(top_k_index.reshape(-1))``, which does not promise
    tie order. If a kernel ever breaks ties differently than the recomputation,
    rows would be attributed to the wrong tokens — and this 0.0 would fail
    loudly instead."""
    bundle = qwen35moe_bundle
    value = executor_for(_read_doc(), bundle, base_texts=[TEXT]).read_value("r")

    block = bundle.model.model.layers[MOE_LAYER].mlp
    seen: dict[str, torch.Tensor] = {}
    handles = [
        block.experts.act_fn.register_forward_hook(
            lambda _m, _i, out: seen.__setitem__("act", out.detach().clone())
        ),
        block.gate.register_forward_hook(
            lambda _m, _i, out: seen.__setitem__("idx", out[2].detach().clone())
        ),
    ]
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    _, perm = torch.sort(seen["idx"].reshape(-1))
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel())
    reference = seen["act"][inv_perm].reshape(1, seen["idx"].shape[0], -1)
    torch.testing.assert_close(value, reference, atol=0.0, rtol=0.0)


def test_reading_the_interior_does_not_change_the_model(qwen35moe_bundle):
    """Observe-only must be bit-identical: the wrapper replaces a process-global
    dispatch entry and patches a module global while active, so "reading changed
    the answer" is a live failure mode."""
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()
    executor = executor_for(_read_doc(), qwen35moe_bundle, base_texts=[TEXT])
    executor.read_value("r")
    with torch.no_grad():
        after = qwen35moe_bundle.model(**encoded).logits.clone()
    assert torch.equal(after, clean)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


def test_a_swap_through_the_interface_moves_the_logits(qwen35moe_bundle):
    moved = _moved(qwen35moe_bundle, _write_doc("expert_activation", {"swap": "v_cf"}))
    assert moved > 1e-4


def test_swapping_a_tap_with_its_own_value_moves_nothing(qwen35moe_bundle):
    """The identity payload, exactly 0.0: an edit that substitutes the same
    tensor must be exactly the identity — through the un-sort, the write math,
    and the re-sort — or the write is landing somewhere it should not."""
    doc = _own_value(_write_doc("expert_activation", {"swap": "v_cf"}))
    assert _moved(qwen35moe_bundle, doc) == 0.0


def test_doubling_the_activation_moves_the_logits(qwen35moe_bundle):
    """📐 The probe's causal spike (act ×2 moved the logits by 0.2486),
    expressed in the vocabulary: add the tap's own value to itself."""
    doc = _own_value(
        _write_doc("expert_activation", {"add_scaled": {"op": "v_cf", "alpha": 1.0}})
    )
    assert _moved(qwen35moe_bundle, doc) > 1e-3


def test_a_read_of_a_written_slot_sees_the_written_value(qwen35moe_bundle):
    """Same-forward read-after-write agreement, the R2.3 contract: the executor
    registers edits before reads, so a document that swaps and reads the same
    slot sees the written value — difference exactly 0.0."""
    doc = _write_doc("expert_activation", {"swap": "v_cf"})
    doc["method"]["reads"]["obs"] = {"site": "tap", "pos": "all"}
    doc["method"]["intervened_models"]["patched"]["reads"].append("obs")
    doc["method"]["save"].append(saved("obs", "patched", "o.safetensors"))
    executor = executor_for(
        doc, qwen35moe_bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    v_cf, obs = executor.read_value("v_cf"), executor.read_value("obs")
    assert float((obs - v_cf).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# generated frames (D7)
# --------------------------------------------------------------------------- #


def test_a_continuation_read_accumulates_one_row_per_step(qwen35moe_bundle):
    """The interior is token-indexed, so a decode step is exactly one position
    per row and the steps stack — unlike ``attention_key``, nothing here grows
    with the prefix."""
    doc = _read_doc()
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 3}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == (1, 3, TOP_K * D_EXPERT)


# --------------------------------------------------------------------------- #
# the dispatch pin, and the equivalence gate that justifies it
# --------------------------------------------------------------------------- #


def test_the_fixture_runs_the_grouped_path_by_default(qwen35moe_bundle):
    """📐 The pin's premise: ``grouped_mm`` is the default dispatch even on CPU
    with no kwarg — the gate is a class check, not a device check. If a
    transformers bump ever changes this default, the whole round's taps move,
    and this is the test that says so first."""
    config = qwen35moe_bundle.model.config
    text = getattr(config, "text_config", None) or config
    assert text._experts_implementation == "grouped_mm"


def test_a_non_grouped_model_is_refused_by_name(qwen35moe_bundle):
    """The dispatch pin: same tensor, wrong provenance. The refusal names the
    knob (``experts_implementation``) so an exotic environment knows what to
    change."""
    bundle = _eager_bundle(qwen35moe_bundle)
    with pytest.raises(ProtocolError, match="experts_implementation='eager'"):
        resolve_site(
            bundle, SiteSpec(component="expert_activation", layers=(MOE_LAYER,))
        )


def test_grouped_and_eager_agree_only_to_float_tolerance(qwen35moe_bundle):
    """📐 4.2e-7 — why the naive path is *refused* rather than silently
    different: the two implementations compute the same numbers in a different
    order, so forcing "eager" for determinism would break the exact-identity
    bar every other pin in this suite holds."""
    eager = _eager_bundle(qwen35moe_bundle)
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        grouped_logits = qwen35moe_bundle.model(**encoded).logits
        eager_logits = eager.model(**encoded).logits
    drift = float((grouped_logits - eager_logits).abs().max())
    assert 0.0 < drift <= 1e-6


# --------------------------------------------------------------------------- #
# containment and scoping
# --------------------------------------------------------------------------- #


def test_the_dispatch_entry_is_restored_on_exit(qwen35moe_bundle):
    """The entry is process-global while installed; leaking it would leave every
    later MoE forward in the process running through our wrapper."""
    import transformers.integrations.moe as moe

    before = moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"]
    experts = qwen35moe_bundle.model.model.layers[MOE_LAYER].mlp.experts
    with experts_interface_taps(
        {id(experts): (ExpertsTap(slot="activation", read=lambda _v, _i: None),)}
    ):
        assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is not before
    assert moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"] is before


def test_a_tap_at_one_layer_leaves_another_layers_experts_alone(qwen35moe_bundle):
    """The wrapper answers for *every* experts module while installed, so
    pass-through for an untapped module is the scoping that can actually fail.
    📐 Layer 1's experts output is bit-identical under a layer-0 tap."""
    bundle = qwen35moe_bundle
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    other = bundle.model.model.layers[1].mlp.experts
    seen: dict[str, torch.Tensor] = {}
    handle = other.register_forward_hook(
        lambda _m, _i, out: seen.__setitem__(
            "t", (out[0] if isinstance(out, tuple) else out).detach().clone()
        )
    )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
        clean = seen["t"].clone()
        executor_for(_read_doc(layer=0), bundle, base_texts=[TEXT]).read_value("r")
        with torch.no_grad():
            bundle.model(**encoded)
        after = seen["t"]
    finally:
        handle.remove()
    assert torch.equal(after, clean)


# --------------------------------------------------------------------------- #
# guards and refusals
# --------------------------------------------------------------------------- #


def test_the_call_count_guard_refuses_any_other_shape(qwen35moe_bundle):
    """📐 The grouped forward calls ``_grouped_linear`` exactly twice and
    ``act_fn`` exactly once; any other count is a different factorization and
    the slot labels would lie (the eager loop fires ``act_fn`` once per hit
    expert)."""
    experts = qwen35moe_bundle.model.model.layers[MOE_LAYER].mlp.experts
    _check_call_counts(experts, 2, 1)  # the measured case: fine
    for gl_calls, act_calls in ((1, 1), (3, 1), (2, 0), (2, 40)):
        with pytest.raises(ProtocolError, match="not \\(2, 1\\)"):
            _check_call_counts(experts, gl_calls, act_calls)


def test_an_ungated_experts_module_is_refused_by_name():
    """A ``has_gate=False`` family also calls ``_grouped_linear`` twice, with
    different meanings (up, then down) — labeling its first call ``[gate | up]``
    would be the silent-wrong-tensor failure the descriptors exist to prevent."""

    class Ungated(torch.nn.Module):
        has_gate = False
        act_fn = torch.nn.SiLU()

    module = Ungated()
    with experts_interface_taps(
        {id(module): (ExpertsTap(slot="activation", read=lambda _v, _i: None),)}
    ):
        import transformers.integrations.moe as moe

        with pytest.raises(ProtocolError, match="has_gate"):
            moe.ALL_EXPERTS_FUNCTIONS["grouped_mm"](
                module,
                torch.zeros(2, 8),
                torch.zeros(2, 2, dtype=torch.int64),
                torch.ones(2, 2),
            )


def test_head_is_refused_because_nothing_here_has_heads(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="no head axis"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="expert_activation", layers=(MOE_LAYER,), head=0),
        )


def test_a_dense_mlp_family_is_refused_architecturally():
    bundle = load_model(TINY_LLAMA)
    with pytest.raises(ProtocolError, match="sparse-MoE"):
        resolve_site(bundle, SiteSpec(component="expert_activation", layers=(1,)))


def test_the_shape_declares_the_fixtures_widths(qwen35moe_bundle):
    """The registry's answer against the loaded config: top-k slots of the
    *routed* inner width — a field of its own, because a MoE checkpoint carries
    three inner widths and on this fixture all three are 32 (⚠️ so only the
    registry unit tests can catch a wrong-field read; this pins the plumbing)."""
    shape = component_shape(qwen35moe_bundle.info, "expert_activation")
    assert shape.width == TOP_K * D_EXPERT
    assert shape.ranking is True
    assert shape.flat_batch is True
    assert shape.head_space is None


# --------------------------------------------------------------------------- #
# The interface-slot components
# --------------------------------------------------------------------------- #

INTERIOR = (
    "expert_gate_proj",
    "expert_up_proj",
    "expert_activation",
    "expert_output",
)

#: 📐 token-major contract widths on the fixture: the projection halves and the
#: activation are moe_intermediate_size (32) per slot, the down-projection's
#: output is hidden (8) per slot.
INTERIOR_WIDTH = {
    "expert_gate_proj": TOP_K * D_EXPERT,
    "expert_up_proj": TOP_K * D_EXPERT,
    "expert_activation": TOP_K * D_EXPERT,
    "expert_output": TOP_K * 8,
}


def _hit_and_missing_expert(bundle: ModelBundle) -> tuple[int, int]:
    """One expert the router chose often at these tokens, and one it never did
    (📐 40 of 128 experts are hit on the 5-token prompt, so both exist)."""
    doc = _read_doc("expert_idx")
    idx = executor_for(doc, bundle, base_texts=[TEXT]).read_value("idx" and "r")
    chosen = set(int(x) for x in idx.reshape(-1))
    hit = int(idx.reshape(-1).mode().values)
    missing = next(i for i in range(128) if i not in chosen)
    return hit, missing


@pytest.mark.parametrize("component", INTERIOR)
def test_every_interior_component_resolves_and_reads(qwen35moe_bundle, component):
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layers=(MOE_LAYER,))
    )
    assert site.kind == "experts"
    value = executor_for(
        _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, INTERIOR_WIDTH[component])


def test_the_projection_halves_share_one_fused_capture(qwen35moe_bundle):
    """The `attention_gate` precedent with a top-k axis in front: gate and up
    are chunks of one projection output, and the registry identity
    ``expert_activation == act_fn(expert_gate_proj)`` pins which chunk is
    which — exactly, because the model's own SiLU is deterministic."""
    doc = _read_doc("expert_gate_proj")
    _also_read(doc, "a", "expert_activation", site="act")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    gate, act = executor.read_value("r"), executor.read_value("a")
    torch.testing.assert_close(torch.nn.functional.silu(gate), act, atol=0.0, rtol=0.0)


def test_the_registry_identity_reconstructs_routed_output_exactly(qwen35moe_bundle):
    """The declared identity, asserted: ``routed_output == Σ_slot
    expert_output · router_scores`` — at the tolerance the family's row
    declares for fp32 (exactly 0.0, because the model computes precisely this
    sum in this order). The identity, its inputs and its tolerance are read
    off the family adapter (``registry.identity``), not this test's
    literals, which is what lets a new family's plugin be held to the same
    identity. 📐 This is also the tie-order guard on the recomputed sort: rows
    attributed to the wrong tokens would break it loudly."""
    from causalab.neural.shared.sites import adapter_of
    from causalab.protocol.registry import identity

    declared = identity(adapter_of(qwen35moe_bundle).family, "routed_output")
    assert declared.name == "routed_sum"
    assert set(declared.inputs) == {"expert_output", "router_scores"}
    atol, rtol = declared.tolerance_for(qwen35moe_bundle.dtype)
    doc = _read_doc("expert_output")
    _also_read(doc, "s", "router_scores", site="scores")
    _also_read(doc, "o", "routed_output", site="routed")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    out = executor.read_value("r").reshape(1, 5, TOP_K, 8)
    scores = executor.read_value("s").reshape(1, 5, TOP_K, 1)
    routed = executor.read_value("o")
    torch.testing.assert_close((out * scores).sum(2), routed, atol=atol, rtol=rtol)


@pytest.mark.parametrize("component", ("expert_gate_proj", "expert_output"))
def test_interior_writes_hold_the_identity_bar(qwen35moe_bundle, component):
    """swap-with-own-value is exactly 0.0 through the fused scatter and the
    re-sort; a counterfactual swap moves the logits (📐 expert_out +1 moved
    them by 1.53 in the probe)."""
    doc = _own_value(_write_doc(component, {"swap": "v_cf"}))
    assert _moved(qwen35moe_bundle, doc) == 0.0
    assert _moved(qwen35moe_bundle, _write_doc(component, {"swap": "v_cf"})) > 1e-4


# --------------------------------------------------------------------------- #
# The `expert:` sub-axis
# --------------------------------------------------------------------------- #


def test_the_expert_face_selects_exactly_the_routed_pairs(qwen35moe_bundle):
    """The ragged face against a manual join: rows where ``expert_idx == e``,
    in (position, slot) order, at exactly 0.0."""
    from causalab.neural.shared.executor import RaggedValue

    hit, _ = _hit_and_missing_expert(qwen35moe_bundle)
    doc = _read_doc("expert_activation")
    _also_read(doc, "i", "expert_idx", site="idxs")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    full, idx = executor.read_value("r"), executor.read_value("i")

    faced = _read_doc("expert_activation")
    faced["method"]["sites"]["tap"]["expert"] = hit
    value = executor_for(faced, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert isinstance(value, RaggedValue)
    mask = idx.reshape(1, 5, TOP_K) == hit
    manual = full.reshape(1, 5, TOP_K, D_EXPERT)[mask]
    assert value.widths == (int(mask.sum()),)
    torch.testing.assert_close(value.flat, manual, atol=0.0, rtol=0.0)


def test_an_expert_no_token_chose_reads_as_width_zero(qwen35moe_bundle):
    """The honest form of the never-fired-hook question: there is no hook to
    not-fire. The router simply sent this expert nothing at these positions,
    and the read says so as data — width-0 rows, no error."""
    from causalab.neural.shared.executor import RaggedValue

    _, missing = _hit_and_missing_expert(qwen35moe_bundle)
    doc = _read_doc("expert_activation")
    doc["method"]["sites"]["tap"]["expert"] = missing
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert isinstance(value, RaggedValue)
    assert value.widths == (0,)
    assert tuple(value.flat.shape) == (0, D_EXPERT)


def test_a_write_under_expert_lands_only_on_that_experts_rows(qwen35moe_bundle):
    """Doubling one hit expert's activations moves the logits; the same write
    aimed at an unchosen expert lands nowhere — exactly 0.0, the data-fact twin
    of the width-0 read."""
    hit, missing = _hit_and_missing_expert(qwen35moe_bundle)

    def masked_doc(expert: int) -> dict:
        doc = _write_doc(
            "expert_activation", {"add_scaled": {"op": "v_cf", "alpha": 1.0}}
        )
        doc["method"]["sites"]["tap"]["expert"] = expert
        # the operand reads the token-major form: the write site owns the mask
        doc["method"]["sites"]["whole"] = {
            "component": "expert_activation",
            "layers": [MOE_LAYER],
        }
        doc["method"]["reads"]["v_cf"] = {"site": "whole", "pos": "all"}
        return _own_value(doc)

    assert _moved(qwen35moe_bundle, masked_doc(hit)) > 1e-4
    assert _moved(qwen35moe_bundle, masked_doc(missing)) == 0.0


def test_the_expert_face_reads_in_the_generated_frame(qwen35moe_bundle):
    """D7 extends to the ragged face: the routing table accumulates per decode
    step, so the face selects over the generated tokens' own routing."""
    from causalab.neural.shared.executor import RaggedValue

    hit, _ = _hit_and_missing_expert(qwen35moe_bundle)
    doc = _read_doc("expert_output")
    doc["method"]["sites"]["tap"]["expert"] = hit
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 3}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert isinstance(value, RaggedValue)
    assert len(value.widths) == 1
    assert value.flat.shape[-1] == 8  # hidden-wide rows


def test_expert_bounds_are_refused_by_name(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="128 experts"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="expert_activation", layers=(MOE_LAYER,), expert=128),
        )


def test_expert_on_a_router_component_is_still_refused(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="no per-expert axis"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="router_scores", layers=(MOE_LAYER,), expert=3),
        )


@pytest.mark.parametrize(
    "field, value", [("featurizer", "f"), ("dims", [0, 1])], ids=["featurizer", "dims"]
)
def test_featurizer_and_dims_are_refused_on_the_expert_face(
    qwen35moe_bundle, field, value
):
    """Both are sized against the token-major ``top_k · d`` axis, and the face's
    rows are ``d``-wide — applying either would index a different space than
    the author named."""
    hit, _ = _hit_and_missing_expert(qwen35moe_bundle)
    doc = _read_doc("expert_activation")
    doc["method"]["sites"]["tap"]["expert"] = hit
    if field == "featurizer":
        doc["method"]["featurizers"] = {"f": {"kind": "standardize"}}
    doc["method"]["reads"]["r"][field] = value
    with pytest.raises(ProtocolError, match="expert"):
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
