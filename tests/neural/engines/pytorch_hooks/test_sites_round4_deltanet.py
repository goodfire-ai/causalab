"""The Gated DeltaNet interior — tier 1: the mixer's module boundaries.

📐 Three of the DeltaNet diagram's boxes are ordinary ``nn.Module`` sides on
``tiny-random/qwen3.5-moe`` (transformers 5.16): ``in_proj_qkv``'s output is
the fused ``[q | k | v]`` projection (widths 128/128/256 — **unequal**, so no
head axis), ``in_proj_z``'s output is the output gate (8 v-heads × 32), and
``out_proj``'s **input** is the post-norm, post-gate mixer value — the exact
analogue of ``attention_premix``, which is why the name. The ``conv1d`` module
never fires (the forward calls the ``causal_conv1d_fn`` module global instead),
which is why the conv output and the kernel boundary are *function* taps
(tier 2), not module taps.

The stream refusals mirror ``_FULL_ATTENTION_ONLY`` in the other
direction: a full-attention layer computes no delta-rule state, and a family
with no linear stream anywhere (llama, GPT-2) hits the same refusal at every
layer.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import component_shape
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .conftest import TINY_GPT2, TINY_LLAMA

pytestmark = pytest.mark.smoke

DELTANET_LAYER = 0
FULL_ATTENTION_LAYER = 3
TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

TIER_ONE = ("delta_qkv", "delta_gate", "delta_premix")

#: 📐 fixture numbers: 8 v-heads, 4 k-heads (2× GVA tiling), head dims 32 —
#: key_dim 128, value_dim 256, conv_dim 512.
TIER_ONE_WIDTH = {"delta_qkv": 512, "delta_gate": 256, "delta_premix": 256}

#: which module side each component taps, for the resolve test
TIER_ONE_TAP = {
    "delta_qkv": ("in_proj_qkv", "out"),
    "delta_gate": ("in_proj_z", "out"),
    "delta_premix": ("out_proj", "in"),
}


def _read_doc(
    component: str, layer: int = DELTANET_LAYER, head: int | None = None
) -> dict:
    tap: dict = {"component": component, "layers": [layer]}
    if head is not None:
        tap["head"] = head
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": tap},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _also_read(doc: dict, name: str, component: str, site: str) -> None:
    """Add a second whole-sequence read of ``component`` at the DeltaNet layer
    to a `_read_doc`, on the same un-intervened forward, and save it."""
    doc["method"]["sites"][site] = {"component": component, "layers": [DELTANET_LAYER]}
    doc["method"]["reads"][name] = {"site": site, "pos": "all"}
    doc["method"]["intervened_models"]["original"]["reads"].append(name)
    doc["method"]["save"].append(saved(name, "original", f"{site}.safetensors"))


def _write_doc(component: str, do: dict, *, layer: int = DELTANET_LAYER) -> dict:
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


def _self_swap(doc: dict) -> dict:
    """Take ``v_cf`` on the un-intervened *base* forward instead of the
    counterfactual one, so the write substitutes the tap's own value."""
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


# --------------------------------------------------------------------------- #
# the taps resolve, at the module sides the forward really uses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_the_tap_is_the_declared_module_side(qwen35moe_bundle, component: str):
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layers=(DELTANET_LAYER,))
    )
    child, side = TIER_ONE_TAP[component]
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    assert site.module is getattr(mixer, child)
    assert site.kind == side


@pytest.mark.parametrize("component", TIER_ONE)
def test_the_read_has_the_measured_width(qwen35moe_bundle, component: str):
    value = executor_for(
        _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, TIER_ONE_WIDTH[component])


def test_the_conv1d_module_never_fires(qwen35moe_bundle):
    """📐 The premise of the tier-2 function taps, pinned where tier 1 can
    see it: the forward calls the ``causal_conv1d_fn`` module global, so a hook
    on the ``conv1d`` module reads nothing — a module tap there would be the
    silent-empty-read failure."""
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    fired: list[bool] = []
    handle = mixer.conv1d.register_forward_hook(lambda *_: fired.append(True))
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    try:
        with torch.no_grad():
            qwen35moe_bundle.model(**encoded)
    finally:
        handle.remove()
    assert fired == []


# --------------------------------------------------------------------------- #
# identity pins (tier 1)
# --------------------------------------------------------------------------- #


def test_the_gate_is_the_z_projection_of_the_norm_exactly(qwen35moe_bundle):
    """The tier-1 pin: ``delta_gate == in_proj_z(attention_input_norm)`` at
    exactly 0.0 — the tap is where it claims to be, on the tensor the mixer
    actually consumes."""
    doc = _read_doc("delta_gate")
    _also_read(doc, "n", "attention_input_norm", site="norm")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    gate, norm_out = executor.read_value("r"), executor.read_value("n")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.in_proj_z(norm_out)
    torch.testing.assert_close(gate, reference, atol=0.0, rtol=0.0)


def test_the_qkv_projection_is_the_same_identity_one_module_over(qwen35moe_bundle):
    doc = _read_doc("delta_qkv")
    _also_read(doc, "n", "attention_input_norm", site="norm")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    qkv, norm_out = executor.read_value("r"), executor.read_value("n")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.in_proj_qkv(norm_out)
    torch.testing.assert_close(qkv, reference, atol=0.0, rtol=0.0)


def test_the_premix_projects_to_the_mixer_output_exactly(qwen35moe_bundle):
    """The premix is ``out_proj``'s input, so ``out_proj(delta_premix)`` must
    be the mixer's output — the analogue of the ``attention_premix`` identity."""
    doc = _read_doc("delta_premix")
    _also_read(doc, "o", "attention_output", site="out")
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    premix, mixer_out = executor.read_value("r"), executor.read_value("o")
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.out_proj(premix)
    torch.testing.assert_close(mixer_out, reference, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_swap_through_the_tap_moves_the_logits(qwen35moe_bundle, component: str):
    assert _moved(qwen35moe_bundle, _write_doc(component, {"swap": "v_cf"})) > 1e-4


@pytest.mark.parametrize("component", TIER_ONE)
def test_swapping_a_tap_with_its_own_value_moves_nothing(
    qwen35moe_bundle, component: str
):
    doc = _self_swap(_write_doc(component, {"swap": "v_cf"}))
    assert _moved(qwen35moe_bundle, doc) == 0.0, component


# --------------------------------------------------------------------------- #
# head bounds
# --------------------------------------------------------------------------- #


def test_the_gate_and_premix_are_value_head_space(qwen35moe_bundle):
    info = qwen35moe_bundle.info
    assert info.linear_num_value_heads == 8
    assert component_shape(info, "delta_gate").head_space == 8
    assert component_shape(info, "delta_premix").head_space == 8
    assert component_shape(info, "delta_qkv").head_space is None


def test_a_head_selects_one_v_head_of_the_gate(qwen35moe_bundle):
    value = executor_for(
        _read_doc("delta_gate", head=3), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 32)


def test_head_on_the_fused_qkv_is_refused_because_its_widths_are_unequal(
    qwen35moe_bundle,
):
    """The fused ``[q | k | v]`` splits are 128/128/256 — no equal per-head
    packing exists, so ``head`` cannot mean a slice of it. The refusal's note
    names the per-head faces (the tier-2 kernel-boundary components)."""
    with pytest.raises(ProtocolError, match="no head axis") as excinfo:
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_qkv", layers=(DELTANET_LAYER,), head=0),
        )
    assert "delta_query" in str(excinfo.value)


def test_an_out_of_range_v_head_is_refused(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="which has 8 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_gate", layers=(DELTANET_LAYER,), head=8),
        )


# --------------------------------------------------------------------------- #
# stream and family refusals — the `_LINEAR_ATTENTION_ONLY` mirror
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_full_attention_layer_refuses_with_the_architectural_reason(
    qwen35moe_bundle, component: str
):
    with pytest.raises(ProtocolError, match="delta-rule state"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component=component, layers=(FULL_ATTENTION_LAYER,)),
        )


@pytest.mark.parametrize("key", [TINY_LLAMA, TINY_GPT2])
@pytest.mark.parametrize("component", TIER_ONE)
def test_a_family_with_no_linear_stream_refuses_at_every_layer(
    key: str, component: str
):
    """llama and gpt2 carry full attention everywhere, so the same per-layer
    refusal is the architectural one: the tower listing in the message shows a
    stream this family never has."""
    bundle = load_model(key)
    with pytest.raises(ProtocolError, match="delta-rule state"):
        resolve_site(bundle, SiteSpec(component=component, layers=(1,)))


def test_a_declared_stream_still_wins_over_the_component_check(qwen35moe_bundle):
    """`stream: full_attention` on a delta component at a linear layer refuses
    on the *stream* mismatch first — the declaration is checked before the
    component's needs, so the message names the per-layer fact."""
    with pytest.raises(ProtocolError, match="per-layer fact"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(
                component="delta_gate",
                layers=(DELTANET_LAYER,),
                stream="full_attention",
            ),
        )


# --------------------------------------------------------------------------- #
# generated frames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", TIER_ONE)
def test_a_continuation_read_accumulates_one_row_per_step(
    qwen35moe_bundle, component: str
):
    """All three are query-position-shaped (one row per token), so decode steps
    stack — nothing here is indexed by the growing prefix."""
    doc = _read_doc(component)
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 3}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == (1, 3, TIER_ONE_WIDTH[component])


# --------------------------------------------------------------------------- #
# tier 2 — the kernel boundary
# --------------------------------------------------------------------------- #

TIER_TWO = (
    "delta_conv",
    "delta_query",
    "delta_key",
    "delta_value",
    "delta_beta",
    "delta_decay",
    "delta_kernel_output",
)

#: 📐 contract widths on the fixture: conv_dim 512; q/k/v tiled to 8 v-heads of
#: 32 (q/k share the key head dim, tiled BEFORE the kernel); the gates are one
#: scalar per head.
TIER_TWO_WIDTH = {
    "delta_conv": 512,
    "delta_query": 256,
    "delta_key": 256,
    "delta_value": 256,
    "delta_beta": 8,
    "delta_decay": 8,
    "delta_kernel_output": 256,
}


@pytest.mark.parametrize("component", TIER_TWO)
def test_the_kernel_tap_is_a_delta_slot_not_a_module_side(
    qwen35moe_bundle, component: str
):
    """No module boundary: the tensor is an argument or return of the
    kernel-boundary globals, and the site carries the *mixer* — what identifies
    which forward's calls to tap."""
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layers=(DELTANET_LAYER,))
    )
    assert site.kind == "delta"
    assert site.interface_slot is not None
    assert site.module is qwen35moe_bundle.mixer_at(DELTANET_LAYER)


@pytest.mark.parametrize("component", TIER_TWO)
def test_the_kernel_read_has_the_measured_width(qwen35moe_bundle, component: str):
    value = executor_for(
        _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, TIER_TWO_WIDTH[component])


def test_reading_the_kernel_boundary_does_not_change_the_model(qwen35moe_bundle):
    """The wrappers swap module globals while installed; observe-only must be
    bit-identical (📐 measured 0.0 — the argument for tapping the running path
    rather than forcing a naive one)."""
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()
    for component in TIER_TWO:
        executor_for(
            _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
        ).read_value("r")
        with torch.no_grad():
            after = qwen35moe_bundle.model(**encoded).logits.clone()
        assert torch.equal(after, clean), component


# --------------------------------------------------------------------------- #
# tier-2 identity pins
# --------------------------------------------------------------------------- #


def _multi_read(
    bundle: ModelBundle, components: dict[str, str]
) -> dict[str, torch.Tensor]:
    """Read several layer-0 components in one document: name -> value."""
    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": list(components)}
            },
            "sites": {
                f"{name}_site": {"component": component, "layers": [DELTANET_LAYER]}
                for name, component in components.items()
            },
            "reads": {
                name: {"site": f"{name}_site", "pos": "all"} for name in components
            },
            "save": [
                saved(name, "original", f"{name}.safetensors") for name in components
            ],
        },
    }
    executor = executor_for(doc, bundle, base_texts=[TEXT])
    return {name: executor.read_value(name) for name in components}


def test_the_conv_split_and_tile_reproduces_q_k_v_exactly(qwen35moe_bundle):
    """Tier 2: ``split(delta_conv, [128, 128, 256])``, reshaped to heads and
    GVA-tiled, IS ``(delta_query, delta_key, delta_value)`` — at exactly 0.0,
    which is also why the untiled q/k are not components (one box, one
    address)."""
    reads = _multi_read(
        qwen35moe_bundle,
        {
            "conv": "delta_conv",
            "q": "delta_query",
            "k": "delta_key",
            "v": "delta_value",
        },
    )
    conv = reads["conv"]  # contract (1, 5, 512): position-major again
    q_ref, k_ref, v_ref = torch.split(conv, [128, 128, 256], dim=-1)

    def tile(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(1, 5, 4, 32).repeat_interleave(2, dim=2).reshape(1, 5, 256)

    torch.testing.assert_close(tile(q_ref), reads["q"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(tile(k_ref), reads["k"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(v_ref, reads["v"], atol=0.0, rtol=0.0)


def test_the_gates_are_the_projections_transformed_exactly(qwen35moe_bundle):
    """Tier 2: ``delta_beta == sigmoid(in_proj_b(norm))`` and
    ``delta_decay == -exp(A_log) · softplus(in_proj_a(norm) + dt_bias)`` — the
    justification for keeping the raw projections out of the vocabulary:
    both are closed-form steps from tensors that are components."""
    reads = _multi_read(
        qwen35moe_bundle,
        {"n": "attention_input_norm", "beta": "delta_beta", "decay": "delta_decay"},
    )
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        beta_ref = mixer.in_proj_b(reads["n"]).sigmoid()
        decay_ref = -mixer.A_log.float().exp() * torch.nn.functional.softplus(
            mixer.in_proj_a(reads["n"]).float() + mixer.dt_bias
        )
    torch.testing.assert_close(reads["beta"], beta_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reads["decay"], decay_ref, atol=0.0, rtol=0.0)


def test_norm_gating_the_kernel_output_reproduces_the_premix_exactly(
    qwen35moe_bundle,
):
    """Tier 2: ``norm(delta_kernel_output, delta_gate) == delta_premix`` —
    the kernel's return really is the pre-norm, pre-gate ``core_attn_out``."""
    reads = _multi_read(
        qwen35moe_bundle,
        {
            "out": "delta_kernel_output",
            "gate": "delta_gate",
            "premix": "delta_premix",
        },
    )
    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    with torch.no_grad():
        reference = mixer.norm(
            reads["out"].reshape(-1, 32), reads["gate"].reshape(-1, 32)
        ).reshape(1, 5, 256)
    torch.testing.assert_close(reads["premix"], reference, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# writes at the kernel boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component", ("delta_conv", "delta_value", "delta_beta", "delta_kernel_output")
)
def test_a_kernel_boundary_swap_moves_the_logits_and_self_swap_does_not(
    qwen35moe_bundle, component: str
):
    """📐 the probe's causal spikes (kernel-arg v ×2: 0.2190, conv ×2: 0.2191),
    with the identity payload at exactly 0.0 — both through the same wrapper."""
    assert _moved(qwen35moe_bundle, _write_doc(component, {"swap": "v_cf"})) > 1e-4
    doc = _self_swap(_write_doc(component, {"swap": "v_cf"}))
    assert _moved(qwen35moe_bundle, doc) == 0.0, component


def test_a_read_of_a_written_kernel_slot_sees_the_written_value(qwen35moe_bundle):
    """Same-forward read-after-write agreement across a third tap mechanism —
    the executor registers edits before reads at the kernel boundary too."""
    doc = _write_doc("delta_value", {"swap": "v_cf"})
    doc["method"]["reads"]["obs"] = {"site": "tap", "pos": "all"}
    doc["method"]["intervened_models"]["patched"]["reads"].append("obs")
    doc["method"]["save"].append(saved("obs", "patched", "o.safetensors"))
    executor = executor_for(
        doc, qwen35moe_bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    assert (
        float((executor.read_value("obs") - executor.read_value("v_cf")).abs().max())
        == 0.0
    )


def test_a_tap_at_one_layer_leaves_another_linear_layers_mixer_alone(
    qwen35moe_bundle,
):
    """The globals are swapped process-wide while installed and the fixture has
    three linear layers, so this is the scoping test that can actually fail
    (a single-layer version cannot): tap layer 0, layer 1's mixer output must be
    bit-identical."""
    bundle = qwen35moe_bundle
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    other = bundle.mixer_at(1)
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
        executor_for(_read_doc("delta_value"), bundle, base_texts=[TEXT]).read_value(
            "r"
        )
        with torch.no_grad():
            bundle.model(**encoded)
        after = seen["t"]
    finally:
        handle.remove()
    assert torch.equal(after, clean)


# --------------------------------------------------------------------------- #
# containment: the guards the design has to carry
# --------------------------------------------------------------------------- #


def test_all_four_globals_are_restored_on_exit(qwen35moe_bundle):
    import importlib

    from causalab.neural.engines.pytorch_hooks.delta_interface import (
        DeltaTap,
        delta_kernel_taps,
    )

    mixer = qwen35moe_bundle.mixer_at(DELTANET_LAYER)
    modeling = importlib.import_module(type(mixer).__module__)
    names = (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    )
    before = {name: getattr(modeling, name) for name in names}
    with delta_kernel_taps({mixer: (DeltaTap(slot="value", read=lambda _t: None),)}):
        assert (
            getattr(modeling, "torch_chunk_gated_delta_rule")
            is not before["torch_chunk_gated_delta_rule"]
        )
    for name in names:
        assert getattr(modeling, name) is before[name], name


def test_a_kernelized_mixer_is_refused_by_name():
    """``kernelize()`` replaces the class forward wholesale, and no
    module-global patch applies inside a hub kernel — detected as a forward
    defined outside the mixer's own modeling module."""
    from causalab.neural.engines.pytorch_hooks.delta_interface import (
        DeltaTap,
        delta_kernel_taps,
    )

    class Kernelized(torch.nn.Module):
        pass

    # a forward from another module, the shape kernelize() leaves behind
    Kernelized.forward = torch.nn.functional.relu
    with pytest.raises(ProtocolError, match="kernelize"):
        with delta_kernel_taps(
            {Kernelized(): (DeltaTap(slot="value", read=lambda _t: None),)}
        ):
            pass  # pragma: no cover — entry refuses


def test_a_family_without_the_kernel_globals_is_refused_by_name():
    """A modeling file that does not export the four globals cannot be tapped —
    refused rather than served another family's kernels."""
    from causalab.neural.engines.pytorch_hooks.delta_interface import (
        DeltaTap,
        delta_kernel_taps,
    )

    class NoKernels(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    with pytest.raises(ProtocolError, match="causal_conv1d_fn"):
        with delta_kernel_taps(
            {NoKernels(): (DeltaTap(slot="value", read=lambda _t: None),)}
        ):
            pass  # pragma: no cover — entry refuses


# --------------------------------------------------------------------------- #
# head bounds and stream refusals extend to the kernel boundary
# --------------------------------------------------------------------------- #


def test_the_gates_feature_axis_is_the_head_axis(qwen35moe_bundle):
    assert component_shape(qwen35moe_bundle.info, "delta_beta").head_space == 8
    value = executor_for(
        _read_doc("delta_beta", head=5), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 1)


def test_head_on_the_conv_is_refused_like_the_qkv(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="no head axis"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_conv", layers=(DELTANET_LAYER,), head=0),
        )


@pytest.mark.parametrize("component", TIER_TWO)
def test_kernel_components_refuse_on_a_full_attention_layer(
    qwen35moe_bundle, component: str
):
    with pytest.raises(ProtocolError, match="delta-rule state"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component=component, layers=(FULL_ATTENTION_LAYER,)),
        )


# --------------------------------------------------------------------------- #
# generated frames: decode natively runs the recurrent kernel and conv-update
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component",
    ("delta_conv", "delta_key", "delta_value", "delta_beta", "delta_kernel_output"),
)
def test_a_continuation_read_accumulates_at_the_kernel_boundary(
    qwen35moe_bundle, component: str
):
    """📐 Every cached decode step runs the recurrent kernel and conv-update
    natively (chunk 3 / recurrent 6 / conv_fn 3 / conv_update 6 over a 3-token
    generate), and both are wrapped by the same taps — so per-step shapes are
    constant and the steps stack. Note ``delta_key`` accumulates here, unlike
    ``attention_key``: the kernel receives one step's k, not the prefix."""
    doc = _read_doc(component)
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 3}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == (1, 3, TIER_TWO_WIDTH[component])


# --------------------------------------------------------------------------- #
# tier 3 — the per-step interior: state, readout, update
# --------------------------------------------------------------------------- #

STATE_TIER = ("delta_kv_mem", "delta_state_update", "delta_state")


def _state_write_doc(do: dict, *, pos: dict | str = "all") -> dict:
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {
                "original": {"input": "base", "reads": ["v", "clean"]},
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {"component": "delta_state", "layers": [DELTANET_LAYER]},
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v": {"site": "tap", "pos": pos},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {"patch": {"site": "tap", "pos": pos, "do": do}},
            "save": [
                saved("v", "original", "v.safetensors"),
                saved("clean", "original", "clean.safetensors"),
                saved("after", "patched", "after.safetensors"),
            ],
        },
    }


def test_the_state_reads_whole_matrices_per_step(qwen35moe_bundle):
    """The second ``is_feature_space == False`` citizen after the attention
    pattern — but unlike the pattern it keeps its one position axis, so the
    gather works on the steps axis and the value keeps its native layout:
    one d_k × d_v matrix per head per step."""
    value = executor_for(
        _read_doc("delta_state"), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 8, 32, 32)


@pytest.mark.parametrize("component", ("delta_kv_mem", "delta_state_update"))
def test_the_derived_faces_read_as_ordinary_head_spaces(
    qwen35moe_bundle, component: str
):
    value = executor_for(
        _read_doc(component), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 256)


def test_a_state_read_leaves_the_base_forward_bit_identical(qwen35moe_bundle):
    """The shadow's headline: the chunked kernel still runs and the loop runs in its
    *shadow* — reads cost O(seq) extra kernel calls at the tapped layer only,
    and the model's own numbers do not move at all."""
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        clean = qwen35moe_bundle.model(**encoded).logits.clone()
    executor_for(
        _read_doc("delta_state"), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    with torch.no_grad():
        after = qwen35moe_bundle.model(**encoded).logits.clone()
    assert torch.equal(after, clean)


def test_the_reconstruction_identity_pins_the_derived_faces(qwen35moe_bundle):
    """Tier 3: ``S_t == S_{t-1}·exp(g_t) + k̂_t ⊗ delta_t`` at every step,
    exactly — the two-lines-pinned-by-identity pattern, with the pin
    against the *library's own returned states* rather than a golden. The
    identity — which faces reconstruct the state, and the fp32 tolerance — is
    the family adapter's declared row (``registry.identity``), so the
    same suite holds a new family's plugin to the identity it declares."""
    import importlib

    from causalab.neural.shared.sites import adapter_of
    from causalab.protocol.registry import identity

    declared = identity(adapter_of(qwen35moe_bundle).family, "delta_state")
    assert declared.name == "delta_state_recurrence"
    assert set(declared.inputs) == {"delta_decay", "delta_key", "delta_state_update"}
    atol, rtol = declared.tolerance_for(qwen35moe_bundle.dtype)

    reads = _multi_read(
        qwen35moe_bundle,
        {
            "S": "delta_state",
            "du": "delta_state_update",
            "k": "delta_key",
            "g": "delta_decay",
        },
    )
    modeling = importlib.import_module(
        type(qwen35moe_bundle.mixer_at(DELTANET_LAYER)).__module__
    )
    states = reads["S"]  # (1, 5, 8, 32, 32)
    update = reads["du"].reshape(1, 5, 8, 32)
    k_hat = modeling.l2norm(reads["k"].reshape(1, 5, 8, 32), dim=-1, eps=1e-6)
    decay = reads["g"]  # (1, 5, 8)
    previous = torch.zeros(8, 32, 32)
    for t in range(5):
        predicted = (
            previous * decay[0, t].exp()[:, None, None]
            + k_hat[0, t][..., None] * update[0, t][:, None, :]
        )
        torch.testing.assert_close(predicted, states[0, t], atol=atol, rtol=rtol)
        previous = states[0, t]


def test_stepping_the_recurrent_kernel_reproduces_the_chunked_output(
    qwen35moe_bundle,
):
    """Tier 3: the stacked stepwise output equals the tier-2
    ``delta_kernel_output`` to ≤ 1e-9 (📐 4.66e-10) — the two kernels compute
    the same function in fp32, which is what makes the shadow's states faithful
    to what the chunked path computed."""
    import importlib

    reads = _multi_read(
        qwen35moe_bundle,
        {
            "q": "delta_query",
            "k": "delta_key",
            "v": "delta_value",
            "beta": "delta_beta",
            "g": "delta_decay",
            "out": "delta_kernel_output",
        },
    )
    modeling = importlib.import_module(
        type(qwen35moe_bundle.mixer_at(DELTANET_LAYER)).__module__
    )
    heads = lambda t: t.reshape(1, 5, 8, 32)  # noqa: E731 — local reshape
    state = None
    outs = []
    for t in range(5):
        out_t, state = modeling.torch_recurrent_gated_delta_rule(
            heads(reads["q"])[:, t : t + 1],
            heads(reads["k"])[:, t : t + 1],
            heads(reads["v"])[:, t : t + 1],
            g=reads["g"][:, t : t + 1],
            beta=reads["beta"][:, t : t + 1],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(out_t)
    stacked = torch.cat(outs, dim=1).reshape(1, 5, 256)
    assert float((stacked - reads["out"]).abs().max()) <= 1e-9


# --------------------------------------------------------------------------- #
# state writes: the one deliberate path-forcing in the vocabulary
# --------------------------------------------------------------------------- #


def test_a_state_self_swap_costs_exactly_the_substitution_drift(qwen35moe_bundle):
    """Writing the state's own value back substitutes the stepwise loop for
    the chunked call — path-forcing, the one place the exact-identity bar is
    deliberately traded away — and costs 📐 ~5e-7 on the logits, pinned as a
    bound: more would mean the loop drifted, zero would mean the substitution
    silently did not run."""
    executor = executor_for(
        _state_write_doc({"swap": "v"}), qwen35moe_bundle, base_texts=[TEXT]
    )
    drift = float(
        (executor.read_value("after") - executor.read_value("clean")).abs().max()
    )
    assert 0.0 < drift <= 1e-6


def test_doubling_the_state_moves_the_logits(qwen35moe_bundle):
    executor = executor_for(
        _state_write_doc({"add_scaled": {"op": "v", "alpha": 1.0}}),
        qwen35moe_bundle,
        base_texts=[TEXT],
    )
    moved = float(
        (executor.read_value("after") - executor.read_value("clean")).abs().max()
    )
    assert moved > 1e-3


def test_a_positional_state_write_lands_on_its_step_and_feeds_forward(
    qwen35moe_bundle,
):
    """Doubling the state at step 2 only: steps 0–1 are bit-identical (the
    substitution reproduces the unedited prefix exactly), step 2 is exactly
    doubled, and step 3 moves *causally* — the edit threads into what the next
    step decays and writes into."""
    doc = _state_write_doc({"add_scaled": {"op": "v", "alpha": 1.0}}, pos={"index": 2})
    for name, model in (("v_all", "original"), ("obs", "patched")):
        doc["method"]["reads"][name] = {"site": "tap", "pos": "all"}
        doc["method"]["intervened_models"][model]["reads"].append(name)
        doc["method"]["save"].append(saved(name, model, f"{name}.safetensors"))
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    v_all, obs = executor.read_value("v_all"), executor.read_value("obs")
    assert float((obs[0, :2] - v_all[0, :2]).abs().max()) == 0.0
    assert float((obs[0, 2] - 2 * v_all[0, 2]).abs().max()) == 0.0
    assert float((obs[0, 3] - v_all[0, 3]).abs().max()) > 1e-4


def test_a_renormalized_state_write_keeps_the_steps_pre_write_norm(
    qwen35moe_bundle,
):
    """``add_scaled`` + ``renormalize`` on the state at step 2: the edited
    matrix keeps the norm the step had before either write (§2.8), over the
    flattened ``heads·d_k·d_v`` row the per-step writer edits. The per-step
    writer's renormalize was once the identity, since it read the running
    edit as its reference."""
    doc = _state_write_doc({"add_scaled": {"op": 1.0, "alpha": 1.0}}, pos={"index": 2})
    doc["method"]["writes"]["renorm"] = {
        "site": "tap",
        "pos": {"index": 2},
        "do": {"renormalize": True},
    }
    doc["method"]["intervened_models"]["patched"]["writes"].append("renorm")
    for name, model in (("v_all", "original"), ("obs", "patched")):
        doc["method"]["reads"][name] = {"site": "tap", "pos": "all"}
        doc["method"]["intervened_models"][model]["reads"].append(name)
        doc["method"]["save"].append(saved(name, model, f"{name}.safetensors"))
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    v_all, obs = executor.read_value("v_all"), executor.read_value("obs")
    pre, written = v_all[0, 2].flatten(), obs[0, 2].flatten()
    torch.testing.assert_close(written.norm(), pre.norm(), atol=1e-5, rtol=1e-4)
    # anti-vacuity: the step moved, and the bumped norm is far from the target
    assert float((written - pre).abs().max()) > 1e-3
    assert float((pre + 1.0).norm() - pre.norm()) > 1.0


def test_a_misaligned_state_operand_is_refused_not_broadcast(qwen35moe_bundle):
    """The j-th operand step lands on the j-th addressed step, so both sides
    must cover the same steps — refused by name rather than silently applying
    step 0's matrix to step 2."""
    doc = _state_write_doc({"add_scaled": {"op": "v", "alpha": 1.0}}, pos={"index": 2})
    doc["method"]["reads"]["v"]["pos"] = "all"  # 5 steps against a 1-step write
    with pytest.raises(ProtocolError, match="same steps"):
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("after")


@pytest.mark.parametrize("component", ("delta_kv_mem", "delta_state_update"))
def test_the_derived_faces_are_read_only_with_the_lowering_refusal(
    qwen35moe_bundle, component: str
):
    """A memory readout has no independent existence, and the
    update's write lowers exactly onto a state edit — both refusals point at
    'delta_state'."""
    doc = _state_write_doc({"swap": "v"})
    doc["method"]["sites"]["tap"]["component"] = component
    with pytest.raises(ProtocolError, match="delta_state") as excinfo:
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("after")
    assert "no write may change" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the state's sub-axes and refusals
# --------------------------------------------------------------------------- #


def test_a_head_selects_one_matrix_stack(qwen35moe_bundle):
    value = executor_for(
        _read_doc("delta_state", head=2), qwen35moe_bundle, base_texts=[TEXT]
    ).read_value("r")
    assert tuple(value.shape) == (1, 5, 32, 32)


def test_an_out_of_range_state_head_is_refused(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="8 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="delta_state", layers=(DELTANET_LAYER,), head=8),
        )


def test_featurizer_and_dims_are_refused_on_the_state(qwen35moe_bundle):
    """Generated from the declared axes: the trailing axes form a matrix per
    head, not a feature vector, so there is no basis to fit or index."""
    doc = _read_doc("delta_state")
    doc["method"]["featurizers"] = {"f": {"kind": "standardize"}}
    doc["method"]["reads"]["r"]["featurizer"] = "f"
    with pytest.raises(ProtocolError, match="matrix"):
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    doc = _read_doc("delta_state")
    doc["method"]["reads"]["r"]["dims"] = [0, 1]
    with pytest.raises(ProtocolError, match="matrix|features"):
        executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")


# --------------------------------------------------------------------------- #
# generated frames: decode needs none of the machinery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component, shape",
    [
        ("delta_state", (1, 3, 8, 32, 32)),
        ("delta_kv_mem", (1, 3, 256)),
        ("delta_state_update", (1, 3, 256)),
    ],
)
def test_generated_state_reads_are_plain_per_step_captures(
    qwen35moe_bundle, component: str, shape: tuple[int, ...]
):
    """📐 Every cached decode step runs the recurrent kernel natively, so the
    per-step interior in the generated frame is the stock path — the state is
    each call's own return, the faces derive from its initial_state and
    arguments, and nothing is substituted or shadowed."""
    doc = _read_doc(component)
    doc["method"]["positions"] = {
        "window": {"generated": {"max_new_tokens": 3}, "all": True}
    }
    doc["method"]["reads"]["r"]["pos"] = "window"
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == shape


def test_the_cross_path_pin_decode_states_match_test_side_stepping(
    qwen35moe_bundle,
):
    """The cross-path pin: the decode's *native* recurrent path produces the
    same per-step states as stepping the kernel over the decode's own inputs
    from the prefill's final state — the prompt-frame shadow and the
    decode-frame captures are two faces of one computation."""
    import importlib

    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "positions": {
                "window": {"generated": {"max_new_tokens": 3}, "all": True},
                "last": {"index": -1},
            },
            "sites": {
                f"{name}_site": {"component": component, "layers": [DELTANET_LAYER]}
                for name, component in {
                    "S": "delta_state",
                    "q": "delta_query",
                    "k": "delta_key",
                    "v": "delta_value",
                    "beta": "delta_beta",
                    "g": "delta_decay",
                }.items()
            },
            "reads": {
                name: {"site": f"{name}_site", "pos": "window"}
                for name in ("S", "q", "k", "v", "beta", "g")
            },
        },
    }
    doc["method"]["reads"]["S0"] = {"site": "S_site", "pos": "last"}
    doc["method"]["intervened_models"] = {
        "original": {"input": "base", "reads": list(doc["method"]["reads"])}
    }
    doc["method"]["save"] = [
        saved(name, "original", f"{name}.safetensors")
        for name in doc["method"]["reads"]
    ]
    executor = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT])
    reads = {name: executor.read_value(name) for name in doc["method"]["reads"]}
    modeling = importlib.import_module(
        type(qwen35moe_bundle.mixer_at(DELTANET_LAYER)).__module__
    )

    def heads(name: str) -> torch.Tensor:
        return reads[name].reshape(1, 3, 8, 32)

    state = reads["S0"][:, 0]  # the prefill's final state, (1, 8, 32, 32)
    for t in range(3):
        _, state = modeling.torch_recurrent_gated_delta_rule(
            heads("q")[:, t : t + 1],
            heads("k")[:, t : t + 1],
            heads("v")[:, t : t + 1],
            g=reads["g"][:, t : t + 1],
            beta=reads["beta"][:, t : t + 1],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # ⚠️ Not exact, and honestly so: the decode's native path starts from
        # the CHUNKED prefill's cache state, while the prompt-frame S0 read is
        # the recurrent shadow's — the two kernels agree to 📐 4.66e-10, and
        # three steps of drift measured 1.9e-9. The bound is the cross-path
        # agreement, not machinery slack.
        torch.testing.assert_close(state, reads["S"][:, t], atol=1e-8, rtol=0.0)
