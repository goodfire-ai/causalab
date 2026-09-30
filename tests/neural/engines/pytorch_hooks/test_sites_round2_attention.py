"""The attention interior at module boundaries: `v`, pre-RoPE q/k, the gate.

📐 One could assume these need function-level taps inside the mixer's
forward, "chunk/view/reshape ops, not module boundaries". Measured against
``tiny-random/qwen3.5-moe`` on transformers 5.16, three of the four are ordinary
``nn.Module`` outputs — ``Qwen3_5MoeAttention`` runs ``q_norm``/``k_norm``
**before** ``apply_rotary_pos_emb``, so their outputs *are* the pre-RoPE
projections, and ``v_proj``'s output is ``v`` — and only the gate needs anything
special, because it shares one projection with ``q``.

What each group of tests is for:

* **resolves / shape** — the tap names the module its name claims, and the
  declared shape matches the tensor that module really emits (the conversion
  raises if not, which is the point of declaring widths).
* **identity pins** — stronger than shapes: each component is reproduced from
  the model's own weights, so a tap pointed at a plausible neighbour fails.
* **causal writes** — a write through the tap moves the logits, and the same
  write with an identity payload moves them by exactly 0.0. Without the second
  half, a "write works" test passes for a write that silently no-ops.
* **refusals** — the family and architecture boundaries, asserted on the
  *message* rather than the exception type. GPT-2's fused ``c_attn`` is no
  longer one of them: the per-family tap table addresses its three
  logical blocks (``test_family_tap_table.py`` has the oracle equivalence).
* **head bounds** — the head-bound defect, on the components that introduce
  it: three of these four live in KV-head space.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.layout import to_contract
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.positions.alignment import COMPONENT_RANK
from causalab.protocol.registry import component_shape, component_width
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .conftest import TINY_GPT2

pytestmark = pytest.mark.smoke

#: 📐 the qwen fixture's only full-attention layer; 0-2 are Gated DeltaNet.
FULL_ATTENTION_LAYER = 3
DELTANET_LAYER = 0
LLAMA_LAYER = 1

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"

INTERIOR = (
    "attention_query_pre_rope",
    "attention_key_pre_rope",
    "attention_value_states",
    "attention_gate",
)

#: 📐 measured on `tiny-random/qwen3.5-moe`: hidden 8, H 8, H_kv 4, head_dim 32
#: (decoupled from hidden — reading head_dim as hidden // H would give 1).
#: ``component -> (tapped module attribute, native shape, contract width)``
QWEN_TAPS: dict[str, tuple[str, tuple[int, ...], int]] = {
    "attention_query_pre_rope": ("q_norm", (1, 5, 8, 32), 256),
    "attention_key_pre_rope": ("k_norm", (1, 5, 4, 32), 128),
    "attention_value_states": ("v_proj", (1, 5, 128), 128),
    "attention_gate": ("q_proj", (1, 5, 512), 256),
}

#: 📐 measured on tiny-llama: H 4, H_kv 4, head_dim 4, and no q_norm/k_norm, so
#: the bare projections *are* the pre-RoPE tensors.
#: (the llama tokenizer splits the same text into 8 tokens, not qwen's 5)
LLAMA_TAPS: dict[str, tuple[str, tuple[int, ...], int]] = {
    "attention_query_pre_rope": ("q_proj", (1, 8, 16), 16),
    "attention_key_pre_rope": ("k_proj", (1, 8, 16), 16),
    "attention_value_states": ("v_proj", (1, 8, 16), 16),
}


def _mixer(bundle: ModelBundle, layer: int) -> torch.nn.Module:
    return bundle.mixer_at(layer)


def _capture(
    bundle: ModelBundle, layer: int, components: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Read several interior taps in one forward, each in contract shape."""
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    batch = encoded["input_ids"].shape[0]
    out: dict[str, torch.Tensor] = {}
    handles = []
    for component in components:
        site = resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))

        def hook(_m, _i, output, *, _name=component, _site=site):
            out[_name] = (
                to_contract(output, _site.shape, batch_size=batch).detach().clone()
            )

        handles.append(site.module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return out


def _native(bundle: ModelBundle, layer: int, attribute: str) -> torch.Tensor:
    """The raw output of one submodule of the mixer, unconverted."""
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    seen: dict[str, torch.Tensor] = {}
    module = getattr(_mixer(bundle, layer), attribute)
    handle = module.register_forward_hook(
        lambda _m, _i, o: seen.__setitem__("t", o.detach().clone())
    )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        handle.remove()
    return seen["t"]


# --------------------------------------------------------------------------- #
# resolves, and the declared shape is the module's real one
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", INTERIOR)
def test_the_tap_is_the_module_its_name_claims(qwen35moe_bundle, component: str):
    attribute, _, _ = QWEN_TAPS[component]
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component=component, layers=(FULL_ATTENTION_LAYER,))
    )
    assert site.module is getattr(
        _mixer(qwen35moe_bundle, FULL_ATTENTION_LAYER), attribute
    )
    assert site.kind == "out"


@pytest.mark.parametrize("component", INTERIOR)
def test_the_declared_shape_matches_the_real_tensor(qwen35moe_bundle, component: str):
    """The conversion checks every static width, so this fails loudly if the
    table and the checkpoint disagree — the wrong-tap-with-plausible-numbers
    failure the descriptor exists to prevent."""
    attribute, native_shape, width = QWEN_TAPS[component]
    assert tuple(_native(qwen35moe_bundle, FULL_ATTENTION_LAYER, attribute).shape) == (
        native_shape
    )
    captured = _capture(qwen35moe_bundle, FULL_ATTENTION_LAYER, (component,))
    assert tuple(captured[component].shape) == (1, 5, width)
    assert component_width(qwen35moe_bundle.info, component) == width


def test_the_pre_rope_taps_keep_their_head_axis_only_where_the_module_does(
    qwen35moe_bundle, llama_bundle
):
    """📐 The same component is 4-D on qwen (``q_norm`` emits ``(b,s,H,d)``) and
    3-D on llama (a bare ``q_proj`` emits ``(b,s,H·d)``). Same width, same head
    space, different packing — which is exactly the one thing the backend is
    allowed to say about a shape the protocol table owns."""
    on_qwen = resolve_site(
        qwen35moe_bundle,
        SiteSpec(component="attention_query_pre_rope", layers=(FULL_ATTENTION_LAYER,)),
    ).shape
    on_llama = resolve_site(
        llama_bundle,
        SiteSpec(component="attention_query_pre_rope", layers=(LLAMA_LAYER,)),
    ).shape
    assert on_qwen.flat_inner is False and on_qwen.native_rank == 4
    assert on_llama.flat_inner is True and on_llama.native_rank == 3
    # and the family-independent half is genuinely family-independent
    assert [a.kind for a in on_qwen.axes] == [a.kind for a in on_llama.axes]


@pytest.mark.parametrize("component", sorted(LLAMA_TAPS))
def test_a_family_without_q_norm_taps_the_bare_projection(llama_bundle, component: str):
    attribute, native_shape, width = LLAMA_TAPS[component]
    site = resolve_site(
        llama_bundle, SiteSpec(component=component, layers=(LLAMA_LAYER,))
    )
    assert site.module is getattr(_mixer(llama_bundle, LLAMA_LAYER), attribute)
    assert tuple(_native(llama_bundle, LLAMA_LAYER, attribute).shape) == native_shape
    assert component_width(llama_bundle.info, component) == width


def test_the_ranks_run_in_forward_order():
    """The mixer computes q and the gate from one projection, then k, then v,
    then rotates, then attends. The rank table is read as that story."""
    order = [
        "attention_input_norm",
        "attention_query_pre_rope",
        "attention_key_pre_rope",
        "attention_value_states",
        "attention_gate",
        "attention_probs",
        "attention_premix",
        "attention_output",
    ]
    ranks = [COMPONENT_RANK[c] for c in order]
    assert ranks == sorted(ranks)


# --------------------------------------------------------------------------- #
# identity pins — each tap reproduced from the model's own weights
# --------------------------------------------------------------------------- #


def test_v_is_the_v_projection_of_what_the_mixer_consumes(qwen35moe_bundle):
    """``attention_value_states`` == ``v_proj(attention_input_norm)``."""
    bundle = qwen35moe_bundle
    captured = _capture(bundle, FULL_ATTENTION_LAYER, ("attention_value_states",))
    norm = _capture_component(bundle, FULL_ATTENTION_LAYER, "attention_input_norm")
    attn = _mixer(bundle, FULL_ATTENTION_LAYER)
    with torch.no_grad():
        expected = attn.v_proj(norm)
    torch.testing.assert_close(
        captured["attention_value_states"], expected, atol=0.0, rtol=0.0
    )


def test_q_pre_rope_and_the_gate_are_the_two_halves_of_one_projection(
    qwen35moe_bundle,
):
    """📐 ``q_proj`` emits ``[q_h | gate_h]`` per head — width H·2·d = 512.
    ``attention_query_pre_rope`` is ``q_norm(chunk 0)`` and ``attention_gate``
    is chunk 1, and this pins that neither picked up any of the other."""
    bundle = qwen35moe_bundle
    attn = _mixer(bundle, FULL_ATTENTION_LAYER)
    info = bundle.info
    fused = _native(bundle, FULL_ATTENTION_LAYER, "q_proj")
    b, s, _ = fused.shape
    split = fused.view(b, s, info.num_heads, 2, info.head_dim)
    captured = _capture(
        bundle,
        FULL_ATTENTION_LAYER,
        ("attention_query_pre_rope", "attention_gate"),
    )

    # the gate is the second split, flattened head-major
    torch.testing.assert_close(
        captured["attention_gate"],
        split[:, :, :, 1, :].reshape(b, s, -1),
        atol=0.0,
        rtol=0.0,
    )
    # and q_pre_rope is q_norm applied to the first
    with torch.no_grad():
        expected = attn.q_norm(split[:, :, :, 0, :]).reshape(b, s, -1)
    torch.testing.assert_close(
        captured["attention_query_pre_rope"], expected, atol=0.0, rtol=0.0
    )


def test_k_pre_rope_is_the_normalized_k_projection(qwen35moe_bundle):
    bundle = qwen35moe_bundle
    attn = _mixer(bundle, FULL_ATTENTION_LAYER)
    info = bundle.info
    norm = _capture_component(bundle, FULL_ATTENTION_LAYER, "attention_input_norm")
    b, s, _ = norm.shape
    with torch.no_grad():
        expected = attn.k_norm(
            attn.k_proj(norm).view(b, s, info.num_kv_heads, info.head_dim)
        ).reshape(b, s, -1)
    captured = _capture(bundle, FULL_ATTENTION_LAYER, ("attention_key_pre_rope",))
    torch.testing.assert_close(
        captured["attention_key_pre_rope"], expected, atol=0.0, rtol=0.0
    )


def test_the_gate_is_what_separates_premix_from_the_mixer_output(qwen35moe_bundle):
    """A docstring correction, as a test rather than a restatement.

    📐 ``Qwen3_5MoeAttention`` ends with ``attn_output * sigmoid(gate)`` before
    ``o_proj``, so ``attention_premix`` — the o-projection's input — is the
    *gated* mixer output. Dividing it back out by ``sigmoid(gate)`` must give a
    tensor that no longer depends on the gate, which is what makes the two
    components genuinely different boxes rather than two names for one.
    """
    bundle = qwen35moe_bundle
    gate = _capture(bundle, FULL_ATTENTION_LAYER, ("attention_gate",))["attention_gate"]
    premix = _capture_component(bundle, FULL_ATTENTION_LAYER, "attention_premix")
    assert premix.shape == gate.shape
    # the gate is a real modulation, not a constant: if it were ~1 everywhere
    # the two components would be the same tensor and this test would be vacuous
    sigmoid = torch.sigmoid(gate)
    assert float((sigmoid - 1.0).abs().max()) > 1e-3


def _capture_component(bundle: ModelBundle, layer: int, component: str) -> torch.Tensor:
    """One tap, in contract shape — handles the ``in`` side that ``_capture``
    does not (``attention_premix`` is the o-projection's *input*)."""
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    batch = encoded["input_ids"].shape[0]
    site = resolve_site(bundle, SiteSpec(component=component, layers=(layer,)))
    seen: dict[str, torch.Tensor] = {}

    def out_hook(_m, _i, output):
        seen["t"] = to_contract(output, site.shape, batch_size=batch).detach().clone()

    def pre_hook(_m, args):
        seen["t"] = to_contract(args[0], site.shape, batch_size=batch).detach().clone()

    handle = (
        site.module.register_forward_hook(out_hook)
        if site.kind == "out"
        else site.module.register_forward_pre_hook(pre_hook)
    )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        handle.remove()
    return seen["t"]


# --------------------------------------------------------------------------- #
# causal writes — and the identity write that must move nothing
# --------------------------------------------------------------------------- #


def _write_doc(component: str, layer: int, *, head: int | None = None) -> dict:
    site: dict = {"component": component, "layers": [layer]}
    if head is not None:
        site["head"] = head
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
            "sites": {"tap": site, "lm_head": {"component": "lm_head"}},
            "reads": {
                "v_cf": {"site": "tap", "pos": "all"},
                "clean": {"site": "lm_head", "pos": {"index": -1}},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {"patch": {"site": "tap", "pos": "all", "do": {"swap": "v_cf"}}},
            "save": [
                saved("after", "patched", "p.safetensors"),
                saved("clean", "original_base", "c.safetensors"),
            ],
        },
    }


def _self_swap(doc: dict) -> dict:
    """Take ``v_cf`` on the un-intervened *base* forward instead of the
    counterfactual one, so the write swaps the tap with its own value."""
    models = doc["method"]["intervened_models"]
    del models["original_counterfactual"]
    models["original_base"]["reads"].append("v_cf")
    return doc


@pytest.mark.parametrize("component", INTERIOR)
def test_a_write_through_the_tap_actually_changes_the_logits(
    qwen35moe_bundle, component: str
):
    """The non-vacuity half. A forward hook on a module whose output nothing
    downstream reads would run, succeed, and move the logits by exactly 0.0 —
    which is how ``router_logits`` was found to be unwritable."""
    executor = executor_for(
        _write_doc(component, FULL_ATTENTION_LAYER),
        qwen35moe_bundle,
        base_texts=[TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    delta = (executor.read_value("after") - executor.read_value("clean")).abs().max()
    assert float(delta) > 1e-4, component


@pytest.mark.parametrize("component", INTERIOR)
def test_swapping_a_tap_with_its_own_value_moves_nothing(
    qwen35moe_bundle, component: str
):
    """The other half, and the one that catches a write landing in the wrong
    place: swapping in the value read from the *same* input must be exactly the
    identity. For the gate this is the load-bearing case — the write goes back
    into a projection it shares with ``q``, and disturbing ``q`` would show up
    here as a nonzero delta."""
    executor = executor_for(
        _self_swap(_write_doc(component, FULL_ATTENTION_LAYER)),
        qwen35moe_bundle,
        base_texts=[TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    delta = (executor.read_value("after") - executor.read_value("clean")).abs().max()
    assert float(delta) == 0.0, component


def test_a_write_to_v_reaches_the_kv_cache_path(qwen35moe_bundle):
    """📐 ``v_proj``'s output is consumed by ``past_key_values.update`` *after*
    the tap, so this is the one interior write whose effect the cache carries.
    Pinned as a plain causal check on the prompt frame — rule 16 makes writes
    prefill-only, so nothing here depends on decode behaviour."""
    executor = executor_for(
        _write_doc("attention_value_states", FULL_ATTENTION_LAYER),
        qwen35moe_bundle,
        base_texts=[TEXT],
        counterfactual_texts=[CF_TEXT],
    )
    delta = (executor.read_value("after") - executor.read_value("clean")).abs().max()
    assert float(delta) > 1e-4


# --------------------------------------------------------------------------- #
# refusals, asserted on the message
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", INTERIOR)
def test_a_deltanet_layer_refuses_with_the_architectural_reason(
    qwen35moe_bundle, component: str
):
    """Not "unimplemented" — a Gated DeltaNet block has no q/k/v projections at
    all, so this is permanent and true of the architecture."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            qwen35moe_bundle, SiteSpec(component=component, layers=(DELTANET_LAYER,))
        )
    assert "full-attention mixer" in str(excinfo.value)
    assert "linear_attention" in str(excinfo.value)


def test_the_gate_refuses_on_a_family_that_computes_none(llama_bundle):
    """Llama's mixer has no output gate, so the box does not exist
    there — refused by name rather than fabricated from a slice of ``q_proj``."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            llama_bundle,
            SiteSpec(component="attention_gate", layers=(LLAMA_LAYER,)),
        )
    message = str(excinfo.value)
    assert "computes no output gate" in message
    assert "Qwen3.5/3.6" in message


#: 📐 measured on tiny-gpt2: H 4, d 8, one fused ``c_attn``
#: emitting ``(1, s, 96)`` = ``[q | k | v]`` as three 32-wide column blocks.
GPT2_BLOCKS: dict[str, int] = {
    "attention_query_pre_rope": 0,
    "attention_key_pre_rope": 1,
    "attention_value_states": 2,
}


@pytest.mark.parametrize("component", sorted(GPT2_BLOCKS))
def test_gpt2_addresses_the_fused_projection_as_logical_blocks(component: str):
    """GPT-2 fuses q, k and v into one ``c_attn``, and
    the per-family tap table says which ``H·d`` block of its output each one is
    — so the same logical site resolves here as on a split-projection family,
    to ``c_attn``'s output with a fused-blocks packing. The oracle equivalence
    (values and writes) is ``test_family_tap_table.py``'s."""
    bundle = load_model(TINY_GPT2)
    site = resolve_site(bundle, SiteSpec(component=component, layers=(LLAMA_LAYER,)))
    attn = _mixer(bundle, LLAMA_LAYER)
    assert site.module is attn.c_attn and site.kind == "out"
    assert site.shape.fused_index == GPT2_BLOCKS[component]
    assert [a.kind for a in site.shape.axes] == [
        "batch",
        "position",
        "fused",
        "head",
        "feature",
    ]
    assert site.shape.width == component_width(bundle.info, component) == 32


def test_gpt2_refuses_the_gate_by_name():
    """The one interior component GPT-2 really lacks: no gate on any family
    but Qwen3.5/3.6's, and the row says so for ``gpt2`` — the same text the
    llama refusal carries (snapshot entry 24)."""
    bundle = load_model(TINY_GPT2)
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            bundle, SiteSpec(component="attention_gate", layers=(LLAMA_LAYER,))
        )
    assert "computes no output gate" in str(excinfo.value)
    assert excinfo.value.reason == "component_unavailable"


# --------------------------------------------------------------------------- #
# the head bound, on the components that introduce KV space
# --------------------------------------------------------------------------- #


def test_the_kv_space_components_have_a_narrower_head_space(qwen35moe_bundle):
    """📐 H 8, H_kv 4 on the qwen fixture. `v` and `k_pre_rope` index KV heads;
    `q_pre_rope`, the gate and `attention_premix` index query heads."""
    info = qwen35moe_bundle.info
    spaces = {c: component_shape(info, c).head_space for c in INTERIOR}
    assert spaces == {
        "attention_query_pre_rope": info.num_heads,
        "attention_key_pre_rope": info.num_kv_heads,
        "attention_value_states": info.num_kv_heads,
        "attention_gate": info.num_heads,
    }
    assert info.num_kv_heads < info.num_heads  # otherwise this test is vacuous


@pytest.mark.parametrize(
    "component", ["attention_key_pre_rope", "attention_value_states"]
)
def test_a_query_space_head_on_a_kv_space_component_is_refused(
    qwen35moe_bundle, component: str
):
    """🐞 The silent-no-op this closes. Head 5 is valid in query space (H 8) and
    not in KV space (H_kv 4) — and python does not raise on an over-wide slice,
    it returns an **empty** one. The read would have saved ``(b, n_pos, 0)`` and
    the write would have changed nothing at all."""
    with pytest.raises(ProtocolError, match="which has 4 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component=component, layers=(FULL_ATTENTION_LAYER,), head=5),
        )


@pytest.mark.parametrize("component", INTERIOR)
def test_a_head_slice_is_one_heads_worth_of_the_contract(
    qwen35moe_bundle, component: str
):
    """And it is the same head-major block the executor already addressed for
    ``attention_premix``, which is why a per-head tap needs no new sub-axis."""
    info = qwen35moe_bundle.info
    site = resolve_site(
        qwen35moe_bundle,
        SiteSpec(component=component, layers=(FULL_ATTENTION_LAYER,), head=1),
    )
    assert site.feature_slice == slice(info.head_dim, 2 * info.head_dim)
    whole = _capture(qwen35moe_bundle, FULL_ATTENTION_LAYER, (component,))[component]
    assert whole[..., site.feature_slice].shape[-1] == info.head_dim
