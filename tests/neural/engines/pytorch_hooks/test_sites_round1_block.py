"""Block components of the hookpoint vocabulary, and per-layer stream dispatch.

Four block taps — ``input_ids``, ``attention_input_norm``, ``block_mid``,
``mlp_input_norm`` — and ``attention_output``, which resolves its mixer per
layer because the mixer is a *per-layer* fact.

The shape assertions here are measurements against a real ``qwen3_5_moe``
checkpoint, so a mismatch is a finding rather than a stale expectation. The
identity assertions are stronger than shapes: they pin that each tap names the
tensor its *name* claims, by reproducing the decoder layer's residual algebra
from the taps alone.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.shared.layout import (
    LayoutError,
    from_contract,
    tap_tensor,
    to_contract,
)
from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.rules.errors import ProtocolError, ValidationError
from causalab.protocol.positions.alignment import COMPONENT_RANK
from causalab.protocol.registry import shapes as shapes
from causalab.protocol.registry import component_width
from causalab.protocol.schema import (
    COMPONENTS,
    LAYERLESS_COMPONENTS,
    PROTOCOL_VERSION,
    SiteSpec,
)

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .conftest import TINY_LLAMA, TINY_QWEN35_MOE

pytestmark = pytest.mark.smoke

#: The four block taps, plus the one that resolves its mixer per layer.
NEW_COMPONENTS = ("input_ids", "attention_input_norm", "block_mid", "mlp_input_norm")
BLOCK_COMPONENTS = NEW_COMPONENTS + ("attention_output",)

TEXT = "the quick brown fox jumps"


def _spec(component: str, layer: int | None = None, **kw: object) -> SiteSpec:
    if component in LAYERLESS_COMPONENTS:
        return SiteSpec(component=component, **kw)
    return SiteSpec(component=component, layers=(0 if layer is None else layer,), **kw)


def _capture(
    bundle: ModelBundle, specs: dict[str, SiteSpec]
) -> dict[str, torch.Tensor]:
    """Read many sites in one forward pass, each in contract shape."""
    encoded = bundle.tokenizer(TEXT, return_tensors="pt")
    batch = encoded["input_ids"].shape[0]
    out: dict[str, torch.Tensor] = {}
    handles = []
    for name, spec in specs.items():
        site = resolve_site(bundle, spec)

        def hook(_mod, inp, output=None, *, _name=name, _site=site):
            payload = inp if output is None else output
            if output is None and isinstance(payload, tuple) and len(payload) == 1:
                payload = payload[0]
            tensor = tap_tensor(payload, _site.tuple_index)
            out[_name] = (
                to_contract(tensor, _site.shape, batch_size=batch).detach().clone()
            )

        handles.append(
            site.module.register_forward_pre_hook(hook)
            if site.kind == "in"
            else site.module.register_forward_hook(hook)
        )
    try:
        with torch.no_grad():
            bundle.model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return out


# --------------------------------------------------------------------------- #
# the vocabulary itself
# --------------------------------------------------------------------------- #


def test_the_four_components_joined_the_closed_vocabulary():
    for component in NEW_COMPONENTS:
        assert component in COMPONENTS, f"{component} missing from Component"


def test_input_ids_is_layerless_and_the_others_are_not():
    """``input_ids`` is the model's *input* — there is no layer at which
    to read it, and naming one is a parse error."""
    assert "input_ids" in LAYERLESS_COMPONENTS
    for component in ("attention_input_norm", "block_mid", "mlp_input_norm"):
        assert component not in LAYERLESS_COMPONENTS


def test_the_new_ranks_order_the_block_as_the_forward_pass_runs():
    """Group elision (§4) picks a group's deepest tap, so the ranks must follow
    the decoder layer's actual order — the same order the identities below
    reproduce."""
    order = [
        "input_ids",
        "embeddings",
        "block_input",
        "attention_input_norm",
        "attention_output",
        "block_mid",
        "mlp_input_norm",
        "mlp_input",
        "mlp_output",
        "block_output",
    ]
    ranks = [COMPONENT_RANK[c] for c in order]
    assert ranks == sorted(ranks), dict(zip(order, ranks))


# --------------------------------------------------------------------------- #
# widths: three are residual-shaped, one is not a feature space at all
# --------------------------------------------------------------------------- #


def test_the_three_norm_taps_are_residual_width(qwen35moe_bundle):
    hidden = qwen35moe_bundle.info.hidden_size
    for component in ("attention_input_norm", "block_mid", "mlp_input_norm"):
        assert component_width(qwen35moe_bundle.info, component) == hidden


def test_input_ids_refuses_a_width_because_it_is_not_a_feature_space(qwen35moe_bundle):
    """Integer ids on a position axis. No featurizer can attach, and the
    refusal must say *why* rather than read as a missing table entry."""
    with pytest.raises(ValidationError) as excinfo:
        component_width(qwen35moe_bundle.info, "input_ids")
    assert "not a feature space" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the mixer is a per-layer fact
# --------------------------------------------------------------------------- #


def test_the_fixture_tower_really_is_hybrid(qwen35moe_bundle):
    """📐 The premise every assertion below rests on."""
    assert qwen35moe_bundle.streams == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )


def test_attention_output_follows_the_stream_per_layer(qwen35moe_bundle):
    """The per-layer fix. ``block.self_attn`` for every non-GPT-2 model raised
    AttributeError on 3 of these 4 layers — the tap now asks the layer."""
    for layer, stream in enumerate(qwen35moe_bundle.streams):
        site = resolve_site(qwen35moe_bundle, _spec("attention_output", layer))
        block = qwen35moe_bundle.blocks[layer]
        if stream == "full_attention":
            assert site.module is block.self_attn
        else:
            assert site.module is block.linear_attn
            # and the regression this replaces: there is no self_attn to reach
            assert not hasattr(block, "self_attn")


def test_a_site_naming_the_wrong_stream_refuses(qwen35moe_bundle):
    """``stream`` has parsed since schema.py gained it and nothing read it.
    It reads it now, and a contradiction is an error."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(
            qwen35moe_bundle,
            _spec("attention_output", 0, stream="full_attention"),
        )
    assert "layer 0" in str(excinfo.value)

    with pytest.raises(ProtocolError):
        resolve_site(
            qwen35moe_bundle,
            _spec("attention_output", 3, stream="linear_attention"),
        )


def test_a_site_naming_the_right_stream_resolves(qwen35moe_bundle):
    for layer, stream in enumerate(qwen35moe_bundle.streams):
        site = resolve_site(
            qwen35moe_bundle, _spec("attention_output", layer, stream=stream)
        )
        assert site.module is qwen35moe_bundle.mixer_at(layer)


def test_a_block_carrying_both_mixer_kinds_refuses_instead_of_guessing(
    qwen35moe_bundle,
):
    """A hypothetical block with both children must not resolve silently.

    No family the box map covers ships one, so this is a guard rather than
    a regression — but `stream_at` answers by probing children, and a fixed
    probe order would call such a block "full_attention" without a word. Every
    per-layer tap would then attach to one of its two mixers and go on
    producing plausible numbers, which is the failure mode the whole
    stream-per-layer fix exists to remove.

    Built by grafting a `linear_attn` child onto the fixture's one real
    full-attention layer, so the block is a genuine torch module in both
    respects and not a stub that only looks like one.
    """
    full_layer = qwen35moe_bundle.streams.index("full_attention")
    linear_layer = qwen35moe_bundle.streams.index("linear_attention")
    block = qwen35moe_bundle.blocks[full_layer]
    donor = qwen35moe_bundle.blocks[linear_layer].linear_attn

    block.add_module("linear_attn", donor)
    try:
        with pytest.raises(ProtocolError) as excinfo:
            qwen35moe_bundle.stream_at(full_layer)
        message = str(excinfo.value)
        assert "self_attn" in message and "linear_attn" in message
        # and mixer_at must not answer either, since it resolves through stream_at
        with pytest.raises(ProtocolError):
            qwen35moe_bundle.mixer_at(full_layer)
    finally:
        del block._modules["linear_attn"]

    # the fixture is intact again — the graft must not leak into later tests
    assert qwen35moe_bundle.stream_at(full_layer) == "full_attention"


def test_mixer_at_and_stream_at_never_disagree(qwen35moe_bundle, llama_bundle):
    """`mixer_at` picks the child that matches the stream `stream_at` named.

    They used to probe independently, in different orders (`attn` before
    `self_attn` in one, the reverse in the other) — harmless for every family
    here, and exactly the kind of duplicated table that stops being harmless
    once a family has both names.
    """
    for bundle in (qwen35moe_bundle, llama_bundle):
        for layer, stream in enumerate(bundle.streams):
            block = bundle.blocks[layer]
            mixer = bundle.mixer_at(layer)
            if stream == "full_attention":
                assert mixer is getattr(block, "self_attn", None) or mixer is getattr(
                    block, "attn", None
                )
            else:
                assert mixer is block.linear_attn


# --------------------------------------------------------------------------- #
# refuse for the permanent reason before the temporary one
# --------------------------------------------------------------------------- #


def test_attention_probs_at_a_deltanet_layer_refuses_on_the_architecture(
    qwen35moe_bundle,
):
    """A Gated DeltaNet block computes no attention matrix, so this is not a
    missing feature — it stays false now that ``attention_probs`` exists."""
    with pytest.raises(ProtocolError) as excinfo:
        resolve_site(qwen35moe_bundle, _spec("attention_probs", 0))
    message = str(excinfo.value)
    assert "full-attention mixer" in message
    assert "linear_attention" in message


def test_attention_probs_at_a_full_attention_layer_resolves(qwen35moe_bundle):
    """The other half of the ordering. This test used to assert
    ``NotImplementedError`` — a statement about missing work, which stopped
    holding once the component was implemented. What must survive is the *asymmetry*: at layer 3
    the tensor exists and the tap resolves, while at a DeltaNet layer it refuses
    for a reason that is permanent (the test above)."""
    site = resolve_site(qwen35moe_bundle, _spec("attention_probs", 3))
    assert site.module is qwen35moe_bundle.mixer_at(3)
    assert site.tuple_index == 1


# --------------------------------------------------------------------------- #
# the taps name what they claim: the decoder layer's residual algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer", [0, 3], ids=["deltanet", "full_attention"])
def test_the_taps_reproduce_the_residual_algebra(qwen35moe_bundle, layer):
    """The sanity check that shapes cannot give.

    A Qwen3.5-MoE decoder layer runs::

        h = input_layernorm(resid_pre)      -> attention_input_norm
        h = mixer(h)                        -> attention_output
        resid_mid = resid_pre + h           -> block_mid
        h = post_attention_layernorm(resid_mid) -> mlp_input_norm
        h = mlp(h)                          -> mlp_output
        resid_post = resid_mid + h          -> block_output

    So the three new taps are pinned by identities against taps that already
    existed. If ``block_mid`` were accidentally the layernorm's *output* rather
    than its input (the one-character mistake available here), the second
    identity fails.
    """
    block = qwen35moe_bundle.blocks[layer]
    got = _capture(
        qwen35moe_bundle,
        {
            name: _spec(name, layer)
            for name in (
                "block_input",
                "attention_input_norm",
                "attention_output",
                "block_mid",
                "mlp_input_norm",
                "mlp_output",
                "block_output",
            )
        },
    )

    torch.testing.assert_close(
        got["attention_input_norm"], block.input_layernorm(got["block_input"])
    )
    torch.testing.assert_close(
        got["block_mid"], got["block_input"] + got["attention_output"]
    )
    torch.testing.assert_close(
        got["mlp_input_norm"], block.post_attention_layernorm(got["block_mid"])
    )
    torch.testing.assert_close(
        got["block_output"], got["block_mid"] + got["mlp_output"]
    )


@pytest.mark.parametrize("layer", [0, 3], ids=["deltanet", "full_attention"])
def test_every_new_block_tap_is_residual_shaped(qwen35moe_bundle, layer):
    """📐 Every box in the DecoderLayer panel of
    ``docs/qwen36-35b-a3b-architecture.html`` is ``(1, 6, 8)`` on
    this fixture — 6 positions of hidden size 8."""
    got = _capture(
        qwen35moe_bundle,
        {
            n: _spec(n, layer)
            for n in ("attention_input_norm", "block_mid", "mlp_input_norm")
        },
    )
    hidden = qwen35moe_bundle.info.hidden_size
    for name, tensor in got.items():
        assert tensor.dim() == 3, name
        assert tensor.shape[0] == 1 and tensor.shape[-1] == hidden, (name, tensor.shape)


def test_the_three_norm_taps_are_three_different_tensors(qwen35moe_bundle):
    """Two of them share a module and differ only by side, so a copy-paste slip
    would make them alias — the failure a shape assertion cannot see."""
    got = _capture(
        qwen35moe_bundle,
        {
            n: _spec(n, 0)
            for n in ("attention_input_norm", "block_mid", "mlp_input_norm")
        },
    )
    values = list(got.values())
    for i, first in enumerate(values):
        for second in values[i + 1 :]:
            assert not torch.equal(first, second)


# --------------------------------------------------------------------------- #
# input_ids
# --------------------------------------------------------------------------- #


def test_input_ids_reads_the_ids_the_tokenizer_produced(qwen35moe_bundle):
    encoded = qwen35moe_bundle.tokenizer(TEXT, return_tensors="pt")
    got = _capture(qwen35moe_bundle, {"input_ids": _spec("input_ids")})["input_ids"]
    # contract shape: the degenerate feature axis of width 1
    assert got.shape == (*encoded["input_ids"].shape, 1)
    assert not got.dtype.is_floating_point, "token ids are integers"
    torch.testing.assert_close(got.squeeze(-1), encoded["input_ids"])


def test_input_ids_taps_the_embedding_input_on_both_families():
    """The ids cross exactly one module boundary, and it differs by family."""
    for key, attr in (
        (TINY_QWEN35_MOE, "embed_tokens"),
        (TINY_LLAMA, "embed_tokens"),
    ):
        bundle = load_model(key)
        site = resolve_site(bundle, _spec("input_ids"))
        assert site.module is getattr(bundle.model.model, attr)
        assert site.kind == "in"
        assert site.shape == shapes.bs(integral=True, note=site.shape.note)


# --------------------------------------------------------------------------- #
# the shape with no feature axis
# --------------------------------------------------------------------------- #

#: ``input_ids``' shape: ``(batch, position)``, one integer per position.
_BS = shapes.bs(integral=True)


def test_a_featureless_tap_round_trips_and_returns_a_view():
    """``to_contract`` must return a view, so an in-place
    edit reaches native storage. ``unsqueeze`` does; ``reshape`` might not."""
    native = torch.arange(12).reshape(3, 4)
    contract = to_contract(native, _BS, batch_size=3)
    assert contract.shape == (3, 4, 1)
    assert contract._base is native or contract.data_ptr() == native.data_ptr()
    torch.testing.assert_close(from_contract(contract, _BS, batch_size=3), native)


@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.zeros(2, 3, 4)], ids=["1d", "3d"]
)
def test_a_featureless_tap_refuses_a_shape_it_cannot_mean(bad):
    """A declared shape that contradicts the tensor raises rather than
    reinterpreting — the wrong-tap-with-plausible-numbers failure."""
    with pytest.raises(LayoutError):
        to_contract(bad, _BS, batch_size=2)


# --------------------------------------------------------------------------- #
# family parity: the gate says all five read on tiny-llama too
# --------------------------------------------------------------------------- #


def test_all_five_components_resolve_on_a_non_hybrid_family(llama_bundle):
    assert set(llama_bundle.streams) == {"full_attention"}
    for component in BLOCK_COMPONENTS:
        site = resolve_site(llama_bundle, _spec(component))
        assert site.module is not None


def test_all_five_components_read_on_a_non_hybrid_family(llama_bundle):
    got = _capture(llama_bundle, {n: _spec(n) for n in BLOCK_COMPONENTS})
    assert set(got) == set(BLOCK_COMPONENTS)
    hidden = llama_bundle.info.hidden_size
    for name, tensor in got.items():
        width = 1 if name == "input_ids" else hidden
        assert tensor.shape[-1] == width, (name, tensor.shape)


def test_the_llama_residual_algebra_holds_too(llama_bundle):
    """The identities are architecture-independent — same algebra, other family."""
    block = llama_bundle.blocks[0]
    got = _capture(
        llama_bundle,
        {
            n: _spec(n, 0)
            for n in (
                "block_input",
                "attention_input_norm",
                "attention_output",
                "block_mid",
                "mlp_input_norm",
                "mlp_output",
                "block_output",
            )
        },
    )
    torch.testing.assert_close(
        got["attention_input_norm"], block.input_layernorm(got["block_input"])
    )
    torch.testing.assert_close(
        got["block_mid"], got["block_input"] + got["attention_output"]
    )
    torch.testing.assert_close(
        got["mlp_input_norm"], block.post_attention_layernorm(got["block_mid"])
    )


# --------------------------------------------------------------------------- #
# end to end through the executor — the path raw hooks do not exercise
# --------------------------------------------------------------------------- #


def _read_doc(component: str, layer: int | None = None, featurizer: bool = False):
    site: dict = {"component": component}
    if layer is not None:
        site["layers"] = layer
    read: dict = {"site": "tap", "pos": {"index": 1}}
    doc: dict = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": read},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }
    if featurizer:
        doc["method"]["featurizers"] = {"f": {"kind": "subspace", "k": 1}}
        read["featurizer"] = "f"
    return doc


@pytest.mark.parametrize(
    "component", ["attention_input_norm", "block_mid", "mlp_input_norm"]
)
def test_a_document_reads_each_new_norm_component(llama_bundle, component: str):
    """End to end: a real document, parsed and validated, reads the component."""
    executor = executor_for(_read_doc(component, 0), llama_bundle, base_texts=[TEXT])
    value = executor.read_value("r")
    assert value.shape == (1, 1, llama_bundle.info.hidden_size)


def test_a_document_reads_input_ids_even_though_it_has_no_width(llama_bundle):
    """The refusal in ``component_width`` must not make reading impossible.

    A read with no featurizer needs no width — ``build_stack`` returns an
    Identity stack and ignores it — so the executor asks for the width lazily.
    Eagerly asking made a plain ``input_ids`` read raise the "not a feature
    space" error, which is not what that refusal is for: it exists to reject a
    *featurizer*, not a read.
    """
    executor = executor_for(_read_doc("input_ids"), llama_bundle, base_texts=[TEXT])
    value = executor.read_value("r")
    assert value.shape == (1, 1, 1)
    assert not value.dtype.is_floating_point


def test_a_featurizer_on_input_ids_still_refuses(llama_bundle):
    """The other half: the not-a-feature-space rule survives the laziness
    above."""
    with pytest.raises(ValidationError) as excinfo:
        executor_for(
            _read_doc("input_ids", featurizer=True), llama_bundle, base_texts=[TEXT]
        ).read_value("r")
    assert "not a feature space" in str(excinfo.value)


def test_a_featurizer_on_a_norm_component_is_fine(llama_bundle):
    """Anti-vacuity for the test above: the refusal is about the component, not
    about featurizers being broken here."""
    executor = executor_for(
        _read_doc("block_mid", 0, featurizer=True), llama_bundle, base_texts=[TEXT]
    )
    assert executor.read_value("r").shape == (1, 1, 1)
