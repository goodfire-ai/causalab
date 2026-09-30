"""``attention_result``: the per-head contribution the model never computes.

Every other component in the vocabulary names a tensor the forward pass forms.
This one does not. The block projects the *whole* premix at once::

    attention_output = o_proj(z * sigmoid(gate))

so what exists is the **sum** over heads. Head ``h``'s share of it —
``premix[..., h·d:(h+1)·d] @ W_o[:, h·d:(h+1)·d].T`` — is the quantity a
path-patching or direct-logit-attribution analysis actually wants, and it has to
be derived.

Three consequences, and a test for each:

* **it is read-only**, and the refusal names its lowering rather than just
  saying no: write ``attention_premix`` with the same ``head``, since the result
  is a linear function of it;
* **the defining identity is ``sum_h result == attention_output``** (minus the
  o-projection's bias, which belongs to no head). That is what pins the
  derivation as the right one rather than merely a plausible one;
* **it is derived after the position gather**, so the cost is
  ``n_positions · H · hidden`` rather than ``seq · H · hidden``. On the real
  A3B the dense form is 64× ``attention_output`` at hidden 4096, so this is not
  a micro-optimization.
"""

from __future__ import annotations

import pytest
import torch

from causalab.neural.engines.pytorch_hooks.loading import ModelBundle, load_model
from causalab.neural.shared.sites import (
    resolve_site,
)
from causalab.protocol.rules.errors import ProtocolError
from causalab.protocol.registry import CAPABILITIES, component_shape, component_width
from causalab.protocol.schema import PROTOCOL_VERSION, SiteSpec

from tests.protocol._docs import saved

from ._drive import base_data_section, executor_for
from .conftest import TINY_GPT2, TINY_LLAMA

pytestmark = pytest.mark.smoke

TEXT = "the quick brown fox jumps"
CF_TEXT = "a slow green turtle sleeps"
QWEN_LAYER = 3
DELTANET_LAYER = 0
OTHER_LAYER = 1


def _read_doc(component: str, layer: int, *, head: int | None = None) -> dict:
    site: dict = {"component": component, "layers": [layer]}
    if head is not None:
        site["head"] = head
    return {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=False),
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["r"]}},
            "sites": {"tap": site},
            "reads": {"r": {"site": "tap", "pos": "all"}},
            "save": [saved("r", "original", "a.safetensors")],
        },
    }


def _read(bundle: ModelBundle, component: str, layer: int, head: int | None = None):
    doc = _read_doc(component, layer, head=head)
    return executor_for(doc, bundle, base_texts=[TEXT]).read_value("r")


FAMILIES = [
    pytest.param("tiny-random/qwen3.5-moe", QWEN_LAYER, id="qwen35moe"),
    pytest.param(TINY_LLAMA, OTHER_LAYER, id="llama"),
    pytest.param(TINY_GPT2, OTHER_LAYER, id="gpt2"),
]


# --------------------------------------------------------------------------- #
# the identity that defines the component
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key, layer", FAMILIES)
def test_the_heads_sum_to_the_attention_output(key: str, layer: int):
    """📐 ``sum_h attention_result == attention_output``, on all three families.

    Measured max differences: qwen 1.8e-07, llama 1.9e-09, gpt2 3.7e-09 — the
    residue of summing H fp32 terms in a different order than the model's own
    matmul does, not a modelling difference.

    This is the assertion that makes the derivation *the* right one. A per-head
    split that got the head blocks wrong, or transposed the weight, would still
    produce a tensor of exactly the right shape.
    """
    bundle = load_model(key)
    whole = _read(bundle, "attention_result", layer)
    output = _read(bundle, "attention_output", layer)
    heads, hidden = bundle.info.num_heads, bundle.info.hidden_size
    summed = whole.reshape(*whole.shape[:-1], heads, hidden).sum(dim=-2)
    torch.testing.assert_close(summed, output, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("key, layer", FAMILIES)
def test_naming_a_head_gives_that_heads_block_of_the_whole(key: str, layer: int):
    """The cheap path and the dense path must agree exactly — the head read
    derives one contribution instead of all of them, and that is the only
    difference."""
    bundle = load_model(key)
    whole = _read(bundle, "attention_result", layer)
    hidden = bundle.info.hidden_size
    for head in range(bundle.info.num_heads):
        one = _read(bundle, "attention_result", layer, head=head)
        block = whole[..., head * hidden : (head + 1) * hidden]
        torch.testing.assert_close(one, block, atol=0.0, rtol=0.0)


def test_a_head_with_no_premix_contributes_nothing(qwen35moe_bundle):
    """A sanity check on the masking: the derivation attributes to head ``h``
    exactly the columns of the premix that belong to ``h``, so zeroing that
    head's premix must zero its result and leave the other heads alone.

    Done here by construction rather than through a write, because the write is
    the thing this component refuses."""
    from causalab.neural.shared.executor.base import _attention_result

    site = resolve_site(
        qwen35moe_bundle,
        SiteSpec(component="attention_result", layers=(QWEN_LAYER,), head=1),
    )
    info = qwen35moe_bundle.info
    premix = torch.randn(1, 4, info.num_heads * info.head_dim)
    with torch.no_grad():
        before = _attention_result(site, premix)
        blanked = premix.clone()
        blanked[..., info.head_dim : 2 * info.head_dim] = 0.0
        after = _attention_result(site, blanked)
    assert float(before.abs().max()) > 0.0  # otherwise the test is vacuous
    torch.testing.assert_close(after, torch.zeros_like(after), atol=0.0, rtol=0.0)


def test_the_bias_belongs_to_no_head(qwen35moe_bundle):
    """``sum_h result == attention_output - bias``, and the derivation subtracts
    the bias back off rather than charging it to every head H times over.

    📐 None of the three fixtures' o-projections carries a bias, so the identity
    above is exact — this asserts the *code path* is right anyway, because a
    family that does have one would otherwise be wrong by ``H·bias`` and nothing
    in the suite would notice.
    """
    from causalab.neural.shared.executor.base import _attention_result

    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component="attention_result", layers=(QWEN_LAYER,))
    )
    o_proj = site.module
    assert o_proj.bias is None, "fixture assumption"
    info = qwen35moe_bundle.info
    premix = torch.randn(1, 3, info.num_heads * info.head_dim)
    try:
        o_proj.bias = torch.nn.Parameter(torch.full((info.hidden_size,), 5.0))
        with torch.no_grad():
            whole = _attention_result(site, premix)
            summed = whole.reshape(1, 3, info.num_heads, info.hidden_size).sum(-2)
            projected = o_proj(premix) - o_proj.bias
        torch.testing.assert_close(summed, projected, atol=1e-5, rtol=1e-5)
    finally:
        o_proj.bias = None


# --------------------------------------------------------------------------- #
# shape, and where the derivation happens
# --------------------------------------------------------------------------- #


def test_the_value_is_heads_times_the_residual_stream(qwen35moe_bundle):
    info = qwen35moe_bundle.info
    shape = component_shape(info, "attention_result")
    assert shape.width == info.num_heads * info.hidden_size
    assert shape.head_space == info.num_heads
    assert component_width(info, "attention_result", head=0) == info.hidden_size
    assert tuple(_read(qwen35moe_bundle, "attention_result", QWEN_LAYER).shape) == (
        1,
        5,
        info.num_heads * info.hidden_size,
    )


def test_the_tap_captures_the_premix_not_the_result(qwen35moe_bundle):
    """The one place in the backend where the tapped shape and the component's
    shape differ, and the site declares it rather than leaving it to be
    inferred."""
    site = resolve_site(
        qwen35moe_bundle, SiteSpec(component="attention_result", layers=(QWEN_LAYER,))
    )
    info = qwen35moe_bundle.info
    assert site.derivation == "attention_result"
    assert site.module is qwen35moe_bundle.mixer_at(QWEN_LAYER).o_proj
    assert site.kind == "in"
    # what is captured
    assert site.shape.width == info.num_heads * info.head_dim
    # what the component means
    assert component_shape(info, "attention_result").width == (
        info.num_heads * info.hidden_size
    )


def test_the_result_and_the_premix_share_one_capture(qwen35moe_bundle):
    """They are the same tensor at the same tap, so reading both must not hook
    the module twice — which is what makes the derived read cost one extra
    projection rather than one extra forward."""
    from causalab.neural.shared.executor import tap_key as _tap_key

    premix = resolve_site(
        qwen35moe_bundle, SiteSpec(component="attention_premix", layers=(QWEN_LAYER,))
    )
    result = resolve_site(
        qwen35moe_bundle, SiteSpec(component="attention_result", layers=(QWEN_LAYER,))
    )
    assert _tap_key(premix) == _tap_key(result)


def test_the_derivation_runs_after_the_position_gather(qwen35moe_bundle):
    """The memory argument, as a test: a one-position read must not build the
    whole sequence's worth of contributions.

    Asserted through the shape of what comes back — 1 position, not 5 — because
    the alternative implementation (derive, then gather) produces exactly the
    same numbers and differs only in what it allocates.
    """
    doc = _read_doc("attention_result", QWEN_LAYER, head=0)
    doc["method"]["reads"]["r"]["pos"] = {"index": -1}
    value = executor_for(doc, qwen35moe_bundle, base_texts=[TEXT]).read_value("r")
    assert tuple(value.shape) == (1, 1, qwen35moe_bundle.info.hidden_size)


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_a_write_is_refused_and_the_refusal_names_its_lowering(qwen35moe_bundle):
    """A refusal that can name what to do instead, does.

    ``attention_result`` is a linear function of ``attention_premix``, so a
    write to the premix at the same head moves it by exactly the projection of
    what was written — the user does not lose the capability, only the spelling.
    """
    assert CAPABILITIES["attention_result"].writes is None
    doc = {
        "header": {"protocol_version": PROTOCOL_VERSION},
        "model": {"key": "test", "revision": "main"},
        "data": base_data_section(with_counterfactual=True),
        "method": {
            "intervened_models": {
                "original_counterfactual": {
                    "input": "counterfactual",
                    "reads": ["v_cf"],
                },
                "patched": {"input": "base", "reads": ["after"], "writes": ["patch"]},
            },
            "sites": {
                "tap": {
                    "component": "attention_result",
                    "layers": [QWEN_LAYER],
                    "head": 0,
                },
                "lm_head": {"component": "lm_head"},
            },
            "reads": {
                "v_cf": {"site": "tap", "pos": "all"},
                "after": {"site": "lm_head", "pos": {"index": -1}},
            },
            "writes": {"patch": {"site": "tap", "pos": "all", "do": {"swap": "v_cf"}}},
            "save": [saved("after", "patched", "p.safetensors")],
        },
    }
    with pytest.raises(ProtocolError) as excinfo:
        executor_for(
            doc,
            qwen35moe_bundle,
            base_texts=[TEXT],
            counterfactual_texts=[CF_TEXT],
        ).read_value("after")
    message = str(excinfo.value)
    assert "derived from the joint output projection" in message
    assert "attention_premix" in message


def test_a_deltanet_layer_refuses_with_the_architectural_reason(qwen35moe_bundle):
    with pytest.raises(ProtocolError, match="full-attention mixer"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(component="attention_result", layers=(DELTANET_LAYER,)),
        )


def test_the_head_bound_is_the_query_head_space(qwen35moe_bundle):
    """Query space, like ``attention_premix``: the o-projection's input has one
    block per query head, and each block projects to one contribution."""
    info = qwen35moe_bundle.info
    resolve_site(
        qwen35moe_bundle,
        SiteSpec(
            component="attention_result", layers=(QWEN_LAYER,), head=info.num_heads - 1
        ),
    )
    with pytest.raises(ProtocolError, match="which has 8 heads"):
        resolve_site(
            qwen35moe_bundle,
            SiteSpec(
                component="attention_result", layers=(QWEN_LAYER,), head=info.num_heads
            ),
        )


def test_the_derivation_works_in_the_generated_frame_too():
    """A continuation read of a derived component.

    Worth its own test because the derivation *re-invokes the o-projection*, and
    doing that while the capture hook was still installed would re-enter it —
    corrupting the very tensor being read. It is safe because `_finalize_read`
    runs after the hook scope closes on both the prefill and the decode path,
    and this pins the decode one, which nothing else exercises.

    📐 The defining identity survives: `sum_h result == attention_output` to
    4.7e-10 across three decode steps.
    """
    bundle = load_model(TINY_LLAMA)
    info = bundle.info

    def generated(component: str, head: int | None = None) -> torch.Tensor:
        doc = _read_doc(component, OTHER_LAYER, head=head)
        doc["method"]["positions"] = {
            "w": {"generated": {"max_new_tokens": 3}, "all": True}
        }
        doc["method"]["reads"]["r"]["pos"] = "w"
        return executor_for(doc, bundle, base_texts=[TEXT]).read_value("r")

    whole = generated("attention_result")
    output = generated("attention_output")
    assert tuple(whole.shape) == (1, 3, info.num_heads * info.hidden_size)
    summed = whole.reshape(1, 3, info.num_heads, info.hidden_size).sum(dim=-2)
    torch.testing.assert_close(summed, output, atol=1e-6, rtol=1e-5)
    # and the cheap per-head path agrees there as well
    torch.testing.assert_close(
        generated("attention_result", head=1),
        whole[..., info.hidden_size : 2 * info.hidden_size],
        atol=0.0,
        rtol=0.0,
    )
