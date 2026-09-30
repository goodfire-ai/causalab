"""Read and swap complete neuron outputs at the down-projection input."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from causalab.neural.engines.pytorch_hooks.loading import load_model
from causalab.neural.shared.sites import resolve_site
from causalab.protocol.schema import SiteSpec

from ._drive import executor_for
from .conftest import TINY_LLAMA
from .test_sites_round3_moe_interior import (
    CF_TEXT,
    MOE_LAYER,
    TEXT,
    _moved,
    _own_value,
    _read_doc,
    _write_doc,
)

pytestmark = pytest.mark.smoke


def test_routed_neuron_output_matches_independent_expert_products(qwen35moe_bundle):
    bundle = qwen35moe_bundle
    value = executor_for(
        _read_doc("expert_neuron_output"), bundle, base_texts=[TEXT]
    ).read_value("r")
    experts = bundle.model.model.layers[MOE_LAYER].mlp.experts
    seen = {}
    handle = experts.register_forward_pre_hook(
        lambda _m, args: seen.update(hidden=args[0].detach(), idx=args[1].detach())
    )
    try:
        with torch.no_grad():
            bundle.model(**bundle.tokenizer(TEXT, return_tensors="pt"))
    finally:
        handle.remove()
    expected = []
    for hidden, ids in zip(seen["hidden"], seen["idx"], strict=True):
        slots = []
        for expert in ids:
            gate, up = F.linear(hidden, experts.gate_up_proj[expert]).chunk(2)
            slots.append(F.silu(gate) * up)
        expected.append(torch.cat(slots))
    expected = torch.stack(expected).unsqueeze(0)
    torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-5)
    activation = executor_for(_read_doc(), bundle, base_texts=[TEXT]).read_value("r")
    assert float((value - activation).abs().max()) > 1e-3


@pytest.mark.parametrize("component", ["expert_neuron_output", "mlp_neuron_output"])
def test_complete_neuron_swap_moves_logits_and_self_swap_is_exact(
    qwen35moe_bundle, component
):
    bundle = (
        qwen35moe_bundle
        if component == "expert_neuron_output"
        else load_model(TINY_LLAMA)
    )
    doc = _write_doc(component, {"swap": "v_cf"})
    doc["method"]["reads"]["v_cf"]["pos"] = -1
    doc["method"]["writes"]["patch"]["pos"] = -1
    assert _moved(bundle, doc) > 1e-5
    _own_value(doc)  # the self-swap: the operand is the tap's own base value
    assert _moved(bundle, doc) == 0.0


def test_dense_neuron_output_is_the_complete_product():
    bundle = load_model(TINY_LLAMA)
    mlp = bundle.model.model.layers[MOE_LAYER].mlp
    seen = {}
    handles = [
        mlp.act_fn.register_forward_hook(
            lambda _m, _args, out: seen.update(activation=out.detach())
        ),
        mlp.up_proj.register_forward_hook(
            lambda _m, _args, out: seen.update(up=out.detach())
        ),
    ]
    try:
        value = executor_for(
            _read_doc("mlp_neuron_output"), bundle, base_texts=[TEXT]
        ).read_value("r")
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(value, seen["activation"] * seen["up"], atol=0, rtol=0)
    assert float((value - seen["activation"]).abs().max()) > 1e-3
    site = resolve_site(bundle, SiteSpec(component="mlp_neuron_output", layers=(0,)))
    assert site.module is mlp.down_proj and site.kind == "in"


def test_routed_product_swap_matches_a_direct_apply_gate_patch(
    qwen35moe_bundle, monkeypatch
):
    bundle = qwen35moe_bundle
    experts = bundle.model.model.layers[MOE_LAYER].mlp.experts
    source = (
        executor_for(_read_doc("expert_neuron_output"), bundle, base_texts=[CF_TEXT])
        .read_value("r")
        .reshape(-1, bundle.info.moe_intermediate_size)
    )
    routing = executor_for(
        _read_doc("expert_idx"), bundle, base_texts=[TEXT]
    ).read_value("r")
    _, order = torch.sort(routing.reshape(-1))
    monkeypatch.setattr(experts, "_apply_gate", lambda _value: source[order])
    with torch.no_grad():
        expected = bundle.model(**bundle.tokenizer(TEXT, return_tensors="pt")).logits[
            :, -1:
        ]
    monkeypatch.undo()
    doc = _write_doc("expert_neuron_output", {"swap": "v_cf"})
    # Match the raw oracle's full-sequence head so its GEMM shape is identical.
    doc["method"]["reads"]["after"]["pos"] = "all"
    executor = executor_for(
        doc, bundle, base_texts=[TEXT], counterfactual_texts=[CF_TEXT]
    )
    torch.testing.assert_close(
        executor.read_value("after")[:, -1:], expected, atol=0, rtol=0
    )


@pytest.mark.parametrize("component", ["expert_activation", "expert_neuron_output"])
def test_expert_taps_preserve_the_activation_call_with_gate_fusion(
    qwen35moe_bundle, monkeypatch, component
):
    from causalab.neural.engines.pytorch_hooks.kernels import moe_glue

    bundle = qwen35moe_bundle
    expected = executor_for(_read_doc(component), bundle, base_texts=[TEXT]).read_value(
        "r"
    )
    admissions = []
    fused_calls = []

    def plan(**kwargs):
        # Exercise the CUDA gate decision on CPU with the same tensor product.
        allowed = kwargs["default_silu_gate"]
        admissions.append(allowed)
        return moe_glue.GluePlan(gate=allowed)

    def fused_gate(value):
        fused_calls.append(True)
        gate, up = value.chunk(2, dim=-1)
        return F.silu(gate) * up

    monkeypatch.setattr(moe_glue, "plan_moe_glue", plan)
    monkeypatch.setattr(moe_glue, "fused_gate", fused_gate)
    actual = executor_for(_read_doc(component), bundle, base_texts=[TEXT]).read_value(
        "r"
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert False in admissions and True in admissions
    assert fused_calls
    doc = _write_doc(component, {"swap": "v_cf"})
    assert _moved(bundle, doc) > 1e-5
    _own_value(doc)  # the self-swap: the operand is the tap's own base value
    assert _moved(bundle, doc) == 0.0
