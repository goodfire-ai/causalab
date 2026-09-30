"""Resource conservation and causal dependency checks for the SOL model."""

import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from causalab.sol.qwen36 import Qwen36A3B
from causalab.sol.hardware import h100_sxm
from causalab.sol.model import Workload, reference
from examples.sol.audit.assumption_sensitivity import analyze, components


@pytest.mark.unit
@pytest.mark.parametrize("batch,sequence", [(1, 1), (8, 128), (16, 65)])
def test_partition_conserves_existing_forward_resources(batch, sequence):
    model = Qwen36A3B(batch=batch, sequence=sequence)
    blocks, head = components(model)
    full = sum((m + e for m, e in blocks), head)
    phase = model.forward("forward")
    assert full.bf16 == pytest.approx(phase.flops["bf16_dense"])
    assert full.fp32 == pytest.approx(phase.flops["fp32"])
    assert full.fixed_bytes == phase.weight_bytes
    assert full.expert_bytes == phase.routed_weight_bytes
    assert full.activation_bytes == phase.activation_bytes


@pytest.mark.unit
def test_sensitivity_preserves_baseline_and_orders_overlap_assumptions():
    # Synthetic concentrated top-8 routing: all tokens visit the same eight
    # experts. This tests the traffic model without a recorded GPU run.
    model = Qwen36A3B(batch=8, sequence=128)
    evidence = {
        "batch": model.batch,
        "sequence": model.sequence,
        "expert_token_counts": [[1024] * 8 + [0] * 248 for _ in range(40)],
    }
    result = analyze(evidence)
    times = {k: v["seconds"] for k, v in result["scenarios"].items()}
    original = reference(
        Workload(
            "inference",
            model.batch,
            [model.forward("forward")],
            ["synthetic routing"],
            ["test"],
        ),
        h100_sxm(),
        1,
        1,
    )
    assert times["whole_forward_uniform"] == pytest.approx(original["sol_seconds"])
    assert times["whole_forward_uniform"] <= times["sequential_blocks_uniform"]
    assert times["sequential_blocks_uniform"] <= times["sequential_mixer_moe_uniform"]
    assert (
        times["sequential_mixer_moe_uniform"]
        <= times["serialized_existing_resources_uniform"]
    )
    assert times["whole_forward_observed_routing"] < times["whole_forward_uniform"]


@pytest.mark.numerical_unit
@pytest.mark.parametrize("suffix_type", ["full_attention", "linear_attention"])
def test_last_position_dependency_and_restricted_head_gradient(suffix_type):
    torch.manual_seed(123)
    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=3,
        layer_types=["linear_attention", suffix_type, suffix_type],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        pad_token_id=0,
    )
    model = Qwen3_5MoeForCausalLM(config).eval().requires_grad_(False)
    model.set_attn_implementation("eager")
    tokens = torch.randint(1, 32, (2, 4))
    intervention = torch.randn(2, 16, requires_grad=True)

    def patch(_module, _args, output):
        changed = output.clone()
        changed[:, -1, :] = intervention
        return changed

    handle = model.model.layers[0].register_forward_hook(patch)
    handles = []

    def run(keep):
        logits = model(tokens, use_cache=False, logits_to_keep=keep).logits
        loss = logits[:, -1, :].square().sum()
        gradient = torch.autograd.grad(loss, intervention)[0]
        return logits.detach(), gradient

    try:
        all_logits, gradient = run(0)
        last_logits, last_gradient = run(1)
        torch.testing.assert_close(
            last_logits, all_logits[:, -1:], atol=1e-7, rtol=1e-5
        )
        torch.testing.assert_close(last_gradient, gradient, atol=1e-7, rtol=1e-5)
        assert gradient.abs().max() > 0

        # Earlier positions have no dependency on the intervention parameters.
        # Detaching them is a graph pruning check, not an optimized suffix kernel.
        def detach_earlier(_module, _args, output):
            return torch.cat((output[:, :-1, :].detach(), output[:, -1:, :]), dim=1)

        for layer in model.model.layers[1:]:
            handles.append(layer.register_forward_hook(detach_earlier))
        pruned_logits, pruned_gradient = run(0)
        torch.testing.assert_close(pruned_logits, all_logits, atol=0, rtol=0)
        torch.testing.assert_close(pruned_gradient, gradient, atol=1e-7, rtol=1e-5)
        for h in handles:
            h.remove()
        handles.clear()
        with torch.no_grad():
            intervention.add_(0.3 * torch.randn_like(intervention))
            changed = model(tokens, use_cache=False).logits
        torch.testing.assert_close(changed[:, :-1], all_logits[:, :-1], atol=0, rtol=0)
        assert not torch.equal(changed[:, -1], all_logits[:, -1])
    finally:
        handle.remove()
        for h in handles:
            h.remove()
