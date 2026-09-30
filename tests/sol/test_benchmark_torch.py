"""All seven actual hook/featurizer paths on a tiny random CPU hybrid Qwen."""

import pytest
import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from causalab.sol.benchmark_torch import TorchOperations
from causalab.sol.qwen36 import Qwen36A3B, TEXT_CONFIG, qwen36_workloads

pytestmark = pytest.mark.smoke


@pytest.fixture
def tiny_driver():
    torch.manual_seed(123)
    config = Qwen3_5MoeTextConfig(
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
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
    model = Qwen3_5MoeForCausalLM(config)
    model.set_attn_implementation("eager")
    return TorchOperations(
        model,
        batch=2,
        sequence=4,
        updates=2,
        source_batches=1,
        eval_batches=2,
        suffix_layers=2,
        rank=2,
        device="cpu",
    )


@pytest.mark.parametrize(
    "name",
    [
        "inference",
        "activation_harvest",
        "interchange",
        "subspace_apply",
        "dbm_apply",
        "subspace_train",
        "dbm_train",
    ],
)
def test_real_operation_counts_finite_outputs_reset_and_hook_cleanup(tiny_driver, name):
    # Only workload names are used here: CPU timing is never compared with GPU peaks.
    work = next(w for w in qwen36_workloads(Qwen36A3B()) if w.name == name)
    case = tiny_driver.case(work)
    case.reset()
    assert case.run() == case.expected_counts
    case.validate()
    assert not tiny_driver.layers[0]._forward_hooks
    if name.endswith("_train"):
        kind = name.removesuffix("_train")
        after = {
            key: p.detach().clone() for key, p in tiny_driver.stage.state_dict().items()
        }
        assert any(
            not torch.equal(p, tiny_driver.initial[kind][key])
            for key, p in after.items()
        )
        case.reset()
        assert tiny_driver.cache == {}
        for key, p in tiny_driver.stage.state_dict().items():
            assert torch.equal(p, tiny_driver.initial[kind][key])
        assert case.run() == case.expected_counts
        case.validate()
        for key, p in tiny_driver.stage.state_dict().items():
            torch.testing.assert_close(p, after[key])


def test_pinned_full_model_parameter_count_without_allocating_weights():
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**TEXT_CONFIG))
    assert sum(p.numel() for p in model.parameters()) == sum(
        Qwen36A3B().parameter_breakdown().values()
    )


def test_cayley_reference_matches_actual_forward_gemms_and_accesses(tiny_driver):
    from causalab.neural.shared.featurizers import Cayley
    from causalab.sol.recipes import DenseTransformer, dense_catalog_workloads

    d, k = 16, 2
    transform = Cayley(torch.eye(d, k))
    with torch.profiler.profile(with_flops=True) as profile:
        transform(torch.zeros(d, k))
    # Independent operator accounting: excludes the inverse, as the ledger does.
    actual = sum(e.flops for e in profile.key_averages() if e.key == "aten::mm")
    assert actual == 10 * d * k**2 + 10 * k**3

    stage = tiny_driver.stages["subspace"]
    accesses = []
    handle = stage.parametrizations.weight[0].register_forward_hook(
        lambda *_: accesses.append(1)
    )
    try:
        work = next(
            w for w in qwen36_workloads(Qwen36A3B()) if w.name == "subspace_apply"
        )
        case = tiny_driver.case(work)
        case.reset()
        accesses.clear()
        case.run()
        assert len(accesses) == 3
        ledger = dense_catalog_workloads(
            DenseTransformer(2, d, 32, 32, 4, 2),
            updates=2,
            source_batches=1,
            eval_batches=2,
            suffix_layers=1,
            rank=k,
        )
        apply = next(w for w in ledger if w.name == "subspace_apply")
        phase = next(p for p in apply.phases if p.name.startswith("Cayley"))
        assert phase.flops["fp32"] == len(accesses) * actual
    finally:
        handle.remove()


def test_hook_cleanup_on_model_failure(tiny_driver):
    case = tiny_driver.case(
        next(w for w in qwen36_workloads(Qwen36A3B()) if w.name == "interchange")
    )
    case.reset()
    # Invalid token ids force a real embedding failure after the hook is installed.
    tiny_driver.source[0].fill_(1000)
    with pytest.raises((IndexError, RuntimeError)):
        case.run()
    assert not tiny_driver.layers[0]._forward_hooks


@pytest.mark.parametrize(
    "name",
    [
        "inference",
        "activation_harvest",
        "interchange",
        "subspace_apply",
        "dbm_apply",
        "subspace_train",
        "dbm_train",
    ],
)
def test_actual_execution_counts_and_cache_reset(tiny_driver, name):
    driver = tiny_driver
    work = next(w for w in qwen36_workloads(Qwen36A3B()) if w.name == name)
    blocks, head_rows = [], []
    handles = [
        layer.register_forward_pre_hook(lambda *_: blocks.append(1))
        for layer in driver.layers
    ]
    handles.append(
        driver.model.lm_head.register_forward_pre_hook(
            lambda _m, args: head_rows.append(args[0].shape[0] * args[0].shape[1])
        )
    )
    try:
        case = driver.case(work)
        case.reset()
        assert case.run() == case.expected_counts
        case.validate()
        if name.endswith("_train"):
            assert len(driver.base_cache) == 1
        assert len(blocks) == driver.counts["block_executions"]
        assert sum(head_rows) == driver.counts["head_token_rows"]
        assert driver.model.model.layers is driver.layers
        case.reset()
        assert not driver.cache and not driver.base_cache
    finally:
        for handle in handles:
            handle.remove()


def test_v2_restores_model_after_cached_suffix_failure(tiny_driver):
    work = next(w for w in qwen36_workloads(Qwen36A3B()) if w.name == "subspace_train")
    case = tiny_driver.case(work)
    case.reset()

    def fail(*_):
        raise RuntimeError("suffix failed")

    handle = tiny_driver.layers[1].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="suffix failed"):
            case.run()
        assert tiny_driver.model.model.layers is tiny_driver.layers
        assert not tiny_driver.layers[0]._forward_hooks
    finally:
        handle.remove()
