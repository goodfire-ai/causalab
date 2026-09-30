"""Independent graph-work checks for the execution contract."""

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
)

from causalab.sol.benchmark_qwen import parser, prepare_workloads
from causalab.sol.qwen36 import Qwen36A3B, qwen36_workloads
from causalab.sol.qwen36_v2 import CONTRACT, delta_chunk_gemm_flops
from causalab.sol.hardware import h100_sxm
from causalab.sol.model import reference


@pytest.mark.numerical_unit
@pytest.mark.parametrize("sequence", [32, 64, 65, 128, 192])
def test_delta_gemm_derivation_against_autograd(sequence):
    torch.manual_seed(0)
    shape = (1, sequence, 1, 8)
    q, k, v = [torch.randn(shape, requires_grad=True) for _ in range(3)]
    g = (-torch.rand(1, sequence, 1)).requires_grad_()
    beta = torch.rand(1, sequence, 1, requires_grad=True)
    with torch.profiler.profile(with_flops=True) as forward:
        out, _ = torch_chunk_gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=True
        )
    with torch.profiler.profile(with_flops=True) as backward:
        out[:, -1].square().sum().backward()
    for profile, is_backward in ((forward, False), (backward, True)):
        actual = sum(e.flops for e in profile.key_averages() if e.key == "aten::bmm")
        assert actual == delta_chunk_gemm_flops(
            1, sequence, 1, 8, 8, backward=is_backward
        )


@pytest.mark.unit
def test_default_cli_and_v2_ledger_work_counts():
    args = parser().parse_args(["--output", "result.json"])
    assert not hasattr(args, "contract")
    model, works = prepare_workloads(args)
    train = next(w for w in works if w.name == "subspace_train")
    phases = train.phases

    def count(label):
        return sum(
            p.repeats
            for p in phases
            if p.name.startswith(label) and p.name.endswith(": MoE")
        )

    assert count("source capture") == 10 * 27
    assert count("base prefix fill") == 10 * 27
    assert count("cached base suffix") == 100 * 13
    assert count("frozen suffix backward") == 100 * 13
    assert count("evaluation source") == 10 * 27
    assert count("evaluation base") == 10 * 40
    heads = [p for p in phases if p.name.endswith("last-token head")]
    assert sum(p.repeats for p in heads) == 100 + 100 + 10
    assert all(p.flops["bf16_dense"] == 2 * model.batch * 2048 * 248320 for p in heads)
    assert any(
        p.name == "cache writes and reads" and p.activation_bytes > 0 for p in phases
    )
    assert all(
        w.tensor_parallel_sizes == [1] and w.contract["id"] == CONTRACT for w in works
    )
    with pytest.raises(ValueError):
        reference(train, h100_sxm(), 1, 2)
    ref = reference(train, h100_sxm(), 1, 1)
    assert ref["fits"] is None and ref["memory_status"] == "unknown"
    harvest = next(w for w in works if w.name == "activation_harvest")
    assert all("head" not in p.name for p in harvest.phases)
    assert sum(p.repeats for p in harvest.phases if p.name.endswith(": MoE")) == 27


@pytest.mark.unit
def test_only_current_contract_is_available():
    with pytest.raises(ValueError, match="chunk64"):
        qwen36_workloads(Qwen36A3B(delta_algorithm="recurrent"))
    with pytest.raises(ValueError, match="even"):
        qwen36_workloads(Qwen36A3B(), eval_batches=3)
    with pytest.raises(SystemExit):
        parser().parse_args(
            ["--output", "result.json", "--contract", "full_forward_v1"]
        )
    import inspect

    assert "contract" not in inspect.signature(qwen36_workloads).parameters
