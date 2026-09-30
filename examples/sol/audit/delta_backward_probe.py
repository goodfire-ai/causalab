"""CPU operator-count probe for the installed Delta fallback, not a GPU benchmark."""

import torch
import json
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule,
)

torch.manual_seed(0)
q = torch.randn(1, 128, 1, 128, dtype=torch.float32, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
g = (-torch.rand(1, 128, 1)).requires_grad_()
beta = torch.rand(1, 128, 1, requires_grad=True)
with torch.profiler.profile(with_flops=True) as p:
    out, _ = torch_chunk_gated_delta_rule(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True
    )
with torch.profiler.profile(with_flops=True) as b:
    out[:, -1].square().sum().backward()


def rows(prof):
    return {
        e.key: {"flops": e.flops, "count": e.count}
        for e in prof.key_averages()
        if e.key in ("aten::mm", "aten::bmm", "aten::mul", "aten::add")
    }


print(
    json.dumps(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "shape": list(q.shape),
            "dtype": str(q.dtype),
            "loss": "sum of squared last-token outputs; no initial/final state",
            "forward": rows(p),
            "backward": rows(b),
        },
        indent=2,
    )
)
