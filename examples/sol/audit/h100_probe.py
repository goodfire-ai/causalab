"""Collect batch-8 H100 routing and operator evidence using cached pinned weights.

Run with the prepared CUDA Python environment. This is a standalone diagnostic,
not a benchmark median or a training profile. See examples/sol/README.md.
"""

import json
import inspect
import hashlib
import time
from typing import Any
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as impl

model: Any = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    local_files_only=True,
)
model = model.eval().requires_grad_(False).cuda()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rng = torch.Generator().manual_seed(0)
x = torch.randint(0, 200000, (8, 128), generator=rng).cuda()
counts = []
handles = []
for layer in model.model.layers:

    def hook(m, args, output):
        counts.append(torch.bincount(output[2].flatten(), minlength=256).detach())

    handles.append(layer.mlp.gate.register_forward_hook(hook))
with torch.no_grad():
    model(x, use_cache=False)
for h in handles:
    h.remove()
counts = [c.cpu().tolist() for c in counts]
with torch.no_grad():
    torch.cuda.synchronize()
    t = time.perf_counter()
    model(x, use_cache=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        model(x, use_cache=False)
        torch.cuda.synchronize()
events = p.key_averages()
rows = [
    dict(
        name=e.key,
        count=e.count,
        self_cpu_us=e.self_cpu_time_total,
        self_device_us=e.self_device_time_total,
    )
    for e in events
]
r = dict(
    batch=8,
    sequence=128,
    unprofiled_seconds=elapsed,
    experts_implementation=model.config._experts_implementation,
    delta_function=str(impl.torch_chunk_gated_delta_rule),
    module_sha256=hashlib.sha256(open(inspect.getfile(impl), "rb").read()).hexdigest(),
    expert_token_counts=counts,
    operators=sorted(rows, key=lambda r: r["self_device_us"], reverse=True),
)
print(json.dumps(r), flush=True)
