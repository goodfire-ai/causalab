"""Offline one-GPU sensitivity analysis; scenarios are not certified SOL bounds.

Run: uv run python -m examples.sol.audit.assumption_sensitivity --evidence probe.json
Uses caller-supplied diagnostic routing. No model, torch, or GPU required.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from causalab.sol.hardware import h100_sxm
from causalab.sol.qwen36 import Qwen36A3B


@dataclass(frozen=True)
class Costs:
    bf16: float = 0
    fp32: float = 0
    fixed_bytes: float = 0
    expert_bytes: float = 0
    activation_bytes: float = 0

    def __add__(self, other: "Costs") -> "Costs":
        return Costs(
            *(a + b for a, b in zip(vars(self).values(), vars(other).values()))
        )

    def seconds(
        self, touched_fraction: float = 1, *, serialized: bool = False
    ) -> float:
        hw = h100_sxm()
        compute = (
            self.bf16 / hw.flops_per_second["bf16_dense"]
            + self.fp32 / hw.flops_per_second["fp32"]
        )
        memory = (
            self.fixed_bytes
            + self.expert_bytes * touched_fraction
            + self.activation_bytes
        ) / hw.hbm_bytes_per_second
        return compute + memory if serialized else max(compute, memory)


def components(model: Qwen36A3B) -> tuple[list[tuple[Costs, Costs]], Costs]:
    """Partition existing ledger into sequential mixer/MoE blocks plus head.

    Conserves its resource totals; does not introduce a complete kernel ledger.
    Mixer includes layer streams; MoE owns dispatched activation streams.
    """
    n, d = model.batch * model.sequence, model.width
    terminal_params = dict.fromkeys(model.parameter_breakdown(), 0)
    terminal_params.update(lm_head=d * 248320, norms_and_delta_scalars=d)
    terminal_ops = dict.fromkeys(model.operation_breakdown(), 0.0)
    terminal_ops["bf16_projection_and_expert_flops"] = 2 * n * d * 248320
    blocks = []
    for i in range(model.layers):
        p, ops = model.parameter_breakdown(i), model.operation_breakdown(i)
        after_p = (
            model.parameter_breakdown(i + 1)
            if i + 1 < model.layers
            else terminal_params
        )
        after_ops = (
            model.operation_breakdown(i + 1) if i + 1 < model.layers else terminal_ops
        )
        p = {k: v - after_p[k] for k, v in p.items()}
        ops = {k: v - after_ops[k] for k, v in ops.items()}
        moe_fixed = p["shared_experts"] + p["routers_and_shared_gates"]
        moe = Costs(
            bf16=2 * n * (moe_fixed + p["routed_experts"] * 8 / 256),
            fixed_bytes=2 * moe_fixed,
            expert_bytes=2 * p["routed_experts"],
            activation_bytes=4 * n * 8 * d,
        )
        mixer = Costs(
            bf16=ops["bf16_projection_and_expert_flops"]
            + ops["bf16_attention_flops"]
            - moe.bf16,
            fp32=ops["fp32_delta_core_flops"] + ops["fp32_depthwise_convolution_flops"],
            fixed_bytes=2
            * (sum(p.values()) - p["embedding"] - p["routed_experts"] - moe_fixed),
            activation_bytes=4 * n * d,
        )
        blocks.append((mixer, moe))
    head = Costs(
        bf16=terminal_ops["bf16_projection_and_expert_flops"],
        fixed_bytes=2 * (d * 248320 + d),
        activation_bytes=2 * n * 248320,
    )
    return blocks, head


def analyze(evidence: dict) -> dict:
    model = Qwen36A3B(batch=evidence["batch"], sequence=evidence["sequence"])
    blocks, head = components(model)
    full = sum((m + e for m, e in blocks), head)
    fractions = [
        sum(x > 0 for x in row) / 256 for row in evidence["expert_token_counts"]
    ]
    if len(fractions) != 40:
        raise ValueError("routing evidence must contain all 40 layers")
    uniform = 1 - (1 - 8 / 256) ** (model.batch * model.sequence)
    observed = sum(fractions) / 40
    scenarios = {
        "whole_forward_uniform": full.seconds(uniform),
        "whole_forward_observed_routing": full.seconds(observed),
        "sequential_blocks_uniform": sum((m + e).seconds(uniform) for m, e in blocks)
        + head.seconds(),
        "sequential_mixer_moe_uniform": sum(
            m.seconds() + e.seconds(uniform) for m, e in blocks
        )
        + head.seconds(),
        "sequential_mixer_moe_observed_routing": sum(
            m.seconds() + e.seconds(f) for (m, e), f in zip(blocks, fractions)
        )
        + head.seconds(),
        "serialized_existing_resources_uniform": full.seconds(uniform, serialized=True),
    }
    # Avoided ledger costs; these do not predict runtime.
    prefix = sum((m + e for m, e in blocks[:27]), Costs())
    suffix = sum((m + e for m, e in blocks[27:]), head)
    n, d, k, v, c = model.batch * model.sequence, model.width, 128, 128, 64
    last_head = 2 * model.batch * d * 248320
    return {
        "scope": "Single-GPU whole-forward sensitivity diagnostic, separate from the benchmark execution contract. Sequential partitions are schedule assumptions, not universal lower bounds. Observed routing is one base batch, not training evidence.",
        "geometry": {
            "batch": model.batch,
            "sequence": model.sequence,
            "suffix_blocks": 13,
        },
        "resource_totals": vars(full),
        "scenarios": {name: {"seconds": t} for name, t in scenarios.items()},
        "semantic_work_opportunities": {
            "full_head_flops": head.bf16,
            "last_token_head_flops": last_head,
            "avoided_head_flops_fraction": 1 - last_head / head.bf16,
            "full_logits_bytes": head.activation_bytes,
            "last_token_logits_bytes": 2 * model.batch * 248320,
            "source_capture_avoids_blocks": 13,
            "source_capture_avoided_bf16_flops_per_batch": suffix.bf16,
            "source_capture_avoided_fp32_flops_per_batch": suffix.fp32,
            "repeated_base_prefix_avoided_forwards": 90,
            "repeated_base_prefix_avoided_bf16_flops_total": 90 * prefix.bf16,
            "repeated_base_prefix_avoided_fp32_flops_total": 90 * prefix.fp32,
            "base_prefix_cache_bytes_for_10_batches": 10 * 2 * n * d,
            "last_token_suffix_expert_assignments_per_layer": model.batch * 8,
            "current_suffix_expert_assignments_per_layer": n * 8,
            "notes": "Prefix cache is block-26 output only; excludes suffix KV/recurrent caches, retrieval and workspace. Last-token dependence permits fewer changing-token projections but does not remove attention reads of earlier KV or Delta state.",
        },
        "delta_state_capacity": {
            "current_30_layer_state_bytes": 4 * model.batch * 32 * k * v * 30,
            "one_layer_state_bytes": 4 * model.batch * 32 * k * v,
            "notes": "use_cache=False does not retain 30 inference states simultaneously. Training saved tensors are separate and not accounted by either number.",
        },
        "delta_triangular_loop": {
            "current_cubic_flops_per_head_chunk": 2 * c**3 / 3,
            "exact_loop_mult_add_flops_per_head_chunk": 2
            * sum(i * i for i in range(1, c)),
            "logical_tensor_bytes_all_layers": model.batch
            * 32
            * ((model.sequence + c - 1) // c)
            * 30
            * sum(20 * i * i + 36 * i for i in range(1, c)),
            "notes": "Logical read/write traffic for row clone, submatrix clone, multiply, sum, add, copy. NOT compulsory HBM traffic: intermediate arrays can hit L2. Excludes backward, masks and other Delta work.",
        },
        "expert_geometry": {
            "uniform_assignments_per_expert": n * 8 / 256,
            "observed_touched_experts_mean": observed * 256,
            "h100_dense_bf16_flops_per_hbm_byte": h100_sxm().flops_per_second[
                "bf16_dense"
            ]
            / h100_sxm().hbm_bytes_per_second,
            "hypothetical_m64_padding_ratio": sum(
                sum(((x + 63) // 64) * 64 for x in row)
                for row in evidence["expert_token_counts"]
            )
            / (40 * n * 8),
            "notes": "M64 padding is a shape sensitivity, not measured extra FLOPs or HBM bytes; grouped kernels may use other tiles/schedules.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.evidence.read_text()))
    print(json.dumps(result, indent=2))
