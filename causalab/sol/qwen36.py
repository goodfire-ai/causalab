"""Qwen3.6-35B-A3B text-tower work model. No model download/import required.

Shapes come from the pinned official config and the repository's tap diagram.
FLOP/traffic formulas are analytical assumptions, not vendor measurements.
"""

from dataclasses import asdict, dataclass
from typing import Any

from causalab.sol.model import Phase, Workload, validate_integer

REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CONFIG_SOURCE = (
    f"https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/{REVISION}/config.json"
)
KERNEL_SOURCE = "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attn_output_gate": True,
    "tie_word_embeddings": False,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
}


@dataclass(frozen=True)
class Qwen36A3B:
    batch: int = 16
    sequence: int = 128
    delta_algorithm: str = "chunk64"

    @property
    def layers(self) -> int:
        return 40

    @property
    def width(self) -> int:
        return 2048

    def __post_init__(self) -> None:
        validate_integer("batch", self.batch)
        validate_integer("sequence", self.sequence)
        if self.sequence > 262144:
            raise ValueError("sequence exceeds the pinned model context limit")
        if self.delta_algorithm not in {"chunk64", "recurrent"}:
            raise ValueError("delta_algorithm must be chunk64 or recurrent")

    def parameter_breakdown(self, start_layer: int = 0) -> dict[str, int]:
        validate_integer("start_layer", start_layer, 0)
        if start_layer >= self.layers:
            raise ValueError("start_layer must precede the last layer")
        layer_types = TEXT_CONFIG["layer_types"][start_layer:]
        full = layer_types.count("full_attention")
        delta = layer_types.count("linear_attention")
        layers = full + delta
        d, q, kv, key, value = self.width, 16 * 256, 2 * 256, 16 * 128, 32 * 128
        return {
            "routed_experts": layers * 256 * 3 * d * 512,
            "shared_experts": layers * 3 * d * 512,
            "routers_and_shared_gates": layers * (d * 256 + d),
            "full_attention_projections": full * d * (3 * q + 2 * kv),
            "delta_projections": delta * d * (2 * key + 3 * value + 2 * 32),
            "delta_convolution": delta * (2 * key + value) * 4,
            "norms_and_delta_scalars": layers * 2 * d
            + full * 2 * 256
            + delta * (2 * 32 + 128)
            + d,
            "embedding": d * 248320 if start_layer == 0 else 0,
            "lm_head": d * 248320,
        }

    def operation_breakdown(
        self, start_layer: int = 0, *, backward: bool = False
    ) -> dict[str, float]:
        p = self.parameter_breakdown(start_layer)
        types = TEXT_CONFIG["layer_types"][start_layer:]
        full, delta = types.count("full_attention"), types.count("linear_attention")
        n = self.batch * self.sequence
        # Routed FLOPs use top-8, whereas resident memory uses all 256 experts.
        active = p["routed_experts"] * 8 // 256
        matrices = sum(
            p[k]
            for k in (
                "shared_experts",
                "routers_and_shared_gates",
                "full_attention_projections",
                "delta_projections",
                "lm_head",
            )
        )
        # Full square attention; both operands of attention GEMMs need gradients.
        attention = 4 * self.batch * self.sequence**2 * (16 * 256) * full
        if self.delta_algorithm == "recurrent":
            # FP32 state decay, prediction, rank-one update and output reduction.
            delta_work = n * 32 * delta * (7 * 128 * 128 + 2 * 128)
        else:
            # Explicit 64-token WY/inverse chunk algorithm: major GEMMs plus
            # triangular inverse cubic floor. Not an exact FLA kernel count.
            c, k, v = 64, 128, 128
            chunks = (self.sequence + c - 1) // c
            delta_work = (
                self.batch
                * 32
                * delta
                * chunks
                * (
                    6 * c * c * k
                    + 4 * c * c * v
                    + 6 * c * k * v
                    + 2 * c**3 / 3
                    + 2 * k * v
                )
            )
        return {
            "bf16_projection_and_expert_flops": 2 * n * (matrices + active),
            "bf16_attention_flops": attention * (2 if backward else 1),
            "fp32_delta_core_flops": delta_work * (2 if backward else 1),
            "fp32_depthwise_convolution_flops": 2 * n * p["delta_convolution"],
        }

    def _phase(
        self, name: str, repeats: int, start_layer: int = 0, *, backward: bool = False
    ) -> Phase:
        p = self.parameter_breakdown(start_layer)
        ops = self.operation_breakdown(start_layer, backward=backward)
        layers = self.layers - start_layer
        n, d = self.batch * self.sequence, self.width
        # Embedding lookup traffic is activation-like; don't scan its table.
        fixed_weights = sum(p.values()) - p["routed_experts"] - p["embedding"]
        logits = 2 * n * 248320
        # Compulsory layer streams + dispatched expert input/output + logits.
        activations = 4 * n * d * layers + 4 * n * 8 * d * layers + logits
        # Inference floor includes full logits, routed intermediates, recurrent state.
        state = 4 * self.batch * 32 * 128 * 128 * 30
        transient = logits + 2 * n * (8 * (d + 2 * 512) + d) + state
        return Phase(
            name,
            repeats,
            {
                "bf16_dense": ops["bf16_projection_and_expert_flops"]
                + ops["bf16_attention_flops"],
                "fp32": ops["fp32_delta_core_flops"]
                + ops["fp32_depthwise_convolution_flops"],
            },
            activation_bytes=activations,
            weight_bytes=2 * fixed_weights,
            resident_sharded_bytes=2 * sum(self.parameter_breakdown().values()),
            transient_bytes=transient,
            tp_payload_bytes=2 * n * d,
            tp_collectives=2 * layers,
            routed_weight_bytes=2 * p["routed_experts"],
            routed_experts=256,
            routed_top_k=8,
            routed_tokens=n,
        )

    def forward(self, name: str, repeats: int = 1) -> Phase:
        return self._phase(name, repeats)

    def suffix_backward(self, layers: int, repeats: int) -> Phase:
        validate_integer("suffix layers", layers)
        if layers > self.layers:
            raise ValueError("suffix layers exceed model depth")
        return self._phase(
            "frozen suffix backward", repeats, self.layers - layers, backward=True
        )

    def metadata(self) -> dict:
        return {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "revision": REVISION,
            "config_source": CONFIG_SOURCE,
            "text_config": TEXT_CONFIG,
            "execution": asdict(self),
            "parameter_breakdown": self.parameter_breakdown(),
            "forward_operation_breakdown": self.operation_breakdown(),
        }


def qwen36_workloads(
    model: Qwen36A3B,
    *,
    updates: int = 100,
    source_batches: int = 10,
    eval_batches: int = 20,
    suffix_layers: int = 13,
    rank: int = 8,
) -> list[Workload]:
    from causalab.sol.qwen36_v2 import workloads

    return workloads(
        model,
        updates=updates,
        source_batches=source_batches,
        eval_batches=eval_batches,
        suffix_layers=suffix_layers,
        rank=rank,
    )
