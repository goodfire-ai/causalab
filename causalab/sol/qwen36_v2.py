"""Versioned last-token / cold-prefix-cache work ledger.

These are execution-path major-work estimates, not hardware lower bounds. Model
stages execute sequentially; resource overlap within a stage remains idealized.
"""

from dataclasses import replace

from causalab.sol.model import Phase, Workload, validate_integer
from causalab.sol.qwen36 import Qwen36A3B, TEXT_CONFIG, CONFIG_SOURCE, KERNEL_SOURCE
from causalab.sol.recipes import feature_phases

CONTRACT = "last_token_cached_prefix_v2"


def delta_chunk_gemm_flops(
    batch: int,
    sequence: int,
    heads: int,
    key: int,
    value: int,
    *,
    backward: bool = False,
) -> int:
    """Live major GEMMs of the unfused no-cache chunk graph, including padding.

    First-chunk initial state is constant; last-chunk final state is unused.
    This counts executed dense GEMMs, including zero gradients, not a minimal
    last-token recurrence or all elementwise backward operations.
    """
    c = 64
    chunks = (sequence + c - 1) // c
    forward = chunks * (6 * c * c * key + 4 * c * c * value + 6 * c * key * value)
    return batch * heads * (2 * forward - 8 * c * key * value if backward else forward)


def stages(
    model: Qwen36A3B,
    name: str,
    repeats: int,
    start: int = 0,
    end: int = 40,
    *,
    head: bool = True,
    backward: bool = False,
) -> list[Phase]:
    n, d = model.batch * model.sequence, model.width
    resident = 2 * sum(model.parameter_breakdown().values())
    terminal = dict.fromkeys(model.parameter_breakdown(), 0)
    terminal.update(lm_head=d * 248320, norms_and_delta_scalars=d)
    result = []
    for i in range(start, end):
        p = model.parameter_breakdown(i)
        after = model.parameter_breakdown(i + 1) if i < 39 else terminal
        p = {k: v - after[k] for k, v in p.items()}
        is_delta = TEXT_CONFIG["layer_types"][i] == "linear_attention"
        projection_params = p["full_attention_projections"] + p["delta_projections"]
        bf16_core = (
            0
            if is_delta
            else 4 * model.batch * model.sequence**2 * 16 * 256 * (2 if backward else 1)
        )
        fp32_core = 0.0
        if is_delta:
            c, k, v = 64, 128, 128
            chunks = (model.sequence + c - 1) // c
            triangular = 2 * sum(j * j for j in range(1, c))
            # For no initial/final state, the first chunk has constant state;
            # the final state update is dead in backward. Count live GEMMs.
            bmm = delta_chunk_gemm_flops(
                model.batch, model.sequence, 32, k, v, backward=backward
            )
            fp32_core = bmm + model.batch * 32 * chunks * (
                triangular * (2 if backward else 1) + 2 * k * v
            )
        mixer = Phase(
            f"{name}: {'Delta' if is_delta else 'attention'} mixer",
            repeats,
            {
                "bf16_dense": 2 * n * projection_params + bf16_core,
                "fp32": fp32_core + 2 * n * p["delta_convolution"],
            },
            activation_bytes=4 * n * d,
            weight_bytes=2
            * (
                projection_params
                + p["delta_convolution"]
                + p["norms_and_delta_scalars"]
            ),
            resident_sharded_bytes=resident,
            transient_bytes=4 * model.batch * 32 * 128 * 128
            if is_delta
            else 2 * model.batch * 16 * model.sequence**2,
        )
        moe_fixed = p["shared_experts"] + p["routers_and_shared_gates"]
        moe = Phase(
            f"{name}: MoE",
            repeats,
            {"bf16_dense": 2 * n * (moe_fixed + p["routed_experts"] * 8 // 256)},
            activation_bytes=4 * n * 8 * d,
            weight_bytes=2 * moe_fixed,
            resident_sharded_bytes=resident,
            transient_bytes=2 * n * 8 * (d + 2 * 512),
            routed_weight_bytes=2 * p["routed_experts"],
            routed_experts=256,
            routed_top_k=8,
            routed_tokens=n,
        )
        result.extend([mixer, moe])
    if head:
        result.append(
            Phase(
                f"{name}: last-token head",
                repeats,
                {"bf16_dense": 2 * model.batch * d * 248320},
                activation_bytes=2 * model.batch * 248320,
                weight_bytes=2 * (d * 248320 + d),
                resident_sharded_bytes=resident,
                transient_bytes=2 * model.batch * 248320,
            )
        )
    # Identical stage bounds add linearly. Compact repetitions without claiming
    # that all Delta blocks actually execute before all attention blocks.
    grouped: dict[str, Phase] = {}
    for phase in result:
        previous = grouped.get(phase.name)
        grouped[phase.name] = (
            phase
            if previous is None
            else replace(previous, repeats=previous.repeats + phase.repeats)
        )
    return list(grouped.values())


def workloads(
    model: Qwen36A3B,
    *,
    updates: int,
    source_batches: int,
    eval_batches: int,
    suffix_layers: int,
    rank: int,
) -> list[Workload]:
    for name, value in dict(
        updates=updates,
        source_batches=source_batches,
        eval_batches=eval_batches,
        suffix_layers=suffix_layers,
        rank=rank,
    ).items():
        validate_integer(name, value)
    if rank > model.width:
        raise ValueError("rank exceeds model width")
    if model.delta_algorithm != "chunk64":
        raise ValueError("v2 execution contract requires chunk64")
    if not 0 < suffix_layers < 40 or source_batches > updates or eval_batches % 2:
        raise ValueError(
            "v2 requires suffix < 40, source_batches <= updates and even evaluation batches"
        )
    cut = 40 - suffix_layers
    contract = {
        "id": CONTRACT,
        "reference_version": 2,
        "outputs": "last-token logits; captures at block output",
        "tap_layer": cut - 1,
        "source_elision": True,
        "training_base_prefix_cache": "cold each timed trial",
        "suffix_execution": "full sequence; frozen backbone input-gradient backward",
        "schedule": "sequential block mixer, MoE and head; ideal resource overlap within each stage",
        "routing": "uniform independent top-8 estimate, not measured HBM traffic",
        "memory": "persistent residency plus rough transient estimates; saved autograd/workspace unknown",
        "tp": "unsupported until an execution and collective plan is implemented",
    }
    assumptions = [
        "Versioned execution-path major-work estimate, not a certified minimum-work or kernel-level bound.",
        "BF16 backbone, FP32 featurizer and analytical Delta core; FMA=2 FLOPs; last-token head only.",
        "Source and activation captures stop at the tap. Training caches full base prefix outputs after cold fills inside each timed trial.",
        "Suffix still executes full-sequence layers; no last-token KV/state algorithm or gradient graph pruning assumed.",
        "Chunk Delta backward GEMMs derived for constant initial state and unused final state; triangular/elementwise backward remains approximate.",
        "Uniform independent routing and ideal one-read expert weights. Logical streams are traffic estimates, not compulsory HBM transfers.",
        "Separate mixer/MoE/head phases; BF16 and FP32 execution within a stage assumed serialized; HBM/compute may overlap.",
        "Excludes norms, activations, softmax, loss, additional intermediate traffic and launch overhead. Cache read/write streams included.",
        "Memory feasibility is unknown unless persistent weights/state alone exceed capacity. Saved autograd, workspace and allocator overhead unmodeled.",
        "DP supported; TP excluded. Featurizer backward retains the explicit approximate 2x factor.",
    ]
    result = []
    scalar = feature_phases(model.batch, model.width, rank)
    for name in (
        "inference",
        "activation_harvest",
        "interchange",
        "subspace_apply",
        "dbm_apply",
        "subspace_train",
        "dbm_train",
    ):
        train = name.endswith("_train")
        parameters = model.width * (rank if name.startswith("subspace") else 1)
        apply_features = (
            [scalar["subspace"], scalar["cayley"]]
            if name.startswith("subspace")
            else [scalar["dbm"]]
            if name.startswith("dbm")
            else []
        )
        features = [
            replace(
                p,
                transient_bytes=0,
                resident_replicated_bytes=0,
                repeats=updates if train else 1,
                flops={
                    mode: value * (3 if train else 1) for mode, value in p.flops.items()
                },
            )
            for p in apply_features
        ]
        if train:
            phases = stages(
                model, "source capture", source_batches, end=cut, head=False
            )
            phases += stages(
                model, "base prefix fill", source_batches, end=cut, head=False
            )
            phases += stages(model, "cached base suffix", updates, start=cut)
            phases += stages(
                model, "frozen suffix backward", updates, start=cut, backward=True
            )
            phases += features
            phases.append(
                Phase(
                    "Adam update and gradient synchronization",
                    updates,
                    {"fp32": 15 * parameters},
                    weight_bytes=28 * parameters,
                    gradient_bytes=4 * parameters,
                )
            )
            phases += stages(
                model, "evaluation source", eval_batches // 2, end=cut, head=False
            )
            phases += stages(model, "evaluation base", eval_batches // 2)
            phases += [replace(p, repeats=eval_batches // 2) for p in apply_features]
            # Full base prefix plus last-position source cache. DP partitions both.
            cache = (
                2 * source_batches * model.batch * model.width * (model.sequence + 1)
            )
            phases.append(
                Phase(
                    "cache writes and reads",
                    1,
                    activation_bytes=cache
                    + 2 * updates * model.batch * model.width * (model.sequence + 1),
                )
            )
            resident = 2 * sum(model.parameter_breakdown().values())
            phases = [
                replace(
                    p,
                    resident_sharded_bytes=resident,
                    resident_replicated_bytes=16 * parameters,
                    transient_bytes=p.transient_bytes + cache,
                )
                for p in phases
            ]
        elif name == "activation_harvest":
            phases = stages(model, "activation capture", 1, end=cut, head=False)
        else:
            phases = []
            if name != "inference":
                phases += stages(model, "source capture", 1, end=cut, head=False)
            phases += stages(model, "base forward", 1)
            phases += features
        result.append(
            Workload(
                name,
                model.batch,
                phases,
                assumptions,
                [
                    CONFIG_SOURCE,
                    KERNEL_SOURCE,
                    "docs/qwen36-35b-a3b-architecture.html",
                    "causalab/neural/engines/pytorch_hooks/train.py",
                    "causalab/neural/shared/featurizers.py",
                ],
                tensor_parallel_sizes=[1],
                contract=contract,
            )
        )
    return result
