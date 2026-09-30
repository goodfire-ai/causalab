"""Transparent dense-transformer example recipes; not an arbitrary HF adapter."""

from dataclasses import dataclass, replace
from typing import Protocol

from causalab.sol.model import Phase, Workload, validate_integer


class ForwardModel(Protocol):
    @property
    def layers(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def batch(self) -> int: ...
    def forward(self, name: str, repeats: int = 1) -> Phase: ...
    def suffix_backward(self, layers: int, repeats: int) -> Phase: ...


@dataclass(frozen=True)
class DenseTransformer:
    layers: int
    width: int
    mlp_width: int
    vocabulary: int
    sequence: int
    batch: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            validate_integer(name, value)

    def forward(self, name: str, repeats: int = 1) -> Phase:
        # MHA Q,K,V,O plus gated MLP; full-sequence vocabulary projection.
        b, s, d, m, layers, v = (
            self.batch,
            self.sequence,
            self.width,
            self.mlp_width,
            self.layers,
            self.vocabulary,
        )
        params = layers * (4 * d * d + 3 * d * m) + d * v
        flops = 2 * b * s * params + 4 * b * s * s * d * layers
        return Phase(
            name,
            repeats,
            {"bf16_dense": flops},
            activation_bytes=2 * b * s * d * layers * 2,
            weight_bytes=2 * params,
            resident_sharded_bytes=2 * (params + d * v),
            transient_bytes=2 * b * s * (d + v),
            tp_payload_bytes=2 * b * s * d,
            tp_collectives=2 * layers,
        )

    def suffix_backward(self, layers: int, repeats: int) -> Phase:
        return replace(self, layers=layers).forward("frozen suffix backward", repeats)


def feature_phases(n: int, d: int, k: int) -> dict[str, Phase]:
    # One-position source+base encode and base inverse: five (N,d)@(d,k)
    # products, plus residual reconstruction. Parametrization is separate.
    projection = Phase(
        "subspace projections",
        1,
        {"fp32": 10 * n * d * k},
        activation_bytes=4 * n * (6 * d + 3 * k),
        weight_bytes=4 * d * k,
        resident_replicated_bytes=4 * d * k,
    )
    gate = Phase(
        "gate swap",
        1,
        {"fp32": 6 * n * d},
        activation_bytes=4 * n * d * 5,
        weight_bytes=4 * d,
        resident_replicated_bytes=4 * d,
    )
    # featurize(source), featurize(base), inverse: each materializes Q once.
    # Cayley.forward has five width-dependent and five k-by-k GEMMs.
    cayley = Phase(
        "Cayley materialization GEMM floor",
        1,
        {"fp32": 3 * (10 * d * k * k + 10 * k**3)},
        resident_replicated_bytes=8 * d * k,
    )
    return {"subspace": projection, "dbm": gate, "cayley": cayley}


def feature_workloads(
    model: ForwardModel,
    *,
    updates: int,
    source_batches: int,
    eval_batches: int,
    suffix_layers: int,
    rank: int,
    assumptions: list[str],
    sources: list[str],
) -> list[Workload]:
    """Build canonical recipes with explicitly supplied execution counts.

    ``eval_batches`` counts batches across all evaluation passes, including any
    final reporting. ``source_batches`` counts unique cold cache fills.
    Early stopping is not predicted; source runs count as full forwards.
    """
    for name, value in {
        "updates": updates,
        "source_batches": source_batches,
        "eval_batches": eval_batches,
        "suffix_layers": suffix_layers,
        "rank": rank,
    }.items():
        validate_integer(name, value)
    if suffix_layers > model.layers or rank > model.width:
        raise ValueError("suffix_layers/rank exceed model dimensions")
    source = model.forward("source capture", source_batches)
    base = model.forward("base/intervened forward")

    def work(
        name: str, phases: list[Phase], extra: list[str] | None = None
    ) -> Workload:
        return Workload(name, model.batch, phases, assumptions + (extra or []), sources)

    n, d, k = model.batch, model.width, rank
    features = feature_phases(n, d, k)
    projection, gate, cayley = features["subspace"], features["dbm"], features["cayley"]
    results = [
        work("inference", [base]),
        work("activation_harvest", [replace(base, name="forward and capture")]),
        work("interchange", [replace(source, repeats=1), base]),
        work("subspace_apply", [replace(source, repeats=1), base, projection, cayley]),
        work("dbm_apply", [replace(source, repeats=1), base, gate]),
    ]
    # Frozen backbone: input gradient GEMMs cost approximately one forward
    # over the affected suffix, not two full-model backward forwards.
    suffix = model.suffix_backward(suffix_layers, updates)
    for name, feature, parameters in [
        ("subspace_train", projection, d * k),
        ("dbm_train", gate, d),
    ]:
        feature_train = replace(
            feature, repeats=updates, flops={"fp32": 3 * feature.flops["fp32"]}
        )
        optimizer = Phase(
            "Adam update and gradient synchronization",
            updates,
            {"fp32": 15 * parameters},
            weight_bytes=28 * parameters,
            resident_sharded_bytes=base.resident_sharded_bytes,
            resident_replicated_bytes=16 * parameters,
            gradient_bytes=4 * parameters,
        )
        phases = [
            source,
            replace(base, repeats=updates),
            suffix,
            feature_train,
            optimizer,
            model.forward("evaluation (total batches)", eval_batches),
        ]
        extra = [
            "Frozen suffix backward approximated by suffix forward GEMMs; no backbone weight gradients.",
            "Feature backward approximated as twice feature forward; Adam uses 15 scalar FLOPs/parameter.",
            "Source cache fill count and aggregate evaluation count are supplied, not inferred from epochs.",
            "Recipes aggregate phase counts; dependency order is represented only by additive phase bounds.",
        ]
        if name == "subspace_train":
            phases.append(
                replace(
                    cayley,
                    repeats=updates,
                    flops={"fp32": 3 * cayley.flops["fp32"]},
                )
            )
            extra.append(
                "Three basis accesses/update; Cayley GEMM floor only, excluding inverse/norm and initialization; backward factor 2. Evaluation featurizer work excluded."
            )
        # Account for saved source captures and optimizer state in every phase.
        phases = [
            replace(
                p,
                resident_replicated_bytes=p.resident_replicated_bytes + 16 * parameters,
                resident_sharded_bytes=max(
                    p.resident_sharded_bytes, base.resident_sharded_bytes
                ),
                transient_bytes=p.transient_bytes + source_batches * n * d * 2,
            )
            for p in phases
        ]
        results.append(work(name, phases, extra))
    return results


def dense_catalog_workloads(
    model: DenseTransformer,
    *,
    updates: int,
    source_batches: int,
    eval_batches: int,
    suffix_layers: int,
    rank: int,
) -> list[Workload]:
    assumptions = [
        "Synthetic dense MHA/gated-MLP architecture; bf16 weights, fp32 featurizers; FMA=2 FLOPs.",
        "Fixed padded global batch and sequence; ideal balanced DP/TP with sharded activations.",
        "Major GEMMs only: excludes norms, softmax, RNG, losses, launch/host/I/O and cache transfers.",
        "Activation traffic is an optimistic compulsory floor, not kernel-level traffic.",
        "Full forwards including full-sequence lm_head; source elision is not assumed.",
        "No checkpoint recompute; transient memory is a floor, not allocator-fit certification.",
        "TP uses two ring all-reduces per transformer block; embedding/head collectives omitted.",
    ]
    sources = [
        "causalab/protocol/plan.py",
        "causalab/neural/shared/executor_base.py",
        "causalab/neural/shared/featurizers.py",
        "causalab/neural/engines/pytorch_hooks/train.py",
    ]
    return feature_workloads(
        model,
        updates=updates,
        source_batches=source_batches,
        eval_batches=eval_batches,
        suffix_layers=suffix_layers,
        rank=rank,
        assumptions=assumptions,
        sources=sources,
    )
