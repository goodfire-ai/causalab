"""Analytical resource bounds, not runtime predictions. All units are SI.

A phase is one dependency-ordered unit of work for a global batch. Independent
resources may overlap perfectly within a phase; phases execute sequentially.
DP partitions examples, TP partitions tensor work. Only explicitly sharded
storage is divided by TP. No pipeline parallelism or inter-node execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


def _positive(name: str, value: float, *, zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )


def validate_integer(name: str, value: object, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Hardware:
    name: str
    # Per GPU, dense (not sparse) rates at the specified arithmetic mode.
    flops_per_second: dict[str, float]
    hbm_bytes_per_second: float
    memory_bytes: float
    # Per-rank effective injection rate for the assumed collective topology.
    link_bytes_per_second: float
    collective_latency_seconds: float
    provenance: str

    def __post_init__(self) -> None:
        if not self.name or not self.provenance or not self.flops_per_second:
            raise ValueError("hardware needs name, provenance and arithmetic rates")
        for name, rate in self.flops_per_second.items():
            _positive(name, rate)
        for name in ("hbm_bytes_per_second", "memory_bytes", "link_bytes_per_second"):
            _positive(name, getattr(self, name))
        _positive(
            "collective_latency_seconds", self.collective_latency_seconds, zero=True
        )


@dataclass(frozen=True)
class Phase:
    name: str
    repeats: int
    flops: dict[str, float] = field(default_factory=dict)
    # Global-batch activation traffic; divided by DP and TP.
    activation_bytes: float = 0
    # Weight/parameter traffic per replica per execution; divided only by TP.
    weight_bytes: float = 0
    # Persistent / peak transient memory, using the same partition rules.
    resident_sharded_bytes: float = 0
    resident_replicated_bytes: float = 0
    transient_bytes: float = 0
    # One TP all-reduce's payload at DP=1; divided by DP, not TP.
    tp_payload_bytes: float = 0
    tp_collectives: int = 0
    # Trainable-parameter gradients, once per phase execution (update).
    gradient_bytes: float = 0
    # Optional uniform-independent top-k MoE occupancy model. This is the
    # all-expert weight pool; expected fraction touched depends on local tokens.
    routed_weight_bytes: float = 0
    routed_experts: int = 0
    routed_top_k: int = 0
    routed_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase needs a name")
        validate_integer("repeats", self.repeats, 0)
        validate_integer("tp_collectives", self.tp_collectives, 0)
        for name, value in asdict(self).items():
            if name not in {"name", "repeats", "flops", "tp_collectives"}:
                _positive(name, value, zero=True)
        for name, value in self.flops.items():
            _positive(name, value, zero=True)
        for name in ("routed_experts", "routed_top_k", "routed_tokens"):
            validate_integer(name, getattr(self, name), 0)
        routing = (
            self.routed_weight_bytes,
            self.routed_experts,
            self.routed_top_k,
            self.routed_tokens,
        )
        if any(routing) and (
            not all(routing) or self.routed_top_k > self.routed_experts
        ):
            raise ValueError(
                "routing requires positive pool, experts, top-k <= experts and tokens"
            )
        if bool(self.tp_payload_bytes) != bool(self.tp_collectives):
            raise ValueError("TP payload and collective count must both be specified")


@dataclass(frozen=True)
class Workload:
    name: str
    batch_size: int
    phases: list[Phase]
    assumptions: list[str]
    sources: list[str]
    tensor_parallel_sizes: list[int] = field(default_factory=lambda: list(range(1, 9)))
    contract: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_integer("batch_size", self.batch_size)
        if not self.tensor_parallel_sizes:
            raise ValueError("tensor_parallel_sizes must not be empty")
        for size in self.tensor_parallel_sizes:
            validate_integer("tensor parallel size", size)
            if size > 8:
                raise ValueError("tensor parallel size must be <= 8")
        if not self.name or not self.phases or not self.assumptions or not self.sources:
            raise ValueError(
                "workload needs name, phases, assumptions and source paths"
            )


def _allreduce(payload: float, ranks: int, hw: Hardware, count: int = 1) -> float:
    if ranks == 1 or not payload or not count:
        return 0.0
    # Ideal ring: two traversals, p-1 rounds each. Latency is per round.
    return count * (
        2 * (ranks - 1) / ranks * payload / hw.link_bytes_per_second
        + 2 * (ranks - 1) * hw.collective_latency_seconds
    )


def reference(work: Workload, hw: Hardware, dp: int, tp: int) -> dict:
    validate_integer("dp", dp)
    validate_integer("tp", tp)
    if dp * tp > 8 or work.batch_size % dp or tp not in work.tensor_parallel_sizes:
        raise ValueError(
            "require dp*tp <= 8, batch_size divisible by dp and supported tp"
        )
    rows = []
    for phase in work.phases:
        # Distinct arithmetic modes share execution resources conservatively.
        compute = sum(
            n / hw.flops_per_second[mode] for mode, n in phase.flops.items()
        ) / (dp * tp)
        touched_fraction = (
            (
                1
                - (1 - phase.routed_top_k / phase.routed_experts)
                ** (phase.routed_tokens / dp)
            )
            if phase.routed_experts
            else 0
        )
        weight_bytes = phase.weight_bytes + phase.routed_weight_bytes * touched_fraction
        memory = (
            (phase.activation_bytes / dp + weight_bytes) / tp / hw.hbm_bytes_per_second
        )
        tp_time = _allreduce(phase.tp_payload_bytes / dp, tp, hw, phase.tp_collectives)
        dp_time = _allreduce(phase.gradient_bytes / tp, dp, hw)
        communication = tp_time + dp_time  # shared node fabric
        capacity = (
            phase.resident_sharded_bytes / tp
            + phase.resident_replicated_bytes
            + phase.transient_bytes / (dp * tp)
        )
        persistent = phase.resident_sharded_bytes / tp + phase.resident_replicated_bytes
        resources = {"compute": compute, "hbm": memory, "communication": communication}
        rows.append(
            {
                "name": phase.name,
                "weight_bytes_per_gpu": weight_bytes / tp,
                "expected_experts_per_layer": touched_fraction * phase.routed_experts,
                "repeats": phase.repeats,
                "resource_seconds_per_repeat": resources,
                "bottleneck": max(resources, key=lambda k: resources[k]),
                "ideal_overlap_seconds": phase.repeats * max(resources.values()),
                "serialized_resources_seconds": phase.repeats * sum(resources.values()),
                "required_bytes_per_gpu": capacity,
                "fits": False
                if persistent > hw.memory_bytes and phase.repeats
                else None,
                "memory_status": "insufficient_persistent_capacity"
                if persistent > hw.memory_bytes and phase.repeats
                else "unknown",
                "estimated_memory_within_capacity": capacity <= hw.memory_bytes,
            }
        )
    infeasible = any(row["fits"] is False for row in rows)
    return {
        "workload": work.name,
        "dp": dp,
        "tp": tp,
        "gpus": dp * tp,
        "fits": False if infeasible else None,
        "memory_status": "insufficient_persistent_capacity"
        if infeasible
        else "unknown",
        "reference_kind": "conditional_major_work_estimate",
        "distribution_status": "hypothetical_tensor_sharding"
        if tp > 1
        else "data_parallel",
        "contract": work.contract,
        "sol_seconds": sum(row["ideal_overlap_seconds"] for row in rows)
        if not infeasible
        else None,
        "serialized_resources_seconds": sum(
            row["serialized_resources_seconds"] for row in rows
        )
        if not infeasible
        else None,
        "phases": rows,
        "assumptions": work.assumptions,
        "sources": work.sources,
    }


def catalog(workloads: list[Workload], hardware: Hardware) -> dict:
    return {
        "schema_version": 1,
        "hardware": asdict(hardware),
        "interpretation": "Conditional major-work estimates, not certified lower bounds or measured latency. fits=null means memory feasibility is unknown; transient estimates do not certify fit. Serialized resources is not an upper bound.",
        "workloads": [asdict(work) for work in workloads],
        "references": [
            reference(work, hardware, dp, tp)
            for work in workloads
            for dp in range(1, 9)
            for tp in range(1, 8 // dp + 1)
            if work.batch_size % dp == 0 and tp in work.tensor_parallel_sizes
        ],
    }
