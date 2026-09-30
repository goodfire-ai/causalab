"""Synchronized wall-clock measurements against an explicit work ledger.

Torch-free orchestration: a runtime owns device/rank synchronization. Timing
includes Python dispatch and device work, but excludes setup/reset, validation,
and aggregation of rank times. Each sample measures one complete catalog workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
import time
from typing import Callable, Mapping, Protocol

from causalab.sol.model import Hardware, Workload, validate_integer, reference


class Runtime(Protocol):
    def synchronize(self) -> None: ...
    def barrier(self) -> None: ...
    def reset_peak_memory(self) -> None: ...
    def sample(self, seconds: float) -> dict: ...


@dataclass
class Case:
    workload: Workload
    reset: Callable[[], None]
    run: Callable[[], Mapping[str, int]]
    validate: Callable[[], None]
    expected_counts: Mapping[str, int]


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def summarize(seconds: list[float], sol_seconds: float) -> dict:
    if not seconds or any(not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError("measurements must be finite positive seconds")
    if not math.isfinite(sol_seconds) or sol_seconds <= 0:
        raise ValueError("reference must be finite positive seconds")
    ordered = sorted(seconds)
    index = 0.95 * (len(ordered) - 1)
    lower = math.floor(index)
    p95 = ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (
        index - lower
    )
    median = statistics.median(seconds)
    return {
        "min_seconds": min(seconds),
        "median_seconds": median,
        "p95_seconds": p95,
        "max_seconds": max(seconds),
        "mean_seconds": statistics.mean(seconds),
        "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "sol_seconds": sol_seconds,
        "percent_sol": 100 * sol_seconds / median,
        "best_percent_sol": 100 * sol_seconds / min(seconds),
        "slowdown_vs_sol": median / sol_seconds,
        "status": "reference_exceeded" if min(seconds) < sol_seconds else "ok",
    }


def measure(
    case: Case,
    hardware: Hardware,
    runtime: Runtime,
    *,
    dp: int = 1,
    tp: int = 1,
    warmup: int = 1,
    repeats: int = 5,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] = lambda _: None,
) -> dict:
    """Each trial resets state, synchronizes all participants and measures once.

    Runtime.sample must return `rank_seconds`, one finite duration per rank,
    and may add peak-memory fields. The slowest rank determines sample latency.
    A custom TP backend can use this API with its actual layout and collectives.
    """
    validate_integer("warmup", warmup, 0)
    validate_integer("repeats", repeats)
    ref = reference(case.workload, hardware, dp, tp)
    if ref["sol_seconds"] is None or ref["sol_seconds"] <= 0:
        raise ValueError("workload has no positive feasible reference for this layout")
    samples = []
    for trial in range(warmup + repeats):
        progress(
            f"{case.workload.name}: {'warmup' if trial < warmup else 'sample'} "
            f"{trial + 1}/{warmup + repeats}"
        )
        case.reset()
        runtime.synchronize()
        runtime.barrier()
        runtime.reset_peak_memory()
        start = clock()
        observed = dict(case.run())
        runtime.synchronize()
        elapsed = clock() - start
        # Collectives and validation are outside the timed interval.
        sample = runtime.sample(elapsed)
        if observed != dict(case.expected_counts):
            raise ValueError(
                f"{case.workload.name}: observed work {observed} != "
                f"expected {dict(case.expected_counts)}"
            )
        case.validate()
        times = sample["rank_seconds"]
        if len(times) != dp * tp:
            raise ValueError("rank timing count does not match DP*TP")
        if any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("rank timings must be finite and positive")
        sample["seconds"] = max(times)
        sample["observed_counts"] = observed
        if trial >= warmup:
            samples.append(sample)
    identity = {
        "hardware": asdict(hardware),
        "workload": asdict(case.workload),
        "dp": dp,
        "tp": tp,
    }
    return {
        "workload": case.workload.name,
        "dp": dp,
        "tp": tp,
        "reference_sha256": fingerprint(identity),
        "reference": ref,
        "workload_ledger": asdict(case.workload),
        "warmup": warmup,
        "repeats": repeats,
        "timing_scope": "synchronized operation wall time; excludes reset, validation and timing aggregation",
        "expected_counts": dict(case.expected_counts),
        "samples": samples,
        **summarize([sample["seconds"] for sample in samples], ref["sol_seconds"]),
    }


def markdown_report(report: dict) -> str:
    rows = [
        "# Measured efficiency against the conditional reference",
        "",
        "`% SOL = 100 × reference seconds / measured median seconds`. "
        "Samples use the slowest rank; higher is better. Values over 100% "
        "indicate a reference/workload mismatch or a faster algorithm, and are not clipped.",
        "",
        "| Operation | SOL (ms) | Median (ms) | p95 (ms) | % SOL | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if report.get("execution_contract"):
        rows[2:2] = [f"Execution contract: `{report['execution_contract']}`.", ""]
    for result in report["results"]:
        rows.append(
            f"| {result['workload']} | {1000 * result['sol_seconds']:.3f} | "
            f"{1000 * result['median_seconds']:.3f} | {1000 * result['p95_seconds']:.3f} | "
            f"{result['percent_sol']:.2f}% | {result['status']} |"
        )
    if report.get("error"):
        rows += ["", "Run incomplete. See the JSON error and completed results."]
    rows += [
        "",
        "The JSON records hardware, software, reference assumptions, input geometry, "
        "raw rank timings, observed operation counts and measured peak memory. "
        "These percentages are conditional on that reference, not measured GPU utilization.",
        "Reference times are major-work estimates, not certified lower bounds. Memory feasibility remains unknown unless persistent residency alone exceeds capacity.",
        "",
    ]
    return "\n".join(rows)
