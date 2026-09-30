"""Measurement math and timing contracts; time/device boundaries are simulated."""

from dataclasses import replace

import pytest

from causalab.sol.benchmark import Case, markdown_report, measure, summarize
from causalab.sol.model import Hardware, Phase, Workload

pytestmark = pytest.mark.unit


class ClockRuntime:
    def __init__(self, events, ranks=1):
        self.events, self.ranks = events, ranks

    def synchronize(self):
        self.events.append("sync")

    def barrier(self):
        self.events.append("barrier")

    def reset_peak_memory(self):
        self.events.append("peak_reset")

    def sample(self, seconds):
        self.events.append("aggregate")
        return {"rank_seconds": [seconds * (rank + 1) for rank in range(self.ranks)]}


def test_summarize_is_inverse_latency_and_does_not_clip():
    result = summarize([2.0, 4.0, 6.0], 1.0)
    assert result["percent_sol"] == 25
    assert result["best_percent_sol"] == 50
    assert result["slowdown_vs_sol"] == 4
    assert result["p95_seconds"] == pytest.approx(5.8)
    result = summarize([0.5, 1.0, 1.5], 2.0)
    assert result["percent_sol"] == 200
    assert result["status"] == "reference_exceeded"


def test_reset_sync_order_warmup_exclusion_and_slowest_rank():
    events = []
    clock_values = iter([0, 100, 200, 202, 300, 304])
    hardware = Hardware("fixture", {"fp32": 100}, 100, 1000, 100, 0, "test")
    work = Workload("w", 2, [Phase("f", 1, {"fp32": 200})], ["test"], ["test"])
    case = Case(
        work,
        lambda: events.append("reset"),
        lambda: events.append("run") or {"forwards": 1},
        lambda: events.append("validate"),
        {"forwards": 1},
    )
    result = measure(
        case,
        hardware,
        ClockRuntime(events, 2),
        dp=2,
        warmup=1,
        repeats=2,
        clock=lambda: next(clock_values),
    )
    assert (
        events
        == [
            "reset",
            "sync",
            "barrier",
            "peak_reset",
            "run",
            "sync",
            "aggregate",
            "validate",
        ]
        * 3
    )
    assert [s["seconds"] for s in result["samples"]] == [4, 8]
    assert result["percent_sol"] == pytest.approx(100 / 6)
    assert len(result["reference_sha256"]) == 64
    assert "16.67%" in markdown_report({"results": [result]})
    broken = replace(case, expected_counts={"forwards": 2})
    with pytest.raises(ValueError, match="observed work"):
        measure(
            broken,
            hardware,
            ClockRuntime([]),
            warmup=0,
            repeats=1,
            clock=iter([0, 1]).__next__,
        )


@pytest.mark.parametrize(
    "samples,sol", [([], 1), ([0], 1), ([-1], 1), ([float("nan")], 1), ([1], 0)]
)
def test_refuse_invalid_measurements(samples, sol):
    with pytest.raises(ValueError):
        summarize(samples, sol)


def test_hardware_detection_refuses_other_skus():
    from causalab.sol.benchmark_qwen import device_profile

    assert device_profile("NVIDIA H100 80GB HBM3", 80_000_000_000) == "h100"
    assert device_profile("NVIDIA B200", 180_000_000_000) == "b200"
    for name, memory in [
        ("NVIDIA H100 PCIe", 80_000_000_000),
        ("NVIDIA H100 NVL", 94_000_000_000),
        ("NVIDIA H100 SXM MIG", 80_000_000_000),
        ("NVIDIA A100", 80_000_000_000),
    ]:
        with pytest.raises(ValueError):
            device_profile(name, memory)
