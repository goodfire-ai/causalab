"""Absolute descriptive statistics for measurement samples."""

import statistics
from typing import Any


def describe(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_variance": statistics.variance(values) if len(values) > 1 else None,
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }
