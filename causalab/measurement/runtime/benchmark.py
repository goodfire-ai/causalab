"""Bind the shared benchmark identity to the documents each worker consumes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..study.scheduler import digest


def benchmark_identity(
    workflow: Mapping[str, Any],
    protocols: Mapping[str, Any],
    observations: Mapping[str, Any],
) -> str:
    return digest(
        {
            "workflow": {
                key: value for key, value in workflow.items() if key != "measurement"
            },
            "protocols": protocols,
            "observations": observations,
        }
    )


@dataclass(eq=False)
class BenchmarkIdentityError(ValueError):
    expected: str
    observed: str

    def __str__(self) -> str:
        return (
            "benchmark definition changed after preparation: "
            f"expected {self.expected}, observed {self.observed}; "
            "restore the workflow and intervention specifications or start a fresh study"
        )
