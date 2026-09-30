"""PyTorch profiler adapter; importing it does not initialize Torch or CUDA."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from typing import Any

from .base import CaptureRequest, ProfilerBackend, ProfilerUnavailable


def normalize_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    if set(raw) - {"record_shapes", "with_stack"}:
        raise ValueError("unknown torch profiler option")
    result = {name: raw.get(name, False) for name in ("record_shapes", "with_stack")}
    if any(type(value) is not bool for value in result.values()):
        raise ValueError("torch profiler record_shapes/with_stack must be boolean")
    return result


def probe() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ProfilerUnavailable("torch is not installed") from exc
    return {"tool": "torch.profiler", "version": version}


def command(request: CaptureRequest) -> list[str]:
    return list(request.command)


def artifacts(request: CaptureRequest):
    return [request.output_dir / "trace.json"]


BACKEND = ProfilerBackend(
    name="torch",
    artifact_format="chrome_trace_json",
    normalize_options=normalize_options,
    probe=probe,
    command=command,
    artifacts=artifacts,
    capture_control="torch",
    startup_coverage="starts after Torch import and profiler CUDA initialization; before Worker/model construction",
    perturbation="instrumented execution; durations are not benchmark timings",
)
