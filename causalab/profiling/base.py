"""Torch-free contract for native profiler adapters.

Adapters describe tools and construct argv; the measurement controller owns
process lifetime, study identity, output verification and scientific comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfilerUnavailable(RuntimeError):
    """The requested profiler cannot be used in this compute environment."""


@dataclass(frozen=True)
class CaptureRequest:
    command: tuple[str, ...]
    output_dir: Path
    mode: str
    options: Mapping[str, Any]
    device: str


@dataclass(frozen=True)
class ProfilerBackend:
    name: str
    artifact_format: str
    normalize_options: Callable[[Mapping[str, Any]], dict[str, Any]]
    probe: Callable[[], dict[str, Any]]
    command: Callable[[CaptureRequest], list[str]]
    artifacts: Callable[[CaptureRequest], list[Path]]
    # External tools use CUDA profiler API boundaries for warm captures.
    capture_control: str
    startup_coverage: str
    perturbation: str
    environment: Mapping[str, str] = field(default_factory=dict)
