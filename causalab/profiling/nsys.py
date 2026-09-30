"""Managed Nsight Systems timeline capture, without trace analysis.

CLI semantics: https://docs.nvidia.com/nsight-systems/UserGuide/
The controller owns process lifetime and verifies each returned report path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .base import CaptureRequest, ProfilerBackend, ProfilerUnavailable


def normalize_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(raw) - {"cuda_graph_trace", "cuda_memory_usage"}
    if unknown:
        raise ValueError(f"unknown nsys options: {sorted(unknown)}")
    graph = raw.get("cuda_graph_trace", "node")
    if not isinstance(graph, str) or graph not in {"node", "graph"}:
        raise ValueError("nsys cuda_graph_trace must be 'node' or 'graph'")
    memory = raw.get("cuda_memory_usage", False)
    if not isinstance(memory, bool):
        raise ValueError("nsys cuda_memory_usage must be a boolean")
    return {"cuda_graph_trace": graph, "cuda_memory_usage": memory}


def _executable() -> str:
    executable = shutil.which("nsys")
    if executable is None:
        raise ProfilerUnavailable("nsys was not found on the compute host PATH")
    return executable


def probe() -> dict[str, Any]:
    executable = _executable()
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProfilerUnavailable(f"nsys --version failed: {exc}") from exc
    version = (result.stdout or result.stderr).strip()
    if result.returncode or not version:
        raise ProfilerUnavailable(
            f"nsys --version failed (exit {result.returncode}): {version[:1000]}"
        )
    return {
        "executable": executable,
        "version": version,
        "capabilities": {
            "cold_process_launch": True,
            "warm_cuda_profiler_api": True,
            "traces": ["cuda", "nvtx", "osrt"],
            "cpu_sampling": False,
            "cpu_context_switches": False,
        },
    }


def artifacts(request: CaptureRequest) -> list[Path]:
    return [request.output_dir / "capture.nsys-rep"]


def command(request: CaptureRequest) -> list[str]:
    options = normalize_options(request.options)
    if request.mode not in {"cold", "warm"}:
        raise ValueError("nsys capture mode must be 'cold' or 'warm'")
    if not request.command or request.command[0].startswith("-"):
        raise ValueError("nsys requires a child executable command")
    output = str(request.output_dir / "capture")
    # Nsight expands percent macros even without a shell. Reject them so the
    # controller's exact artifact path and the tool's output cannot diverge.
    if "%" in output:
        raise ValueError("nsys output directory cannot contain '%' macros")
    argv = [
        _executable(),
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--stats=false",
        "--export=none",
        "--force-overwrite=false",
        "--kill=none",
        f"--cuda-graph-trace={options['cuda_graph_trace']}",
        f"--cuda-memory-usage={str(options['cuda_memory_usage']).lower()}",
        f"--output={output}",
    ]
    if request.mode == "warm":
        # Stop collection without stopping the child: it still writes its
        # numerical observations and exits normally after cudaProfilerStop.
        argv += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
    else:
        argv.append("--capture-range=none")
    return [*argv, *request.command]


BACKEND = ProfilerBackend(
    name="nsys",
    artifact_format="nsys-rep",
    normalize_options=normalize_options,
    probe=probe,
    command=command,
    artifacts=artifacts,
    capture_control="cuda_profiler_api",
    startup_coverage=(
        "Cold: CUDA, NVTX and OS runtime tracing from process launch, including "
        "model initialization and first execution. Warm: CUDA profiler API "
        "range after initialization and warmup. CPU instruction sampling and "
        "context-switch collection are disabled; OS runtime tracing is not a "
        "complete CPU execution trace."
    ),
    perturbation=(
        "Separate instrumented execution; do not use its duration as benchmark "
        "timing. CUDA graph node tracing and optional CUDA memory tracing can "
        "add substantial overhead. No native Torch profiler runs concurrently."
    ),
)
