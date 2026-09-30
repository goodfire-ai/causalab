"""Bounded Nsight Compute counter capture, separate from performance timing.

CLI contract: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
Replay caveats: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .base import CaptureRequest, ProfilerBackend, ProfilerUnavailable

_OPTIONS = {
    "set",
    "sections",
    "metrics",
    "kernel_name",
    "kernel_name_base",
    "launch_skip",
    "launch_count",
    "replay_mode",
    "cache_control",
    "clock_control",
}


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_.]*", value
    ):
        raise ValueError(f"ncu {name} must be a nonempty identifier")
    return value


def _identifiers(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"ncu {name} must be a list of identifiers")
    result = [_identifier(item, name) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"ncu {name} must not contain duplicates")
    return result


def normalize_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate JSON options without importing Torch or invoking the tool.

    Sets/sections/metric names are checked syntactically here; availability is
    specific to the installed tool and GPU, and tool errors remain explicit.
    Kernel filters accept NCU's exact name or ``regex:...`` syntax.
    """
    unknown = set(raw) - _OPTIONS
    if unknown:
        raise ValueError(f"unknown ncu options: {sorted(unknown)}")
    sections = _identifiers(raw.get("sections", []), "sections")
    metrics = _identifiers(raw.get("metrics", []), "metrics")
    metric_set = raw.get("set", None if sections or metrics else "basic")
    if metric_set is not None:
        metric_set = _identifier(metric_set, "set")
    if metric_set is None and not (sections or metrics):
        raise ValueError("ncu requires a set, sections, or metrics")
    kernel_name = raw.get("kernel_name")
    if kernel_name is not None and (
        not isinstance(kernel_name, str)
        or not kernel_name.strip()
        or any(char in kernel_name for char in "\x00\r\n")
        or kernel_name == "regex:"
    ):
        raise ValueError("ncu kernel_name must be a nonempty name or regex: expression")
    kernel_name_base = raw.get("kernel_name_base", "demangled")
    if kernel_name_base not in ("function", "demangled", "mangled"):
        raise ValueError("ncu kernel_name_base must be function, demangled, or mangled")
    launch_skip = raw.get("launch_skip", 0)
    launch_count = raw.get("launch_count", 10)
    for name, value, minimum in (
        ("launch_skip", launch_skip, 0),
        ("launch_count", launch_count, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"ncu {name} must be an integer >= {minimum}")
    replay_mode = raw.get("replay_mode", "kernel")
    if replay_mode != "kernel":
        raise ValueError(
            "ncu replay_mode must be kernel: application/range replay is not "
            "supported by the single-execution capture artifact contract"
        )
    cache_control = raw.get("cache_control", "all")
    if cache_control not in ("all", "none"):
        raise ValueError("ncu cache_control must be all or none")
    clock_control = raw.get("clock_control", "none")
    if clock_control != "none":
        raise ValueError(
            "ncu clock_control must be none; captures do not change GPU clocks"
        )
    return {
        "set": metric_set,
        "sections": sections,
        "metrics": metrics,
        "kernel_name": kernel_name,
        "kernel_name_base": kernel_name_base,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "replay_mode": replay_mode,
        "cache_control": cache_control,
        "clock_control": clock_control,
    }


def _executable() -> str:
    executable = shutil.which("ncu")
    if executable is None:
        raise ProfilerUnavailable("ncu was not found on the compute host PATH")
    return executable


def probe() -> dict[str, Any]:
    executable = _executable()
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProfilerUnavailable(f"ncu version probe failed: {exc}") from exc
    version = (result.stdout + "\n" + result.stderr).strip()
    if not version:
        raise ProfilerUnavailable("ncu version probe returned no version information")
    return {
        "executable": executable,
        "version": version,
        "capabilities": {
            "capture_control": "cuda_profiler_api",
            "replay_modes": ["kernel"],
            "coverage": "bounded matching kernel launches; not an application timeline",
            "performance_timing": False,
            "counter_permissions": "checked during capture; never changed by the harness",
            "default_cache_control": "all",
            "clock_control": "none",
            "section_definitions": "installed stock sections, not user-edited copies",
        },
        "environment": {"NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT": "1"},
    }


def artifacts(request: CaptureRequest) -> list[Path]:
    # NCU expands file macros anywhere in --export, even in directory names.
    if "%" in str(request.output_dir):
        raise ValueError("ncu output_dir must not contain '%' file macros")
    return [request.output_dir / "capture.ncu-rep"]


def command(request: CaptureRequest) -> list[str]:
    if request.mode not in ("cold", "warm"):
        raise ValueError("ncu capture mode must be cold or warm")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", request.device):
        raise ProfilerUnavailable("ncu requires a CUDA device")
    if not request.command:
        raise ValueError("ncu capture requires a child command")
    if not request.command[0] or request.command[0].startswith("-"):
        raise ValueError("ncu child executable must be a path or executable name")
    options = normalize_options(request.options)
    # Disable user config and kernel renaming files to preserve authored options.
    argv = [
        _executable(),
        "--config-file",
        "off",
        "--rename-kernels",
        "off",
        "--target-processes",
        "application-only",
        "--profile-from-start",
        "on" if request.mode == "cold" else "off",
        "--replay-mode",
        options["replay_mode"],
        "--cache-control",
        options["cache_control"],
        "--clock-control",
        options["clock_control"],
        "--kernel-name-base",
        options["kernel_name_base"],
        "--launch-skip",
        str(options["launch_skip"]),
        "--launch-count",
        str(options["launch_count"]),
        # The full workflow must finish and produce comparable observations.
        "--kill",
        "off",
        "--nvtx",
        "--export",
        str(artifacts(request)[0]),
    ]
    if options["set"] is not None:
        argv.extend(["--set", options["set"]])
    for section in options["sections"]:
        argv.extend(["--section", section])
    if options["metrics"]:
        argv.extend(["--metrics", ",".join(options["metrics"])])
    if options["kernel_name"] is not None:
        argv.append(f"--kernel-name={options['kernel_name']}")
    return [*argv, *request.command]


BACKEND = ProfilerBackend(
    name="ncu",
    artifact_format="ncu-rep",
    normalize_options=normalize_options,
    probe=probe,
    command=command,
    artifacts=artifacts,
    capture_control="cuda_profiler_api",
    startup_coverage=(
        "Cold: bounded matching kernels from process launch, which may all be "
        "initialization kernels. Warm: bounded matching kernels between CUDA "
        "profiler API boundaries after warmup. Neither is a CPU/startup timeline."
    ),
    perturbation=(
        "Diagnostic counters only: kernel replay serializes launches and may "
        "replay kernels many times, perturbing concurrency and cache state. "
        "cache_control=all flushes GPU caches between passes; none retains "
        "uncontrolled cache history across passes. GPU clocks are not changed. "
        "Metric sections use installed stock definitions rather than editable user copies. "
        "NCU durations must not be used for before/after performance comparisons."
    ),
    environment={"NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT": "1"},
)
