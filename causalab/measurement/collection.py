"""Collection primitives shared by operation and workflow measurements.

Callers supply an operation and its reset context. Comparison runs as a workflow
analysis step. Imports remain torch-free until collection starts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Literal

from .device import require_single_device_runtime


@dataclass(frozen=True)
class Operation:
    """One prepared operation, with observation extraction outside the clock.

    Observation keys identify logical examples/sites. Tensor coordinates must have
    the same semantic meaning across arms; exclude padding before returning them.
    ``prepare`` must restore all scientific state for each pass, including RNG,
    parameters and optimizer. The collector never guesses how to reset a fit.
    """

    run: Callable[[], Any]
    observe: Callable[[Any], Mapping[str, Any]]
    profile_context: Callable[[], AbstractContextManager[Mapping[str, Any]]] | None = (
        None
    )
    numerics_context: Callable[[], AbstractContextManager[Mapping[str, Any]]] | None = (
        None
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_record(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete receipt; never leave a truncated JSON record."""
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data)
    temporary.replace(path)


def execution_identity(device: Any) -> dict[str, Any]:
    import torch

    from causalab.provenance import runtime_identity

    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "node": platform.node(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware.update(
            name=props.name,
            memory_bytes=props.total_memory,
            uuid=str(getattr(props, "uuid", "unknown")),
        )
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            hardware["driver_versions"] = sorted(set(driver.split())) or None
        except (OSError, subprocess.SubprocessError):
            hardware["driver_versions"] = None
    packages = {}
    for name in (
        "torch",
        "safetensors",
        "numpy",
        "transformers",
        "triton",
        "flash-linear-attention",
        "fla-core",
        "flash-attn",
        "causal-conv1d",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "implementation": asdict(runtime_identity()),
        "environment": {
            "python": platform.python_version(),
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "cudnn_runtime": torch.backends.cudnn.version(),
            "default_dtype": str(torch.get_default_dtype()),
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "autocast_enabled": torch.is_autocast_enabled(device.type),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "runtime_variables": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_TF32_OVERRIDE",
                    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                    # The package's execution switches (docs/measurement.md
                    # lists them). Literal names, so an arm whose code
                    # predates a switch still records it, as unset. Keep an
                    # allowlist: other env vars may contain credentials and
                    # do not belong in receipts.
                    "CAUSALAB_MOE_GLUE",
                    "CAUSALAB_FUSED_NORMS",
                    "CAUSALAB_GDN_SHORT_SEQ",
                    "CAUSALAB_PROJECT_HEAD_UNDER_GRAD",
                    "CAUSALAB_COMPILE_CACHE",
                )
            },
        },
        "hardware": hardware,
        "dispatch": {"status": "unknown", "reason": "no backend observer attached"},
    }


def collect(
    prepare: Callable[[int, Path], AbstractContextManager[Operation]],
    output: Path,
    *,
    case: str,
    input_identity: str,
    scope: str,
    reset_policy: str,
    seeds: Sequence[int] = (0,),
    repeats: int = 3,
    warmups: int = 1,
    device: str = "cpu",
    profile: bool = False,
    profile_options: Mapping[str, Any] | None = None,
    observation_specs: Mapping[str, Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
    mode: Literal["single", "comparison"] = "comparison",
    observation_policy: Literal["required", "not_requested"] = "required",
) -> Path:
    """Collect clean timing and optional observation passes for one source case.

    Preparation, cleanup, observation extraction and serialization are untimed.
    Every pass, including warmup and trace capture, starts from fresh scientific
    state. Traces use the first measured seed/repeat in a separate pass.
    The output directory must be new; the study scheduler owns resume.

    Supports one CPU/CUDA device. ``context`` records caller-supplied model, data
    and geometry facts separately from observed runtime provenance.
    """
    if mode not in {"single", "comparison"} or observation_policy not in {
        "required",
        "not_requested",
    }:
        raise ValueError("invalid measurement mode or observation policy")
    numerical = observation_policy == "required"
    if not numerical and (mode != "single" or observation_specs):
        raise ValueError(
            "omitted observations require a timing-only single measurement"
        )
    for name, value in (("repeats", repeats), ("warmups", warmups)):
        if type(value) is not int or value < (1 if name == "repeats" else 0):
            raise ValueError(f"{name} has an invalid count")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    for name, value in (
        ("case", case),
        ("input_identity", input_identity),
        ("scope", scope),
        ("reset_policy", reset_policy),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if type(profile) is not bool:
        raise ValueError("profile must be boolean")
    # Check serializability before executing anything or creating outputs.
    declared = json.loads(json.dumps(dict(context or {}), allow_nan=False))
    specs = json.loads(json.dumps(dict(observation_specs or {}), allow_nan=False))
    options = dict(profile_options or {})
    if set(options) - {"record_shapes", "with_stack", "reason"}:
        raise ValueError("unknown profiling option")
    for flag in ("record_shapes", "with_stack"):
        if type(options.get(flag, False)) is not bool:
            raise ValueError(f"profile option {flag} must be boolean")

    import torch
    from safetensors.torch import save_file

    require_single_device_runtime(device)
    target = torch.device(device)
    identity = execution_identity(target)
    factory = inspect.unwrap(prepare)
    try:
        source = inspect.getsourcefile(factory)
    except TypeError:
        source = None
    identity["adapter"] = {
        "module": getattr(factory, "__module__", None),
        "qualname": getattr(factory, "__qualname__", None),
        "source_sha256": file_hash(Path(source))
        if source and Path(source).is_file()
        else None,
        "closure_state": "caller-declared context; not automatically attested",
    }
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "measurement.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "observation_policy": observation_policy,
        "status": "running",
        "case": case,
        "input_identity": input_identity,
        "scope": scope,
        "reset_policy": reset_policy,
        "alignment": "logical_keys_and_caller_aligned_tensor_coordinates",
        "provenance": identity,
        "context": declared,
        "observation_specs": specs,
        "plan": {"seeds": list(seeds), "repeats": repeats, "warmups": warmups},
        "passes": "independently reset timing, numerics, and optional profiling"
        if numerical
        else "independently reset timing and optional profiling; no numerical-only passes",
        "started_at_ns": time.time_ns(),
        "warmups": [],
        "samples": [],
        "trace": {"status": "not_requested"},
    }
    write_record(receipt, record)

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def execute(
        seed: int, directory: Path, *, observe: bool
    ) -> tuple[float, Any, Any, Any]:
        directory.mkdir()
        with prepare(seed, directory) as operation:
            synchronize()
            if target.type == "cuda":
                torch.cuda.reset_peak_memory_stats(target)
            with (
                operation.numerics_context()
                if observe and operation.numerics_context is not None
                else nullcontext(None)
            ) as numerics_context:
                started = time.perf_counter_ns()
                result = operation.run()
                synchronize()
                seconds = (time.perf_counter_ns() - started) / 1e9
            if seconds <= 0:
                raise ValueError("timer did not advance during operation")
            memory = (
                None
                if target.type == "cpu"
                else {
                    "allocated_bytes": torch.cuda.max_memory_allocated(target),
                    "reserved_bytes": torch.cuda.max_memory_reserved(target),
                }
            )
            tensors = None
            if observe:
                values = operation.observe(result)
                if not values or any(type(k) is not str or not k for k in values):
                    raise ValueError("observations need nonempty logical string keys")
                tensors = {}
                for key, tensor in values.items():
                    if not isinstance(tensor, torch.Tensor) or not tensor.numel():
                        raise ValueError(
                            f"observation {key!r} must be a nonempty tensor"
                        )
                    if tensor.is_complex() or tensor.layout != torch.strided:
                        raise ValueError("observations must be dense real tensors")
                    tensors[key] = tensor.detach().cpu().contiguous().clone()
            return seconds, tensors, memory, numerics_context

    try:
        for i in range(warmups):
            elapsed, _, _, _ = execute(seeds[0], output / f"warmup_{i}", observe=False)
            record["warmups"].append({"repeat": i, "seconds": elapsed})
            write_record(receipt, record)
        for seed in seeds:
            for repeat in range(repeats):
                label = f"seed_{seed}_repeat_{repeat}"
                elapsed, _, memory, _ = execute(
                    seed, output / f"{label}_timing", observe=False
                )
                sample = {
                    "seed": seed,
                    "repeat": repeat,
                    "seconds": elapsed,
                    "peak_memory": memory,
                    "timing_directory": f"{label}_timing",
                }
                if numerical:
                    _, tensors, _, numerics_context = execute(
                        seed, output / f"{label}_numerics", observe=True
                    )
                    artifact = output / f"{label}.safetensors"
                    save_file(tensors, str(artifact))
                    sample.update(
                        numerics_directory=f"{label}_numerics",
                        numerics_context=numerics_context,
                        observations={
                            "file": artifact.name,
                            "sha256": file_hash(artifact),
                        },
                    )
                else:
                    sample["observation_status"] = "not_requested"
                record["samples"].append(sample)
                write_record(receipt, record)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_record(receipt, record)
        raise

    if profile:
        trace = output / "trace.json"
        trace_record: dict[str, Any] = {
            "status": "running",
            "seed": seeds[0],
            "repeat": 0,
        }
        if not numerical:
            trace_record["observation_status"] = "not_requested"
        record["trace"] = trace_record
        write_record(receipt, record)
        try:
            work = output / "profile"
            work.mkdir()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if target.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with (
                prepare(seeds[0], work) as operation,
                (
                    operation.profile_context()
                    if operation.profile_context is not None
                    else nullcontext({"policy": "whole-case annotation only"})
                ) as instrumentation,
            ):
                synchronize()
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=options.get("record_shapes", False),
                    with_stack=options.get("with_stack", False),
                ) as profiler:
                    with torch.profiler.record_function(f"case:{case}"):
                        result = operation.run()
                        synchronize()
                # Captured outputs attest this pass; never compare timing under the profiler.
                if numerical:
                    values = {
                        k: v.detach().cpu().contiguous().clone()
                        for k, v in operation.observe(result).items()
                    }
                    profiled = output / "profile.safetensors"
                    save_file(values, str(profiled))
                    trace_record["observations"] = {
                        "file": profiled.name,
                        "sha256": file_hash(profiled),
                    }
                else:
                    trace_record["observation_status"] = "not_requested"
                profiler.export_chrome_trace(str(trace))
            trace_record.update(
                status="completed",
                file=trace.name,
                sha256=file_hash(trace),
                tool="torch.profiler",
                tool_version=torch.__version__,
                activities=[a.name for a in activities],
                record_shapes=options.get("record_shapes", False),
                with_stack=options.get("with_stack", False),
                reason=options.get("reason", "whole selected case"),
                profile_memory=False,
                scope=f"case:{case}",
                instrumentation=instrumentation,
            )
        except Exception as exc:
            trace_record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        except BaseException as exc:
            record.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
            record["trace"].update(status="interrupted")
            write_record(receipt, record)
            raise
    record.update(status="completed", ended_at_ns=time.time_ns())
    write_record(receipt, record)
    return receipt
