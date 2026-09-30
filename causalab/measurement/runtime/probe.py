"""Untimed execution evidence: actual inputs, Python dispatch and operator names.

Input snapshots and Python profiling are observers, never part of clean timing.
Coverage is explicit: Python dispatch covers the calling thread, while the Torch
profiler also records device kernels. A package being installed is not evidence
that its backend was called.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
import hashlib
from pathlib import Path
import sys
from typing import Any

from ..collection import file_hash
from ..device import require_single_device, require_single_device_runtime


@contextmanager
def observe_execution(bundles):
    import torch

    result: dict[str, Any] = {
        "scope": "instrumented execution, not a clean timing pass",
        "python_thread_coverage": "calling thread; Python callbacks on autograd worker threads may be absent",
        "input_capture": "integer model inputs and attention masks cloned outside CUDA graph capture; content hashed after execution; replay inputs are not independently sampled by model hooks",
        "graph_capture_callbacks_skipped": 0,
    }
    cuda_available = torch.cuda.is_available()
    snapshots, handles = [], []
    calls: Counter[str] = Counter()
    sources = {}
    previous = sys.getprofile()
    selected_names = (
        "attention_forward",
        "gated_delta_rule",
        "grouped_mm",
        "graph_replay",
    )

    def profile(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event != "call":
            return
        name = frame.f_code.co_name
        namespace = frame.f_globals.get("__name__", "")
        if not namespace.startswith(
            ("torch", "transformers", "fla", "causalab.neural")
        ):
            return
        instance = frame.f_locals.get("self")
        cls = type(instance).__qualname__ if instance is not None else ""
        selected = any(part in name for part in selected_names)
        selected |= name == "forward" and any(
            part in cls.lower() for part in ("expert", "sparsemoe", "mlp")
        )
        selected |= (
            name in ("replay", "capture_begin", "capture_end")
            and "graph" in cls.lower()
        )
        if selected:
            label = f"{namespace}.{cls + '.' if cls else ''}{name}"
            calls[label] += 1
            path = Path(frame.f_code.co_filename)
            if path.is_file():
                sources[str(path.resolve())] = None

    def hook(model_key):
        def capture(module, args, kwargs):
            if cuda_available and torch.cuda.is_current_stream_capturing():
                # Graph-pool reuse can overwrite captured clones. Snapshot only
                # ordinary forwards and report the excluded coverage.
                result["graph_capture_callbacks_skipped"] += 1
                return
            values = dict(kwargs)
            if args and "input_ids" not in values and isinstance(args[0], torch.Tensor):
                values["input_ids"] = args[0]
            captured = {}
            mask = values.get("attention_mask")
            padding_key = (
                "attention_mask.linear_attention"
                if isinstance(mask, Mapping)
                else "attention_mask"
            )
            with torch.profiler.record_function("observer:model_inputs"):
                for name in (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "cache_position",
                    "token_type_ids",
                ):
                    value = values.get(name)
                    if isinstance(value, torch.Tensor):
                        captured[name] = value.detach().clone()
                if isinstance(mask, Mapping):
                    for kind, value in mask.items():
                        if isinstance(value, torch.Tensor):
                            captured[f"attention_mask.{kind}"] = value.detach().clone()
            if captured:
                snapshots.append((model_key, captured, padding_key, mask is not None))

        return capture

    try:
        for key, bundle in bundles.items():
            handles.append(
                bundle.model.register_forward_pre_hook(hook(key), with_kwargs=True)
            )
        sys.setprofile(profile)
        yield result
    finally:
        sys.setprofile(previous)
        for handle in handles:
            handle.remove()
        inputs, logical_rows = [], []
        for key, tensors, padding_key, has_mask in snapshots:
            row: dict[str, Any] = {"model": key, "tensors": {}}
            cpu = {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}
            for name, tensor in cpu.items():
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                row["tensors"][name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            inputs.append(row)
            ids, mask = cpu.get("input_ids"), cpu.get(padding_key)
            if ids is not None and ids.ndim == 2:
                if has_mask and (mask is None or mask.shape != ids.shape):
                    continue  # A transformed attention mask cannot identify padding.
                for index, tokens in enumerate(ids):
                    if mask is not None:
                        tokens = tokens[mask[index].bool()]
                    logical_rows.append({"model": key, "token_ids": tokens.tolist()})
        result.update(
            model_inputs=inputs,
            logical_token_rows=logical_rows,
            python_dispatch=dict(sorted(calls.items())),
            source_files={path: file_hash(Path(path)) for path in sorted(sources)},
        )


def probe_case(
    prepare,
    bundles,
    directory: Path,
    *,
    case: str,
    seed: int,
    device: str,
    profile: bool = True,
):
    """Warm and reset once before observing a fresh execution of the same case."""
    import torch

    require_single_device_runtime(device)
    for bundle in bundles.values():
        geometry = getattr(bundle, "geometry", None)
        placement = getattr(bundle, "devices", None)
        require_single_device(
            getattr(placement, "spelling", device), world=getattr(geometry, "world", 1)
        )
    directory.mkdir(parents=True, exist_ok=False)
    target = torch.device(device)
    with prepare(case, seed, directory / "warmup") as operation:
        operation.run()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
    operators = Counter()
    kernels = Counter()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if target.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with prepare(case, seed, directory / "observed") as operation:
        with observe_execution(bundles) as observed:
            with (
                torch.profiler.profile(activities=activities)
                if profile
                else nullcontext()
            ) as profiler:
                operation.run()
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
        if profile:
            events = profiler.events()
            if events is None:
                raise ValueError(
                    f"execution probe for {case} produced no profiler events"
                )
            for event in events:
                if "CUDA" in str(event.device_type):
                    kernels[event.name] += 1
                elif event.name.startswith(("aten::", "cudaGraph")):
                    operators[event.name] += 1
    if not observed["logical_token_rows"]:
        raise ValueError(
            f"execution probe for {case} observed no token inputs; "
            "this execution path needs a compatible input observer"
        )
    if profile and target.type == "cuda" and not kernels:
        raise ValueError(f"execution probe for {case} recorded no CUDA kernel evidence")
    return {
        "case": case,
        "seed": seed,
        "conditions": "one warmup followed by a fresh scientific reset; resident model; profiling outside measurement timers",
        "native_profile": {"status": "completed" if profile else "not_requested"},
        "observed": observed,
        "operators": dict(sorted(operators.items())),
        "device_kernels": dict(sorted(kernels.items())),
    }


def reuse_evidence(record):
    """Dynamic counts remain evidence, rather than a claimed static workload."""
    import json

    observed = record["observed"]
    unique_inputs = {
        json.dumps(row, sort_keys=True): row for row in observed["model_inputs"]
    }
    logical_rows = {
        json.dumps(row, sort_keys=True): row for row in observed["logical_token_rows"]
    }
    return {
        "case": record["case"],
        "seed": record["seed"],
        "conditions": record["conditions"],
        "native_profile": record["native_profile"],
        "model_inputs": [unique_inputs[key] for key in sorted(unique_inputs)],
        "logical_token_rows": [logical_rows[key] for key in sorted(logical_rows)],
        "python_dispatch": sorted(observed["python_dispatch"]),
        "source_files": observed["source_files"],
        "operators": sorted(record["operators"]),
        "device_kernels": sorted(record["device_kernels"]),
    }
