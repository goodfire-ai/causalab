"""One native capture in an otherwise isolated arm process, with file IPC."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import traceback

from ..collection import file_hash, write_record
from ..device import require_single_device_runtime
from ..runtime.cache import COLD_RESET_POLICY, RESIDENT_RESET_POLICY, cache_provenance
from ...profiling import get_backend


@contextmanager
def _cuda_capture(torch, device):
    torch.cuda.synchronize(device)
    runtime = torch.cuda.cudart()
    torch.cuda.check_error(runtime.cudaProfilerStart())
    try:
        yield
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            torch.cuda.check_error(runtime.cudaProfilerStop())


def capture(config):
    """Torch cold capture begins before Worker construction; warm excludes setup."""
    require_single_device_runtime(config["device"])
    import torch
    from safetensors.torch import save_file

    from ..runtime.worker import Worker
    from ..runtime.observations import observation_specs
    from .ranges import phase_ranges

    numerical = config["plan"].get("observation_policy", "required") == "required"
    command = config["capture"]
    target = Path(command["directory"])
    backend = get_backend(command["backend"])
    device = config["device"]
    mode = command["mode"]
    native = backend.capture_control == "torch"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = None

    def context():
        nonlocal profiler
        if native:
            profiler = torch.profiler.profile(
                activities=activities, **command["options"]
            )
            return profiler
        return _cuda_capture(torch, device) if mode == "warm" else nullcontext()

    def synchronize():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    cold_context = context() if mode == "cold" else nullcontext()
    with ExitStack() as lifetime:
        with cold_context:
            worker = Worker(config)
            warmups = max(1, config["plan"]["warmups"]) if mode == "warm" else 0
            for index in range(warmups):
                work = target / f"warmup_{index}"
                work.mkdir()
                with worker.prepare(
                    command["case"], command["seed"], work
                ) as operation:
                    operation.run()
                    synchronize()
            work = target / "work"
            work.mkdir()
            operation = lifetime.enter_context(
                worker.prepare(command["case"], command["seed"], work)
            )
            # prepare restores RNG and reconstructs fits/optimizers after warmup.
            synchronize()
            with phase_ranges(
                worker.engine,
                bundles=worker.bundles,
                operation_step=worker.plan["cases"][command["case"]].get("step"),
            ) as instrumentation:
                with context() if mode == "warm" else nullcontext():
                    with torch.profiler.record_function(f"case:{command['case']}"):
                        result = operation.run()
                        synchronize()
            # Cold native capture covers setup + required workflow saves; observation
            # extraction, tensor export and receipt hashing below remain outside it.
        evidence = {}
        if numerical:
            values = {
                key: value.detach().cpu().contiguous().clone()
                for key, value in operation.observe(result).items()
            }
            specs = observation_specs(
                work / worker.loaded.document.output_dir,
                worker.plan["observations"],
                step=worker.plan["cases"][command["case"]].get("step"),
            )
            observed = target / "observations.safetensors"
            save_file(values, str(observed))
            evidence.update(
                observations={"file": observed.name, "sha256": file_hash(observed)},
                observation_specs=specs,
            )
        else:
            evidence["observation_status"] = "not_requested"
        output_root = work / worker.loaded.document.output_dir
        evidence["output_files"] = {
            path.relative_to(target).as_posix(): file_hash(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        }
    if profiler is not None:
        profiler.export_chrome_trace(str(target / "trace.json"))
    return {
        "status": "completed",
        "identity": worker.identity,
        "tool": {"tool": "torch.profiler", "version": torch.__version__}
        if native
        else None,
        **evidence,
        "instrumentation": instrumentation,
        "coverage": {
            "case": command["case"],
            "warmups": warmups,
            "scope": "fresh process initialization, model load and first workflow execution"
            if mode == "cold"
            else "resident prepared operation; imports, model loading, warmup and operation preparation excluded",
            "reset_policy": COLD_RESET_POLICY
            if mode == "cold"
            else RESIDENT_RESET_POLICY,
            "cache_policy": cache_provenance(
                "cold_process" if mode == "cold" else "resident"
            ),
            "retained_state": "OS/HF disk caches retained; warm runtime/model/allocator and host memos retained after warmup; engine graph reuse follows normal execution and is not forced",
            "checkpoint_hashing": "reuses prior clean-session attested checkpoint hashes; excluded from capture",
            "execution_probes": "reuses prior clean-session evidence; no nested native profiler",
            "capture_control": backend.capture_control,
            "diagnostic_tail": (
                (
                    "external cold capture continues through observation extraction, tensor export, hashes and result serialization until process exit"
                    if numerical
                    else "external cold capture continues through output hashes and result serialization until process exit"
                )
                if mode == "cold" and not native
                else (
                    "observation extraction, tensor export, hashes and result serialization excluded"
                    if numerical
                    else "output hashes and result serialization excluded"
                )
            ),
        },
    }


def serve(config):
    require_single_device_runtime(config["device"])
    destination = Path(config["capture"]["result"])
    try:
        response = capture(config)
    except Exception as exc:
        traceback.print_exc()
        write_record(
            destination, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
    write_record(destination, response)
