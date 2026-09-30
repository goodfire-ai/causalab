"""Isolated profiler subprocesses, independent of successful clean measurements."""

from __future__ import annotations

from contextlib import closing, contextmanager
import errno
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from ..collection import file_hash, write_record
from ..device import require_single_device
from ..paths import worker_bootstrap
from ...profiling import CaptureRequest, ProfilerUnavailable, get_backend


@contextmanager
def _forward_termination():
    """Let a controller SIGTERM unwind capture cleanup before leaving its lease."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("profiling controller received SIGTERM")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _wait_unreaped(process: subprocess.Popen, timeout: float) -> None:
    """Observe exit while retaining the child PID until group cleanup finishes."""
    if hasattr(os, "waitid"):
        deadline = time.monotonic() + timeout
        while (
            os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            is None
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            time.sleep(min(remaining, 0.05))
        return
    # Python on macOS lacks waitid; kqueue observes exit without reaping too.
    with closing(select.kqueue()) as queue:
        event = select.kevent(
            process.pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        try:
            events = queue.control([event], 1, timeout)
        except ProcessLookupError:
            return  # An already-exited, still-unreaped child.
        if not events:
            raise subprocess.TimeoutExpired(process.args, timeout)
        if events[0].flags & select.KQ_EV_ERROR and events[0].data != errno.ESRCH:
            raise OSError(events[0].data, "could not observe capture process exit")


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        # Darwin's killpg excludes zombies and returns EPERM for an otherwise
        # empty group. Do not confuse that with denied access to a live leader.
        if sys.platform != "darwin":
            raise
        try:
            _wait_unreaped(process, 0)
        except subprocess.TimeoutExpired:
            raise exc


def _stop_group(process: subprocess.Popen) -> None:
    if process.returncode is not None:
        return  # Never signal a PID already released by wait/poll.
    try:
        _signal_group(process, signal.SIGTERM)
        try:
            _wait_unreaped(process, 5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # Retain even an exited leader until the final signal: descendants may
        # ignore SIGTERM, and reaping would allow an empty group's ID to be reused.
        try:
            _signal_group(process, signal.SIGKILL)
        finally:
            process.wait()


def execute(command, *, cwd, log, timeout, lease_fd=None, environment=None):
    """Own a separate capture group so timeout/cancellation reaches tool children."""
    with _forward_termination():
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                **(environment or {}),
            },
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            pass_fds=() if lease_fd is None else (lease_fd,),
        )
        try:
            _wait_unreaped(process, timeout)
        finally:
            _stop_group(process)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def run_captures(python, config, identity, case, seed, directory, receipt):
    """Publish every requested backend/mode outcome without replacing clean data."""
    require_single_device(config["device"])
    record = json.loads(receipt.read_text())
    record["captures"] = []
    record["mode"] = config["plan"].get("mode", "comparison")
    record["observation_policy"] = config["plan"].get("observation_policy", "required")
    numerical = record["observation_policy"] == "required"
    profile = config["plan"]["profile"]
    modes = (
        ("cold", "warm") if config["plan"]["cases"][case]["cold_process"] else ("warm",)
    )
    backends = profile["backends"]
    record["capture_plan"] = {"backends": list(backends), "modes": list(modes)}
    bootstrap = worker_bootstrap(Path(config["controller_root"]))
    for name, options in backends.items():
        for mode in modes:
            target = directory / "captures" / name / mode
            target.mkdir(parents=True, exist_ok=False)
            capture: dict[str, Any] = {
                "id": f"{name}:{mode}",
                "pair_id": name,
                "backend": name,
                "case": case,
                "mode": mode,
                "seed": seed,
                "repeat": 0,
                "status": "running",
                "options": options,
                "artifacts": [],
                "reason": profile.get("reason", "whole selected case"),
            }
            if not numerical:
                capture["observation_status"] = "not_requested"
            record["captures"].append(capture)
            write_record(receipt, record)
            try:
                backend = get_backend(name)
                options = backend.normalize_options(options)
                capture.update(
                    options=options,
                    artifact_format=backend.artifact_format,
                    environment=dict(backend.environment),
                    coverage={
                        "startup": backend.startup_coverage
                        if mode == "cold"
                        else "resident model after unprofiled warmup; startup excluded",
                        "perturbation": backend.perturbation,
                    },
                )
                if backend.capture_control == "cuda_profiler_api" and not config[
                    "device"
                ].startswith("cuda"):
                    raise ProfilerUnavailable(f"{name} requires a CUDA device")
                capture["tool"] = backend.probe()
                result_path = target / "result.json"
                capture_config = {
                    **config,
                    "verified_model_files": {
                        key: value["files"] for key, value in identity["models"].items()
                    },
                    "execution_probe": identity["execution_probe"],
                    "capture": {
                        "backend": name,
                        "mode": mode,
                        "case": case,
                        "seed": seed,
                        "directory": str(target),
                        "result": str(result_path),
                        "options": options,
                    },
                }
                path = target / "config.json"
                write_record(path, capture_config)
                request = CaptureRequest(
                    command=(python, str(bootstrap), str(path)),
                    output_dir=target,
                    mode=mode,
                    options=options,
                    device=config["device"],
                )
                capture["command"] = backend.command(request)
                log_path = target / "capture.log"
                try:
                    with log_path.open("w") as log:
                        execute(
                            capture["command"],
                            cwd=config["package_root"],
                            log=log,
                            timeout=profile.get("timeout_seconds", 3600),
                            lease_fd=config.get("lease_fd"),
                            environment=backend.environment,
                        )
                finally:
                    if log_path.is_file():
                        capture["logs"] = [
                            {
                                "file": log_path.relative_to(directory).as_posix(),
                                "sha256": file_hash(log_path),
                            }
                        ]
                response = json.loads(result_path.read_text())
                if response.get("status") != "completed":
                    raise RuntimeError(
                        response.get("error", "capture worker did not complete")
                    )
                if response["identity"] != identity:
                    raise ValueError(
                        "profile process did not execute the same attested arm"
                    )
                artifacts = backend.artifacts(request)
                if not artifacts or any(
                    not artifact.is_file() or not artifact.stat().st_size
                    for artifact in artifacts
                ):
                    raise ValueError("profiler produced no nonempty native artifact")
                capture["artifacts"] = [
                    {
                        "file": artifact.relative_to(directory).as_posix(),
                        "sha256": file_hash(artifact),
                    }
                    for artifact in artifacts
                ]
                if numerical:
                    observed = target / "observations.safetensors"
                    if (
                        not observed.is_file()
                        or file_hash(observed) != response["observations"]["sha256"]
                    ):
                        raise ValueError("profile output observation hash mismatch")
                    capture["observations"] = {
                        "file": observed.relative_to(directory).as_posix(),
                        "sha256": file_hash(observed),
                    }
                    capture["observation_specs"] = response["observation_specs"]
                elif (
                    response.get("observation_status") != "not_requested"
                    or "observations" in response
                ):
                    raise ValueError(
                        "profile observation policy differs from authored plan"
                    )
                else:
                    capture["observation_status"] = "not_requested"
                from ..analysis.receipts import verified_artifact

                capture["output_files"] = {}
                for filename, sha in response.get("output_files", {}).items():
                    artifact = verified_artifact(
                        target, {"file": filename, "sha256": sha}
                    )
                    capture["output_files"][
                        artifact.relative_to(directory).as_posix()
                    ] = sha
                capture.update(
                    status="completed",
                    instrumentation=response["instrumentation"],
                    identity=response["identity"],
                )
                if name == "torch":
                    capture["tool"] = response.get("tool") or capture["tool"]
                capture["coverage"].update(response["coverage"])
            except ProfilerUnavailable as exc:
                capture.update(status="unavailable", error=str(exc))
            except Exception as exc:
                capture.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            except BaseException as exc:
                capture.update(
                    status="interrupted", error=f"{type(exc).__name__}: {exc}"
                )
                write_record(receipt, record)
                raise
            write_record(receipt, record)
