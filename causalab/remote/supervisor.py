"""Standalone stdlib-only supervisor uploaded with a committed source snapshot.

Remote launchers invoke this over SSH without torch or GPU dependencies.
Each job directory is isolated; cancellation targets the supervisor's own child
process group. ``causalab.sol.remote`` is the user-facing launcher
(``docs/speed_of_light.md``, "Run on remote GPUs"); the host is the user's own.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
ARTIFACTS = {"report.json", "report.md", "run.log", "state.json", "job.json"}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def status(job: Path) -> dict:
    state = json.loads((job / "state.json").read_text())
    if state["status"] not in TERMINAL and (job / "supervisor.json").exists():
        pid = json.loads((job / "supervisor.json").read_text())["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            state = {
                **state,
                "status": "interrupted",
                "detail": "supervisor is no longer running",
            }
    return state


def launch(job: Path) -> dict:
    spec = json.loads((job / "job.json").read_text())
    if not spec["command"] or spec["timeout_seconds"] <= 0:
        raise ValueError("job needs a command and positive timeout")
    # Retrying an uncertain SSH launch must never create a second GPU job.
    with (job / "launch.lock").open("x"):
        pass
    write_json(job / "state.json", {"status": "submitted", "submitted_at": stamp()})
    try:
        with (job / "run.log").open("ab") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "work", str(job)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        write_json(job / "supervisor.json", {"pid": child.pid})
        return {"status": "submitted", "supervisor_pid": child.pid}
    except Exception as exc:
        write_json(
            job / "state.json",
            {"status": "failed", "error": str(exc), "ended_at": stamp()},
        )
        raise


def work(job: Path) -> None:
    with (job / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _work(job)


@contextmanager
def measurement_lease(job: Path) -> Iterator[None]:
    """Exclude supervisors and measurement writers through export or restart."""
    with ExitStack() as stack:
        for path in (
            job / ".supervisor.lock",
            job / "results/.experiment.lock",
            job / "results/.workers.lock",
        ):
            if not path.parent.exists():
                continue
            lock = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "previous supervisor, controller or workers are still running"
                ) from exc
        yield


def restart_measurement(job: Path) -> dict:
    """Restart the same immutable study after all prior processes release leases."""
    with measurement_lease(job):
        state = status(job)
        if state["status"] not in TERMINAL:
            raise ValueError("resume requires a terminal measurement job")
        spec = json.loads((job / "job.json").read_text())
        if spec.get("kind") != "measurement" or spec.get("scheduler") != "direct":
            raise ValueError("resume is supported only for direct measurement jobs")
        command = list(spec["command"])
        if (job / "results").exists():
            if not (job / "results/preparation.json").is_file():
                raise ValueError(
                    "resume requires a valid measurement preparation receipt"
                )
            if "--resume" not in command:
                command.append("--resume")
        history = job / "recovery" / uuid.uuid4().hex
        history.mkdir(parents=True)
        for name in ("state.json", "job.json", "supervisor.json"):
            path = job / name
            if path.is_file():
                (history / name).write_bytes(path.read_bytes())
        write_json(job / "job.json", {**spec, "command": command})
        (job / "cancel.request").unlink(missing_ok=True)
        (job / "launch.lock").unlink(missing_ok=True)
        return launch(job)


def _work(job: Path) -> None:
    spec = json.loads((job / "job.json").read_text())
    state = {
        "status": "running",
        "started_at": stamp(),
        "source_commit": spec["source_commit"],
        "scheduler": spec["scheduler"],
        "supervisor_pid": os.getpid(),
    }
    write_json(job / "state.json", state)
    env = dict(os.environ)
    env["CAUSALAB_SOL_SOURCE_COMMIT"] = spec["source_commit"]
    env["PYTHONUNBUFFERED"] = "1"
    # Do not inherit a login shell's distributed rank identity.
    for key in (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    if spec.get("hf_cache"):
        env["HF_HUB_CACHE"] = spec["hf_cache"]
    if not spec.get("allow_download", False):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    reason = None
    started = time.monotonic()
    try:
        with subprocess.Popen(
            spec["command"],
            cwd=job / "source",
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        ) as child:
            while child.poll() is None:
                if (job / "cancel.request").exists():
                    reason = "cancelled"
                elif time.monotonic() - started >= spec["timeout_seconds"]:
                    reason = "timed_out"
                if reason:
                    # The Popen child is still owned/unreaped here. Never use
                    # an unverified stored PID for process-group termination.
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        child.wait()
                    break
                time.sleep(0.2)
            code = child.returncode
        state.update(
            status=reason or ("completed" if code == 0 else "failed"),
            exit_code=code,
            ended_at=stamp(),
        )
    except Exception as exc:
        state.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", ended_at=stamp()
        )
    write_json(job / "state.json", state)


def fetch(job: Path) -> dict:
    files = {}
    truncated = False
    for name in sorted(ARTIFACTS):
        path = job / name
        if path.is_file():
            if name == "run.log":
                with path.open("rb") as log:
                    size = path.stat().st_size
                    truncated = size > 1024 * 1024
                    log.seek(max(0, size - 1024 * 1024))
                    data = log.read()
            else:
                data = path.read_bytes()
            files[name] = base64.b64encode(data).decode("ascii")
    return {"state": status(job), "files": files, "log_truncated": truncated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["init", "launch", "work", "status", "fetch", "cancel", "resume"],
    )
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = args.job.expanduser().resolve()
    if args.action == "init":
        spec = json.load(sys.stdin)
        with (job / "job.json").open("x") as out:
            json.dump(spec, out, indent=2, allow_nan=False)
        write_json(job / "state.json", {"status": "prepared", "prepared_at": stamp()})
        result = {"status": "prepared"}
    elif args.action == "launch":
        result = launch(job)
    elif args.action == "work":
        work(job)
        return
    elif args.action == "resume":
        result = restart_measurement(job)
    elif args.action == "status":
        result = status(job)
    elif args.action == "fetch":
        result = fetch(job)
    else:
        result = status(job)
        if result["status"] not in TERMINAL:
            (job / "cancel.request").touch()
            result = {**result, "cancel_requested": True}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
