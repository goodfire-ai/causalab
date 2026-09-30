"""Launch a committed SOL harness snapshot over SSH; inspect and fetch later.

The user names the SSH host, remote paths and scheduler; nothing here assumes a
particular machine or cluster. ``docs/speed_of_light.md`` ("Run on remote
GPUs") documents the commands.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile
import time
import uuid

from causalab.sol.benchmark_qwen import parser as benchmark_parser, prepare_workloads
from causalab.sol.model import validate_integer
from causalab.remote.supervisor import ARTIFACTS, TERMINAL
from causalab.remote.transport import SSH, query, remote_path, worker_command


def command_for(
    python: str,
    gpus: int,
    arguments: list[str],
    *,
    scheduler: str,
    time_limit: str,
    env_script: str | None = None,
) -> list[str]:
    validate_integer("gpus", gpus)
    if gpus > 8:
        raise ValueError("one node supports at most eight GPUs")
    if (
        not python
        or python.startswith("-")
        or ("/" in python and not python.startswith("/"))
    ):
        raise ValueError(
            "--python must be a command on remote PATH or an absolute executable path"
        )
    if any(a == "--output" or a.startswith("--output=") for a in arguments):
        raise ValueError(
            "remote launcher owns --output; use fetch --output-dir for local reports"
        )
    parsed = benchmark_parser().parse_args(["--output", "../report.json", *arguments])
    prepare_workloads(parsed)
    if parsed.batch % gpus:
        raise ValueError("global benchmark batch must divide across the requested GPUs")
    program = [python, "-m"]
    if gpus > 1:
        program += [
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpus}",
            "-m",
        ]
    program += ["causalab.sol.benchmark_qwen", "--output", "../report.json", *arguments]
    if env_script:
        program = [
            "bash",
            "-c",
            f"source {shlex.quote(env_script)} && exec {shlex.join(program)}",
        ]
    if scheduler == "slurm":
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", time_limit):
            raise ValueError("SLURM time limit must use HHH:MM:SS")
        # Respect cluster defaults: no partition, CPU or memory override.
        program = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            f"--gpus={gpus}",
            f"--time={time_limit}",
            "--job-name=causalab-sol",
            "--export=ALL",
            *program,
        ]
    elif scheduler != "direct":
        raise ValueError("scheduler must be direct or slurm")
    return program


def collect(receipt: dict, output: Path) -> dict:
    payload = query(receipt, "fetch")
    if set(payload["files"]) - ARTIFACTS:
        raise ValueError("remote returned an unexpected artifact filename")
    decoded = {
        name: base64.b64decode(data, validate=True)
        for name, data in payload["files"].items()
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, data in decoded.items():
        (output / name).write_bytes(data)
    (output / "remote.json").write_text(
        json.dumps(
            {
                "receipt": receipt,
                "state": payload["state"],
                "log_truncated": payload["log_truncated"],
            },
            indent=2,
        )
        + "\n"
    )
    return payload["state"]


def launch(args) -> dict:
    validate_integer("timeout_seconds", args.timeout_seconds)
    arguments = (
        args.benchmark_args[1:]
        if args.benchmark_args[:1] == ["--"]
        else args.benchmark_args
    )
    command = command_for(
        args.python,
        args.gpus,
        arguments,
        scheduler=args.scheduler,
        time_limit=args.time_limit,
        env_script=args.env_script,
    )
    ssh = SSH(args.host)
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{args.revision}^{{commit}}",
        ],
        cwd=repo,
        text=True,
    ).strip()
    job = remote_path(args.remote_root, uuid.uuid4().hex)
    receipt = {
        "schema_version": 1,
        "host": args.host,
        "remote_job": job,
        "control_python": args.control_python,
        "source_commit": commit,
        "command": command,
        "scheduler": args.scheduler,
    }
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return receipt
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:causalab/remote/supervisor.py"],
        cwd=repo,
        check=True,
    )
    # Write the receipt before upload so interrupted launches can be queried.
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as out:
        json.dump(receipt, out, indent=2)
    print(f"Deploying committed snapshot {commit}; receipt: {args.receipt}", flush=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=repo,
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        hasher = hashlib.sha256()
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        archive.seek(0)
        root = str(PurePosixPath(job).parent)
        source = str(PurePosixPath(job) / "source")
        ssh.call(
            f"mkdir -p -- {shlex.quote(root)} && mkdir -- {shlex.quote(job)} "
            f"&& mkdir -- {shlex.quote(source)} && tar -xf - -C {shlex.quote(source)}",
            stdin=archive,
        )
    spec = {
        **receipt,
        "source_archive_sha256": digest,
        "timeout_seconds": args.timeout_seconds,
        "hf_cache": args.hf_cache,
        "allow_download": "--allow-download" in arguments,
    }
    ssh.call(worker_command(receipt, "init"), data=json.dumps(spec).encode())
    result = query(receipt, "launch")
    print(json.dumps(result))
    return receipt


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    start = subs.add_parser(
        "launch", help="upload a committed snapshot and start a detached job"
    )
    start.add_argument(
        "--host", required=True, help="SSH alias or user@host; uses normal SSH config"
    )
    start.add_argument(
        "--receipt", type=Path, required=True, help="new local JSON recovery receipt"
    )
    start.add_argument(
        "--remote-root",
        default=".cache/causalab-sol",
        help="remote home-relative or absolute directory",
    )
    start.add_argument(
        "--revision",
        default="HEAD",
        help="committed Git revision to upload; working edits are not copied",
    )
    start.add_argument(
        "--python",
        default="python3",
        help="prepared GPU environment's Python executable on remote compute",
    )
    start.add_argument(
        "--control-python", default="python3", help="stdlib-only Python on the SSH host"
    )
    start.add_argument("--gpus", type=int, default=1)
    start.add_argument("--scheduler", choices=["direct", "slurm"], default="direct")
    start.add_argument(
        "--time-limit", default="02:00:00", help="SLURM allocation wall time"
    )
    start.add_argument(
        "--timeout-seconds",
        type=int,
        default=14400,
        help="supervisor deadline including SLURM queue time",
    )
    cache = start.add_mutually_exclusive_group()
    cache.add_argument("--hf-cache", help="remote HF_HUB_CACHE path")
    cache.add_argument(
        "--env-script",
        help="remote bash script to source inside the compute allocation",
    )
    start.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without contacting remote compute",
    )
    start.add_argument(
        "benchmark_args", nargs=argparse.REMAINDER, help="benchmark flags after --"
    )
    for action in ("status", "fetch", "cancel", "wait"):
        sub = subs.add_parser(action)
        sub.add_argument("--receipt", type=Path, required=True)
        if action == "fetch":
            sub.add_argument("--output-dir", type=Path, required=True)
        if action == "wait":
            sub.add_argument("--poll-seconds", type=int, default=10)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == "launch":
        launch(args)
        return
    receipt = json.loads(args.receipt.read_text())
    if args.action == "fetch":
        result = collect(receipt, args.output_dir)
    elif args.action == "wait":
        validate_integer("poll_seconds", args.poll_seconds)
        while True:
            result = query(receipt, "status")
            print(json.dumps(result), flush=True)
            if result["status"] in TERMINAL:
                break
            time.sleep(args.poll_seconds)
    else:
        result = query(receipt, args.action)
    print(json.dumps(result, indent=2))
    if result["status"] in TERMINAL - {"completed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
