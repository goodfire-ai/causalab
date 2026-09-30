"""Shared stdlib SSH transport and supervisor commands."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess


class SSH:
    def __init__(self, host: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", host):
            raise ValueError("host must be an SSH config alias or user@host")
        self.host = host

    def call(
        self, command: str, *, data: bytes | None = None, stdin=None, output=None
    ) -> bytes:
        result = subprocess.run(
            [
                "ssh",
                # Large native trace exports benefit from transport compression;
                # SSH delivers the original bytes to the manifest verifier.
                *(["-C"] if output is not None else []),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                self.host,
                command,
            ],
            input=data,
            stdin=stdin,
            stdout=output if output is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise RuntimeError(
                f"SSH failed ({result.returncode}): {result.stderr.decode(errors='replace')}"
            )
        return result.stdout if output is None else b""


def remote_path(root: str, job_id: str) -> str:
    path = PurePosixPath(root)
    if (
        not root
        or ".." in path.parts
        or "~" in root
        or any(c in root for c in "\n\r\0")
    ):
        raise ValueError(
            "remote root must be an absolute path or a path relative to remote home; no ~ or .."
        )
    return str(path / job_id)


def worker_command(receipt: dict, action: str) -> str:
    return shlex.join(
        [
            receipt["control_python"],
            str(
                PurePosixPath(receipt["remote_job"])
                / "source/causalab/remote/supervisor.py"
            ),
            action,
            receipt["remote_job"],
        ]
    )


def query(receipt: dict, action: str) -> dict:
    return json.loads(SSH(receipt["host"]).call(worker_command(receipt, action)))
