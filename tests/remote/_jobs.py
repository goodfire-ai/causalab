"""Real standalone supervisor fixtures shared with its clients."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from causalab.remote import supervisor, transport


@pytest.fixture(name="job")
def supervisor_job(tmp_path):
    directory = tmp_path / "job with 'quotes' and $characters"
    source = directory / "source/causalab/remote"
    source.mkdir(parents=True)
    shutil.copy2(Path(supervisor.__file__), source / "supervisor.py")
    return directory


def invoke(job, action, spec=None):
    # Exercise the exact shell quoting used by SSH, with a local shell standing
    # in for the remote shell. The supervisor is a real detached subprocess.
    receipt = {"remote_job": str(job), "control_python": sys.executable}
    result = subprocess.run(
        ["sh", "-c", transport.worker_command(receipt, action)],
        input=None if spec is None else json.dumps(spec),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def start(job, code, timeout=10):
    invoke(
        job,
        "init",
        {
            "command": [sys.executable, "-c", code],
            "timeout_seconds": timeout,
            "source_commit": "a" * 40,
            "scheduler": "direct",
        },
    )
    return invoke(job, "launch")


def finish(job):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = invoke(job, "status")
        if state["status"] in supervisor.TERMINAL:
            return state
        time.sleep(0.05)
    invoke(job, "cancel")
    pytest.fail("supervisor did not terminate")
