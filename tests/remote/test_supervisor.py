"""Detached stdlib supervision, interruption and measurement recovery."""

import base64
import fcntl
import json
import sys

import pytest

from causalab.remote import supervisor
from tests.remote._jobs import finish, invoke, start
from tests.remote._jobs import supervisor_job as supervisor_job

pytestmark = pytest.mark.unit


def test_nonzero_exit_keeps_partial_report(job):
    start(
        job,
        "from pathlib import Path; Path('../report.json').write_text('{}'); raise SystemExit(7)",
    )
    result = finish(job)
    assert result["status"] == "failed" and result["exit_code"] == 7
    assert base64.b64decode(invoke(job, "fetch")["files"]["report.json"]) == b"{}"


def test_measurement_resume_reuses_job_and_refuses_active_worker_lease(job):
    code = (
        "import sys; from pathlib import Path; "
        "root=Path('../results'); root.mkdir(exist_ok=True); "
        "(root/'preparation.json').write_text('{}'); "
        "raise SystemExit(0 if '--resume' in sys.argv else 7)"
    )
    invoke(
        job,
        "init",
        {
            "kind": "measurement",
            "command": [sys.executable, "-c", code],
            "timeout_seconds": 10,
            "source_commit": "a" * 40,
            "scheduler": "direct",
        },
    )
    invoke(job, "launch")
    assert finish(job)["status"] == "failed"
    with (job / "results/.workers.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_SH)
        with pytest.raises(ValueError, match="still running"):
            supervisor.restart_measurement(job)
    assert invoke(job, "resume")["status"] == "submitted"
    assert finish(job)["status"] == "completed"
    spec = json.loads((job / "job.json").read_text())
    assert spec["command"].count("--resume") == 1
    previous = next((job / "recovery").glob("*/state.json"))
    assert json.loads(previous.read_text())["exit_code"] == 7


def test_cancel_and_deadline(job):
    start(job, "import time; time.sleep(60)")
    assert invoke(job, "cancel")["cancel_requested"]
    assert finish(job)["status"] == "cancelled"


def test_deadline(job):
    start(job, "import time; time.sleep(60)", timeout=1)
    assert finish(job)["status"] == "timed_out"
