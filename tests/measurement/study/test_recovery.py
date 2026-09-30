"""Real OS leases survive the parent descriptor closing and prevent overlap."""

import fcntl
import subprocess
import sys

import pytest

from causalab.measurement.study.controller import ProcessSession, run

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("raise SystemExit(3)", "exited with code 3"),
        ("pass", "exited with code 0"),
        (
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
            "terminated by SIGTERM",
        ),
    ],
)
def test_worker_eof_reports_exit_status(tmp_path, code, expected):
    session = ProcessSession.__new__(ProcessSession)
    session.log = None
    session.logs = tmp_path
    session.child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        with pytest.raises(RuntimeError, match=expected):
            session.read()
    finally:
        session.close()


def test_exited_worker_with_buffered_input_can_be_closed():
    session = ProcessSession.__new__(ProcessSession)
    session.log = None
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    session.child = child
    child.stdin.write("buffered shutdown")
    child.wait(timeout=5)
    session.close()
    assert session.child is None
    assert child.stdin.closed and child.stdout.closed


def test_inherited_worker_lease_prevents_orphan_overlap(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    lease = (output / ".workers.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_SH)
    child = subprocess.Popen(
        [sys.executable, "-c", "print('ready', flush=True); input()"],
        pass_fds=(lease.fileno(),),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        lease.close()  # Simulates controller exit; only the child holds the lease.
        with pytest.raises(ValueError, match="previous measurement workers"):
            run(
                tmp_path / "study.json", tmp_path / "bindings.json", output, resume=True
            )
        child.communicate("done\n", timeout=5)
        with (output / ".workers.lock").open("a") as check:
            fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        lease.close()
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
