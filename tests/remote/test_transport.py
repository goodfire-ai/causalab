"""SSH boundaries and remote command quoting."""

import shlex
import subprocess

import pytest

from causalab.remote import transport

pytestmark = pytest.mark.unit


def test_ssh_does_not_disable_host_verification_and_quotes_worker_paths(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b'{"status":"completed"}', b"")

    monkeypatch.setattr(transport.subprocess, "run", run)
    receipt = {
        "host": "user@gpu",
        "control_python": "/path to/python",
        "remote_job": "/tmp/a'; echo bad; '",
    }
    assert transport.query(receipt, "status")["status"] == "completed"
    argv = calls[0][0]
    assert "BatchMode=yes" in argv and not any(
        "StrictHostKeyChecking=no" in a for a in argv
    )
    assert shlex.split(argv[-1])[-1] == receipt["remote_job"]
    for host in ("-oProxyCommand=bad", "gpu; touch bad", "gpu\nother"):
        with pytest.raises(ValueError):
            transport.SSH(host)
    with pytest.raises(ValueError):
        transport.remote_path("../outside", "job")
