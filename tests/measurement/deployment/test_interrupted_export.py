"""Native export reconciles dead supervisors while holding writer leases."""

import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from causalab.remote.supervisor import status

pytestmark = pytest.mark.unit
TRANSFER = (
    Path(__file__).resolve().parents[3] / "causalab/measurement/deployment/transfer.py"
)


def make_job(root: Path, state: str) -> None:
    (root / "results").mkdir()
    (root / "results/trace.json").write_text('{"traceEvents":[]}')
    (root / "state.json").write_text(json.dumps({"status": state}))
    # A reaped child provides a real PID whose supervisor has exited.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    (root / "supervisor.json").write_text(json.dumps({"pid": child.pid}))


def export_job(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(TRANSFER), "export", str(root)],
        capture_output=True,
    )


def test_export_reconciles_interrupted_state_and_manifest(tmp_path):
    make_job(tmp_path, "running")
    expected = status(tmp_path)
    assert expected["status"] == "interrupted"
    result = export_job(tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        manifest = json.load(archive.extractfile("export.json"))
        state = json.load(archive.extractfile("state.json"))
        assert manifest["state"] == state == expected
        assert archive.extractfile("results/trace.json").read() == b'{"traceEvents":[]}'


@pytest.mark.parametrize("state", ["running", "completed", "failed", "interrupted"])
@pytest.mark.parametrize(
    "lease_path",
    [".supervisor.lock", "results/.experiment.lock", "results/.workers.lock"],
)
def test_export_refuses_any_active_writer_even_with_terminal_state(
    tmp_path, state, lease_path
):
    make_job(tmp_path, state)
    with (tmp_path / lease_path).open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_SH)
        result = export_job(tmp_path)
    assert result.returncode != 0
    assert b"still running" in result.stderr
    assert result.stdout == b""


def test_export_holds_all_writer_leases_until_streaming_finishes(tmp_path, monkeypatch):
    from causalab.measurement.deployment import transfer as measurement_transfer

    make_job(tmp_path, "completed")
    streamed = []

    def stream(job: Path, state: dict) -> None:
        for name in (
            ".supervisor.lock",
            "results/.experiment.lock",
            "results/.workers.lock",
        ):
            with (job / name).open("a") as lease:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        streamed.append(state["status"])

    monkeypatch.setattr(measurement_transfer, "_export", stream)
    measurement_transfer.export(tmp_path)
    assert streamed == ["completed"]
