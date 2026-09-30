"""Shared remote control works without model dependencies or package installation."""

from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_transport_imports_without_model_or_sol_dependencies(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from causalab.remote.transport import SSH, remote_path, worker_command; "
            "assert remote_path('/jobs', 'run') == '/jobs/run'; "
            "assert 'source/causalab/remote/supervisor.py' in worker_command("
            "{'control_python': 'python3', 'remote_job': '/jobs/run'}, 'status'); "
            "assert not any(name == 'torch' or name.startswith('causalab.sol') "
            "for name in sys.modules)",
            str(ROOT),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_supervisor_is_a_standalone_stdlib_script(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "causalab/remote/supervisor.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "resume" in result.stdout


def test_export_from_source_text_explains_required_checkout(tmp_path):
    source = ROOT / "causalab/measurement/deployment/transfer.py"
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", source.read_text(), "export", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "export must run from the deployed checkout" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_transfer_receives_source_text_then_exports_from_relocated_checkout(tmp_path):
    import hashlib
    import io
    import json
    import tarfile

    source = ROOT / "causalab/measurement/deployment/transfer.py"
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for relative in (
            "causalab/measurement/deployment/transfer.py",
            "causalab/remote/supervisor.py",
        ):
            archive.add(ROOT / relative, arcname=f"source/{relative}")
    data = payload.getvalue()
    job = tmp_path / "job"
    received = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            source.read_text(),
            "receive",
            str(job),
            hashlib.sha256(data).hexdigest(),
        ],
        input=data,
        cwd=tmp_path,
        capture_output=True,
    )
    assert received.returncode == 0, received.stderr.decode()
    (job / "state.json").write_text('{"status":"completed"}')
    (job / "results").mkdir()
    (job / "results/trace.json").write_text('{"traceEvents":[]}')
    exported = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(job / "source/causalab/measurement/deployment/transfer.py"),
            "export",
            str(job),
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert exported.returncode == 0, exported.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(exported.stdout)) as archive:
        manifest = json.load(archive.extractfile("export.json"))
        assert manifest["state"] == {"status": "completed"}
        assert archive.extractfile("results/trace.json").read() == b'{"traceEvents":[]}'
