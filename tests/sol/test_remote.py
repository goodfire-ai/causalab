"""SOL command construction and artifact collection over shared remote control."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from causalab.sol import remote
from causalab.remote import supervisor
from tests.remote._jobs import finish, invoke, start
from tests.remote._jobs import supervisor_job as supervisor_job

pytestmark = pytest.mark.unit


def test_direct_and_slurm_use_one_torchrun_and_global_batch():
    direct = remote.command_for(
        "/env with spaces/bin/python",
        8,
        ["--batch", "16"],
        scheduler="direct",
        time_limit="02:00:00",
    )
    assert direct[:3] == ["/env with spaces/bin/python", "-m", "torch.distributed.run"]
    assert direct.count("--nproc-per-node=8") == 1
    assert "../report.json" in direct
    slurm = remote.command_for(
        "python3",
        2,
        [],
        scheduler="slurm",
        time_limit="01:30:00",
        env_script="/shared/config with spaces.sh",
    )
    assert slurm[:5] == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=2",
        "--time=01:30:00",
    ]
    assert not any("partition" in arg or "--mem" in arg for arg in slurm)
    assert "source '/shared/config with spaces.sh' && exec python3" in slurm[-1]
    for flags in (
        ["--output", "/tmp/wrong.json"],
        ["--output=/tmp/wrong.json"],
        ["--batch", "3"],
    ):
        with pytest.raises(ValueError):
            remote.command_for(
                "python3", 2, flags, scheduler="direct", time_limit="01:00:00"
            )


def test_fetch_rejects_path_traversal_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote,
        "query",
        lambda *_: {
            "files": {"../escape": "eA=="},
            "state": {"status": "completed"},
            "log_truncated": False,
        },
    )
    with pytest.raises(ValueError, match="unexpected artifact"):
        remote.collect({}, tmp_path / "reports")
    assert not (tmp_path / "reports").exists()


def test_detached_run_records_revision_and_retrieves_artifacts(
    job, monkeypatch, tmp_path
):
    code = (
        "import os,json; from pathlib import Path; "
        "Path('../report.json').write_text(json.dumps({'commit':os.environ['CAUSALAB_SOL_SOURCE_COMMIT']})); "
        "Path('../report.md').write_text('measured'); print('benchmark log')"
    )
    assert start(job, code)["status"] == "submitted"
    # The launch SSH-equivalent subprocess has exited. Work survives it.
    assert finish(job)["status"] == "completed"
    payload = invoke(job, "fetch")
    monkeypatch.setattr(remote, "query", lambda *_: payload)
    target = tmp_path / "download"
    assert remote.collect({}, target)["status"] == "completed"
    assert json.loads((target / "report.json").read_text())["commit"] == "a" * 40
    assert "benchmark log" in (target / "run.log").read_text()
    # An uncertain launch cannot start a second worker in the same directory.
    with pytest.raises(FileExistsError):
        supervisor.launch(job)


def test_archive_revision_is_used_in_benchmark_provenance(monkeypatch):
    from causalab.sol.benchmark_qwen import _software  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setenv("CAUSALAB_SOL_SOURCE_COMMIT", "b" * 40)
    assert _software()["git_commit"] == "b" * 40
    assert _software()["source_kind"] == "remote_git_archive"


def test_archive_upload_launch_and_fetch_with_local_ssh_boundary(monkeypatch, tmp_path):
    # A tiny committed repo exercises git archive and the actual worker. Only
    # the SSH transport and GPU program are replaced at the system boundary.
    repo = tmp_path / "repository"
    package = repo / "causalab/sol"
    package.mkdir(parents=True)
    (repo / "causalab/__init__.py").touch()
    (package / "__init__.py").touch()
    shared = repo / "causalab/remote"
    shared.mkdir()
    shutil.copy2(Path(supervisor.__file__), shared / "supervisor.py")
    (package / "benchmark_qwen.py").write_text(
        "import argparse,json,os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()\n"
        "Path(a.output).write_text(json.dumps({'commit':os.environ['CAUSALAB_SOL_SOURCE_COMMIT']}))\n"
        "Path(a.output).with_suffix('.md').write_text('report')\n"
    )
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr(remote, "__file__", str(package / "remote.py"))

    def local_ssh(_self, command, *, data=None, stdin=None):
        completed = subprocess.run(
            ["sh", "-c", command],
            input=data,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    monkeypatch.setattr(remote.SSH, "call", local_ssh)
    args = remote.parser().parse_args(
        [
            "launch",
            "--host",
            "fixture",
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--remote-root",
            str(tmp_path / "remote home"),
            "--python",
            sys.executable,
            "--control-python",
            sys.executable,
        ]
    )
    receipt = remote.launch(args)
    assert finish(Path(receipt["remote_job"]))["status"] == "completed"
    downloaded = tmp_path / "downloaded"
    remote.collect(receipt, downloaded)
    assert (
        json.loads((downloaded / "report.json").read_text())["commit"]
        == receipt["source_commit"]
    )
    spec = json.loads((downloaded / "job.json").read_text())
    assert len(spec["source_archive_sha256"]) == 64
    assert not (Path(receipt["remote_job"]) / "source/.git").exists()
