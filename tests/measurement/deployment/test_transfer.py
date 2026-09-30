"""Transport safety and integrity with actual standalone stdlib processes."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from causalab.measurement.collection import file_hash
from causalab.measurement.deployment.transfer import extract

pytestmark = pytest.mark.unit


def test_extract_strips_unsafe_permissions_and_ownership(tmp_path):
    archive = tmp_path / "permissions.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("executable")
        info.mode = 0o6777
        info.uid, info.gid = 12345, 12345
        stream.addfile(info)
    root = tmp_path / "out"
    root.mkdir()
    extract(archive, root)
    mode = (root / "executable").stat()
    assert mode.st_mode & 0o7022 == 0
    assert mode.st_mode & 0o100
    assert mode.st_uid == os.getuid()


def test_extract_requires_security_patched_control_python(tmp_path, monkeypatch):
    monkeypatch.delattr(tarfile, "data_filter")
    with pytest.raises(RuntimeError, match="patched Python"):
        extract(tmp_path / "unused.tar", tmp_path)


def test_standalone_receive_and_terminal_stream_export(tmp_path):
    worker = (
        Path(__file__).resolve().parents[3]
        / "causalab/measurement/deployment/transfer.py"
    )
    archive = tmp_path / "payload.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo("study/workflow.json")
        info.size = 2
        stream.addfile(info, io.BytesIO(b"{}"))
    job = tmp_path / "remote"
    subprocess.run(
        [sys.executable, str(worker), "receive", str(job), file_hash(archive)],
        input=archive.read_bytes(),
        check=True,
    )
    assert (job / "study/workflow.json").read_text() == "{}"
    (job / "state.json").write_text('{"status":"completed"}')
    (job / "results").mkdir()
    (job / "results/trace.json").write_text('{"traceEvents":[]}')
    exported = tmp_path / "export.tar"
    with exported.open("wb") as output:
        subprocess.run(
            [sys.executable, str(worker), "export", str(job)], stdout=output, check=True
        )
    destination = tmp_path / "fetched"
    destination.mkdir()
    extract(exported, destination)
    manifest = json.loads((destination / "export.json").read_text())
    assert manifest["files"]["results/trace.json"] == file_hash(
        job / "results/trace.json"
    )
    assert "payload.tar" not in manifest["files"]
    (job / "state.json").write_text('{"status":"running"}')
    refusal = subprocess.run(
        [sys.executable, str(worker), "export", str(job)], capture_output=True
    )
    assert refusal.returncode != 0
    assert b"terminal job" in refusal.stderr


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("/absolute", "file"),
        ("symlink", "symlink"),
        ("link", "hardlink"),
    ],
)
def test_extract_refuses_escaping_and_link_members(tmp_path, name, kind):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(name)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "../escape"
        elif kind == "hardlink":
            info.type = tarfile.LNKTYPE
            info.linkname = "/absolute"
        stream.addfile(info)
    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        extract(archive, root)
    assert list(root.iterdir()) == []


def test_extract_refuses_normalized_duplicate_and_overwrite(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as stream:
        for name in ("x", "dir/../x"):
            stream.addfile(tarfile.TarInfo(name))
    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        extract(archive, root)
    with tarfile.open(archive, "w") as stream:
        stream.addfile(tarfile.TarInfo("x"))
    (root / "x").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        extract(archive, root)
    assert (root / "x").read_text() == "preserve"
