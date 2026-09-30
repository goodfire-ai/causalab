"""Source integrity and native target identity are checked before reuse."""

import json
import subprocess
import sys

import pytest

from causalab.measurement.collection import write_record
from causalab.measurement.deployment.installation import (
    archive_arm,
    build_source,
    interpreter_identity,
    load_installation,
    load_source,
)
from causalab.measurement.study.scheduler import manifest

pytestmark = pytest.mark.unit


def test_source_export_never_builds_and_detects_changed_archive(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "pyproject.toml").write_text("deliberately not buildable")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "source",
        ],
        cwd=repo,
        check=True,
    )
    source = tmp_path / "source"
    record = archive_arm(repo, "HEAD", source)
    assert set(p.name for p in source.iterdir()) == {"source.tar", "source.json"}
    assert load_source(source, record["source_commit"]) == record
    with pytest.raises(ValueError, match="source commit"):
        load_source(source, "0" * 40)
    with (source / "source.tar").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        build_source(source, tmp_path / "installation", python=sys.executable)
    assert not (tmp_path / "installation").exists()


def test_prepared_native_installation_checks_target_interpreter(tmp_path):
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "extension.so").write_bytes(b"native wheel bytes")
    commit = "a" * 40
    record = {
        "source_commit": commit,
        "installed_files": manifest(installed),
        "build_runtime": interpreter_identity(sys.executable),
    }
    write_record(tmp_path / "installation.json", record)
    assert (
        load_installation(tmp_path, commit, python=sys.executable)["build_runtime"]
        == record["build_runtime"]
    )
    for field in ("abi", "platform", "python"):
        changed = json.loads(json.dumps(record))
        changed["build_runtime"][field] = "different"
        write_record(tmp_path / "installation.json", changed)
        with pytest.raises(ValueError, match="target interpreter/platform"):
            load_installation(tmp_path, commit, python=sys.executable)


def test_worker_attests_its_arm_from_the_deployment_record(tmp_path):
    from causalab.measurement.runtime.worker import attest_installation

    installed = tmp_path / "installed"
    (installed / "causalab").mkdir(parents=True)
    (installed / "causalab" / "__init__.py").write_text("")
    commit = "a" * 40
    record = {
        "source_commit": commit,
        "package_root": str(installed),
        "installed_files": manifest(installed),
    }
    write_record(tmp_path / "installation.json", record)
    config = {
        "installation": str(tmp_path / "installation.json"),
        "source_commit": commit,
    }
    assert attest_installation(config, installed) == commit
    with pytest.raises(ValueError, match="arm revision mismatch"):
        attest_installation({**config, "source_commit": "b" * 40}, installed)
    with pytest.raises(ValueError, match="arm installation mismatch"):
        attest_installation(config, tmp_path)
    (installed / "causalab" / "__init__.py").write_text("# tampering\n")
    with pytest.raises(ValueError, match="deployed source installation changed"):
        attest_installation(config, installed)
