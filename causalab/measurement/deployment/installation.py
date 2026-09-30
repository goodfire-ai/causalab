"""Immutable source bundles and private wheels built for the executing interpreter.

SSH transports source bytes, never the submitting host's native extensions.
Build dependencies and the selected source's Rust toolchain must be prepared on
the compute host. No installs modify the supplied dependency environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.study.scheduler import manifest
from causalab.measurement.deployment.transfer import extract


def interpreter_identity(python: str) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import json,sys,sysconfig; print(json.dumps({"
                "'implementation':sys.implementation.name,"
                "'python':list(sys.version_info[:2]),"
                "'abi':sysconfig.get_config_var('SOABI'),"
                "'platform':sysconfig.get_platform()}))",
            ],
            text=True,
        )
    )


def archive_arm(repository: Path, revision: str, destination: Path) -> dict[str, Any]:
    """Export a committed revision without building or importing it."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=repository,
        text=True,
    ).strip()
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "source.tar"
    with archive.open("wb") as stream:
        subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=repository,
            stdout=stream,
            check=True,
        )
    record = {
        "source_commit": commit,
        "source_archive_sha256": file_hash(archive),
        "repository_url": repository.resolve().as_uri(),
        "requested_revision": revision,
    }
    write_record(destination / "source.json", record)
    return record


def load_source(source: Path, commit: str | None = None) -> dict[str, Any]:
    record = json.loads((source / "source.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", record["source_commit"]):
        raise ValueError("source bundle requires a pinned 40-character source commit")
    if commit is not None and record["source_commit"] != commit:
        raise ValueError("source bundle does not match the study's source commit")
    if file_hash(source / "source.tar") != record["source_archive_sha256"]:
        raise ValueError("source archive changed")
    return record


def load_installation(
    destination: Path, commit: str, *, python: str | None = None
) -> dict[str, Any]:
    """Verify installed bytes and their target ABI before reuse."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(
            "a prepared installation requires a pinned 40-character source commit"
        )
    record = json.loads((destination / "installation.json").read_text())
    if record["source_commit"] != commit:
        raise ValueError(
            "prepared installation does not match the study's source commit"
        )
    if manifest(destination / "installed") != record["installed_files"]:
        raise ValueError("deployed source installation changed")
    if python is not None and record.get("build_runtime") != interpreter_identity(
        python
    ):
        raise ValueError(
            "prepared installation does not match the target interpreter/platform; rebuild it on the compute host"
        )
    return {**record, "package_root": str((destination / "installed").resolve())}


def build_source(source: Path, destination: Path, *, python: str) -> dict[str, Any]:
    """Build on the execution host for its selected Python, outside all clocks."""
    source_record = load_source(source)
    receipt = destination / "installation.json"
    if receipt.exists():
        record = load_installation(
            destination, source_record["source_commit"], python=python
        )
        if record["source_archive_sha256"] != source_record["source_archive_sha256"]:
            raise ValueError("prepared installation has a different source archive")
        return record
    if destination.exists():
        destination.rename(
            destination.with_name(f"{destination.name}.incomplete-{uuid.uuid4().hex}")
        )
    destination.mkdir(parents=True, exist_ok=False)
    runtime = interpreter_identity(python)
    archive = destination / "source.tar"
    shutil.copyfile(source / "source.tar", archive)
    if file_hash(archive) != source_record["source_archive_sha256"]:
        raise ValueError("source archive changed during staging")
    tree = destination / "source"
    tree.mkdir()
    extract(archive, tree)
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("source deployment requires uv on the compute host")
    env = dict(os.environ, CARGO_NET_OFFLINE="true", PYTHONDONTWRITEBYTECODE="1")
    env.pop("VIRTUAL_ENV", None)
    rustup_bin = Path.home() / ".cargo" / "bin"
    if shutil.which("cargo") is None and (rustup_bin / "cargo").is_file():
        env["PATH"] = str(rustup_bin) + os.pathsep + env.get("PATH", "")
    # Share compilation artifacts between arms without mixing their installed
    # source trees. Cargo keys these artifacts by target, dependencies and input.
    env["CARGO_TARGET_DIR"] = str(
        destination.parent.parent.parent / ".measurement-build-cache"
    )
    subprocess.run(
        [
            uv,
            "build",
            "--offline",
            "--wheel",
            "--python",
            python,
            "--out-dir",
            str(destination / "wheel"),
            str(tree),
        ],
        env=env,
        check=True,
    )
    wheels = list((destination / "wheel").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("source deployment expected one wheel")
    installed = destination / "installed"
    # uv validates native wheel tags for this Python and performs wheel
    # relocation, including platform wheels with .data entries.
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--no-deps",
            "--no-compile-bytecode",
            "--python",
            python,
            "--target",
            str(installed),
            str(wheels[0]),
        ],
        env=env,
        check=True,
    )
    metadata = list(installed.glob("causalab-*.dist-info"))
    if len(metadata) != 1:
        raise ValueError("wheel must install exactly one Causalab distribution")
    write_record(
        metadata[0] / "direct_url.json",
        {
            "url": source_record["repository_url"],
            "vcs_info": {
                "vcs": "git",
                "requested_revision": source_record["requested_revision"],
                "commit_id": source_record["source_commit"],
            },
        },
    )
    if interpreter_identity(python) != runtime:
        raise ValueError("target interpreter changed during the build")
    record = {
        **source_record,
        "wheel_sha256": file_hash(wheels[0]),
        "wheel_filename": wheels[0].name,
        "build_runtime": runtime,
        "package_root": str(installed.resolve()),
        "installed_files": manifest(installed),
    }
    write_record(receipt, record)
    return record


def build_arm(
    repository: Path, revision: str, destination: Path, *, python: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="causalab-measure-source-") as temporary:
        source = Path(temporary) / "bundle"
        archive_arm(repository, revision, source)
        return build_source(source, destination, python=python)
