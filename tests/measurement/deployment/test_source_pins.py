"""Source pins are checked against committed arms before contacting a host."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

from hypothesis import given, settings, strategies as st
import pytest

from causalab.measurement.deployment import remote
from causalab.measurement.deployment.installation import archive_arm
from causalab.measurement.deployment.source_pins import (
    SourcePinError,
    check_source_pins,
)
from tests.measurement.deployment.test_remote_pins import _study

pytestmark = pytest.mark.unit


def _revisions(root: Path) -> tuple[str, str]:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    module = root / "causalab/example.py"
    module.parent.mkdir()
    commits = []
    for value in (1, 2):
        module.write_text(f"def main(inputs, outputs): return {value}\n")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                str(value),
            ],
            cwd=root,
            check=True,
        )
        commits.append(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
        )
    return commits[0], commits[1]


def _pinned(category: str) -> dict:
    digest = hashlib.sha256(b"def main(inputs, outputs): return 1\n").hexdigest()
    return {
        "version": "1",
        "output_dir": "run",
        "steps": {
            "report": {
                "type": "script",
                "script": {"module": "causalab.example"},
                "inputs": {},
                "outputs": {"out": {"file": "out.json", "columns": {"n": "int64"}}},
            }
        },
        "pins": {category: {"causalab.example": digest}},
    }


@pytest.mark.parametrize("category", ["code", "scripts"])
def test_checks_actual_archive_bytes_without_importing_arm(
    tmp_path: Path, category: str
) -> None:
    repo = tmp_path / "repo"
    before, after = _revisions(repo)
    study = _pinned(category)
    if category == "code":
        study["steps"] = {
            "run": {"type": "intervention_protocol", "document": "unused.json"}
        }
    for arm, revision in (("before", before), ("after", after)):
        destination = tmp_path / arm
        archive_arm(repo, revision, destination)
        if arm == "before":
            check_source_pins(study, destination / "source.tar", arm=arm)
        else:
            with pytest.raises(
                SourcePinError, match=f"after.*pins.{category}.causalab.example"
            ):
                check_source_pins(study, destination / "source.tar", arm=arm)
    assert study["pins"] == _pinned(category)["pins"]


@pytest.mark.parametrize("key", ["third_party.helper", "causalab.example#helper.py"])
def test_refuses_source_pins_not_resolvable_from_the_arm(
    tmp_path: Path, key: str
) -> None:
    repo = tmp_path / "repo"
    before, _ = _revisions(repo)
    archive_arm(repo, before, tmp_path / "source")
    study = _pinned("code")
    study["pins"]["code"] = {key: "a" * 64}
    with pytest.raises(SourcePinError, match="cannot preflight"):
        check_source_pins(study, tmp_path / "source/source.tar", arm="before")


@pytest.mark.parametrize("baseline_valid", [False, True])
def test_launch_holds_baseline_pins_and_allows_candidate_source_changes(
    tmp_path: Path, monkeypatch, baseline_valid: bool
) -> None:
    workflow, _ = _study(tmp_path)
    raw = json.loads(workflow.read_text())
    repo = tmp_path / "repo"
    before, after = _revisions(repo)
    raw["measurement"]["arms"] = {
        "before": {"revision": before},
        "after": {"revision": after},
    }
    raw["pins"]["code"] = _pinned("code")["pins"]["code"]
    if not baseline_valid:
        raw["pins"]["code"]["causalab.example"] = "0" * 64
    workflow.write_text(json.dumps(raw))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    arm: {"repository": str(repo), "python": "/remote/python"}
                    for arm in ("before", "after")
                },
                "device": "cpu",
                "data_root": "/data",
                "artifacts_root": "/artifacts",
            }
        )
    )
    calls = []

    class ReachedSSH(RuntimeError):
        pass

    def unexpected_ssh(*args, **kwargs):
        calls.append(args)
        raise ReachedSSH("baseline source preflight completed")

    monkeypatch.setattr(remote.SSH, "call", unexpected_ssh)
    args = remote.parser().parse_args(
        [
            "launch",
            str(workflow),
            "--bindings",
            str(bindings),
            "--host",
            "compute",
            "--receipt",
            str(tmp_path / "job.json"),
        ]
    )
    if baseline_valid:
        # The changed candidate archive is valid: its source pins are frozen
        # against its own installation, never against the baseline's hashes.
        with pytest.raises(ReachedSSH, match="baseline source preflight completed"):
            remote.launch(args)
        assert len(calls) == 1
    else:
        with pytest.raises(SourcePinError, match="before.*pins.code.causalab.example"):
            remote.launch(args)
        assert calls == []
    assert not args.receipt.exists()


def test_unsupported_steps_refused_before_document_pin_check(tmp_path: Path) -> None:
    workflow, _ = _study(tmp_path)
    raw = json.loads(workflow.read_text())
    raw["steps"]["nested"] = {"type": "workflow", "document": "missing.json"}
    raw["pins"]["documents"]["missing.json"] = "a" * 64
    workflow.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="supports protocol steps"):
        remote._freeze_study(workflow, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


@pytest.mark.parametrize("damage", ["missing", "unused", "absent"])
def test_script_census_and_missing_modules_are_refused(
    tmp_path: Path, damage: str
) -> None:
    repo = tmp_path / "repo"
    before, _ = _revisions(repo)
    archive_arm(repo, before, tmp_path / "source")
    study = _pinned("scripts")
    match damage:
        case "missing":
            study["pins"] = {}
            reason = "script has no authored pin"
        case "unused":
            study["steps"] = {
                "run": {"type": "intervention_protocol", "document": "unused.json"}
            }
            reason = "pinned script is unused"
        case "absent":
            study["pins"]["scripts"] = {"causalab.missing": "a" * 64}
            reason = "module is absent"
    with pytest.raises(SourcePinError, match=reason):
        check_source_pins(study, tmp_path / "source/source.tar", arm="before")


def test_package_source_uses_same_hash_as_protocol_code(tmp_path: Path) -> None:
    from causalab.protocol.identity import source_sha256

    repo = tmp_path / "repo"
    _revisions(repo)
    package = repo / "causalab/example/__init__.py"
    package.parent.mkdir()
    package.write_text("raise AssertionError('preflight must not import me')\n")
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
            "package",
        ],
        cwd=repo,
        check=True,
    )
    archive_arm(repo, "HEAD", tmp_path / "source")
    study = _pinned("scripts")
    study["pins"]["scripts"]["causalab.example"] = source_sha256(package)
    check_source_pins(study, tmp_path / "source/source.tar", arm="before")


@settings(max_examples=12, deadline=None)
@given(source=st.binary(max_size=128))
def test_archive_bytes_are_the_pin_invariant(source: bytes) -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "source.tar"
        study = _pinned("scripts")
        study["pins"]["scripts"]["causalab.example"] = hashlib.sha256(
            source
        ).hexdigest()
        for content in (source, source + b"\n"):
            with tarfile.open(archive, "w") as stream:
                member = tarfile.TarInfo("causalab/example.py")
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))
            if content == source:
                check_source_pins(study, archive, arm="before")
            else:
                with pytest.raises(SourcePinError, match="source bytes differ"):
                    check_source_pins(study, archive, arm="after")


@pytest.mark.parametrize(
    ("damage", "remedy"),
    [
        ("external", "local execution"),
        ("closure", "local execution"),
        ("absent", "module name and selected revision"),
        ("nonregular", "regular Python source"),
        ("mismatch", "revisions satisfying the authored source pins"),
        ("missing", "causalab pin <workflow>"),
        ("unused", "causalab pin <workflow>"),
    ],
)
def test_source_pin_remedy_matches_failure(
    tmp_path: Path, damage: str, remedy: str
) -> None:
    study = _pinned("scripts")
    content = b"def main(inputs, outputs): return 1\n"
    match damage:
        case "external" | "closure":
            key = (
                "third_party.helper"
                if damage == "external"
                else "causalab.example#helper.py"
            )
            study["pins"] = {"code": {key: "a" * 64}}
        case "absent":
            study["pins"]["scripts"] = {"causalab.missing": "a" * 64}
        case "mismatch":
            content += b"# changed\n"
        case "missing":
            study["pins"] = {}
        case "unused":
            study["steps"] = {
                "run": {"type": "intervention_protocol", "document": "unused.json"}
            }
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("causalab/example.py")
        if damage == "nonregular":
            member.type = tarfile.SYMTYPE
            member.linkname = "elsewhere.py"
            stream.addfile(member)
        else:
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    with pytest.raises(SourcePinError) as caught:
        check_source_pins(study, archive, arm="before")
    error = caught.value
    assert error.arm == "before"
    assert remedy in error.remedy
    assert error.remedy in str(error)
    if damage != "mismatch":
        assert "revisions satisfying" not in str(error)
