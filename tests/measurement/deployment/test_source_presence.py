"""Candidate source may change, but required archived modules must exist."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from causalab.measurement.deployment import remote, source_pins
from causalab.measurement.deployment.installation import archive_arm
from causalab.measurement.deployment.source_pins import SourcePinError
from tests.measurement.deployment.test_remote_pins import _study
from tests.measurement.deployment.test_source_pins import _pinned, _revisions

pytestmark = pytest.mark.unit


def _commit(root: Path) -> str:
    # Retain an ordinary archive member when deleting the only test module.
    # Git's empty-tree tar has no source members to inspect.
    (root / "README.md").write_text("Source presence fixture.\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "candidate source",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


@pytest.mark.parametrize("category", ["code", "scripts"])
def test_presence_accepts_distinct_committed_source_bytes(tmp_path, category):
    repo = tmp_path / "repo"
    _, after = _revisions(repo)
    archive_arm(repo, after, tmp_path / "candidate")
    source_pins.check_source_presence(
        _pinned(category), tmp_path / "candidate/source.tar", arm="after"
    )


@pytest.mark.parametrize("key", ["third_party.helper", "causalab.example#helper.py"])
def test_presence_preserves_external_and_closure_restrictions(tmp_path, key):
    repo = tmp_path / "repo"
    _, after = _revisions(repo)
    archive_arm(repo, after, tmp_path / "candidate")
    study = _pinned("code")
    study["pins"]["code"] = {key: "a" * 64}
    with pytest.raises(SourcePinError, match="cannot preflight"):
        source_pins.check_source_presence(
            study, tmp_path / "candidate/source.tar", arm="after"
        )


@pytest.mark.parametrize("arm", ["after", "eager"])
@pytest.mark.parametrize("category", ["code", "scripts"])
@pytest.mark.parametrize("damage", [None, "missing", "symlink"])
def test_launch_preflights_candidate_presence_before_ssh_and_receipt(
    tmp_path, monkeypatch, arm, category, damage
):
    workflow, _ = _study(tmp_path)
    raw = json.loads(workflow.read_text())
    repo = tmp_path / "repo"
    before, after = _revisions(repo)
    candidate = after
    if damage is not None:
        module = repo / "causalab/example.py"
        module.unlink()
        if damage == "symlink":
            module.symlink_to("elsewhere.py")
        candidate = _commit(repo)
    arms = {"before": {"revision": before}, "after": {"revision": after}}
    arms[arm] = {"revision": candidate}
    raw["measurement"]["arms"] = arms
    raw["pins"][category] = _pinned(category)["pins"][category]
    if category == "scripts":
        raw["steps"]["report"] = _pinned(category)["steps"]["report"]
    workflow.write_text(json.dumps(raw))
    bindings = tmp_path / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "arms": {
                    name: {"repository": str(repo), "python": "/remote/python"}
                    for name in arms
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

    def reached_ssh(*args, **kwargs):
        calls.append(args)
        raise ReachedSSH("all arm preflights completed")

    monkeypatch.setattr(remote.SSH, "call", reached_ssh)
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
    if damage is None:
        with pytest.raises(ReachedSSH, match="all arm preflights completed"):
            remote.launch(args)
        assert len(calls) == 1
    else:
        with pytest.raises(SourcePinError) as caught:
            remote.launch(args)
        assert caught.value.arm == arm
        assert caught.value.category == category
        assert caught.value.key == "causalab.example"
        assert ("absent" if damage == "missing" else "regular") in caught.value.reason
        assert calls == []
    assert not args.receipt.exists()
