"""Single-source packaging preserves its authored shape and source constraints."""

from copy import deepcopy
import io
import json
import shutil
import sys
import tarfile

import pytest

from causalab.measurement.collection import write_record
from causalab.measurement.census import CensusError
from causalab.measurement.deployment import remote
from causalab.measurement.deployment.source_pins import SourcePinError
from causalab.measurement.runtime.pins import PinContract
from tests.measurement.deployment.test_source_pins import _pinned, _revisions
from tests.measurement.runtime.test_pin_contract import PACKAGE_ROOT, MODULE, census
from tests.measurement.runtime.test_worker_pin_lifecycle import _worker

pytestmark = pytest.mark.unit


def single_inputs(tmp_path):
    repo = tmp_path / "repo"
    before, after = _revisions(repo)
    study = _pinned("scripts")
    study["measurement"] = {
        "version": 1,
        "mode": "single",
        "source": {"revision": before},
        "cases": {"workflow": {"kind": "workflow"}},
        "seeds": [7],
        "repeats": 1,
        "profile": False,
    }
    study["pins"] = study.pop("pins")
    document = tmp_path / "workflow.json"
    write_record(document, study)
    binding = {
        "source": {"repository": str(repo), "python": "/remote/venv/bin/python"},
        "device": "cpu",
        "data_root": "/remote/data",
        "artifacts_root": "/remote/artifacts",
    }
    bindings = tmp_path / "bindings.json"
    write_record(bindings, binding)
    return document, bindings, before, after


def launch_args(document, bindings, receipt):
    return remote.parser().parse_args(
        [
            "launch",
            str(document),
            "--bindings",
            str(bindings),
            "--host",
            "compute",
            "--receipt",
            str(receipt),
        ]
    )


def test_single_launch_packages_one_source_and_selects_its_interpreter(
    tmp_path, monkeypatch
):
    document, bindings, revision, _ = single_inputs(tmp_path)
    received = {}
    archived = []
    original_archive = remote.archive_arm

    def archive(repository, revision, destination):
        archived.append(destination.name)
        return original_archive(repository, revision, destination)

    class Transport:
        def __init__(self, host):
            assert host == "compute"

        def call(self, command, **kwargs):
            if "stdin" in kwargs:
                with tarfile.open(
                    fileobj=io.BytesIO(kwargs["stdin"].read())
                ) as payload:
                    received["names"] = payload.getnames()
                    received["bindings"] = json.load(
                        payload.extractfile("study/bindings.json")
                    )
                    received["study"] = json.load(
                        payload.extractfile("study/workflow.json")
                    )
                return b""
            if "data" in kwargs:
                received["job"] = json.loads(kwargs["data"])
                return b""
            return json.dumps("/remote/home").encode()

    monkeypatch.setattr(remote, "SSH", Transport)
    monkeypatch.setattr(remote, "archive_arm", archive)
    monkeypatch.setattr(remote, "query", lambda *args: {"status": "submitted"})
    result = remote.launch(launch_args(document, bindings, tmp_path / "job.json"))
    assert result["status"] == "submitted"
    assert archived == ["source"]
    assert set(received["job"]["sources"]) == {"source"}
    assert received["job"]["command"][0] == "/remote/venv/bin/python"
    assert set(received["bindings"]) == {
        "source",
        "device",
        "data_root",
        "artifacts_root",
    }
    assert received["bindings"]["source"]["source"].endswith("/sources/source")
    assert received["study"]["measurement"]["source"]["revision"] == revision
    assert "arms" not in received["study"]["measurement"]
    assert "sources/source/source.tar" in received["names"]
    assert not any(
        "sources/before" in name or "sources/after" in name
        for name in received["names"]
    )


def test_single_launch_refuses_changed_authored_source_before_ssh(
    tmp_path, monkeypatch
):
    document, bindings, _, changed = single_inputs(tmp_path)
    study = json.loads(document.read_text())
    study["measurement"]["source"]["revision"] = changed
    write_record(document, study)
    calls = []
    monkeypatch.setattr(remote.SSH, "call", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(SourcePinError) as caught:
        remote.launch(launch_args(document, bindings, tmp_path / "job.json"))
    assert caught.value.arm == "source"
    assert "source bytes differ" in caught.value.reason
    assert calls == []
    assert not (tmp_path / "job.json").exists()


def test_single_pin_policy_holds_authored_and_frozen_source():
    actual = census()
    contract = PinContract.resolve(
        actual,
        actual,
        arm="source",
        source_pin_anchor="source",
        package_root=PACKAGE_ROOT,
    )
    altered = {**actual, "code": {MODULE: "0" * 64}}
    with pytest.raises(CensusError, match="pins.code"):
        PinContract.resolve(
            altered,
            actual,
            arm="source",
            source_pin_anchor="source",
            package_root=PACKAGE_ROOT,
        )
    with pytest.raises(CensusError, match="pins.code"):
        contract.check(altered)


def test_single_worker_applies_anchor_policy_on_initial_load(tmp_path):
    worker = _worker(tmp_path, arm="source")
    worker.config["plan"] = {"source_pin_anchor": "source"}
    for name in worker.raw["pins"]["scripts"]:
        worker.raw["pins"]["scripts"][name] = "0" * 64
    with pytest.raises(CensusError, match="pins.scripts"):
        worker.load(7)


def test_single_local_preparation_installs_one_source_and_preserves_resume_contract(
    tmp_path, monkeypatch
):
    from causalab.measurement.study import controller
    from causalab.measurement.analysis import reports

    document, bindings, revision, _ = single_inputs(tmp_path)
    authored = document.read_bytes()
    calls = []
    configs = []
    scheduled = []

    def build(repository, commit, destination, *, python):
        calls.append((commit, destination.name))
        return {"package_root": str(tmp_path / "installed"), "source_commit": commit}

    class Session:
        def __init__(self, python, config, logs):
            configs.append(config)
            self.identity = {"execution_probe": {}}

        def close(self):
            pass

    def schedule(plan, contract, output, open_session, **kwargs):
        scheduled.append((deepcopy(plan), deepcopy(contract), kwargs["resume"]))
        assert set(plan["arms"]) == {"source"}
        with open_session("source"):
            pass
        return {}

    monkeypatch.setattr(controller, "build_arm", build)
    monkeypatch.setattr(controller, "ProcessSession", Session)
    monkeypatch.setattr(controller, "run_schedule", schedule)
    monkeypatch.setattr(reports, "write_reports", lambda *args: None)
    output = tmp_path / "run"
    controller.run(document, bindings, output)
    controller.run(document, bindings, output, resume=True)
    assert calls == [(revision, "source"), (revision, "source")]
    assert [entry[2] for entry in scheduled] == [False, True]
    assert configs[0]["plan"]["observation_policy"] == "not_requested"
    assert configs[0]["plan"]["source_pin_anchor"] == "source"
    assert document.read_bytes() == authored
    changed = json.loads(bindings.read_text())
    changed["source"]["python"] = "/different/python"
    write_record(bindings, changed)
    with pytest.raises(ValueError, match="deployment changed"):
        controller.run(document, bindings, output, resume=True)
    assert len(calls) == 2


@pytest.mark.parametrize("action", ["resume", "cancel", "status"])
def test_single_remote_lifecycle_uses_same_job_receipt(tmp_path, monkeypatch, action):
    receipt = {
        "kind": "measurement",
        "sources": {"source": {"source_commit": "a" * 40}},
    }
    path = tmp_path / "job.json"
    write_record(path, receipt)
    calls = []
    monkeypatch.setattr(sys, "argv", ["measure-remote", action, "--receipt", str(path)])
    monkeypatch.setattr(
        remote,
        "query",
        lambda received, command: calls.append((received, command)) or {"status": "ok"},
    )
    remote.main()
    assert calls == [(receipt, action)]


def test_single_fetch_renders_local_report_with_single_plan(tmp_path, monkeypatch):
    from causalab.measurement.analysis import reports
    from causalab.measurement.study.scheduler import manifest

    tree = tmp_path / "export"
    ledger = tree / "results/collections/study.json"
    ledger.parent.mkdir(parents=True)
    plan = {"mode": "single", "source": {"revision": "a" * 40}}
    write_record(
        ledger,
        {
            "status": "completed",
            "plan": plan,
            "collections": {"workflow": {"source": "workflow.source.json"}},
        },
    )
    write_record(ledger.parent / "workflow.source.json", {"receipt": "retained"})
    write_record(
        tree / "export.json",
        {"state": {"status": "completed"}, "files": manifest(tree)},
    )
    archive = tmp_path / "export.tar"
    with tarfile.open(archive, "w") as stream:
        for path in tree.iterdir():
            stream.add(path, arcname=path.name)

    class Transport:
        def __init__(self, host):
            pass

        def call(self, command, *, output):
            with archive.open("rb") as stream:
                shutil.copyfileobj(stream, output)

    received = []
    monkeypatch.setattr(remote, "SSH", Transport)
    monkeypatch.setattr(
        reports,
        "write_reports",
        lambda collections, plan, output: received.append((collections, plan, output)),
    )
    output = tmp_path / "fetched"
    remote.fetch(
        {"host": "compute", "control_python": "python3", "remote_job": "/job"}, output
    )
    collections, actual_plan, destination = received[0]
    assert actual_plan == plan
    assert collections["workflow"]["source"].is_file()
    assert destination == output / "local_reports"
