"""Real timing/profile receipts survive streaming export and local report regeneration."""

from contextlib import contextmanager
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import unquote

import pytest

from causalab.measurement.analysis.receipts import load_measurement
from causalab.measurement.analysis.reports import write_reports
from causalab.measurement.capture import worker as capture_worker
from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.deployment import remote, transfer
from causalab.measurement.study.scheduler import run_schedule
from causalab.remote import supervisor, transport
from tests.measurement.test_single_runtime import make_worker, plan

pytestmark = pytest.mark.unit


class LocalSSH:
    """Replace only the SSH hop; execute the deployed stdlib entry points."""

    def __init__(self, host):
        assert host == "local-test"

    def call(self, command, *, data=None, stdin=None, output=None):
        result = subprocess.run(
            shlex.split(command),
            input=data,
            stdin=stdin,
            stdout=output if output is not None else subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=True,
        )
        return result.stdout if output is None else b""


def deployed_job(tmp_path):
    job = tmp_path / "remote job"
    for module, relative in (
        (transfer, "source/causalab/measurement/deployment/transfer.py"),
        (supervisor, "source/causalab/remote/supervisor.py"),
    ):
        destination = job / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module.__file__, destination)
    return job, {
        "kind": "measurement",
        "host": "local-test",
        "remote_job": str(job),
        "control_python": sys.executable,
    }


def collect_study(job, monkeypatch):
    """Use an offline saved-output operation with real timing and Torch capture."""
    from causalab.measurement.capture import ranges
    from causalab.measurement.runtime import worker as runtime_worker

    authored = {
        **plan(),
        "seeds": [0],
        "repeats": 2,
        "profile": {"cases": ["workflow"], "backends": {"torch": {}}},
    }

    @contextmanager
    def phases(*args, **kwargs):
        yield {"scope": "offline saved-output operation"}

    monkeypatch.setattr(ranges, "phase_ranges", phases)

    def capture_adapter(config):
        worker = make_worker(job)
        worker.engine, worker.bundles = None, {}
        return worker

    monkeypatch.setattr(runtime_worker, "Worker", capture_adapter)

    @contextmanager
    def open_session(source):
        assert source == "source"
        worker = make_worker(job)
        sample = worker.sample

        def clean(case, seed, repeat, directory):
            return sample({"case": case, "seed": seed, "directory": str(directory)})

        def capture(case, seed, repeat, directory):
            directory.mkdir()
            response = capture_worker.capture(
                {
                    "device": "cpu",
                    "plan": authored,
                    "capture": {
                        "backend": "torch",
                        "mode": "warm",
                        "case": case,
                        "seed": seed,
                        "directory": str(directory),
                        "options": {},
                    },
                }
            )
            trace = directory / "trace.json"
            assert json.loads(trace.read_text())["traceEvents"]
            log = directory / "capture.log"
            log.write_text("Torch capture completed\n")
            receipt = directory / "measurement.json"
            write_record(
                receipt,
                {
                    "mode": "single",
                    "observation_policy": "not_requested",
                    "capture_plan": {"backends": ["torch"], "modes": ["warm"]},
                    "captures": [
                        {
                            **response,
                            "id": "torch:warm",
                            "backend": "torch",
                            "mode": "warm",
                            "seed": seed,
                            "repeat": repeat,
                            "artifacts": [
                                {"file": trace.name, "sha256": file_hash(trace)}
                            ],
                            "logs": [{"file": log.name, "sha256": file_hash(log)}],
                        }
                    ],
                },
            )
            return receipt

        worker.sample, worker.capture = clean, capture
        yield worker

    collections = run_schedule(authored, {}, job / "results/collections", open_session)
    write_reports(collections, authored, job / "results/reports")
    write_record(job / "state.json", {"status": "completed"})
    assert not list((job / "results").rglob("*.safetensors"))


def test_fetch_regenerates_single_report_with_portable_timing_and_trace_links(
    tmp_path, monkeypatch
):
    job, receipt = deployed_job(tmp_path)
    collect_study(job, monkeypatch)
    monkeypatch.setattr(remote, "SSH", LocalSSH)
    output = tmp_path / "downloaded"
    assert remote.fetch(receipt, output)["status"] == "completed"
    exported = json.loads((output / "export.json").read_text())
    for name, expected in exported["files"].items():
        assert file_hash(output / name) == expected
    shutil.rmtree(job)

    collection = output / "results/collections/workflow.source.json"
    record, samples = load_measurement(collection, require_observations=False)
    assert len(samples) == 2
    assert record["captures"][0]["observation_status"] == "not_requested"
    expected_artifacts = {
        (collection.parent / name).resolve()
        for sample in samples.values()
        for name in sample["output_files"]
    }
    expected_artifacts.update(
        (collection.parent / ref["file"]).resolve()
        for capture in record["captures"]
        for ref in [*capture["artifacts"], *capture["logs"]]
    )
    expected_artifacts.update(
        (collection.parent / name).resolve()
        for capture in record["captures"]
        for name in capture["output_files"]
    )
    linked = set()
    pages = list((output / "local_reports").glob("*.html"))
    assert pages
    for page in pages:
        text = page.read_text()
        assert not any(
            word in text.lower()
            for word in ("speedup", "before_after", "equivalence verdict")
        )
        for href in re.findall(r'href="([^"]+)"', text):
            target = (page.parent / unquote(href)).resolve()
            assert target.is_relative_to(output)
            assert target.is_file(), href
            linked.add(target)
    assert expected_artifacts <= linked
    assert collection.resolve() in linked
    report = json.loads((output / "local_reports/study.json").read_text())
    measurement = report["cases"]["workflow"]["measurement"]
    assert measurement["collection_status"] == "completed"
    assert measurement["capture_status"] == "completed"
    assert measurement["observation_check"]["status"] == "not_requested"


def test_fetch_refuses_corrupted_stream_before_publishing_or_reporting(
    tmp_path, monkeypatch
):
    job, receipt = deployed_job(tmp_path)
    collect_study(job, monkeypatch)

    class CorruptedSSH(LocalSSH):
        def call(self, command, *, output, **kwargs):
            archive = tmp_path / "unaltered-export.tar"
            with archive.open("wb") as stream:
                super().call(command, output=stream, **kwargs)
            changed = False
            with (
                tarfile.open(archive) as source,
                tarfile.open(fileobj=output, mode="w|") as target,
            ):
                for member in source.getmembers():
                    payload = source.extractfile(member).read()
                    if member.name.endswith("saved.txt") and not changed:
                        payload = b"corrupted in transit"
                        member.size = len(payload)
                        changed = True
                    target.addfile(member, io.BytesIO(payload))
            assert changed
            return b""

    monkeypatch.setattr(remote, "SSH", CorruptedSSH)
    output = tmp_path / "corrupt-download"
    with pytest.raises(ValueError, match="fetched artifact manifest mismatch"):
        remote.fetch(receipt, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".measurement-fetch-*"))


def test_single_job_supervisor_cancels_and_resumes_the_same_source(
    tmp_path, monkeypatch
):
    job, receipt = deployed_job(tmp_path)
    (job / "study").mkdir()
    single = {
        "version": 1,
        "mode": "single",
        "source": {"revision": "a" * 40},
        "cases": {"workflow": {"kind": "workflow", "cold_process": False}},
        "seeds": [0],
        "repeats": 1,
        "profile": False,
    }
    write_record(job / "study/workflow.json", {"measurement": single})
    worker = job / "source/lightweight.py"
    worker.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "root=Path('..'); plan=json.loads((root/'study/workflow.json').read_text())['measurement']\n"
        "assert plan['mode']=='single' and 'arms' not in plan\n"
        "results=root/'results'; results.mkdir(exist_ok=True)\n"
        "prep=results/'preparation.json'\n"
        "if '--resume' in sys.argv:\n"
        " assert json.loads(prep.read_text())['plan']==plan\n"
        " (results/'resumed.json').write_text(json.dumps({'plan':plan,'resume_count':sys.argv.count('--resume')}))\n"
        "else:\n"
        " prep.write_text(json.dumps({'plan':plan,'sources':['source']}))\n"
        " while True: time.sleep(.05)\n"
    )
    spec = {
        **receipt,
        "source_commit": "a" * 40,
        "scheduler": "direct",
        "command": [sys.executable, str(worker)],
        "timeout_seconds": 10,
    }
    LocalSSH("local-test").call(
        transport.worker_command(receipt, "init"), data=json.dumps(spec).encode()
    )
    monkeypatch.setattr(transport, "SSH", LocalSSH)

    def wait_for(predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.02)
        pytest.fail("supervisor condition did not become true")

    def terminal():
        state = transport.query(receipt, "status")
        return state if state["status"] in supervisor.TERMINAL else None

    try:
        assert transport.query(receipt, "launch")["status"] == "submitted"
        wait_for(lambda: (job / "results/preparation.json").is_file())
        assert transport.query(receipt, "cancel")["cancel_requested"]
        assert wait_for(terminal)["status"] == "cancelled"
        assert transport.query(receipt, "resume")["status"] == "submitted"
        assert wait_for(terminal)["status"] == "completed"
        preparation = json.loads((job / "results/preparation.json").read_text())
        resumed = json.loads((job / "results/resumed.json").read_text())
        assert preparation == {"plan": single, "sources": ["source"]}
        assert resumed == {"plan": single, "resume_count": 1}
        histories = list((job / "recovery").glob("*/state.json"))
        assert len(histories) == 1
        assert json.loads(histories[0].read_text())["status"] == "cancelled"
    finally:
        transport.query(receipt, "cancel")
        wait_for(terminal)
