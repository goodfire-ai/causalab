"""Deploy measurement studies over SSH using the shared job supervisor."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any
import uuid

from causalab.measurement.collection import file_hash, write_record
from causalab.measurement.census import strip_pins
from causalab.measurement.plan import execution_plan
from causalab.measurement.deployment.bindings import authored_bindings, target_bindings
from causalab.workflow.document import parse_workflow
from causalab.remote.transport import SSH, query, remote_path, worker_command
from causalab.measurement.deployment.installation import archive_arm
from causalab.measurement.study.scheduler import manifest
from causalab.measurement.deployment.transfer import extract
from causalab.measurement.deployment.source_pins import (
    check_source_pins,
    check_source_presence,
)


@dataclass
class RemoteWorkflowError(ValueError):
    """A workflow comparison cannot be packaged without changing its contract."""

    reason: str
    arm: str | None = None

    def __str__(self) -> str:
        return f"arm {self.arm!r}: {self.reason}" if self.arm else self.reason


def _freeze_document(
    document: Path, destination: Path, *, separate_specifications: bool = False
) -> dict:
    from causalab.protocol.pipeline import read_document
    from causalab.io.sources import load_text
    from causalab.measurement.census import check_pins, strip_pins
    from causalab.workflow.document import ProtocolStep, ScriptStep, parse_workflow

    raw, pins = strip_pins(dict(load_text(document)))
    parsed = parse_workflow(raw)
    if any(
        not isinstance(step, ProtocolStep)
        and (not isinstance(step, ScriptStep) or step.path is not None)
        for step in parsed.steps.values()
    ):
        raise ValueError(
            "remote measurement packaging supports protocol steps and module-located workflow scripts only"
        )
    if pins is not None:
        # Check the source closure before replacing its document names/bytes.
        # Other categories remain pinned for validation against remote resources.
        documents = {
            step.document: file_hash(document.parent / step.document)
            for step in parsed.steps.values()
            if isinstance(step, ProtocolStep)
        }
        check_pins(
            {"documents": pins.get("documents", {})},
            {"documents": documents},
        )
    destination.mkdir()
    frozen_documents: dict[str, str] = {}
    for name, step in parsed.steps.items():
        if isinstance(step, ProtocolStep):
            path = (document.parent / step.document).resolve()
            authored = dict(read_document(path, path.parent, step.set).raw)
            # The launcher writes both metadata files after freezing protocols.
            relative = (
                f"specifications/{name}.json"
                if separate_specifications or name in {"workflow", "bindings"}
                else f"{name}.json"
            )
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            write_record(target, authored)
            frozen_documents[relative] = file_hash(target)
            raw["steps"][name]["document"] = relative
            raw["steps"][name].pop("set", None)
    if pins is not None:
        # the census travels with the frozen study copy, under the key the
        # worker strips before the workflow loader reads the document
        raw["pins"] = {key: dict(value) for key, value in pins.items()}
        if frozen_documents:
            raw["pins"]["documents"] = dict(sorted(frozen_documents.items()))
        else:
            raw["pins"].pop("documents", None)
    return raw


def _freeze_study(document: Path, destination: Path) -> dict:
    from causalab.io.sources import load_text
    from causalab.measurement.census import strip_pins
    from causalab.workflow.document import parse_workflow
    from causalab.measurement.study.definitions import load_definitions

    # a study copy may carry its census; the workflow loader never reads it
    parsed = parse_workflow(strip_pins(dict(load_text(document)))[0])
    if parsed.measurement is None:
        raise ValueError("remote measurement requires an authored measurement block")
    definitions = (
        load_definitions(document, parsed.measurement)
        if parsed.measurement.get("comparison", "code") == "workflow"
        else None
    )
    frozen = _freeze_document(document, destination)
    if definitions is None:
        return frozen
    for arm, spec in parsed.measurement["arms"].items():
        reference = spec.get("workflow")
        if reference is None:
            continue
        selected = definitions[arm].path
        directory = destination / "arms" / arm
        directory.parent.mkdir(exist_ok=True)
        candidate = _freeze_document(selected, directory, separate_specifications=True)
        write_record(directory / "workflow.json", candidate)
        frozen["measurement"]["arms"][arm]["workflow"] = (
            (directory / "workflow.json").relative_to(destination).as_posix()
        )
    return frozen


def _check_arm_sources(study: dict, payload: Path, sources: dict) -> None:
    """Validate every selected source archive before contacting the remote host."""
    authored_plan = parse_workflow(strip_pins(study)[0]).measurement
    assert authored_plan is not None
    plan = execution_plan(authored_plan)
    if plan.get("comparison", "code") == "workflow":
        if len({source["source_commit"] for source in sources.values()}) != 1:
            raise RemoteWorkflowError(
                "workflow comparison requires all arms to resolve to the same code commit"
            )
        for arm, spec in plan["arms"].items():
            reference = spec.get("workflow")
            selected = (
                json.loads((payload / "study" / reference).read_text())
                if reference is not None
                else study
            )
            check_source_pins(
                selected, payload / "sources" / arm / "source.tar", arm=arm
            )
        return
    anchor = plan["source_pin_anchor"]
    check_source_pins(study, payload / "sources" / anchor / "source.tar", arm=anchor)
    for arm in plan["arms"]:
        if arm != anchor:
            check_source_presence(
                study, payload / "sources" / arm / "source.tar", arm=arm
            )


def launch(args) -> dict:
    if args.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    ssh = SSH(args.host)
    controller = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{args.controller_revision}^{{commit}}",
        ],
        cwd=controller,
        text=True,
    ).strip()
    for entrypoint in (
        "causalab/measurement/deployment/transfer.py",
        "causalab/remote/supervisor.py",
    ):
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{entrypoint}"],
            cwd=controller,
            check=True,
        )
    bindings = json.loads(args.bindings.read_text())
    with tempfile.TemporaryDirectory(prefix="causalab-measure-deploy-") as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        source_archive = Path(temporary) / "controller.tar"
        with source_archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", "--format=tar", commit],
                cwd=controller,
                stdout=stream,
                check=True,
            )
        (payload / "source").mkdir()
        extract(source_archive, payload / "source")
        study = _freeze_study(args.document.resolve(), payload / "study")
        authored_plan = parse_workflow(strip_pins(study)[0]).measurement
        assert authored_plan is not None  # _freeze_study requires a measurement block.
        plan = execution_plan(authored_plan)
        bindings = target_bindings(bindings, plan)
        targets = (
            {"source": study["measurement"]["source"]}
            if plan.get("mode") == "single"
            else study["measurement"]["arms"]
        )
        sources = {}
        for arm, spec in targets.items():
            binding = bindings["arms"][arm]
            if set(binding) != {"repository", "python"}:
                raise ValueError(
                    "remote launch requires a local source repository and remote Python for each arm"
                )
            if not PurePosixPath(binding["python"]).is_absolute():
                raise ValueError(
                    "each remote Python must be an absolute executable path"
                )
            repository = (
                args.bindings.resolve().parent / binding["repository"]
            ).resolve()
            source = archive_arm(
                repository, spec["revision"], payload / "sources" / arm
            )
            sources[arm] = {
                "requested_revision": spec["revision"],
                "source_commit": source["source_commit"],
                "source_archive_sha256": source["source_archive_sha256"],
            }
            spec["revision"] = source["source_commit"]
        _check_arm_sources(study, payload, sources)
        anchor = plan["source_pin_anchor"]
        # Every arm's source preflight finishes before any SSH or upload.
        # Resolve the remote home to make every deployment binding independent of cwd.
        home = json.loads(
            ssh.call(
                shlex.join(
                    [
                        args.control_python,
                        "-c",
                        "import json,pathlib; print(json.dumps(str(pathlib.Path.home())))",
                    ]
                )
            )
        )
        root = str(PurePosixPath(home) / args.remote_root)
        job = remote_path(root, uuid.uuid4().hex)
        for arm, binding in bindings["arms"].items():
            bindings["arms"][arm] = {
                "source": str(PurePosixPath(job) / "sources" / arm),
                "python": binding["python"],
            }
        for flag, key, name in (
            (args.upload_data, "data_root", "data"),
            (args.upload_artifacts, "artifacts_root", "artifacts"),
        ):
            if flag is not None:
                manifest(
                    flag.resolve()
                )  # Refuse symlink uploads; copy actual supplied files.
                shutil.copytree(flag.resolve(), payload / name)
                bindings[key] = str(PurePosixPath(job) / name)
            elif not PurePosixPath(bindings[key]).is_absolute():
                raise ValueError(
                    f"{key} must be an absolute remote path unless uploaded"
                )
        write_record(payload / "study/workflow.json", study)
        write_record(payload / "study/bindings.json", authored_bindings(bindings, plan))
        archive = Path(temporary) / "payload.tar"
        with tarfile.open(archive, "w") as stream:
            for path in sorted(payload.iterdir()):
                stream.add(path, arcname=path.name)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "kind": "measurement",
            "host": args.host,
            "remote_job": job,
            "control_python": args.control_python,
            "source_commit": commit,
            "sources": sources,
            "authored_document_sha256": file_hash(args.document),
            "payload_sha256": file_hash(archive),
            "scheduler": "direct",
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        print(
            f"Deploying measurement controller {commit}; receipt: {args.receipt}",
            flush=True,
        )
        transfer = (
            payload / "source/causalab/measurement/deployment/transfer.py"
        ).read_text()
        with archive.open("rb") as stream:
            ssh.call(
                shlex.join(
                    [
                        args.control_python,
                        "-c",
                        transfer,
                        "receive",
                        job,
                        receipt["payload_sha256"],
                    ]
                ),
                stdin=stream,
            )
        python = bindings["arms"][anchor]["python"]
        command = [
            python,
            "-c",
            "from causalab.cli import main; raise SystemExit(main())",
            "measure",
            str(PurePosixPath(job) / "study/workflow.json"),
            "--bindings",
            str(PurePosixPath(job) / "study/bindings.json"),
            "--out",
            str(PurePosixPath(job) / "results"),
        ]
        spec = {
            **receipt,
            "command": command,
            "timeout_seconds": args.timeout_seconds,
            "hf_cache": args.hf_cache,
            "allow_download": args.allow_download,
        }
        ssh.call(worker_command(receipt, "init"), data=json.dumps(spec).encode())
        return {"receipt": str(args.receipt), **query(receipt, "launch")}


def fetch(receipt: dict, output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError("fetch requires a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".measurement-fetch-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        archive = staging / "artifacts.tar"
        transfer = str(
            PurePosixPath(receipt["remote_job"])
            / "source/causalab/measurement/deployment/transfer.py"
        )
        with archive.open("wb") as stream:
            SSH(receipt["host"]).call(
                shlex.join(
                    [
                        receipt["control_python"],
                        transfer,
                        "export",
                        receipt["remote_job"],
                    ]
                ),
                output=stream,
            )
        files = staging / "files"
        files.mkdir()
        extract(archive, files)
        exported = json.loads((files / "export.json").read_text())
        actual = manifest(files)
        actual.pop("export.json")
        if actual != exported["files"]:
            raise ValueError("fetched artifact manifest mismatch")
        write_record(files / "remote.json", receipt)
        files.rename(output)
    # Re-render local links in a separate derived directory; keep downloaded
    # reports byte-identical to their export manifest.
    ledger = output / "results/collections/study.json"
    if ledger.is_file():
        study = json.loads(ledger.read_text())
        if study["status"] == "completed":
            from causalab.measurement.analysis.reports import write_reports

            reports = output / "local_reports"
            write_reports(
                {
                    case: {arm: ledger.parent / path for arm, path in arms.items()}
                    for case, arms in study["collections"].items()
                },
                study["plan"],
                reports,
            )
    return exported["state"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    start = commands.add_parser("launch")
    start.add_argument("document", type=Path)
    start.add_argument("--bindings", type=Path, required=True)
    start.add_argument("--host", required=True)
    start.add_argument("--receipt", type=Path, required=True)
    start.add_argument("--controller-revision", default="HEAD")
    start.add_argument("--remote-root", default=".cache/causalab-measure")
    start.add_argument("--control-python", default="python3")
    start.add_argument("--hf-cache")
    start.add_argument("--allow-download", action="store_true")
    start.add_argument("--timeout-seconds", type=int, default=21600)
    start.add_argument("--upload-data", type=Path)
    start.add_argument("--upload-artifacts", type=Path)
    for action in ("status", "logs", "cancel", "fetch", "resume"):
        command = commands.add_parser(action)
        command.add_argument("--receipt", type=Path, required=True)
        if action == "fetch":
            command.add_argument("--out", type=Path, required=True)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.action == "launch":
        result = launch(args)
    else:
        receipt = json.loads(args.receipt.read_text())
        if receipt.get("kind") != "measurement":
            raise ValueError("expected a measurement job receipt")
        if args.action == "fetch":
            result = fetch(receipt, args.out)
        elif args.action == "logs":
            result = query(receipt, "fetch")
            print(
                base64.b64decode(
                    result["files"].get("run.log", ""), validate=True
                ).decode(errors="replace")
            )
            return
        else:
            result = query(receipt, args.action)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
