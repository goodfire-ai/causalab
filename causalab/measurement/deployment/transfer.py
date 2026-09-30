"""Standalone stdlib-only, streaming measurement payload/artifact transport."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: Path, root: Path) -> None:
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError(
            "archive transfer requires a patched Python with tarfile.data_filter"
        )
    root = root.resolve()
    with tarfile.open(archive) as stream:
        seen = set()
        for member in stream.getmembers():
            target = (root / member.name).resolve()
            if (
                not target.is_relative_to(root)
                or target in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe or duplicate archive entry: {member.name}")
            if target.exists() and not (target.is_dir() and member.isdir()):
                raise ValueError(
                    f"archive would overwrite an existing file: {member.name}"
                )
            seen.add(target)
        stream.extractall(root, filter="data")


def receive(job: Path, expected_hash: str) -> None:
    job.mkdir(parents=True, exist_ok=False)
    archive = job / "payload.tar"
    with archive.open("xb") as output:
        for chunk in iter(lambda: sys.stdin.buffer.read(1024 * 1024), b""):
            output.write(chunk)
    if sha256(archive) != expected_hash:
        raise ValueError("uploaded payload hash mismatch")
    extract(archive, job)


def export(job: Path) -> None:
    # Receive runs from uploaded source text before the checkout exists. Load
    # the shared stdlib supervisor helpers only for export from that checkout.
    worker = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "remote/supervisor.py")
    )
    with worker["measurement_lease"](job):
        state = worker["status"](job)
        if state["status"] not in worker["TERMINAL"]:
            raise ValueError(
                "native artifact export requires a terminal job; use status for a live job"
            )
        # Hash and archive the same reconciled state reported in the manifest.
        worker["write_json"](job / "state.json", state)
        _export(job, state)


def _export(job: Path, state: dict) -> None:
    paths = []
    for name in (
        "results",
        "study",
        "data",
        "artifacts",
        "recovery",
        "state.json",
        "job.json",
        "run.log",
    ):
        path = job / name
        paths.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
    files = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"cannot export a symlink: {path}")
        if path.is_file():
            files[path.relative_to(job).as_posix()] = sha256(path)
    manifest = json.dumps(
        {"schema_version": 1, "state": state, "files": files}, sort_keys=True
    ).encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as stream:
        for name in files:
            stream.add(job / name, arcname=name, recursive=False)
        info = tarfile.TarInfo("export.json")
        info.size = len(manifest)
        stream.addfile(info, io.BytesIO(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("receive", "export"))
    parser.add_argument("job", type=Path)
    parser.add_argument("sha256", nargs="?")
    args = parser.parse_args()
    if args.action == "receive":
        if args.sha256 is None:
            parser.error("receive requires a payload hash")
        receive(args.job, args.sha256)
    else:
        if "__file__" not in globals():
            parser.error("export must run from the deployed checkout, not source text")
        export(args.job)


if __name__ == "__main__":
    main()
