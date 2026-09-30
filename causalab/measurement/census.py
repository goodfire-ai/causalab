"""The measurement census: what a loaded workflow touches, by digest.

A paired study holds every arm to one closure — the documents, scripts,
tables, code modules, and files a workflow load resolved. The census is
computed from the loaded workflow here and travels in the study's own
records (the worker configuration, the frozen remote package), never as a
section of the workflow document: the workflow layer stopped reading a
``pins`` section and holds ``--resume`` to its path inputs by record instead.

Torch-free. Every ``causalab`` import is deferred into the function that
needs it: the measurement runtime is loaded by the bootstrap as
``_measurement_runtime`` against whichever arm is selected, so nothing here
may touch the arm's package at import time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from causalab.workflow.document import LoadedWorkflow

__all__ = [
    "PINS_KEY",
    "PIN_CATEGORIES",
    "CensusError",
    "Pins",
    "check_pins",
    "collect_pins",
    "parse_pins",
    "stamp_pins",
    "strip_pins",
]

#: The key a study record or a frozen remote workflow carries the census under.
PINS_KEY: str = "pins"

#: The closed category vocabulary, in the order the census is written.
PIN_CATEGORIES: tuple[str, ...] = ("documents", "scripts", "datasets", "code", "files")

Pins = Mapping[str, Mapping[str, str]]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CensusError(ValueError):
    """The authored census disagrees with the closure a load resolved: a
    stale, missing, or surplus pin. ``path`` names the entry as
    ``pins.<category>.<key>``. A measurement refusal, not a workflow
    checklist rule: the workflow document carries no census."""

    def __init__(self, message: str, *, path: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def parse_pins(raw: Any, path: str) -> dict[str, dict[str, str]]:
    """The authored census, checked for shape: an object whose keys are
    [`PIN_CATEGORIES`][] and whose values map a resource name to one
    sha256 hex digest. Nothing here reads a file."""
    from causalab.workflow.document import WorkflowError

    if not isinstance(raw, Mapping):
        raise WorkflowError(1, "'pins' is an object of categories", path=path)
    out: dict[str, dict[str, str]] = {}
    for category, entries in raw.items():
        if category not in PIN_CATEGORIES:
            raise WorkflowError(
                1,
                f"unknown pins category {category!r}; the categories are "
                f"{list(PIN_CATEGORIES)}",
                path=f"{path}.{category}",
            )
        if not isinstance(entries, Mapping):
            raise WorkflowError(
                1, f"'pins.{category}' maps a resource to its digest", path=path
            )
        pinned: dict[str, str] = {}
        for key, digest in entries.items():
            if not isinstance(digest, str) or not _HEX64.match(digest):
                raise WorkflowError(
                    1,
                    f"pins.{category}.{key}: a pin is one sha256 hex digest, "
                    f"got {digest!r}",
                    path=f"{path}.{category}",
                )
            pinned[str(key)] = digest
        out[str(category)] = pinned
    return out


def strip_pins(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]] | None]:
    """Split a workflow tree carrying a census under [`PINS_KEY`][] into the
    document the workflow loader reads and the parsed census, or ``None``."""
    document = {key: value for key, value in raw.items() if key != PINS_KEY}
    pins = raw.get(PINS_KEY)
    return document, None if pins is None else parse_pins(pins, PINS_KEY)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(node: Any, code: dict[str, str], files: dict[str, str], step_names) -> None:
    """One pass over a canonical intervention document: every ``code``
    reference and every resolved artifact that is not a run-tree product."""
    from causalab.workflow.document import producer_of

    if isinstance(node, Mapping):
        module = node.get("source_module")
        if isinstance(module, str) and isinstance(node.get("source_sha256"), str):
            code[module] = node["source_sha256"]
            closure = node.get("closure")
            if isinstance(closure, Mapping):
                for member, digest in closure.items():
                    code[f"{module}#{member}"] = str(digest)
            inputs = node.get("data_inputs")
            digests = node.get("data_input_digests")
            if isinstance(inputs, Mapping) and isinstance(digests, Mapping):
                for name, target in inputs.items():
                    if name in digests and producer_of(str(target), step_names) is None:
                        files[str(target)] = str(digests[name])
        file_path = node.get("file_path")
        digest = node.get("content_digest")
        if isinstance(file_path, str) and isinstance(digest, str):
            if producer_of(file_path, step_names) is None:
                files[file_path] = digest
        for value in node.values():
            _walk(value, code, files, step_names)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk(value, code, files, step_names)


def collect_pins(loaded: LoadedWorkflow, datasets: Any) -> dict[str, dict[str, str]]:
    """The census of ``loaded``: what the load touched, by category, keys
    sorted, empty categories omitted. ``datasets`` is the run's dataset
    resolver; a table's pin is the digest of the rows the ref selects, keyed
    by the ref as the document names it (``<table>`` or ``<table>#<split>``) —
    the same quantity the canonical form stamps, and the one the resolver
    answers for a split table, which refuses a bare ref (V22)."""
    from causalab.workflow.document import (
        BehavioralStep,
        ProtocolStep,
        Reference,
        ScriptStep,
        WorkflowStep,
    )

    document = loaded.document
    step_names = tuple(loaded.document.steps)
    documents: dict[str, str] = {}
    scripts: dict[str, str] = {}
    tables: dict[str, str] = {}
    code: dict[str, str] = {}
    files: dict[str, str] = {}
    for name, step in document.steps.items():
        if isinstance(step, (ProtocolStep, BehavioralStep)):
            documents[step.document] = _file_sha256(loaded.workflow_dir / step.document)
            inner = loaded.inner.get(name)
            compiled = getattr(inner, "compiled", inner)
            if compiled is not None:
                for ref in compiled.data:
                    if str(ref) not in tables:
                        tables[str(ref)] = datasets.digest(str(ref))
                _walk(compiled.canonical, code, files, step_names)
        elif isinstance(step, WorkflowStep):
            documents[step.document] = loaded.nested[name].digest
        elif isinstance(step, ScriptStep):
            scripts[step.script] = step.script_sha256
            for member, digest in step.closure.items():
                scripts[f"{step.script}#{member}"] = digest
            for value in step.inputs.values():
                if not isinstance(value, Reference) or value.path is None:
                    continue
                target = str(value.path)
                if target.startswith("/"):
                    continue  # rule 4: not existence-checked at load, not pinned
                files[target] = _file_sha256(loaded.workflow_dir / target)
    census = {
        "documents": documents,
        "scripts": scripts,
        "datasets": tables,
        "code": code,
        "files": files,
    }
    return {
        category: dict(sorted(entries.items()))
        for category in PIN_CATEGORIES
        if (entries := census[category])
    }


def check_pins(authored: Pins, actual: Pins) -> None:
    """Hold the authored census to the observed one, exactly: every pinned
    resource is touched with the pinned digest, and every touched resource
    is pinned. The first disagreement, in category then key order, is the
    refusal, naming ``pins.<category>.<key>``."""
    for category in PIN_CATEGORIES:
        want = authored.get(category, {})
        got = actual.get(category, {})
        for key in sorted(set(want) | set(got)):
            where = f"pins.{category}.{key}"
            if key not in want:
                raise CensusError(
                    f"the workflow touches {key!r} but does not pin it "
                    f"(its digest is {got[key][:12]}…). The census describes "
                    "another closure than the one this document loads",
                    path=where,
                )
            if key not in got:
                raise CensusError(
                    f"pinned, but nothing in the workflow touches {key!r} any more",
                    path=where,
                )
            if want[key] != got[key]:
                raise CensusError(
                    f"{key!r} moved — pinned {want[key][:12]}…, but the "
                    f"bytes the load resolved digest to {got[key][:12]}…",
                    path=where,
                )


def _indent_of(text: str) -> int:
    match = re.search(r"^( +)\S", text, re.M)
    return len(match.group(1)) if match else 2


def stamp_pins(path: Path, pins: Pins) -> Path:
    """Write ``pins`` into the study copy of a workflow at ``path`` as its
    last key, replacing any census already there, and return ``path``. The
    stamped file is a measurement record: the workflow loader reads it only
    after [`strip_pins`][]."""
    from causalab.io.sources import load_text

    text = path.read_text()
    raw: dict[str, Any] = dict(load_text(path))
    raw.pop(PINS_KEY, None)
    raw[PINS_KEY] = {
        category: dict(entries) for category, entries in pins.items() if entries
    }
    if path.suffix in (".yaml", ".yml"):
        import yaml

        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        return path
    body = json.dumps(raw, indent=_indent_of(text), ensure_ascii=text.isascii())
    path.write_text(body + ("\n" if text.endswith("\n") or not text else ""))
    return path
