"""Check authored source constraints against arm archives without importing them.

This is a source-byte preflight, not protocol compilation. Workers still check
full pin censuses against their resolved datasets, artifacts and code references.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tarfile
from typing import Any, Literal, Mapping

from causalab.measurement.census import strip_pins
from causalab.workflow.document import ScriptStep, parse_workflow


@dataclass
class SourcePinError(ValueError):
    """A source integrity constraint cannot hold for a packaged study arm."""

    arm: str
    category: Literal["code", "scripts"]
    key: str
    reason: str
    remedy: str

    def __str__(self) -> str:
        return (
            f"arm {self.arm!r}, pins.{self.category}.{self.key}: {self.reason}. "
            f"{self.remedy}"
        )


def _source_digest(
    archive: tarfile.TarFile,
    module: str,
    *,
    arm: str,
    category: Literal["code", "scripts"],
) -> str:
    parts = module.split(".")
    if parts[0] != "causalab" or not all(part.isidentifier() for part in parts):
        raise SourcePinError(
            arm,
            category,
            module,
            "cannot preflight external modules or sibling closure pins from an "
            "arm archive; only direct causalab module pins are supported",
            "Use local execution for this pin contract, or package the dependency "
            "as a directly pinned causalab module before remote deployment",
        )
    path = "/".join(parts)
    # The resolver prefers a package over a same-named module, as Python does.
    for candidate in (f"{path}/__init__.py", f"{path}.py"):
        try:
            member = archive.getmember(candidate)
        except KeyError:
            continue
        if not member.isfile():
            raise SourcePinError(
                arm,
                category,
                module,
                "source is not a regular file",
                "Package regular Python source in the selected revision before deployment",
            )
        stream = archive.extractfile(member)
        assert stream is not None  # a regular archive member always has bytes
        with stream:
            return hashlib.sha256(stream.read()).hexdigest()
    raise SourcePinError(
        arm,
        category,
        module,
        "module is absent from the arm archive",
        "Correct the module name and selected revision, then re-stamp the intended "
        "workflow with `causalab pin <workflow>`",
    )


def check_source_pins(study: Mapping[str, Any], archive: Path, *, arm: str) -> None:
    """Verify source bytes and the script census for one committed arm.

    All pinned code modules must match, but determining whether a protocol
    touches missing or surplus code pins requires the worker's full compilation.
    External modules and declared sibling closures cannot be resolved from the
    packaged causalab sources, so refuse them rather than trust operator bytes.
    """
    raw, pins = strip_pins(study)
    document = parse_workflow(raw)
    if pins is None:
        return
    with tarfile.open(archive, "r") as source:
        for category in ("code", "scripts"):
            for module, expected in pins.get(category, {}).items():
                actual = _source_digest(source, module, arm=arm, category=category)
                if actual != expected:
                    raise SourcePinError(
                        arm,
                        category,
                        module,
                        f"source bytes differ: pinned {expected}, archived {actual}",
                        "Use revisions satisfying the authored source pins, or "
                        "deliberately revise and re-stamp the intended pin contract "
                        "with `causalab pin <workflow>`",
                    )
        scripts = {
            step.script
            for step in document.steps.values()
            if isinstance(step, ScriptStep)
        }
        pinned = set(pins.get("scripts", {}))
        for module in sorted(scripts ^ pinned):
            reason = (
                "script has no authored pin"
                if module in scripts
                else "pinned script is unused"
            )
            raise SourcePinError(
                arm,
                "scripts",
                module,
                reason,
                "Re-stamp the intended workflow with `causalab pin <workflow>`",
            )


def check_source_presence(study: Mapping[str, Any], archive: Path, *, arm: str) -> None:
    """Check required candidate modules without applying baseline source hashes.

    Baseline preflight holds the authored source contract and script census.
    Other arms may change those source bytes, but must still package regular
    source for each explicitly pinned module before any remote work begins.
    Workers subsequently resolve and freeze their complete per-arm census.
    """
    raw, pins = strip_pins(study)
    parse_workflow(raw)  # the shape check; the census is what this reads
    if pins is None:
        return
    with tarfile.open(archive, "r") as source:
        for category in ("code", "scripts"):
            for module in pins.get(category, {}):
                _source_digest(source, module, arm=arm, category=category)
