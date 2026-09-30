"""Calling a step script the way the runner does.

``main(inputs, outputs)`` with ``outputs`` as real paths, so a test exercises
the same contract the runner uses rather than a stand-in. That is what makes
these tests oracles for shipped behaviour instead of for a wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from causalab.io.tables import write_table

__all__ = ["put_sidecar", "put_table", "run_step"]


def run_step(
    module: Any, inputs: Mapping[str, Any], outputs: Mapping[str, Path]
) -> None:
    module.main(dict(inputs), {slot: Path(p) for slot, p in outputs.items()})


def put_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_table(path, rows)
    return path


def put_sidecar(step_dir: Path, axes: list[str]) -> None:
    """The per-step record the runner publishes — how a script learns the
    producing document's sweep axes (workflow spec §6)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "_step.json").write_text(json.dumps({"axes": axes}) + "\n")
