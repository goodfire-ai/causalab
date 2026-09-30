"""Regenerate tests/workflow/fixtures/resume_identity_pins.json — the pinned
``_step.json`` identity fields ``--resume`` compares (workflow spec §8).

Run with ``uv run python tests/workflow/update_resume_identity_pins.py`` from
the repo root, then review the diff. A changed ``identity`` means a step's
canonical entry hashes differently (spec §7 — a loader migration, never a
routine edit); a changed ``digests`` value means a script's fixed output
bytes moved. The pin is the proof that a change to what the run tree
*records* leaves what ``--resume`` *compares* untouched — the sole reason
it exists is to be diffed by a PR that touches recorded fields.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.protocol._env import FIXTURES, build_env, write_rot_fixture  # noqa: E402
from tests.workflow.test_attempt_publish import (  # noqa: E402
    DECLARED,
    FIRST,
    SECOND,
    THIRD,
    _document,
)

from causalab.io.step_record import SIDECAR  # noqa: E402
from causalab.workflow.document import load_workflow  # noqa: E402
from causalab.workflow.runner import run_workflow  # noqa: E402

PINS_PATH = Path(__file__).parent / "fixtures" / "resume_identity_pins.json"

#: The ``_step.json`` fields the pin records per step: the three ``--resume``
#: compares (``identity``, ``files``, ``digests``) and the record's top-level
#: key set, so a field that leaves or enters the record is a visible diff.
PINNED_FIELDS: tuple[str, ...] = ("identity", "files", "digests")


def chain_dir(root: Path) -> Path:
    """The three-step chain of ``test_attempt_publish`` written under ``root``."""
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name, source in (("first", FIRST), ("second", SECOND), ("third", THIRD)):
        (scripts / f"{name}.py").write_text(source)
    return root


def pins_of(run_root: Path) -> dict[str, dict[str, Any]]:
    """``{step: {identity, files, digests, record_keys}}`` read off the run tree."""
    out: dict[str, dict[str, Any]] = {}
    for step in sorted(DECLARED):
        record = json.loads((run_root / step / SIDECAR).read_text())
        entry: dict[str, Any] = {field: record[field] for field in PINNED_FIELDS}
        entry["record_keys"] = sorted(record)
        out[step] = entry
    return out


def compute() -> dict[str, dict[str, Any]]:
    tmp = Path(tempfile.mkdtemp())
    artifacts = tmp / "artifacts"
    shutil.copytree(FIXTURES / "artifacts", artifacts, dirs_exist_ok=True)
    write_rot_fixture(artifacts)
    env = build_env(artifacts)
    wf_dir = chain_dir(tmp / "wf")
    loaded = load_workflow(_document(), env, workflow_dir=wf_dir)
    run_workflow(loaded, env, tmp / "runs", None)
    return pins_of(tmp / "runs" / "chain")


def main() -> None:
    pins = compute()
    PINS_PATH.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {PINS_PATH} ({len(pins)} steps)")


if __name__ == "__main__":
    main()
