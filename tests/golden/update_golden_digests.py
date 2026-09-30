"""Regenerate tests/golden/golden_digests.json — the paper-golden document pins.

Run with ``uv run python tests/golden/update_golden_digests.py`` from the
repo root, then review the diff: a changed digest means either a document
edit or a fixture-dataset edit (dataset content digests are part of the
canonical form). Both are deliberate, reviewed changes — never a silent
re-pin. The pinned *values* live in paper_goldens.json and never come from
running this stack; this file pins only document identity.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# repo root (NOT tests/ — putting tests/ itself on sys.path would shadow the
# top-level `tasks` package with tests/tasks and break task resolution)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.golden._env import GOLDEN_PROTOCOLS, build_env  # noqa: E402
from tests.protocol._env import steps_of

from causalab.protocol.pipeline import compile_protocol  # noqa: E402


def main() -> None:
    env = build_env(Path(tempfile.mkdtemp()))
    pins: dict[str, dict[str, object]] = {}
    for path in sorted(GOLDEN_PROTOCOLS.glob("*_im.json")):
        loaded = compile_protocol(path, env=env)
        pins[path.name] = {
            "document": loaded.digests.document,
            "points": list(steps_of(loaded, env).digests),
        }
    out = Path(__file__).parent / "golden_digests.json"
    out.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {out} ({len(pins)} documents)")


if __name__ == "__main__":
    main()
