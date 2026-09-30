"""Regenerate tests/protocol/corpus_digests.json — the golden-corpus pins.

Run with ``uv run python tests/protocol/update_corpus_digests.py`` from the
repo root, then review the diff: a changed digest means the canonical form
changed, which spec §7 treats as a loader migration (bump ``version``, ship
a migration), never a silent re-pin.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol._env import CORPUS_DIR, FIXTURES, build_env, write_rot_fixture  # noqa: E402

from tests.protocol._env import steps_of  # noqa: E402
from causalab.protocol.pipeline import compile_protocol  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES / "artifacts", tmp, dirs_exist_ok=True)
    write_rot_fixture(tmp)
    env = build_env(tmp)
    pins: dict[str, dict[str, object]] = {}
    for path in sorted(CORPUS_DIR.glob("*_im.json")):
        loaded = compile_protocol(path, env=env)
        pins[path.name] = {
            "document": loaded.digests.document,
            "points": list(steps_of(loaded, env).digests),
        }
    out = Path(__file__).parent / "corpus_digests.json"
    out.write_text(json.dumps(pins, indent=2) + "\n")
    print(f"wrote {out} ({len(pins)} documents)")


if __name__ == "__main__":
    main()
