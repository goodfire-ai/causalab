"""Regenerate the shipped-document pins: ``tests/protocol/shipped_digests.json``
and the canonical files under ``tests/protocol/canonical/``.

Run with ``uv run python tests/protocol/update_shipped_digests.py`` from the
repo root, then review the diff: a changed canonical file or digest means the
canonical form changed, which spec §7 treats as a loader migration (bump
``version``, ship a migration), never a silent re-pin. The covered set, the
deferred presets and the exclusions are the tables in
``test_shipped_digests.py``; a document is loaded here exactly as the tests
load it (``load_shipped``), against the same fixture environment the corpus
pins are made from (``update_corpus_digests.py``) plus the PCA basis fixture.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.protocol._env import (  # noqa: E402
    FIXTURES,
    build_env,
    write_pca_fixture,
    write_rot_fixture,
)
from tests.protocol.test_shipped_digests import (  # noqa: E402
    CANONICAL_DIR,
    COVERED,
    DEFERRED,
    PINS_PATH,
    load_shipped,
    pin_of,
    shipped_env,
)

from causalab.protocol.schema.explicit import canonical_bytes  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES / "artifacts", tmp, dirs_exist_ok=True)
    write_rot_fixture(tmp)
    write_pca_fixture(tmp)
    env = build_env(tmp)
    if CANONICAL_DIR.exists():
        shutil.rmtree(CANONICAL_DIR)  # a renamed document leaves no stale file
    CANONICAL_DIR.mkdir()
    pins: dict[str, object] = {}
    for name in COVERED:
        loaded = load_shipped(name, env)
        pins[name] = pin_of(loaded, shipped_env(name, env), deferred=name in DEFERRED)
        (CANONICAL_DIR / name).write_bytes(canonical_bytes(loaded.canonical))
    PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PINS_PATH} and {CANONICAL_DIR}/ ({len(pins)} documents)")


if __name__ == "__main__":
    main()
