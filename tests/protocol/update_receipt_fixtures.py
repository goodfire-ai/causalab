"""Regenerate ``tests/protocol/fixtures/receipts/`` — the run receipts
`tests.protocol.test_receipt_bytes` pins byte for byte.

Run with ``uv run python tests/protocol/update_receipt_fixtures.py`` from the
repo root, then review the diff. A receipt's bytes move when the canonical
form moves (spec §7 — a loader migration, never a routine edit) or when the
receipt records a new field; either is a deliberate change the PR body says
why for. The cases, the stub engine and the fixture environment are the
test's own, so a fixture is written here exactly as the test reproduces it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from causalab.protocol import RUN_RECORD_NAME, run_protocol  # noqa: E402
from causalab.protocol.pipeline import compile_protocol  # noqa: E402

from tests.protocol._env import (  # noqa: E402
    CORPUS_DIR,
    FIXTURES,
    build_env,
    fixture_tokenizer,
    write_pca_fixture,
    write_rot_fixture,
)
from tests.protocol.test_receipt_bytes import CASES, _Stub, _fixture  # noqa: E402


def main() -> None:
    # the test's session fixtures (tests/protocol/conftest.py): the artifact
    # root, and the fixture tokenizer standing in for the Hub
    import causalab.io.tokenizer as tokenizer_module

    real = tokenizer_module.load_tokenizer
    tokenizer_module.load_tokenizer = lambda key, revision="main": fixture_tokenizer(
        key, revision, loader=real
    )
    tmp = Path(tempfile.mkdtemp())
    shutil.copytree(FIXTURES / "artifacts", tmp, dirs_exist_ok=True)
    write_rot_fixture(tmp)
    write_pca_fixture(tmp)
    env = build_env(tmp)
    for name, points in CASES:
        loaded = compile_protocol(CORPUS_DIR / name, env=env)
        run_dir = Path(tempfile.mkdtemp())
        run_protocol(loaded, env, _Stub(), run_dir, points=points, record=True)
        target = _fixture(name, points)
        target.write_bytes((run_dir / RUN_RECORD_NAME).read_bytes())
        print(f"wrote {target.relative_to(Path(__file__).resolve().parents[2])}")


if __name__ == "__main__":
    main()
