"""The protocol tests' resolution environment, for the shared layer's
torch-free suites (the engine-side sweep, the per-step rules): the committed
fixture tables and a fresh artifacts root with the generated bundles."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.protocol._env import (
    FIXTURES,
    build_env,
    write_pca_fixture,
    write_rot_fixture,
)


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shared-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    write_pca_fixture(root)
    return root


@pytest.fixture(scope="session")
def env(artifacts_root: Path):
    return build_env(artifacts_root)
