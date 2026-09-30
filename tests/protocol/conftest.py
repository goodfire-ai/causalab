"""Fixtures for the protocol-layer tests: the resolution environment and
the golden corpus, resolved against the committed fixture tables."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.protocol._env import (
    CORPUS_DIR,
    FIXTURES,
    build_env,
    write_pca_fixture,
    write_rot_fixture,
)

CORPUS_FILES = sorted(p.name for p in CORPUS_DIR.glob("*_im.json"))


@pytest.fixture(scope="session")
def artifacts_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("protocol-artifacts")
    shutil.copytree(FIXTURES / "artifacts", root, dirs_exist_ok=True)
    write_rot_fixture(root)
    write_pca_fixture(root)
    return root


@pytest.fixture(scope="session")
def env(artifacts_root: Path):
    return build_env(artifacts_root)


@pytest.fixture(autouse=True)
def _fixture_tokenizer_for_every_door(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run door resolves positions with the model's tokenizer
    (``pipeline.resolve_positions``). The protocol tests name real models
    (``gpt2``, the corpus's ``Qwen/Qwen3-8B``) and run stub engines that
    load nothing, so the default loader would reach the Hub for a tokenizer —
    through the API doors and the CLI's ``file_env`` alike. The fixture
    environment's tokenizer service stands in: a key that loads gets its own
    tokenizer (the corpus model's is ungated and ~10 MB), a key the Hub has
    no repo for the tiny GPT-2's (``tests/protocol/_env.py``)."""
    import causalab.io.tokenizer as tokenizer_module

    from tests.protocol._env import fixture_tokenizer

    real = tokenizer_module.load_tokenizer
    monkeypatch.setattr(
        tokenizer_module,
        "load_tokenizer",
        lambda key, revision="main": fixture_tokenizer(key, revision, loader=real),
    )
