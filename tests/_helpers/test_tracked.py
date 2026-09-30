"""``tests/_helpers/tracked.py``: a layout check counts the directories that
hold a tracked file, so a directory that holds only ignored files (the
``__pycache__`` an old checkout leaves behind) is not a stray, and a tracked
one is."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._helpers.tracked import tracked_child_dirs

pytestmark = pytest.mark.unit


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_only_directories_with_a_tracked_file_count(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _write(tmp_path / ".gitignore", "__pycache__/\n")
    _write(tmp_path / "papers" / "kept" / "page.md")
    _write(tmp_path / "papers" / "stale" / "__pycache__" / "old.cpython-312.pyc")
    _write(tmp_path / "papers" / "index.md")
    _git(tmp_path, "add", ".gitignore", "papers/kept/page.md", "papers/index.md")
    assert tracked_child_dirs(tmp_path / "papers") == {"kept"}
    # a stray the author commits is seen, which is what the layout check is for
    _write(tmp_path / "papers" / "stray" / "notes.md")
    _git(tmp_path, "add", "papers/stray/notes.md")
    assert tracked_child_dirs(tmp_path / "papers") == {"kept", "stray"}
