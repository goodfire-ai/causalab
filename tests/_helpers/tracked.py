"""The files a repository-wide prose check reads: the *tracked* ones.

Two tests walk the whole tree for prose — the vocabulary census
(`tests/protocol/test_vocabulary_census.py`) and the refusal-quote guard
(`tests/workflow/test_cli_refusals.py`) — and both name their carve-outs by
path from the repo root. The paper layout check
(`tests/demos/test_papers.py::test_layout`) counts directories the same way,
through `tracked_child_dirs`. A filesystem walk is the wrong instrument for that:
it enumerates whatever is *present*, and a checkout routinely holds a second
copy of the tree that is not part of it — a worktree under ``worktrees/``
(`.gitignore`), a setuptools ``build/lib/``, a ``dist/`` unpack. Every
root-anchored carve-out misses the nested copy at once, so the check is green
in CI and red on a developer's machine, with a message that misnames the
cause (``worktrees/x/docs/CODEBASE.md`` "is not in the historical
carve-out"; ``build/lib/causalab/analysis/harvest_difference.py`` "reworded a
frozen script").

``git ls-files`` returns the paths the carve-outs are written against, by
construction: tracked files, relative to the root, one copy each. A
gitignored copy is not tracked, an untracked stray is not tracked, and a
nested worktree is another repository.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: The directories the fallback walk prunes. Dotted directories hold
#: virtualenvs and tool state; the rest are the gitignored places a second
#: copy of the tree lands (`.gitignore`: ``/build/``, ``/dist/``,
#: ``worktrees/``). This list is the fallback's known hole — the next such
#: directory is a blind spot until it is added — which is why it is the
#: fallback and not the walk.
PRUNED_DIRECTORIES: frozenset[str] = frozenset(
    {"node_modules", "build", "dist", "worktrees"}
)


def tracked_files(root: Path, *patterns: str) -> list[Path]:
    """Tracked files under ``root`` matching any of ``patterns`` (``'*.md'``).

    Reads ``git ls-files``; patterns are git pathspecs, so ``*.md`` matches at
    every depth. Falls back to a filesystem walk **only when git cannot answer**
    — no ``git`` on ``PATH``, or ``root`` is not inside a repository (a source
    tarball). The fallback prunes `PRUNED_DIRECTORIES` and so has the
    nested-copy hole described in the module docstring; a checkout that is a
    git repository never takes it.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", *patterns],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _walked(root, *patterns)
    return sorted(root / entry.decode() for entry in listed.split(b"\0") if entry)


def tracked_child_dirs(parent: Path) -> set[str]:
    """The names of ``parent``'s child directories that hold a tracked file
    at any depth.

    A layout check lists these rather than every directory on disk. A
    directory that holds only ignored files, such as the ``__pycache__`` a
    renamed package leaves in an old checkout, is then not reported as a
    stray, and CI's clean checkout and a developer's tree agree. Reads
    `tracked_files`, so it takes the same fallback walk outside a
    repository.
    """
    return {
        path.relative_to(parent).parts[0]
        for path in tracked_files(parent, "*")
        if len(path.relative_to(parent).parts) > 1
    }


def _walked(root: Path, *patterns: str) -> list[Path]:
    """The fallback: every file under ``root`` whose name matches a pattern,
    with `PRUNED_DIRECTORIES` and dotted directories pruned from the
    descent rather than filtered from the result — a virtualenv holds more
    files than the tree does."""
    out: list[Path] = []
    for parent, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in PRUNED_DIRECTORIES
        ]
        out.extend(
            Path(parent) / name
            for name in filenames
            if any(Path(name).match(pattern) for pattern in patterns)
        )
    return sorted(out)
