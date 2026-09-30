"""Referenced functions for the ``code`` section's tests (§2.8.1).

Kept tiny and kept here rather than pointing the tests at ``torch.relu``:
resolving a locator reads and AST-parses the *defining module*, and torch's
``__init__.py`` is a megabyte and a half of it.

⚠️ These functions' module bytes are a document identity wherever a test
declares them, so an edit here moves that document's digest. No pinned digest
references this file (the digest-moving tests write their own module into
``tmp_path``), which is what keeps that harmless.
"""

from __future__ import annotations

from typing import Any


def scale(f: Any, factor: float = 1.0) -> Any:
    """The plainest possible declared edit: one tensor, one scalar."""
    return f * factor


def corrupt(f: Any, factor: float = 1.0, *, row_roles: Any = None) -> Any:
    """A declared edit that is *told* which rows are which rather than
    assuming a batch shape (§2.8.1): the ROME corruption function's row
    convention, written down."""
    if row_roles is None:
        return f * factor
    lo, hi = row_roles["corrupted"]
    out = f.clone()
    out[lo:hi] = out[lo:hi] * factor
    return out
