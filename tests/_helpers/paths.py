"""Where the shipped method documents live.

``demos/methods/`` is the method library — one intervention specification per
method under ``protocols/`` and the workflows that chain them under
``workflows/``. It moved there from ``causalab/configs/`` (the package ships
no documents); every test that reads a shipped document takes the path from
here so the next move is one edit.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO / "demos/methods"
PROTOCOLS_DIR = METHODS_DIR / "protocols"
WORKFLOWS_DIR = METHODS_DIR / "workflows"
