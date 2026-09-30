"""Import a paper package's script by path.

The scripts under ``demos/papers/workflows/scripts/<name>/`` are files a
workflow step or a reader runs, not modules of an installed package, so a
test loads them the way the workflow runner does: from their path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
PAPERS = REPO / "demos" / "papers"
SCRIPTS = PAPERS / "workflows" / "scripts"


def load_script(package: str, stem: str) -> ModuleType:
    """``workflows/scripts/<package>/<stem>.py`` as a module, loaded once."""
    name = f"_papers_{package}_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / package / f"{stem}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
