"""Run a workflow script in an isolated process.

The runner supplies the resolved input and output mappings and the script's
declared dependencies. The entry point invokes the script and returns its status."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_main(script: str) -> Any:
    spec = importlib.util.spec_from_file_location("_causalab_isolated_step", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import step script {script!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise SystemExit(f"step script {script!r} has no callable 'main'")
    return main


def run() -> None:
    request = json.load(sys.stdin)
    main = _load_main(str(request["script"]))
    outputs = {slot: Path(str(path)) for slot, path in dict(request["outputs"]).items()}
    main(dict(request["inputs"]), outputs)


if __name__ == "__main__":
    run()
