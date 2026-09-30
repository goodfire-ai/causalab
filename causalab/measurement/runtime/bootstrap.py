"""Standalone worker entry: select the arm before importing any Causalab code."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    package = Path(config["package_root"]).resolve()
    sys.path.insert(0, str(package))
    # The controller stays current even when the tested revision predates it.
    # Its relative imports resolve here; its causalab imports select the arm.
    runtime = types.ModuleType("_measurement_runtime")
    runtime.__path__ = [str(Path(config["controller_root"]) / "causalab")]
    sys.modules[runtime.__name__] = runtime
    module = "capture.worker" if "capture" in config else "runtime.worker"
    worker = importlib.import_module(f"_measurement_runtime.measurement.{module}")
    worker.serve(config)


if __name__ == "__main__":
    main()
