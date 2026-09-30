"""Run with `uv run python -m causalab.sol INPUT.json --output CATALOG.json`."""

import argparse
import json
from pathlib import Path
from typing import Any

from causalab.sol.model import Hardware, Phase, Workload, catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.input.read_text())
    if spec["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    hardware = Hardware(**spec["hardware"])
    workloads = []
    for entry in spec["workloads"]:
        fields: dict[str, Any] = dict(entry)
        fields["phases"] = [Phase(**phase) for phase in entry["phases"]]
        workloads.append(Workload(**fields))
    result = catalog(workloads, hardware)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
