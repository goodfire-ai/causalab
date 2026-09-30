"""Export evaluated DBM masks and metrics from an apply-run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causalab.analysis.export_dbm import export
from causalab.protocol.rules.errors import ProtocolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--register-from-hf",
        action="store_true",
        help="resolve unregistered model keys from their HF configs",
    )
    args = parser.parse_args(argv)
    try:
        result = export(args.manifest.resolve(), register_from_hf=args.register_from_hf)
        serialized = json.dumps(result, separators=(",", ":"), allow_nan=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n")
    except (ValueError, KeyError, OSError, ProtocolError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(result['experiments'])} DBM experiments to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
