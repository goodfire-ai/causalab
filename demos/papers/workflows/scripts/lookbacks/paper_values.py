"""Collect the values the paper draws in Figures 4b, 5b, 6b and 13.

Prakash et al. 2025 (arXiv:2505.14685) draw each full-residual curve from one
result file per sampled layer, which the authors ship in ``Nix07/mind``
(https://github.com/Nix07/mind, ``results/causalToM_novis/Meta-Llama-3-70B-Instruct/``).
Each file holds ``full_rank.accuracy``, the share of the 80 validation pairs
on which the full-residual intervention gives the causal model's answer, so
every value is a multiple of 1/80. The paper samples a subset of layers and
draws straight lines between them.

This script reads a clone of that repository and writes
``artifacts/data/lookbacks/lookbacks_prakash2025_values.json``: the source
URL at the commit read, and one record per curve and layer with the value,
its count over the 80 pairs, the file it comes from and the sha256 of that
file's bytes. The figure script puts each value beside ours in
``lookbacks_plotted.json``. Usage, from ``demos/papers/``, with the clone
outside this repository (a clone inside ``demos/papers/`` is a folder the
package layout refuses)::

    export MIND=~/mind
    git clone https://github.com/Nix07/mind "$MIND" && git -C "$MIND" checkout 0579347e3cf963d13d55edf041c7b595b0dcd88b
    python workflows/scripts/lookbacks/paper_values.py --mind "$MIND"            # writes the values file
    python workflows/scripts/lookbacks/paper_values.py --mind "$MIND" --check    # exit 1 if it differs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["CURVES", "PAIRS", "collect", "encode", "main"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/lookbacks/``.
PAPERS = Path(__file__).resolve().parents[3]
VALUES = (
    PAPERS / "artifacts" / "data" / "lookbacks" / "lookbacks_prakash2025_values.json"
)
REPOSITORY = "https://github.com/Nix07/mind"
RESULTS = "results/causalToM_novis/Meta-Llama-3-70B-Instruct"
#: The validation pairs every value is over (``prepare_dataset``'s ``valid_size``).
PAIRS = 80
#: (figure, curve) of ``lookbacks_plotted.json`` -> the authors' experiment
#: folder. ``source_1`` is the Figure 6b run and ``source_2`` the same run
#: without the freeze, Figure 13. The 6b curve ``late`` freezes the drink
#: tokens in the order the authors' script applies its writes, so it is the
#: curve ``source_1`` measures. The intended freeze (``frozen``) has no
#: counterpart in the paper.
CURVES: dict[tuple[str, str], str] = {
    ("4b", "pointer"): "answer_lookback/pointer",
    ("4b", "payload"): "answer_lookback/payload",
    ("5b", "binding"): "binding_lookback/address_and_payload",
    ("6b", "late"): "binding_lookback/source_1",
    ("13", "unfrozen"): "binding_lookback/source_2",
}


def collect(mind: Path) -> dict[str, Any]:
    """The values file for a clone of ``Nix07/mind``: its commit, and one
    record per curve and sampled layer, in the order of `CURVES` and layer."""
    commit = subprocess.run(
        ["git", "-C", str(mind), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    records = []
    for (figure, curve), experiment in CURVES.items():
        folder = mind / RESULTS / experiment
        files = sorted(folder.glob("*.json"), key=lambda path: int(path.stem))
        if not files:
            raise FileNotFoundError(f"{folder} holds no layer files")
        for path in files:
            raw = path.read_bytes()
            accuracy = json.loads(raw)["full_rank"]["accuracy"]
            count = round(accuracy * PAIRS)
            if abs(count - accuracy * PAIRS) > 1e-6:
                raise ValueError(f"{path}: {accuracy} is not a count over {PAIRS}")
            records.append(
                {
                    "figure": figure,
                    "curve": curve,
                    "layer": int(path.stem),
                    "accuracy": accuracy,
                    "count": count,
                    "file": f"{RESULTS}/{experiment}/{path.name}",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return {
        "source": f"{REPOSITORY}/tree/{commit}/{RESULTS}",
        "commit": commit,
        "pairs": PAIRS,
        "records": records,
    }


def encode(values: dict[str, Any]) -> bytes:
    """The committed bytes: the header keys, then one record per line, so
    that a diff names the value that moved."""
    head = {key: value for key, value in values.items() if key != "records"}
    lines = ["{"]
    lines += [
        f"  {json.dumps(key)}: {json.dumps(value)}," for key, value in head.items()
    ]
    lines.append('  "records": [')
    records = [json.dumps(record) for record in values["records"]]
    lines += [f"    {record}," for record in records[:-1]] + [f"    {records[-1]}"]
    lines += ["  ]", "}"]
    return ("\n".join(lines) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--mind", type=Path, required=True, help="a clone of Nix07/mind"
    )
    parser.add_argument("--out", type=Path, default=VALUES, help="the values file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild in memory and exit 1 if the committed bytes differ",
    )
    args = parser.parse_args(argv)
    fresh = encode(collect(args.mind))
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != fresh:
            print(f"{args.out}: bytes differ from a fresh build", file=sys.stderr)
            return 1
        print(f"{args.out}: reproduces")
        return 0
    args.out.write_bytes(fresh)
    print(f"{args.out}: {len(json.loads(fresh)['records'])} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
