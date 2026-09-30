"""Copy the page's Original values out of onboarding 09's committed run.

The Original of this package is not a paper's crop. It is the component scan
of [onboarding 09](../../../../onboarding_tutorial/09_components.md): one
component of one layer takes its counterfactual value at the answer slot, for
the attention output, the MLP output and the residual stream
(``block_output``) of all 28 layers, on the 64 ``different_symbol`` pairs of
``mcqa/pairs_n64_s0``, Qwen2.5-1.5B-Instruct in bf16. This script copies the
84 values that figure draws into
``artifacts/data/mcqa_components_dbm/component_iia_onboarding09_original.json``,
with the source path and the sha256 of the source bytes, so the package never
reads another demo's files at run time. ``figures.py`` draws the copy in the
style of the replication.

Usage (from ``demos/papers/``)::

    python workflows/scripts/mcqa_components_dbm/copy_original.py
    python workflows/scripts/mcqa_components_dbm/copy_original.py --check   # copy still equals the source?

``--check`` fails when the source file has changed since the copy, which
means the Original no longer matches onboarding 09 and the copy is due again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: ``demos/papers/``: this script sits in ``workflows/scripts/mcqa_components_dbm/``.
PAPERS = Path(__file__).resolve().parents[3]
REPO = PAPERS.parents[1]
#: The plotted values of onboarding 09's component figure, beside the image.
SOURCE = (
    REPO
    / "demos/onboarding_tutorial/artifacts/output/09_components/grid"
    / "09_components_component_iia.json"
)
OUT = (
    PAPERS
    / "artifacts/data/mcqa_components_dbm/component_iia_onboarding09_original.json"
)
COMPONENTS = ("attention_output", "mlp_output", "block_output")
LAYERS = tuple(range(28))


def copy(source: Path) -> dict:
    """The wrapped copy: source path, sha256 of its bytes and the drawn values."""
    raw = source.read_bytes()
    cells = {
        (str(row["sites.target.component"]), int(row["sites.target.layers"])): float(
            row["value"]
        )
        for row in json.loads(raw)
    }
    missing = [
        (component, layer)
        for component in COMPONENTS
        for layer in LAYERS
        if (component, layer) not in cells
    ]
    if missing:
        raise SystemExit(f"{source.name} has no value for {missing}")
    return {
        "source": str(source.relative_to(REPO)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "description": (
            "Onboarding 09: mean IIA over the 64 different_symbol pairs of "
            "mcqa/pairs_n64_s0 of swapping one component of one layer at the "
            "answer slot (position -1) of Qwen/Qwen2.5-1.5B-Instruct in bf16; a "
            "match is the counterfactual letter. Every layer of the 28 for the "
            "attention output, the MLP output and the residual stream "
            "(block_output)."
        ),
        "records": [
            {"component": component, "layer": layer, "iia": cells[component, layer]}
            for component in COMPONENTS
            for layer in LAYERS
        ],
    }


def render(payload: dict) -> str:
    return json.dumps(payload, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed copy differs from a fresh copy",
    )
    args = parser.parse_args(argv)
    fresh = render(copy(args.source))
    if args.check:
        if not args.out.is_file() or args.out.read_text() != fresh:
            print(
                f"{args.out}: differs from a fresh copy of {args.source}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out}: equals a fresh copy")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(fresh)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
