"""Plot an activation manifold and paired steering scores from saved tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--steering", required=True, type=Path, action="append")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def read(run: Path, file: str) -> list[dict]:
        return json.loads((run / file).read_text(encoding="utf-8"))

    spline = read(args.fit, "manifold_path/spline.json")
    centroids = read(args.fit, "manifold_path/centroids.json")
    energy = [row for run in args.steering for row in read(run, "energy/energy.json")]
    pairs = list(dict.fromkeys(row["pair"] for row in energy))
    scores = {(row["pair"], row["method"]): row["energy_sum"] for row in energy}
    if len(scores) != len(energy):
        parser.error("steering run trees contain overlapping pairs")
    differences = [scores[pair, "linear"] - scores[pair, "manifold"] for pair in pairs]
    spline_color, chord_color = "#2c6e9c", "#b05e28"

    fig, (activation, effects) = plt.subplots(
        1, 2, figsize=(10, max(4, len(pairs) * 0.25)), layout="constrained"
    )
    closed = [*spline, spline[0]]
    activation.plot(
        [row["pc1"] for row in closed],
        [row["pc2"] for row in closed],
        color=spline_color,
    )
    for row in centroids:
        x, y = row["pc1"], row["pc2"]
        activation.scatter(x, y, color=spline_color, s=14)
        activation.annotate(
            row["day"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    activation.set(xlabel="PC 1", ylabel="PC 2", title="Activation manifold")
    activation.set_aspect("equal", adjustable="datalim")

    labels = [pair.replace("_", "–") for pair in pairs]
    effects.barh(
        labels,
        differences,
        color=[spline_color if d >= 0 else chord_color for d in differences],
    )
    effects.axvline(0, color="0.35", linewidth=0.7)
    effects.set(xlabel="Chord minus spline energy", title="Per-pair comparison")
    effects.invert_yaxis()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
