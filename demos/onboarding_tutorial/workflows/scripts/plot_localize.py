"""Plot the population scan and summarize variation in answer-slot onset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POSITIONS = {
    -11: "Question end",
    -10: "Symbol 0",
    -9: "Period after symbol 0",
    -8: "Choice 0",
    -6: "Symbol 1",
    -5: "Period after symbol 1",
    -4: "Choice 1",
    -1: "Answer slot (:)",
}


def main(inputs, outputs):
    pairs = json.loads(Path(inputs["pairs"]).read_text())
    rows = json.loads(Path(inputs["table"]).read_text())
    layers = sorted({row["sites.target.layers"] for row in rows})
    positions = list(POSITIONS)
    values = np.full((len(pairs), len(positions), len(layers)), np.nan)
    for row in rows:
        if not row["eligible"]:
            raise ValueError("The tutorial requires every pair to be eligible.")
        pair = int(row["example_id"])
        position = positions.index(json.loads(row["positions.tap"])["index"])
        layer = layers.index(row["sites.target.layers"])
        if not np.isnan(values[pair, position, layer]):
            raise ValueError("Duplicate pair, position, and layer.")
        values[pair, position, layer] = float(row["value"])
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a complete grid of binary match results.")

    slots = np.array([int(pair["answer_position"]) for pair in pairs])
    correct = np.stack(
        [
            values[i, positions.index(-10 if slot == 0 else -6)]
            for i, slot in enumerate(slots)
        ]
    )
    answer = values[:, positions.index(-1)]
    population = values.mean(axis=0)
    first = [next((layers[j] for j, v in enumerate(row) if v), None) for row in answer]
    successful = [layer for layer in first if layer is not None]
    ticks = [j for j, layer in enumerate(layers) if layer % 3 == 0]

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    im = ax.imshow(population, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(positions)), POSITIONS.values())
    ax.set_xticks(ticks, [layers[j] for j in ticks])
    ax.set(
        xlabel="Residual-stream layer",
        ylabel="Patched token position",
        title="Interchange accuracy across 64 pairs",
    )
    ax.axvline(layers.index(22) - 0.5, color="white", linestyle="--", linewidth=1)
    fig.colorbar(
        im, ax=ax, label="IIA · fraction of pairs", ticks=[0, 0.25, 0.5, 0.75, 1]
    )
    fig.savefig(outputs["figure"], dpi=160)
    plt.close(fig)

    def clean_values(key):
        table = json.loads(Path(inputs[key]).read_text())
        result = {
            int(row["example_id"]): float(row["value"])
            for row in table
            if row["eligible"]
        }
        if len(table) != len(pairs) or set(result) != set(range(len(pairs))):
            raise ValueError(f"{key}: expected one eligible result per pair")
        return [result[i] for i in range(len(pairs))]

    clean = {
        key: clean_values(key) for key in ("clean_base", "clean_cf", "unpatched_cf")
    }
    summary = {
        "pairs": len(pairs),
        "clean_accuracy": {
            key: {"correct": int(sum(v)), "fraction": float(np.mean(v))}
            for key, v in clean.items()
        },
        "first_answer_success": dict(
            Counter("never" if v is None else str(v) for v in first)
        ),
        "first_answer_success_variance": {
            "value": float(np.var(successful, ddof=0)) if successful else None,
            "unit": "layer_squared",
            "ddof": 0,
            "included_pairs": len(successful),
            "excluded_pairs": len(first) - len(successful),
            "exclusion": "No successful answer-slot patch in the scanned layers",
        },
        "correct_symbol_success_at_layer_0": {
            str(slot): {
                "successes": int(correct[slots == slot, 0].sum()),
                "pairs": int((slots == slot).sum()),
            }
            for slot in (0, 1)
        },
        "layer_profiles": [
            {
                "layer": layer,
                "correct_symbol_successes": int(correct[:, j].sum()),
                "answer_successes": int(answer[:, j].sum()),
                "symbol_0_group_successes": int(correct[slots == 0, j].sum()),
                "symbol_1_group_successes": int(correct[slots == 1, j].sum()),
            }
            for j, layer in enumerate(layers)
        ],
    }
    plotted = {
        "population": [
            {
                "layer": layer,
                "position": position,
                "role": POSITIONS[position],
                "iia": float(population[p, j]),
            }
            for p, position in enumerate(positions)
            for j, layer in enumerate(layers)
        ],
    }
    for key, data in [("plotted", plotted), ("summary", summary)]:
        Path(outputs[key]).write_text(json.dumps(data, indent=2) + "\n")
