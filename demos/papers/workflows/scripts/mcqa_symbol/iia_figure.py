"""The figures of the ``mcqa_symbol`` page, from the workflow's run tree.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/mcqa_symbol/iia_figure.py    # reads artifacts/output/mcqa_symbol/, writes artifacts/figures/mcqa_symbol/

Every IIA is a mean over pairs of a 0/1 match against ``label_forms``,
aggregated through [`causalab.io.step_record.aggregate`][], the reduction the
workflow's ``select`` step ranks by, so the layer this script marks as chosen
is the layer that step emitted.

Inputs (under the run tree)
    ``clean/``         unpatched accuracy on base and counterfactual prompts,
                       and the rate at which the base prompt already gives the label
    ``apply/``         held-out IIA of the 28 DAS rotations
    ``apply_train/``   the same rotations on the train split
    ``full_patch/``    held-out IIA of swapping the whole block output, per layer
    ``best/``          the chosen layer
    ``dbm_apply/``     held-out IIA of the DBM-DAS fit at that layer
    ``rank/``          the rank the DBM-DAS fit learned
    ``control/``       held-out IIA of three random subspaces at k = 32
    ``control_dbm/``   the same at the learned rank

Outputs (in ``artifacts/figures/mcqa_symbol/``)
    ``iia_all.png``       DAS IIA over the 28 layers, held out and train, the
                          full-vector patch held out, and the k = 32 random
                          controls at the chosen layer
    ``iia_plotted.json``  every value drawn, and the DBM-DAS and control
                          values that the page's table quotes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from causalab.io.step_io import StepError, frame
from causalab.io.step_record import aggregate

__all__ = ["main", "cli"]

NAME = "mcqa_symbol"
#: ``demos/papers/``: this script sits in ``workflows/scripts/mcqa_symbol/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / NAME
FIGURES = PAPERS / "artifacts" / "figures" / NAME

MODEL = "Qwen2.5-1.5B-Instruct"
#: What a match means, for the axis label.
TARGET = "counterfactual letter"
#: IIA of a model that picks one of the base prompt's two letters at random.
#: The label here is a letter the base prompt does not show, so no guess among
#: the prompt's letters reaches it and the chance line is not drawn.
CHANCE: float | None = None
N_LAYERS = 28
LAYER_AXIS = "sites.target.layers"
SEED_AXIS = "featurizers.rot.seed"
#: The residual stream's purple, shared by the three MCQA packages
#: (``workflows/scripts/mcqa_components_dbm/figures.py``, ``COMPONENTS``).
#: DAS sits on ``block_output``, the residual stream.
PURPLE = "#6a3d9a"
#: The same purple mixed half with white, for the train line.
LIGHT_PURPLE = "#b49ecc"
GREY = "#737373"
INK = "#252525"


def mean_iia(path: Path) -> float:
    """The mean IIA of a table with one point."""
    table, axes = aggregate(frame(path), path, "value")
    if len(table) != 1:
        raise StepError(f"{path} holds {len(table)} points over {axes}, not one")
    return float(table["value"].iloc[0])


def by_layer(path: Path) -> dict[int, float]:
    """Mean IIA per layer of an apply table."""
    table, axes = aggregate(frame(path), path, "value")
    if axes != (LAYER_AXIS,):
        raise StepError(f"{path} is swept over {axes}, not {LAYER_AXIS!r} alone")
    return {int(row[LAYER_AXIS]): float(row["value"]) for _, row in table.iterrows()}


def by_seed(path: Path) -> dict[int, float]:
    """Mean IIA per random seed of a control table."""
    table, axes = aggregate(frame(path), path, "value")
    if axes != (SEED_AXIS,):
        raise StepError(f"{path} is swept over {axes}, not {SEED_AXIS!r} alone")
    return {int(row[SEED_AXIS]): float(row["value"]) for _, row in table.iterrows()}


def values(run: Path) -> dict[str, Any]:
    """Everything the figures draw, as one JSON-ready object."""
    test = by_layer(run / "apply" / "iia.json")
    train = by_layer(run / "apply_train" / "iia.json")
    full = by_layer(run / "full_patch" / "iia.json")
    if any(sorted(t) != list(range(N_LAYERS)) for t in (test, train, full)):
        raise StepError(
            f"expected layers 0..{N_LAYERS - 1} in apply, apply_train and full_patch"
        )
    chosen = json.loads((run / "best" / "values.json").read_text())["best_layer"]
    if test[chosen] != max(test.values()):
        raise StepError(f"layer {chosen} is not the held-out maximum")
    rank = json.loads((run / "rank" / "values.json").read_text())
    control = by_seed(run / "control" / "iia.json")
    control_dbm = by_seed(run / "control_dbm" / "iia.json")
    return {
        "model": MODEL,
        "chance": CHANCE,
        "clean": {
            "base_accuracy": mean_iia(run / "clean" / "clean_base.json"),
            "counterfactual_accuracy": mean_iia(run / "clean" / "clean_cf.json"),
            "unpatched_label_rate": mean_iia(run / "clean" / "unpatched_label.json"),
        },
        "das": [
            {"layer": layer, "test_iia": test[layer], "train_iia": train[layer]}
            for layer in range(N_LAYERS)
        ],
        "full_patch": [
            {"layer": layer, "test_iia": full[layer]} for layer in range(N_LAYERS)
        ],
        "chosen_layer": int(chosen),
        "dbm": {
            "learned_rank": int(rank["learned_rank"]),
            "theta": float(rank["theta"]),
            "width": int(rank["width"]),
            "test_iia": mean_iia(run / "dbm_apply" / "iia.json"),
        },
        "control": {
            "k": 32,
            "test_iia_by_seed": control,
            "test_iia_mean": sum(control.values()) / len(control),
        },
        "control_dbm": {
            "k": int(rank["learned_rank"]),
            "test_iia_by_seed": control_dbm,
            "test_iia_mean": sum(control_dbm.values()) / len(control_dbm),
        },
    }


def _frame(ax: Any, ylabel: str) -> None:
    """Shared axes: layers 0 to 27, IIA 0 to 1, a recessive dashed grid."""
    ax.set_xlim(-0.8, N_LAYERS - 0.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.set_xlabel("Residual-stream layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linestyle="--", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _chance(ax: Any) -> None:
    if CHANCE is not None:
        ax.axhline(CHANCE, color=GREY, linestyle=":", linewidth=1.0)
        ax.text(0.2, CHANCE + 0.02, f"chance {CHANCE:g}", color=GREY, fontsize=8)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    return plt


def draw_all(target: Path, drawn: Mapping[str, Any]) -> None:
    """Held-out DAS IIA per layer (solid) over train (dashed), the held-out
    full-vector patch, the chosen layer and the k = 32 random control there."""
    plt = _pyplot()
    rows = drawn["das"]
    layers = [r["layer"] for r in rows]
    chosen = drawn["chosen_layer"]
    figure, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.plot(
        layers,
        [r["train_iia"] for r in rows],
        color=LIGHT_PURPLE,
        linewidth=2,
        linestyle=(0, (4, 2)),
        label="DAS k = 32, train",
    )
    ax.plot(
        [r["layer"] for r in drawn["full_patch"]],
        [r["test_iia"] for r in drawn["full_patch"]],
        color=INK,
        linewidth=1.5,
        marker="s",
        markersize=3.5,
        label="full vector, held out",
    )
    ax.plot(
        layers,
        [r["test_iia"] for r in rows],
        color=PURPLE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="DAS k = 32, held out",
    )
    seeds = list(drawn["control"]["test_iia_by_seed"].values())
    ax.scatter(
        [chosen] * len(seeds),
        seeds,
        color=GREY,
        marker="x",
        s=36,
        zorder=3,
        label="random k = 32, held out",
    )
    ax.axvline(chosen, color="0.75", linewidth=0.8, zorder=0)
    ax.annotate(
        f"layer {chosen}",
        xy=(chosen, 0.4),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    _chance(ax)
    _frame(ax, f"IIA ({TARGET})")
    ax.set_title(f"Answer slot of {MODEL}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


def main(run: Path = RUN, out: Path = FIGURES) -> None:
    drawn = values(run)
    out.mkdir(parents=True, exist_ok=True)
    draw_all(out / "iia_all.png", drawn)
    (out / "iia_plotted.json").write_text(json.dumps(drawn, indent=1) + "\n")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=RUN, help="the workflow's run tree")
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    main(args.run, args.out)
    print(args.out / "iia_all.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
