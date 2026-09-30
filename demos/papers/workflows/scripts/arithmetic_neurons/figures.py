"""Figure 8a for the DBM mask's neurons, and the mask against its references.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/arithmetic_neurons/figures.py            # reads artifacts/output/arithmetic_neurons/, writes artifacts/figures/arithmetic_neurons/

It reads the run tree of ``workflows/arithmetic_neurons.json`` and the
package's own tables and neuron list under ``artifacts/data/arithmetic_neurons/``,
and writes under ``artifacts/figures/arithmetic_neurons/``:

* ``fig8a_replication.png``: the mean full activation ``silu(gate) * up`` of
  each neuron the DBM mask keeps, ordered by the gate's rank, at the last
  token of every addition prompt ``a+b=``, averaged over the prompts with the
  same output sum. A neuron also in the paper's 28 is starred. Colours are
  clipped to [-2, 2], as in the paper.
* ``masks_sweep.png``: the val IIA of each L1 weight's mask over the number
  of neurons it keeps, with the point of the one-standard-error rule ringed,
  over the neurons that point's mask keeps, drawn by
  ``causalab.io.plots.dbm_figure.plot_dbm``, as every DBM page does.
* ``masks_test.png``: the test-split IIA of the fitted mask beside a random
  mask of the same size, every neuron, the paper's 28 and the mask fitted on
  the gate half.
* ``<fig>_plotted.json`` for each figure: the values drawn. The sweep rows
  of ``masks_plotted.json`` flag the ringed point in ``chosen``.

The figure is a command and not a workflow step, because a step's outputs
must stay inside its own directory (workflow spec §2.3) and the committed
figures are under ``artifacts/figures/arithmetic_neurons/``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalab.io.env import FileDatasets
from causalab.io.step_io import frame, write_frame

__all__ = ["cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/arithmetic_neurons/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "arithmetic_neurons"
DATA = PAPERS / "artifacts" / "data" / "arithmetic_neurons"
FIGURES = PAPERS / "artifacts" / "figures" / "arithmetic_neurons"
MODEL = "meta-llama/Llama-3.1-8B"

#: The paper clips Figure 8a's colour map to [-2, 2] (its caption).
CLIP = 2.0
#: The sweep's coordinate, as ``train_eval.json`` and ``fit_diagnostics.json``
#: key it.
AXIS = "train.objective.l1.weight"


def paper_neurons() -> list[dict[str, Any]]:
    """The 28 neurons of Figure 8a, in its order, with their periods."""
    return json.loads((DATA / "neurons_feucht2026.json").read_text())["neurons"]


def table(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA / f"{name}.json").read_text())


def activations(step: str, read: str) -> Any:
    """``(rows, 14336)`` fp32: one saved read of a harvest step, in table order."""
    from safetensors.torch import load_file

    tensor = load_file(str(RUN / step / f"{read}.safetensors"))[read]
    return tensor.float().reshape(tensor.shape[0], -1).numpy()


def mask_units(step: str) -> list[int]:
    """The units an apply step's loaded gate keeps, ordered by rank."""
    rows = json.loads((RUN / step / "rank.json").read_text())
    kept = [r for r in rows if r["hard"]]
    return [int(r["unit"]) for r in sorted(kept, key=lambda r: r["rank"])]


def mean_iia(step: str) -> tuple[float, int]:
    """The mean of an IIA table over its eligible rows, and their count."""
    df = frame(RUN / step / "iia.json")
    df = df[df["eligible"]]
    return float(df["value"].mean()), int(len(df))


def _heatmap(ax: Any, values: Any, cmap: str, vmin: float, vmax: float) -> Any:
    return ax.imshow(
        values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )


def _save(figure: Any, target: Path) -> None:
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Figure 8a
# --------------------------------------------------------------------------- #


def fig8a_values(neurons: Sequence[int]) -> tuple[Any, list[int], list[int]]:
    """Mean full activation per (neuron, output sum) on the addition prompts,
    with the sums and the number of prompts behind each sum."""
    import numpy as np

    rows = table("addition_prompts")
    acts = activations("harvest_addition", "full")
    if acts.shape[0] != len(rows):
        raise ValueError(f"{acts.shape[0]} activation rows for {len(rows)} prompts")
    sums = np.array([int(r["sum"]) for r in rows])
    values = sorted(set(sums.tolist()))
    counts = [int((sums == s).sum()) for s in values]
    means = np.stack([acts[sums == s][:, list(neurons)].mean(axis=0) for s in values])
    return means.T, values, counts


def fig8a(dbm: list[int]) -> Any:
    """Figure 8a's heatmap for the neurons the DBM mask keeps, by rank; a
    neuron also in the paper's 28 is starred."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    in_paper = {n["neuron"] for n in paper_neurons()}
    means, sums, counts = fig8a_values(dbm)
    records = [
        {
            "neuron": int(neuron),
            "in_paper": neuron in in_paper,
            "sum": int(s),
            "mean_activation": float(means[i, j]),
            "n": counts[j],
        }
        for i, neuron in enumerate(dbm)
        for j, s in enumerate(sums)
    ]

    figure, ax = plt.subplots(
        figsize=(9.0, max(3.0, 0.16 * len(dbm) + 1.2)), constrained_layout=True
    )
    image = _heatmap(ax, means, "RdBu_r", -CLIP, CLIP)
    labels = [f"{n}*" if n in in_paper else str(n) for n in dbm]
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ticks = [i for i, s in enumerate(sums) if s % 10 == 0 or s == sums[0]]
    ax.set_xticks(ticks, [str(sums[i]) for i in ticks], fontsize=6, rotation=90)
    ax.set_xlabel("Output Sum")
    ax.set_ylabel("Neuron Index")
    ax.set_title(
        f"The DBM mask's {len(dbm)} neurons, by rank (* also in Feucht et al.'s 28)",
        fontsize=10,
    )
    figure.colorbar(image, ax=ax, shrink=0.6, label="mean activation")
    figure.suptitle(
        f"causalab: {MODEL}, layer 18 MLP, addition prompts a+b=", fontsize=9
    )
    _save(figure, FIGURES / "fig8a_replication.png")
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# The mask against its references
# --------------------------------------------------------------------------- #


def one_standard_error(rows: Sequence[Mapping[str, Any]]) -> int:
    """The index of the sweep row the one-standard-error rule picks.

    The rule takes the largest L1 weight whose IIA is within one binomial
    standard error of the best row's IIA (Hastie et al., *Elements of
    Statistical Learning*, §7.10). Each row holds ``l1_weight``, ``iia`` and
    ``n``, the number of pairs behind ``iia``.
    """
    best = max(rows, key=lambda r: r["iia"])
    se = math.sqrt(best["iia"] * (1 - best["iia"]) / best["n"])
    within = [i for i, r in enumerate(rows) if r["iia"] >= best["iia"] - se]
    return max(within, key=lambda i: rows[i]["l1_weight"])


def sweep_mask(weight: float) -> list[int]:
    """One value per neuron, 1 where the sweep's gate for ``weight`` keeps
    the neuron: ``theta > 0``, the hard mask of the sigmoid gate."""
    from safetensors import safe_open

    key = f"theta[{AXIS.removeprefix('train.')}={float(weight)!r}]"
    with safe_open(str(RUN / "sweep" / "gate.safetensors"), "pt") as handle:
        if key not in handle.keys():
            raise KeyError(f"sweep/gate.safetensors: no entry {key}")
        theta = handle.get_tensor(key).flatten().tolist()
    return [int(value > 0) for value in theta]


def sweep_figure(rows: Sequence[Mapping[str, Any]], kept: Sequence[int]) -> Any:
    """The sweep curve, its ``chosen`` row ringed, over the neurons ``kept``."""
    from causalab.io.plots.dbm_figure import NeuronMask, SweepPoint, plot_dbm

    (chosen,) = [i for i, r in enumerate(rows) if r["chosen"]]
    sweep = [SweepPoint(r["units"], r["iia"], f"{r['l1_weight']:g}") for r in rows]
    return plot_dbm(
        NeuronMask(kept),
        sweep,
        chosen=chosen,
        metric="val IIA",
        title=(
            f"L1 sweep on the val split · causalab: {MODEL}, layer 18 MLP "
            "neurons, weekdays sum"
        ),
    )


def masks(dbm: list[int]) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    diagnostics = json.loads((RUN / "sweep" / "fit_diagnostics.json").read_text())
    evals = json.loads((RUN / "sweep" / "train_eval.json").read_text())
    size = {
        d["coords"][AXIS]: d["featurizers"]["gate"]["hard_mask_size"]
        for d in diagnostics
    }
    # train_eval.json keeps the val IIA but not its pair count; the count is
    # recorded so the one-standard-error choice of the L1 weight can be redone
    # from masks_plotted.json alone
    datasets = FileDatasets(PAPERS / "artifacts" / "data")
    sweep = []
    for e in evals:
        weight = e["coords"][AXIS]
        sweep.append(
            {
                "panel": "sweep",
                "condition": f"l1={weight:g}",
                "l1_weight": weight,
                "split": "val",
                "units": int(size[weight]),
                "iia": float(e["metrics"]["iia"]),
                "n": len(datasets.rows(e["split"])),
                "chosen": False,
            }
        )
    chosen = one_standard_error(sweep)
    sweep[chosen]["chosen"] = True
    kept = sweep_mask(sweep[chosen]["l1_weight"])
    # the ringed mask is drawn as the mask the apply steps score and Figure 8a
    # reads; the fit step refits the chosen weight, so its mask must agree
    if [i for i, k in enumerate(kept) if k] != sorted(dbm):
        raise ValueError(
            f"the sweep's mask at l1={sweep[chosen]['l1_weight']:g} keeps "
            f"{sum(kept)} neurons, not the {len(dbm)} of the fitted gate"
        )
    FIGURES.mkdir(parents=True, exist_ok=True)
    sweep_figure(sweep, kept).savefig(FIGURES / "masks_sweep.png", dpi=200)

    records = list(sweep)
    paper = {n["neuron"] for n in paper_neurons()}
    gate_half = mask_units("apply_gate_half")
    conditions = [
        ("every neuron", "all_neurons", 14336),
        (f"DBM mask ({len(dbm)})", "apply", len(dbm)),
        (f"random {len(dbm)}", "apply_random", len(dbm)),
        ("paper's 28", "paper_neurons", 28),
        (f"DBM on gate half ({len(gate_half)})", "apply_gate_half", len(gate_half)),
    ]
    for label, step, units in conditions:
        iia, n = mean_iia(step)
        records.append(
            {
                "panel": "test",
                "condition": label,
                "l1_weight": None,
                "split": "test",
                "units": units,
                "iia": iia,
                "n": n,
                "chosen": None,
            }
        )
    df = pd.DataFrame.from_records(records)
    overlap = len(set(dbm) & paper)

    rows = df[df["panel"] == "test"]
    figure, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    se = [math.sqrt(p * (1 - p) / n) for p, n in zip(rows["iia"], rows["n"])]
    ax.barh(
        range(len(rows)),
        rows["iia"],
        xerr=se,
        color=["0.5", "C0", "0.75", "C3", "C1"],
    )
    ax.set_yticks(range(len(rows)), rows["condition"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test IIA (± binomial s.e.)")
    ax.set_title(
        f"Held-out interchange; the mask shares {overlap} of the paper's 28",
        fontsize=9,
    )
    ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    figure.suptitle(
        f"causalab: {MODEL}, layer 18 MLP neurons, weekdays sum", fontsize=9
    )
    _save(figure, FIGURES / "masks_test.png")
    return df


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.parse_args(argv)
    dbm = mask_units("apply")
    write_frame(fig8a(dbm), FIGURES / "fig8a_plotted.json")
    write_frame(masks(dbm), FIGURES / "masks_plotted.json")
    print(FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
