"""Figure 3a and its DBM extension: the per-head indirect effect and the fitted head masks.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/function_vectors_fig3a/fig3a_figure.py   # reads artifacts/output/function_vectors_fig3a/, writes artifacts/figures/function_vectors_fig3a/

It reads ``scan_early`` and ``scan_late`` (one head's task mean swapped in per
point), ``dbm_fit_<n>`` (the fitted gates, six tasks per step) and ``dbm_apply_<n>`` (their held-out
replay) and writes, under ``artifacts/figures/function_vectors_fig3a/``:

* ``fig3a_replication.png``: the average indirect effect (AIE) of every GPT-J
  head, layer by head index, in the style of the paper's Figure 3a, with the
  ten heads of highest AIE outlined;
* ``fig3a_dbm.png``: for every head, the number of tasks whose DBM mask keeps
  it, layer by head index as in the paper's Figure 3a, with the paper's ten
  heads outlined, drawn by ``causalab.io.plots.dbm_figure.plot_dbm`` in the
  colours of the encyclopedia's DBM viewer, as every DBM page does;
* ``fig3a_plotted.json``: the values drawn and the numbers the page quotes.

The indirect effect follows Todd et al. 2024, Eq. 3 and 4: per shuffled-label
prompt, the probability of the answer's first token with the head set to its
task mean minus the probability without the swap. The metric tables hold the
cross-entropy of that token, and ``exp(-ce)`` is its probability. The mean is
over each task's 25 prompts, then over the 18 tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "cli",
    "dbm_figure",
    "dbm_masks",
    "heldout_accuracy",
    "indirect_effects",
    "main",
    "rank_heads",
]

#: ``demos/papers/``: this script sits in ``workflows/scripts/function_vectors_fig3a/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "function_vectors_fig3a"
DATA = PAPERS / "artifacts" / "data" / "function_vectors_fig3a"
FIGURES = PAPERS / "artifacts" / "figures" / "function_vectors_fig3a"

LAYERS = 28
HEADS = 16
#: Section 2.3 and footnote 1: |A| = 10 heads for GPT-J.
TOP = 10
SCAN_STEPS = ("scan_early", "scan_late")
TASK = "axes.task"
LAYER = "sites.target.layers"
HEAD = "sites.target.head"
#: The paper outlines its top heads in pink (Figure 3a).
OUTLINE = "#ff00ff"
DPI = 200


def _rows(path: Path) -> Any:
    import pandas as pd

    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} holds no metric rows")
    return pd.DataFrame(rows)


def indirect_effects(patched: Any, corrupted: Any) -> Any:
    """Per (task, layer, head): the mean over prompts of p(answer | head :=
    task mean) − p(answer), from the two cross-entropy tables of the scan.

    ``corrupted`` repeats the un-swapped value at every point; one value per
    (task, example) is kept, and a point whose repeats disagree is refused,
    because the baseline is one forward per prompt."""
    import numpy as np

    base = corrupted.groupby([TASK, "example_id"])["value"].agg(["min", "max"])
    if not np.allclose(base["min"], base["max"], rtol=0, atol=1e-6):
        raise ValueError("the un-swapped cross-entropy differs between points")
    p_base = np.exp(-base["min"].astype(float)).rename("p_base")
    frame = patched.join(p_base, on=[TASK, "example_id"])
    frame = frame.assign(cie=np.exp(-frame["value"].astype(float)) - frame["p_base"])
    return (
        frame.groupby([TASK, LAYER, HEAD])["cie"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "cie", "size": "n"})
        .reset_index()
    )


def aie_grid(per_task: Any) -> Any:
    """The (layer, head) grid of the mean over tasks of the per-task CIE."""
    import numpy as np

    mean = per_task.groupby([LAYER, HEAD])["cie"].mean()
    grid = np.full((LAYERS, HEADS), np.nan)
    for (layer, head), value in mean.items():
        grid[int(layer), int(head)] = float(value)
    if np.isnan(grid).any():
        raise ValueError(f"{int(np.isnan(grid).sum())} heads have no scan value")
    return grid


def rank_heads(grid: Any) -> list[tuple[int, int]]:
    """Every (layer, head), highest AIE first; ties by lower layer, then head."""
    return sorted(
        (
            (layer, head)
            for layer in range(grid.shape[0])
            for head in range(grid.shape[1])
        ),
        key=lambda lh: (-grid[lh], lh[0], lh[1]),
    )


def dbm_masks(fit_dir: Path) -> dict[str, set[tuple[int, int]]]:
    """Per task, the heads the fitted gates keep: ``theta > 0``, the hard mask
    a sigmoid gate applies at evaluation (docs/intervention_protocol.md §2.5)."""
    from safetensors import safe_open

    from causalab.protocol.bundles import parse_entry_key

    masks: dict[str, set[tuple[int, int]]] = {}
    for layer in range(LAYERS):
        with safe_open(
            str(Path(fit_dir) / f"gate_L{layer}.safetensors"), "pt"
        ) as bundle:
            for key in bundle.keys():
                slot, coords = parse_entry_key(key)
                if slot != "theta" or TASK not in coords:
                    continue
                theta = bundle.get_tensor(key).float().flatten().tolist()
                if len(theta) != HEADS:
                    raise ValueError(f"{key}: {len(theta)} thetas, expected {HEADS}")
                kept = masks.setdefault(coords[TASK], set())
                kept.update((layer, h) for h, t in enumerate(theta) if t > 0)
    if not masks:
        raise ValueError(f"{fit_dir}: no per-task theta entries")
    return masks


def heldout_accuracy(apply_dir: Path) -> Any:
    """Per task and condition: top-1 accuracy of the answer's first token."""
    import pandas as pd

    parts = []
    for condition in ("masked", "corrupted", "all_heads"):
        rows = _rows(Path(apply_dir) / f"accuracy_{condition}.json")
        acc = rows.groupby(TASK)["value"].mean().rename(condition)
        parts.append(acc)
    return pd.concat(parts, axis=1).reset_index()


def paper_top(n: int = TOP) -> list[tuple[int, int]]:
    """The first ``n`` heads of the authors' released GPT-J ranking."""
    records = json.loads((DATA / "top_heads_todd2024.json").read_text())["records"]
    return [(int(r["layer"]), int(r["head"])) for r in records[:n]]


def _outline(ax: Any, heads: list[tuple[int, int]], color: str) -> None:
    from matplotlib.patches import Rectangle

    for layer, head in heads:
        ax.add_patch(
            Rectangle(
                (layer - 0.5, head - 0.5),
                1,
                1,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
        )


def draw_grid(
    target: Path,
    grid: Any,
    outlined: list[tuple[int, int]],
    *,
    cmap: str,
    label: str,
    symmetric: bool,
    caption: str,
) -> None:
    """One layer-by-head heatmap in the layout of the paper's Figure 3a."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    if symmetric:
        bound = float(np.nanmax(np.abs(grid))) or 1e-3
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = 0.0, max(float(np.nanmax(grid)), 1.0)
    image = ax.imshow(
        grid.T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper"
    )
    _outline(ax, outlined, OUTLINE)
    ax.set_xticks(range(0, LAYERS, 5))
    ax.set_yticks(range(0, HEADS, 5))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head Index")
    figure.colorbar(image, ax=ax).set_label(label)
    ax.set_title(caption, fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=DPI)
    plt.close(figure)


def dbm_figure(
    counts: Any, masks: int, outlined: list[tuple[int, int]], title: str
) -> Any:
    """The count of masks that keep each head over GPT-J's layers and heads,
    layers across as in Figure 3a, with ``outlined`` marked. There is no
    sweep: each task has one fitted weight."""
    from causalab.io.plots.dbm_figure import DbmLayer, HeadMask, plot_dbm

    mask = HeadMask(
        [DbmLayer(layer, HEADS) for layer in range(LAYERS)],
        {layer: [int(c) for c in counts[layer]] for layer in range(LAYERS)},
        masks=masks,
        layers_across=True,
        outlined=outlined,
    )
    return plot_dbm(mask, title=title)


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    root = Path(inputs["artifacts"])
    patched = pd.concat(
        [_rows(root / s / "ce_patched.json") for s in SCAN_STEPS], ignore_index=True
    )
    corrupted = pd.concat(
        [_rows(root / s / "ce_corrupted.json") for s in SCAN_STEPS], ignore_index=True
    )
    per_task = indirect_effects(patched, corrupted)
    grid = aie_grid(per_task)
    ranking = rank_heads(grid)
    ours = ranking[:TOP]
    paper = paper_top()
    rank_of = {lh: i + 1 for i, lh in enumerate(ranking)}
    tasks = sorted(per_task[TASK].unique())
    plotted: dict[str, Any] = {
        "tasks": tasks,
        "prompts_per_task": int(per_task["n"].min()),
        "aie": [
            [round(float(grid[layer, head]), 6) for head in range(HEADS)]
            for layer in range(LAYERS)
        ],
        "top10": [
            {"layer": lh[0], "head": lh[1], "aie": round(float(grid[lh]), 6)}
            for lh in ours
        ],
        "paper_top10": [
            {
                "layer": lh[0],
                "head": lh[1],
                "rank_here": rank_of[lh],
                "aie_here": round(float(grid[lh]), 6),
            }
            for lh in paper
        ],
        "top10_overlap_with_paper": len(set(ours) & set(paper)),
    }
    draw_grid(
        Path(outputs["replication"]),
        grid,
        ours,
        cmap="RdBu",
        label="AIE",
        symmetric=True,
        caption=f"causalab: GPT-J fp16, {len(tasks)} tasks x {plotted['prompts_per_task']} shuffled-label prompts",
    )
    fit_dirs = sorted(root.glob("dbm_fit_*"))
    apply_dirs = sorted(root.glob("dbm_apply_*"))
    if fit_dirs:
        masks: dict[str, set[tuple[int, int]]] = {}
        for fit_dir in fit_dirs:
            step = dbm_masks(fit_dir)
            if set(step) & set(masks):
                raise ValueError(
                    f"{fit_dir.name} refits {sorted(set(step) & set(masks))}"
                )
            masks.update(step)
        counts = np.zeros((LAYERS, HEADS))
        for kept in masks.values():
            for layer, head in kept:
                counts[layer, head] += 1
        plotted["dbm"] = {
            "mask_size": {t: len(masks[t]) for t in sorted(masks)},
            "mask": {t: sorted([list(lh) for lh in masks[t]]) for t in sorted(masks)},
            "tasks_keeping_head": counts.astype(int).tolist(),
            "overlap_with_paper_top10": {
                t: len(masks[t] & set(paper)) for t in sorted(masks)
            },
            "overlap_with_scan_top10": {
                t: len(masks[t] & set(ours)) for t in sorted(masks)
            },
        }
        target = Path(outputs["dbm"])
        target.parent.mkdir(parents=True, exist_ok=True)
        dbm_figure(
            counts,
            len(masks),
            paper,
            f"causalab: GPT-J fp16, one DBM head mask per task; "
            f"outlined: the paper's {len(paper)} heads",
        ).savefig(target, dpi=DPI)
    if apply_dirs:
        acc = pd.concat([heldout_accuracy(d) for d in apply_dirs], ignore_index=True)
        if acc[TASK].duplicated().any():
            raise ValueError("a task is replayed by two dbm_apply steps")
        plotted["heldout_accuracy"] = {
            "per_task": acc.round(6).to_dict(orient="records"),
            "mean": {
                c: round(float(acc[c].mean()), 6)
                for c in ("masked", "corrupted", "all_heads")
            },
        }
    target = Path(outputs["plotted"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plotted, indent=1) + "\n")
    return plotted


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: scan_early/, scan_late/, dbm_fit_<n>/ and dbm_apply_<n>/ are read",
    )
    args = parser.parse_args(argv)
    main(
        {"artifacts": args.artifacts},
        {
            "replication": FIGURES / "fig3a_replication.png",
            "dbm": FIGURES / "fig3a_dbm.png",
            "plotted": FIGURES / "fig3a_plotted.json",
        },
    )
    print(FIGURES / "fig3a_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
