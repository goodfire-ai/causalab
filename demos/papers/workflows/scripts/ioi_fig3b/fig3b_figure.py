"""Figure 3b: this replication's heatmap, whole and one document at a time.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/ioi_fig3b/fig3b_figure.py            # reads artifacts/output/ioi_fig3b/, writes artifacts/figures/ioi_fig3b/

It reads the two logit-difference tables of each step — ``ld_clean`` (the
un-intervened model) and ``ld_patched`` (the head's direct path to the logits
patched from the pABC prompt) — from ``artifacts/output/ioi_fig3b/scan/`` (layers 0–10) and
``artifacts/output/ioi_fig3b/last/`` (layer 11), and writes, under
``artifacts/figures/ioi_fig3b/``, ``fig3b_replication.png`` (all 144 heads; the
page shows the paper's crop separately), ``fig3b_scan.png`` and
``fig3b_last.png`` (the rows each document draws: layers 0–10 and layer 11,
on the same colour scale) and ``fig3b_plotted.json``,
the exact cells drawn: layer, head, the mean clean and patched logit
differences, the number of pairs each is a mean over, and the paper's
quantity — the **logit-difference variation**, ``(patched − clean) / clean``
of the two means. The paper's Appendix B measures "the logit difference on
the forward pass D compared to the logit difference of the model … as an
average over N > 200 pairs"; the ratio of the two averages is what its
colour bar reads. A head whose patched path *lowers* the logit
difference is negative (the paper's red: Name Movers), one that raises it is
positive (blue: Negative Name Movers).

Two more columns give the uncertainty of each cell, from the same bootstrap
resamples of the pairs for every cell (`resample`). ``se`` is the SD of
the variation over resamples of all the run's pairs: the standard error of
our own draw. ``sd_n100`` is its SD over resamples of 100 pairs: the spread
of one draw of the size the authors' code averages over.

``fig3b_compare.json`` holds every number the page quotes against the paper
(`compare`): each of the 15 heads of Figure 15 against the values that
``paper_values.py`` read off Figures 3b and 15, the χ² over them, the signs,
the orders and how often the resamples keep them.

The figure is a command rather than a workflow step because a step's outputs
must stay inside its own directory (workflow spec §2.3) and the committed
figure sits under ``artifacts/figures/ioi_fig3b/``. ``main(inputs, outputs)`` is still the step
signature — inputs ``scan``, ``last`` (the two step directories) and
``paper`` (the values file, needed for ``compare``); outputs
``replication``, ``scan``, ``last``, ``plotted``, ``compare``, each optional —
so the same function serves either way.

Rows are aggregated through [`causalab.io.step_record.aggregate`][], the
reduction the shipped ``select`` and ``workflow_figures`` scripts use, so a
value quoted from ``plotted`` and one chosen from the table by a later step
cannot disagree about what a row is.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from causalab.io.step_io import StepError, frame, write_frame
from causalab.io.step_record import EXAMPLE_COLUMN, aggregate, axes_for

__all__ = ["main", "cli"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/ioi_fig3b/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "ioi_fig3b"
FIGURES = PAPERS / "artifacts" / "figures" / "ioi_fig3b"
PAPER_VALUES = (
    PAPERS / "artifacts" / "data" / "ioi_fig3b" / "fig3b_wang2022_values.json"
)
FIGURE = "fig3b"
HEAD_AXIS = "sites.sender.head"
LAYER_AXIS = "axes.sender"
LAYERS, HEADS = 12, 12
COLUMNS = ["layer", "head", "ld_clean", "ld_patched", "variation", "n", "se", "sd_n100"]
#: Resamples in each bootstrap, and the seed that fixes them. At 4000
#: resamples the Monte Carlo error of an SD is about 1 / sqrt(2 * 4000), 0.011
#: of its value.
RESAMPLES = 4000
SEED = 0
#: The pairs of one paper draw: the authors' code averages over N = 100
#: (Easy-Transformer ``experiments.py`` lines 78-84 at ``ea15315``).
PAPER_PAIRS = 100
#: Section 3.1 names the Name Movers 9.9, 9.6 and 10.0 and the Negative Name
#: Movers 10.7 and 11.10.
NAME_MOVERS = ("9.9", "9.6", "10.0")
NEGATIVE_NAME_MOVERS = ("10.7", "11.10")
#: The heads of Figure 3b the sign comparison covers: those at least this far
#: from white, about two colour steps.
SIGN_THRESHOLD = 0.01
#: The layers in which the paper shows no direct effect.
EARLY_LAYERS = range(7)


class Resamples(NamedTuple):
    """Bootstrap resamples of the pairs, the same pairs for every cell.

    ``everything`` and ``paper`` are (`RESAMPLES`, 144) arrays of the
    variation, from resamples of all the run's pairs and of `PAPER_PAIRS`
    pairs; the cells are in the order of the table `grid` returns. ``clean``
    is the clean logit difference of each pair."""

    everything: Any
    paper: Any
    clean: Any


def step_cells(step: Path, layer: int | None) -> tuple[Any, Any]:
    """The cells of a step and the pairs behind them.

    The cells are one row per (layer, head): the mean clean and patched
    logit differences over the step's pairs and the count of patched values.
    The pairs are one row per (example, layer, head), with the columns
    ``clean`` and ``patched``. ``layer`` is the constant the step's tables
    carry no column for (the last-layer step), else ``None`` and the layer
    is the ``axes.sender`` coordinate."""
    import pandas as pd

    clean_path, patched_path = step / "ld_clean.json", step / "ld_patched.json"
    clean, patched = frame(clean_path), frame(patched_path)
    for path, df in ((clean_path, clean), (patched_path, patched)):
        if EXAMPLE_COLUMN not in df.columns or "value" not in df.columns:
            raise StepError(f"{path} is not a metric table (needs example, value)")
        axes = axes_for(path)
        if HEAD_AXIS not in axes:
            raise StepError(
                f"{path} does not carry the axis {HEAD_AXIS!r} (has {list(axes)}); "
                "this script reads the steps of workflow.json"
            )
        if layer is None and LAYER_AXIS not in axes:
            raise StepError(
                f"{path} carries no {LAYER_AXIS!r} axis and no layer was given"
            )
    per_clean, axes = aggregate(clean, clean_path, "value")
    per_patched, _ = aggregate(patched, patched_path, "value")
    counts = (
        patched[patched["value"].notna()]
        .groupby(list(axes))
        .size()
        .rename("n")
        .reset_index()
    )
    cells = (
        per_clean.rename(columns={"value": "ld_clean"})
        .merge(per_patched.rename(columns={"value": "ld_patched"}), on=list(axes))
        .merge(counts, on=list(axes), how="left")
    )
    pairs = (
        clean[[EXAMPLE_COLUMN, *axes, "value"]]
        .rename(columns={"value": "clean"})
        .merge(
            patched[[EXAMPLE_COLUMN, *axes, "value"]].rename(
                columns={"value": "patched"}
            ),
            on=[EXAMPLE_COLUMN, *axes],
            how="outer",
        )
    )
    for df in (cells, pairs):
        df["head"] = df[HEAD_AXIS].astype(int)
        df["layer"] = df[LAYER_AXIS].astype(int) if layer is None else int(layer)
    cells["n"] = cells["n"].fillna(0).astype(int)
    return (
        pd.DataFrame(cells)[["layer", "head", "ld_clean", "ld_patched", "n"]],
        pd.DataFrame(pairs)[[EXAMPLE_COLUMN, "layer", "head", "clean", "patched"]],
    )


def resample(clean: Any, patched: Any, size: int, rng: Any) -> Any:
    """Each cell's variation over `RESAMPLES` bootstrap resamples.

    Args:
        clean: The clean logit difference of each pair, a (pairs, cells)
            array with NaN where a pair has no value.
        patched: The patched logit difference, in the same layout.
        size: The pairs each resample draws, with replacement.
        rng: The numpy generator the draws come from.

    Returns:
        A (`RESAMPLES`, cells) array. Each resample uses the same pairs for
        every cell and for both arrays, so a statistic of several cells is
        paired. A cell's means skip its NaN values, as `aggregate` does. The
        SD over resamples at ``size`` equal to the pair count is the
        bootstrap standard error of the variation.
    """
    import numpy as np

    pairs, cells = clean.shape
    # clean and patched side by side, so one sum per resample serves both
    values = np.concatenate([clean, patched], axis=1)
    present = ~np.isnan(values)
    values = np.where(present, values, 0.0)
    counts = None if present.all() else present.astype(float)
    draws = np.empty((RESAMPLES, cells))
    for i in range(RESAMPLES):
        weight = np.bincount(rng.integers(0, pairs, size), minlength=pairs)
        weight = weight.astype(float)
        # einsum, not matmul: matmul through Apple's Accelerate BLAS raised
        # divide-by-zero warnings on finite inputs here (numpy 2.2.6, macOS
        # arm64) with results equal to 1e-14; einsum does not call BLAS.
        # Upstream report: https://github.com/numpy/numpy/issues/28687
        sums = np.einsum("i,ij->j", weight, values)
        means = sums / (
            size if counts is None else np.einsum("i,ij->j", weight, counts)
        )
        draws[i] = (means[cells:] - means[:cells]) / means[:cells]
    return draws


def grid(scan: Path, last: Path) -> tuple[Any, Resamples]:
    """All 144 cells, with the paper's logit-difference variation, its
    bootstrap standard error ``se`` and its spread ``sd_n100`` over draws of
    `PAPER_PAIRS` pairs, and the resamples behind the last two."""
    import numpy as np
    import pandas as pd

    (scan_cells, scan_pairs), (last_cells, last_pairs) = (
        step_cells(scan, None),
        step_cells(last, LAYERS - 1),
    )
    table = pd.concat([scan_cells, last_cells], ignore_index=True)
    if len(table) != LAYERS * HEADS or table.duplicated(["layer", "head"]).any():
        raise StepError(
            f"expected {LAYERS * HEADS} distinct (layer, head) cells, got {len(table)}"
        )
    table["variation"] = (table["ld_patched"] - table["ld_clean"]) / table["ld_clean"]
    table = table.sort_values(["layer", "head"]).reset_index(drop=True)
    pairs = pd.concat([scan_pairs, last_pairs], ignore_index=True)
    cells = pd.MultiIndex.from_frame(table[["layer", "head"]])
    per_pair = {
        column: pairs.pivot(
            index=EXAMPLE_COLUMN, columns=["layer", "head"], values=column
        )
        .reindex(columns=cells)
        .to_numpy(dtype=float)
        for column in ("clean", "patched")
    }
    rng = np.random.default_rng(SEED)
    everything = per_pair["clean"].shape[0]
    draws = Resamples(
        everything=resample(per_pair["clean"], per_pair["patched"], everything, rng),
        paper=resample(per_pair["clean"], per_pair["patched"], PAPER_PAIRS, rng),
        # the clean run does not depend on the head: every cell holds the
        # same clean value of a pair, so one column serves
        clean=np.nanmean(per_pair["clean"], axis=1),
    )
    table["se"] = draws.everything.std(axis=0, ddof=1)
    table["sd_n100"] = draws.paper.std(axis=0, ddof=1)
    return table[COLUMNS], draws


def _name(layer: int, head: int) -> str:
    return f"{layer}.{head}"


def compare(table: Any, draws: Resamples, paper: Mapping[str, Any]) -> dict[str, Any]:
    """The comparison of this run with the paper's two figures.

    ``paper`` is the values file of ``paper_values.py``. A paper value
    matches ours when the gap is at most its reading error ``r`` plus
    ``2 * sqrt(sd_n100**2 + se**2)``: the spread of the paper's 100-pair draw
    and of ours. ``z`` divides the gap by ``sqrt(sd_n100**2 + se**2 + r**2)``,
    and χ² sums ``z**2`` over the 15 heads of Figure 15. An order statistic
    is the fraction of resamples, of all our pairs and of 100, that keep it.
    """
    import numpy as np
    from scipy.stats import chi2

    names = [_name(int(r.layer), int(r.head)) for r in table.itertuples()]
    column = {name: i for i, name in enumerate(names)}
    ours, se, sd = (
        {name: float(v) for name, v in zip(names, table[c].to_numpy(dtype=float))}
        for c in ("variation", "se", "sd_n100")
    )
    figures = {
        figure: {
            _name(r["layer"], r["head"]): float(r["value"])
            for r in paper[figure]["records"]
        }
        for figure in ("fig3b", "fig15")
    }
    reading = {figure: float(paper[figure]["reading_error"]) for figure in figures}
    # Figure 15 draws its heads by decreasing absolute effect
    ranked = sorted(figures["fig15"], key=lambda h: -abs(figures["fig15"][h]))
    top, tail = ranked[:7], ranked[7:]
    magnitude = {key: np.abs(d) for key, d in draws._asdict().items() if key != "clean"}

    def kept(order: list[str], key: str) -> float:
        """The fraction of resamples whose largest heads are ``order``."""
        rank = np.argsort(-magnitude[key], axis=1)[:, : len(order)]
        return float((rank == [column[h] for h in order]).all(axis=1).mean())

    def kept_set(heads: list[str], start: int, key: str) -> float:
        rank = np.argsort(-magnitude[key], axis=1)[:, start : start + len(heads)]
        wanted = sorted(column[h] for h in heads)
        return float((np.sort(rank, axis=1) == wanted).all(axis=1).mean())

    def margin(first: str, second: str) -> dict[str, float]:
        """``|first| - |second|`` here, its paired SE, and how often each
        kind of resample puts ``first`` above ``second``."""
        a, b = column[first], column[second]
        spread = magnitude["everything"][:, a] - magnitude["everything"][:, b]
        return {
            "margin": abs(ours[first]) - abs(ours[second]),
            "se": float(spread.std(ddof=1)),
            "first_in_resamples": float((spread > 0).mean()),
            "first_in_paper_draws": float(
                (magnitude["paper"][:, a] > magnitude["paper"][:, b]).mean()
            ),
        }

    heads = []
    chi_square = {figure: 0.0 for figure in figures}
    outside_without_draw = {figure: [] for figure in figures}
    for head in ranked:
        entry: dict[str, Any] = {
            "head": head,
            "ours": ours[head],
            "se": se[head],
            "sd_n100": sd[head],
        }
        for figure, values in figures.items():
            r, gap = reading[figure], ours[head] - values[head]
            spread = float(np.hypot(sd[head], se[head]))
            z = gap / float(np.sqrt(spread**2 + r**2))
            chi_square[figure] += z * z
            entry[figure] = {
                "paper": values[head],
                "gap": gap,
                "bound": r + 2 * spread,
                "z": z,
                "inside": bool(abs(gap) <= r + 2 * spread),
            }
            if abs(gap) > r + 2 * se[head]:
                outside_without_draw[figure].append(head)
        heads.append(entry)

    # the order pairs of the tail on which both figures agree
    tail_pairs = []
    for first, second in itertools.combinations(tail, 2):
        above = [abs(v[first]) - abs(v[second]) for v in figures.values()]
        if all(x > 0 for x in above) or all(x < 0 for x in above):
            if above[0] < 0:
                first, second = second, first
            tail_pairs.append(
                {"first": first, "second": second, **margin(first, second)}
            )
    shared = np.ones(RESAMPLES, dtype=bool)
    for pair in tail_pairs:
        a, b = column[pair["first"]], column[pair["second"]]
        shared &= magnitude["paper"][:, a] > magnitude["paper"][:, b]
    differ = max(ranked, key=lambda h: abs(figures["fig3b"][h] - figures["fig15"][h]))
    ours_order = sorted(names, key=lambda h: -abs(ours[h]))
    signs = [
        {"head": h, "paper": v, "ours": ours[h]}
        for h, v in figures["fig3b"].items()
        if abs(v) >= SIGN_THRESHOLD
    ]
    faint = [
        {"head": h, "paper": v, "ours": ours[h]}
        for h, v in figures["fig3b"].items()
        if 0 < abs(v) < SIGN_THRESHOLD and np.sign(v) != np.sign(ours[h])
    ]
    early = [h for h in names if int(h.split(".")[0]) in EARLY_LAYERS]
    largest_early = max(early, key=lambda h: abs(ours[h]))
    clean = draws.clean
    return {
        "description": (
            "This run against Figures 3b and 15 of Wang et al. 2022, from "
            "workflows/scripts/ioi_fig3b/fig3b_figure.py: per head, the paper's "
            "values read by paper_values.py, the gap, the bound r + 2 sqrt(sd_n100^2 "
            "+ se^2) and z; chi2 over the 15 heads of Figure 15; the sign of every "
            "Figure 3b head at 0.01 or more; the top-7 order and the set of places "
            "8 to 15; every pair of the tail on whose order both figures agree, "
            "and the share of 100-pair draws that keep all of them; each "
            "statistic's share of the resamples of all our pairs and of 100-pair "
            "draws; the head on which the two figures differ most."
        ),
        "pairs": int(clean.shape[0]),
        "resamples": RESAMPLES,
        "paper_pairs": PAPER_PAIRS,
        "clean": {
            "mean": float(clean.mean()),
            "se": float(clean.std(ddof=1) / np.sqrt(clean.shape[0])),
            "io_over_s": int((clean > 0).sum()),
        },
        "heads": heads,
        "chi2": chi_square,
        "chi2_limit": float(chi2.ppf(0.95, len(ranked))),
        "outside_without_sd_n100": outside_without_draw,
        "extremes": {
            "min": min(names, key=lambda h: ours[h]),
            "max": max(names, key=lambda h: ours[h]),
        },
        "early_layers": {"head": largest_early, "value": ours[largest_early]},
        "signs": {
            "threshold": SIGN_THRESHOLD,
            "heads": len(signs),
            "mismatches": [
                s for s in signs if np.sign(s["paper"]) != np.sign(s["ours"])
            ],
            "fainter_mismatches": faint,
        },
        "top7": {
            "paper": top,
            "ours": ours_order[:7],
            "margins": [
                margin(a, b) | {"first": a, "second": b}
                for a, b in zip(top, top[1:] + [ours_order[7]])
            ],
            "in_resamples": kept(top, "everything"),
            "in_paper_draws": kept(top, "paper"),
        },
        "tail": {
            "paper": tail,
            "ours": ours_order[7:15],
            "set_in_resamples": kept_set(tail, 7, "everything"),
            "set_in_paper_draws": kept_set(tail, 7, "paper"),
            "pairs": tail_pairs,
            "all_pairs_in_paper_draws": float(shared.mean()),
        },
        "figures_differ": {
            "head": differ,
            "gap": figures["fig3b"][differ] - figures["fig15"][differ],
        },
        "name_movers": sum(ours[h] for h in NAME_MOVERS),
        "negative_name_movers": sum(ours[h] for h in NEGATIVE_NAME_MOVERS),
    }


def colour_limit(table: Any) -> float:
    """The symmetric colour bound every image shares: the largest absolute
    variation, as the authors' plotly heatmap sets it
    (``color_continuous_midpoint=0``), so the strongest head saturates."""
    import numpy as np

    return float(np.nanmax(np.abs(table["variation"].to_numpy(dtype=float))))


def draw_panels(
    target: Path, table: Any, layers: list[int], limit: float, caption: str
) -> None:
    """The heatmap rows of ``layers``, cells at or beyond 0.05 labelled in
    percent, on the colour scale ``[-limit, limit]`` with the paper's ticks
    every 25%."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = (
        table.pivot(index="layer", columns="head", values="variation")
        .reindex(index=layers, columns=range(HEADS))
        .to_numpy(dtype=float)
    )
    # a strip of one or two rows is too short for an upright colour bar
    strip = len(layers) <= 2
    figure, ax = plt.subplots(
        figsize=(7.2, 2.4 if strip else 1.6 + 0.42 * len(layers)),
        constrained_layout=True,
    )
    image = ax.imshow(
        matrix, cmap="RdBu", vmin=-limit, vmax=limit, origin="upper", aspect="equal"
    )
    for row, layer in enumerate(layers):
        for head in range(HEADS):
            value = matrix[row, head]
            if value == value and abs(value) >= 0.05:
                ax.text(
                    head,
                    row,
                    f"{100 * value:+.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.6 * limit else "black",
                )
    ax.set_xticks(range(HEADS))
    ax.set_yticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Direct effect on logit difference", fontsize=11)
    steps = int(np.floor(limit / 0.25 + 1e-9))
    ticks = [0.25 * k for k in range(-steps, steps + 1)]
    bar = figure.colorbar(
        image,
        ax=ax,
        ticks=ticks,
        orientation="horizontal" if strip else "vertical",
        shrink=0.6 if strip else 1.0,
    )
    bar.set_label("Logit diff. variation", fontsize=9)
    bar.set_ticklabels([f"{100 * t:.0f}%" for t in ticks])
    figure.suptitle(caption, fontsize=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200)
    plt.close(figure)


#: The heatmap rows each document draws, and the file stem of each drawn alone.
PANEL_LAYERS = {"scan": list(range(LAYERS - 1)), "last": [LAYERS - 1]}
PANEL_FILES = {"scan": f"{FIGURE}_scan", "last": f"{FIGURE}_last"}


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    table, draws = grid(Path(inputs["scan"]), Path(inputs["last"]))
    n_pairs = int(table["n"].max())
    clean = float(table["ld_clean"].mean())
    limit = colour_limit(table)
    caption = (
        f"causalab: GPT-2 small fp32, path patching h → logits at END; "
        f"n = {n_pairs} pairs, clean logit difference {clean:.2f}"
    )
    if "replication" in outputs:
        draw_panels(
            Path(outputs["replication"]), table, list(range(LAYERS)), limit, caption
        )
    for panel, layers in PANEL_LAYERS.items():
        if panel in outputs:
            draw_panels(Path(outputs[panel]), table, layers, limit, caption)
    if "plotted" in outputs:
        write_frame(table, Path(outputs["plotted"]))
    if "compare" in outputs:
        paper = json.loads(Path(inputs["paper"]).read_text())
        target = Path(outputs["compare"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(compare(table, draws, paper), indent=1) + "\n")


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory: scan/ and last/ are read",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=FIGURES,
        help="where the figures, fig3b_plotted.json and fig3b_compare.json go",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=PAPER_VALUES,
        help="the paper's values, from paper_values.py",
    )
    args = parser.parse_args(argv)
    root, figures = args.artifacts, args.figures
    main(
        {"scan": root / "scan", "last": root / "last", "paper": args.paper},
        {
            "replication": figures / f"{FIGURE}_replication.png",
            **{panel: figures / f"{stem}.png" for panel, stem in PANEL_FILES.items()},
            "plotted": figures / f"{FIGURE}_plotted.json",
            "compare": figures / f"{FIGURE}_compare.json",
        },
    )
    print(figures / f"{FIGURE}_replication.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
