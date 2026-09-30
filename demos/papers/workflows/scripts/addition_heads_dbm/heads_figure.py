"""The heads a desiderata-based mask keeps at the readout token of
``NN+MM=`` on Qwen3.5-2B, and the held-out score of every mask and swap.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/addition_heads_dbm/heads_figure.py        # reads artifacts/output/addition_heads_dbm/, writes artifacts/figures/addition_heads_dbm/

It reads the run tree under ``artifacts/output/addition_heads_dbm/`` and
writes, under ``artifacts/figures/addition_heads_dbm/``:

* ``heads_48.png``: the held-out score of each L1 weight's ``fit_all`` mask
  over the number of heads it keeps, and the heads the ringed mask keeps
  (``theta > 0``, the hard mask the apply step swaps through) in a layer by
  head matrix of the whole model, drawn by
  ``causalab.io.plots.dbm_figure.plot_dbm``;
* ``heads_l15.png``: the same for ``fit_l15``, over layer 15's 8 heads;
* ``heads_checks.png``: the held-out score of the exact swaps (``swaps``) and
  of the five random 3-head masks (``apply_random_<s>``);
* ``heads_all.png``: the three panels side by side;
* ``heads_plotted.json``: the values drawn, one row per mask.

A score is the mean over the held-out rows of a ``match``: ``iia`` when the
next token is the counterfactual problem's tens digit, ``null`` when it is
the base problem's. The DBM panels take the colours of the encyclopedia's
DBM viewer, as every DBM page does. The checks panel draws IIA in red.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["main", "cli", "load"]

#: ``demos/papers/``: this script sits in ``workflows/scripts/addition_heads_dbm/``.
PAPERS = Path(__file__).resolve().parents[3]
RUN = PAPERS / "artifacts" / "output" / "addition_heads_dbm"
FIGURES = PAPERS / "artifacts" / "figures" / "addition_heads_dbm"
MODEL = "Qwen/Qwen3.5-2B"
#: The full-attention layers of Qwen3.5-2B (``layer_types`` in its config).
LAYERS = [3, 7, 11, 15, 19, 23]
HEADS = 8
#: The L1 weights each fit sweeps, in the order of the documents.
WEIGHTS = {
    "all": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "l15": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
}
#: ``swaps`` intervened model -> label, in the order drawn.
SWAPS = {
    "all_attn": "all 6 layers, whole",
    "l15": "layer 15, whole",
    "s124": "L15 {1, 2, 4}",
    "s12": "L15 {1, 2}",
    "s03567": "L15 {0, 3, 5, 6, 7}",
}
#: The L1 weight whose mask each DBM panel draws: the lightest weight whose
#: mask keeps {1, 2, 4} of layer 15, the head set the swaps check.
CHOSEN = {"all": 30.0, "l15": 3.0}
RANDOM_DRAWS = 5
DPI = 200
COLUMNS = ["panel", "setting", "weight", "kept", "n_kept", "iia", "null", "n"]
IIA = "#c0392b"
NULL = "#9e9e9e"


def tag(weight: float) -> str:
    """The workflow's step suffix for an L1 weight: ``30.0`` -> ``w30``."""
    return "w" + f"{weight:g}".replace(".", "p")


def score(path: Path) -> tuple[float, int]:
    """Mean ``value`` over the eligible rows of one metric table, and the count."""
    rows = [r for r in json.loads(path.read_text()) if r.get("eligible", True)]
    if not rows:
        raise ValueError(f"{path}: no eligible rows")
    return sum(float(r["value"]) for r in rows) / len(rows), len(rows)


def kept_heads(bundle: Path, entry: str | None) -> list[int]:
    """The heads a gate bundle's hard mask keeps (``theta > 0``, sigmoid),
    for the entry keyed by ``objective.sparsity.weight=<entry>``, or the
    bundle's one entry when ``entry`` is ``None``."""
    from safetensors import safe_open

    with safe_open(str(bundle), "pt") as handle:
        keys = list(handle.keys())
        key = (
            f"theta[objective.sparsity.weight={entry}]"
            if entry is not None
            else keys[0]
        )
        if key not in keys:
            raise KeyError(f"{bundle}: no entry {key} (has {keys})")
        theta = handle.get_tensor(key).tolist()
    if len(theta) != HEADS:
        raise ValueError(f"{bundle}: {len(theta)} units, expected {HEADS} heads")
    return [h for h, value in enumerate(theta) if value > 0]


def _kept_label(kept: dict[int, list[int]]) -> str:
    return "; ".join(
        f"L{layer}: {','.join(map(str, heads))}" for layer, heads in kept.items()
    )


def load(run: Path) -> list[dict[str, Any]]:
    """One row per mask or swap: which heads it moves and its held-out scores."""
    rows: list[dict[str, Any]] = []
    for panel, layers, fit in (("all", LAYERS, "fit_all"), ("l15", [15], "fit_l15")):
        for weight in WEIGHTS[panel]:
            kept: dict[int, list[int]] = {}
            for layer in layers:
                name = (
                    f"gate_{layer}.safetensors"
                    if panel == "all"
                    else "gate.safetensors"
                )
                heads = kept_heads(run / fit / name, repr(float(weight)))
                if heads:
                    kept[layer] = heads
            step = run / f"apply_{panel}_{tag(weight)}"
            iia, n = score(step / "iia.json")
            null, _ = score(step / "null.json")
            rows.append(
                {
                    "panel": panel,
                    "setting": f"weight={weight:g}",
                    "weight": float(weight),
                    "kept": _kept_label(kept),
                    "n_kept": sum(len(h) for h in kept.values()),
                    "iia": iia,
                    "null": null,
                    "n": n,
                }
            )
    for model, label in SWAPS.items():
        iia, n = score(run / "swaps" / f"iia_{model}.json")
        null, _ = score(run / "swaps" / f"null_{model}.json")
        rows.append(
            {
                "panel": "checks",
                "setting": label,
                "weight": None,
                "kept": label,
                "n_kept": {"all_attn": 48, "l15": 8}.get(model, label.count(",") + 1),
                "iia": iia,
                "null": null,
                "n": n,
            }
        )
    for seed in range(RANDOM_DRAWS):
        heads = kept_heads(run / f"random_{seed}" / "gate.safetensors", "3.0")
        step = run / f"apply_random_{seed}"
        iia, n = score(step / "iia.json")
        null, _ = score(step / "null.json")
        label = "random L15 {" + ", ".join(map(str, heads)) + "}"
        rows.append(
            {
                "panel": "checks",
                "setting": label,
                "weight": None,
                "kept": f"L15: {','.join(map(str, heads))}",
                "n_kept": len(heads),
                "iia": iia,
                "null": null,
                "n": n,
            }
        )
    return rows


def _kept(row: dict[str, Any]) -> dict[int, list[int]]:
    """Layer -> kept heads, parsed from a row's ``kept`` label."""
    kept: dict[int, list[int]] = {}
    for part in filter(None, row["kept"].split("; ")):
        layer, heads = part.removeprefix("L").split(": ")
        kept[int(layer)] = [int(head) for head in heads.split(",")]
    return kept


def model_layers() -> list[Any]:
    """Every layer of Qwen3.5-2B with its head count and type, from the
    model registry, checked against ``LAYERS``."""
    from causalab.io.plots.dbm_figure import DbmLayer
    from causalab.protocol.registry import get_model_info

    info = get_model_info(MODEL)
    value_heads = info.linear_num_value_heads
    if info.layer_types is None or value_heads is None:
        raise ValueError(f"{MODEL}: the registry records no Gated DeltaNet layers")
    layers = []
    for index, kind in enumerate(info.layer_types):
        delta = kind == "linear_attention"
        layers.append(
            DbmLayer(
                index,
                value_heads if delta else info.num_heads,
                "gated_delta_net" if delta else "normal_attention",
            )
        )
    full = [layer.index for layer in layers if layer.type == "normal_attention"]
    if full != LAYERS or {layer.heads for layer in layers if layer.index in LAYERS} != {
        HEADS
    }:
        raise ValueError(f"{MODEL}: full-attention layers {full}, expected {LAYERS}")
    return layers


def dbm_panel(rows: list[dict[str, Any]], panel: str, title: str) -> Any:
    """The sweep of one fit and the heads its ``CHOSEN`` weight's mask keeps."""
    from causalab.io.plots.dbm_figure import HeadMask, SweepPoint, plot_dbm

    key = PANELS[panel][0]
    mine = [r for r in rows if r["panel"] == key]
    layers = LAYERS if key == "all" else [15]
    chosen = [r["weight"] for r in mine].index(CHOSEN[key])
    kept = _kept(mine[chosen])
    mask = HeadMask(
        model_layers(),
        {
            layer: [int(head in kept.get(layer, [])) for head in range(HEADS)]
            for layer in layers
        },
    )
    sweep = [SweepPoint(r["n_kept"], r["iia"], f"{r['weight']:g}") for r in mine]
    return plot_dbm(mask, sweep, chosen=chosen, title=title)


def _checks(ax: Any, rows: list[dict[str, Any]]) -> None:
    """Held-out IIA and null rate per exact swap and random mask."""
    import numpy as np

    y = np.arange(len(rows))
    ax.barh(
        y - 0.2,
        [r["iia"] for r in rows],
        height=0.38,
        color=IIA,
        label="IIA: counterfactual's tens digit",
    )
    ax.barh(
        y + 0.2,
        [r["null"] for r in rows],
        height=0.38,
        color=NULL,
        label="null: base's tens digit",
    )
    for yi, row in zip(y, rows):
        ax.text(
            row["iia"] + 0.01, yi - 0.2, f"{row['iia']:.3f}", va="center", fontsize=7.5
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["setting"] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("fraction of held-out pairs", fontsize=9)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.4, -0.16),
        ncol=2,
        frameon=False,
    )
    ax.set_title(PANELS["checks"][1], fontsize=11)


#: Panel name -> (``panel`` value of its rows, title).
PANELS = {
    "48": ("all", "DBM over all 48 attention heads"),
    "l15": ("l15", "DBM over layer 15's 8 heads"),
    "checks": ("checks", "exact head swaps and random 3-head masks"),
}


def _image(figure: Any) -> Any:
    """A figure rendered at ``DPI``, as an RGBA array."""
    import io

    import matplotlib.image

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI)
    buffer.seek(0)
    return matplotlib.image.imread(buffer)


def draw_checks(target: Path, rows: list[dict[str, Any]], caption: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    _checks(ax, [r for r in rows if r["panel"] == "checks"])
    figure.suptitle(caption, fontsize=9)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=DPI)
    plt.close(figure)


def draw_all(target: Path, rows: list[dict[str, Any]], caption: str) -> None:
    """The two DBM panels as rendered images beside the checks panel.

    ``plot_dbm`` draws a whole figure, so each DBM panel is placed as its
    image at the same resolution.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [
        _image(dbm_panel(rows, panel, PANELS[panel][1])) for panel in ("48", "l15")
    ]
    heights = [image.shape[0] / DPI for image in images]
    widths = [image.shape[1] / DPI for image in images]
    checks_width, top = 6.5, 0.35
    width = sum(widths) + checks_width
    height = max(heights) + top
    figure = plt.figure(figsize=(width, height))
    left = 0.0
    for image, w, h in zip(images, widths, heights):
        ax = figure.add_axes(
            (left / width, 1 - (top + h) / height, w / width, h / height)
        )
        ax.imshow(image, interpolation="nearest")
        ax.set_axis_off()
        left += w
    # the checks panel beside the sweep curves, with room for its labels
    ax = figure.add_axes(
        (
            (left + 2.3) / width,
            1.2 / height,
            (checks_width - 2.5) / width,
            (height - top - 1.7) / height,
        )
    )
    _checks(ax, [r for r in rows if r["panel"] == "checks"])
    figure.suptitle(caption, fontsize=10, y=1 - 0.17 / height, va="center")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=DPI)
    plt.close(figure)


def main(run: Path, figures: Path) -> list[dict[str, Any]]:
    """Draw every image and write the plotted values; return the rows."""
    import pandas as pd

    from causalab.io.step_io import write_frame

    rows = load(run)
    n = {r["n"] for r in rows}
    caption = (
        f"causalab: Qwen3.5-2B bf16, readout token '=', held-out pairs n = "
        f"{', '.join(map(str, sorted(n)))}"
    )
    figures.mkdir(parents=True, exist_ok=True)
    draw_all(figures / "heads_all.png", rows, caption)
    for panel in ("48", "l15"):
        title = f"{PANELS[panel][1]} · {caption}"
        dbm_panel(rows, panel, title).savefig(figures / f"heads_{panel}.png", dpi=DPI)
    draw_checks(figures / "heads_checks.png", rows, caption)
    write_frame(pd.DataFrame(rows)[COLUMNS], figures / "heads_plotted.json")
    return rows


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN,
        help="the workflow's output directory; the figures go to artifacts/figures/addition_heads_dbm/",
    )
    args = parser.parse_args(argv)
    main(args.artifacts, FIGURES)
    print(FIGURES / "heads_all.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
