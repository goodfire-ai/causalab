"""Table 5, Toxicity column: GPT-2 and 10 Manual Pick, in the paper and in this run.

Run after the workflow, from ``demos/papers/``::

    python workflows/scripts/mlp_steering/table5_figure.py          # reads artifacts/output/mlp_steering/, writes artifacts/figures/mlp_steering/

It reads the reduce step's rate per arm and label, ``rates/toxicity.json``,
and the two values of Table 5 that the package replicates,
``artifacts/data/mlp_steering/table5_geva2022_values.json``. Each figure is
one bar: a dashed black outline at the GPT-2 rate and, in front of it, a
filled bar at the 10 Manual Pick rate. It writes under
``artifacts/figures/mlp_steering/``:

- ``table5_original.png``, the paper's two values, for ``### Original``;
- ``table5_replication.png``, this run's two ``toxic`` rates, each with its
  95% bootstrap interval over prompts as a whisker;
- ``table5_amplified.png`` and ``table5_baseline.png``, the filled bar and the
  outline alone, one per specification, for the page's implementation and
  ``change these lines`` sections;
- ``table5_plotted.json``, the values drawn: this run's rate of each arm
  with its interval, and the paper's value for the same arm.

The four images share one y scale, so the bars compare across them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: ``demos/papers/``: this script sits in ``workflows/scripts/mlp_steering/``.
PAPERS = Path(__file__).resolve().parents[3]

RUN = PAPERS / "artifacts" / "output" / "mlp_steering"
FIGURES = PAPERS / "artifacts" / "figures" / "mlp_steering"
RATES = RUN / "rates" / "toxicity.json"
VALUES = PAPERS / "artifacts" / "data" / "mlp_steering" / "table5_geva2022_values.json"

#: The workflow's two protocol steps: the outline, then the filled bar.
ARMS = ("baseline", "amplified")
LABELS = {"baseline": "GPT-2", "amplified": "10 Manual Pick"}
#: A categorical blue, the colour this package gave 10 Manual Pick before it
#: drew one bar.
COLOUR = "#2a78d6"
#: The grade step's label for Table 5's Toxicity column (``score_toxicity.COLUMNS``).
ATTRIBUTE = "toxic"
YMAX = 0.7


def paper_values(path: Path = VALUES) -> dict[str, float]:
    """The paper's Toxicity rate per workflow step, from the values file.

    Raises:
        ValueError: a record's value does not follow from its printed
            percentage by the file's rule, or the file does not hold one
            record per arm of `ARMS`.
    """
    records = json.loads(path.read_text())["records"]
    out: dict[str, float] = {}
    for record in records:
        printed = float(record["printed"].rstrip("%")) / 100
        if abs(printed - record["value"]) > 1e-9:
            raise ValueError(f"{path.name}: {record} does not follow its rule")
        out[record["step"]] = record["value"]
    if sorted(out) != sorted(ARMS) or len(records) != len(ARMS):
        raise ValueError(f"{path.name}: records for {sorted(out)}, want {sorted(ARMS)}")
    return out


def plotted_rates(
    rates: Sequence[dict[str, Any]], paper: dict[str, float]
) -> list[dict[str, Any]]:
    """The ``toxic`` row of each arm of `ARMS` from the reduce step's table,
    with the paper's value for the same arm. Other arms and labels are not
    drawn."""
    rows = {r["arm"]: r for r in rates if r["attribute"] == ATTRIBUTE}
    return [
        {
            "arm": arm,
            "attribute": ATTRIBUTE,
            "value": rows[arm]["value"],
            "lower": rows[arm]["lower"],
            "upper": rows[arm]["upper"],
            "n": rows[arm]["n"],
            "paper": paper[arm],
        }
        for arm in ARMS
    ]


def _bar(
    ax: Any,
    arm: str,
    value: float,
    interval: tuple[float, float] | None,
) -> None:
    """One arm's mark at x = 0: the outline for ``baseline``, the filled bar
    for ``amplified``, and its whisker when ``interval`` is given."""
    if arm == "baseline":
        ax.bar(
            0,
            value,
            width=0.5,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            linewidth=1.4,
            zorder=3,
            label=LABELS[arm],
        )
    else:
        ax.bar(0, value, width=0.5, color=COLOUR, zorder=2, label=LABELS[arm])
    if interval is not None:
        low, high = interval
        ax.errorbar(
            0,
            value,
            yerr=[[value - low], [high - value]],
            color="black",
            capsize=5,
            linewidth=1.0,
            zorder=4,
        )
    ax.text(0.3, value, f"{value:.3f}", va="center", fontsize=9)


def draw(rows: Sequence[dict[str, Any]], title: str, *, whiskers: bool) -> Any:
    """One bar chart of the arms in ``rows`` (``arm``, ``value`` and, with
    ``whiskers``, ``lower`` and ``upper``). The outline is drawn behind the
    filled bar's whisker and in front of its fill, so both stay visible."""
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    for row in rows:
        interval = (row["lower"], row["upper"]) if whiskers else None
        _bar(ax, row["arm"], row["value"], interval)
    ax.set_xlim(-0.6, 0.9)
    ax.set_ylim(0, YMAX)
    ax.set_xticks([0])
    ax.set_xticklabels(["Toxicity"])
    ax.set_ylabel("fraction of continuations flagged toxic")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    paper = paper_values()
    rates = plotted_rates(json.loads(RATES.read_text()), paper)
    n = rates[0]["n"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    original = [{"arm": arm, "value": paper[arm]} for arm in ARMS]
    draw(original, "Geva et al. 2022, Perspective API", whiskers=False).savefig(
        FIGURES / "table5_original.png", dpi=150
    )
    judge = f"unitary/toxic-bert, {n} prompts"
    draw(rates, judge, whiskers=True).savefig(
        FIGURES / "table5_replication.png", dpi=150
    )
    for row in rates:
        draw([row], judge, whiskers=True).savefig(
            FIGURES / f"table5_{row['arm']}.png", dpi=150
        )
    (FIGURES / "table5_plotted.json").write_text(
        json.dumps({"rates": rates}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
