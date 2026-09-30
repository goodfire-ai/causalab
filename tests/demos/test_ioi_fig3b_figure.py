"""The statistics of the IOI Figure 3b figure script, on synthetic run trees.

``demos/papers/workflows/scripts/ioi_fig3b/fig3b_figure.py`` turns the two
steps' per-pair logit differences into the plotted cells: the variation of
each (layer, head), its bootstrap standard error ``se`` and its spread
``sd_n100`` over draws of 100 pairs. The page compares the paper's values
with ours through these two columns, so each is checked here against an
independent computation. The comparison file, ``fig3b_compare.json``, is
checked on a synthetic paper whose values are known.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.numerical_unit

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "demos/papers/workflows/scripts/ioi_fig3b/fig3b_figure.py"
LAYERS = HEADS = 12


def _script():
    spec = importlib.util.spec_from_file_location("ioi_fig3b_figure", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_step(step: Path, name: str, values: np.ndarray, layers: bool) -> None:
    """One step's metric table and sidecar, in the runner's row format.

    ``values`` is (pairs, layers, 12), and NaN becomes a null value. With
    ``layers`` the rows carry the ``axes.sender`` coordinate, as the scan's
    do; without it they carry only the head, as the last-layer step's do."""
    rows = []
    for example in range(values.shape[0]):
        for layer in range(values.shape[1]):
            for head in range(HEADS):
                value = float(values[example, layer, head])
                record = {
                    "example_id": str(example),
                    "metric": name,
                    "value": None if math.isnan(value) else value,
                    "sites.sender.head": head,
                }
                if layers:
                    record["axes.sender"] = layer
                rows.append(record)
    step.mkdir(parents=True, exist_ok=True)
    (step / f"{name}.json").write_text(json.dumps(rows))
    axes = ["axes.sender", "sites.sender.head"] if layers else ["sites.sender.head"]
    (step / "_step.json").write_text(json.dumps({"axes": axes}))


def _plotted(root: Path, clean: np.ndarray, patched: np.ndarray) -> dict:
    """The plotted cells, keyed by (layer, head), of a run tree whose steps
    hold ``clean`` and ``patched``, two (pairs, 12, 12) arrays."""
    for name, values in (("ld_clean", clean), ("ld_patched", patched)):
        _write_step(root / "scan", name, values[:, : LAYERS - 1], layers=True)
        _write_step(root / "last", name, values[:, LAYERS - 1 :], layers=False)
    target = root / "plotted.json"
    _script().main({"scan": root / "scan", "last": root / "last"}, {"plotted": target})
    return {(r["layer"], r["head"]): r for r in json.loads(target.read_text())}


def _synthetic(pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Clean logit differences near the model's 3.5, and patched ones that
    move each cell by its own factor, with noise per pair."""
    rng = np.random.default_rng(seed)
    clean = rng.normal(3.5, 1.5, (pairs, 1, 1)) + rng.normal(0, 1e-3, (pairs, 12, 12))
    effect = rng.uniform(-0.7, 0.5, (1, 12, 12))
    patched = clean * (1 + effect) + rng.normal(0, 0.8, (pairs, 12, 12))
    return clean, patched


def test_variation_is_the_ratio_of_the_means(tmp_path: Path) -> None:
    clean, patched = _synthetic(pairs=60, seed=1)
    patched[7, 9, 9] = np.nan
    cells = _plotted(tmp_path, clean, patched)
    assert len(cells) == LAYERS * HEADS
    for (layer, head), cell in cells.items():
        c = np.nanmean(clean[:, layer, head])
        p = np.nanmean(patched[:, layer, head])
        assert cell["ld_clean"] == pytest.approx(c, abs=1e-12)
        assert cell["variation"] == pytest.approx((p - c) / c, abs=1e-12)
        assert cell["n"] == (59 if (layer, head) == (9, 9) else 60)
        assert math.isfinite(cell["se"]) and math.isfinite(cell["sd_n100"])


def test_a_cell_moved_by_a_fixed_factor_has_no_spread(tmp_path: Path) -> None:
    """Patched is the clean value times one factor per cell on every pair.
    A resample that draws the same pairs for both tables gives the factor
    exactly, and an unpaired resample would spread."""
    rng = np.random.default_rng(2)
    clean = rng.normal(3.5, 1.5, (80, 12, 12))
    factor = rng.uniform(-0.7, 0.5, (1, 12, 12))
    cells = _plotted(tmp_path, clean, clean * (1 + factor))
    for (layer, head), cell in cells.items():
        assert cell["variation"] == pytest.approx(factor[0, layer, head], abs=1e-12)
        assert cell["se"] < 1e-12 and cell["sd_n100"] < 1e-12


def test_the_spread_matches_the_delta_method(tmp_path: Path) -> None:
    """For the ratio r = mean(P) / mean(C), the delta method gives
    SD(r) = sqrt(var(P - r C) / m) / mean(C) for a draw of m pairs. The
    bootstrap SD over 4000 resamples has a Monte Carlo error of about 0.011
    of its value, so a tolerance of 0.05 is more than four of those."""
    pairs = 2000
    clean, patched = _synthetic(pairs=pairs, seed=3)
    cells = _plotted(tmp_path, clean, patched)
    for (layer, head), cell in cells.items():
        c, p = clean[:, layer, head], patched[:, layer, head]
        ratio = p.mean() / c.mean()
        spread = np.var(p - ratio * c, ddof=1)
        assert cell["se"] == pytest.approx(
            math.sqrt(spread / pairs) / c.mean(), rel=0.05
        )
        assert cell["sd_n100"] == pytest.approx(
            math.sqrt(spread / 100) / c.mean(), rel=0.05
        )


def test_the_resamples_are_fixed(tmp_path: Path) -> None:
    clean, patched = _synthetic(pairs=50, seed=4)
    first = _plotted(tmp_path / "a", clean, patched)
    second = _plotted(tmp_path / "b", clean, patched)
    for key, cell in first.items():
        assert cell["se"] == second[key]["se"]
        assert cell["sd_n100"] == second[key]["sd_n100"]


#: The 15 heads of a synthetic Figure 15, largest first, with their effects:
#: seven large ones and a tail of eight, each at least 0.008 from the next.
FIFTEEN = {
    "9.9": -0.60, "10.7": 0.45, "9.6": -0.30, "11.10": 0.24, "10.0": -0.20,
    "10.10": -0.14, "11.2": 0.10, "7.9": -0.080, "10.1": -0.072, "8.10": -0.064,
    "10.6": -0.056, "9.7": -0.048, "8.6": -0.040, "10.2": -0.032, "7.3": -0.024,
}  # fmt: skip


def _effects(overrides: dict[str, float] | None = None) -> np.ndarray:
    effect = np.zeros((LAYERS, HEADS))
    for name, value in {**FIFTEEN, **(overrides or {})}.items():
        layer, head = (int(x) for x in name.split("."))
        effect[layer, head] = value
    return effect


def _paper_file(path: Path, fig3b: np.ndarray, fig15: dict[str, float]) -> Path:
    """A values file in the layout of ``paper_values.py``: all 144 cells of
    Figure 3b and the 15 bars of Figure 15."""
    records = [
        {"layer": layer, "head": head, "value": float(fig3b[layer, head])}
        for layer in range(LAYERS)
        for head in range(HEADS)
    ]
    bars = [
        {"layer": int(n.split(".")[0]), "head": int(n.split(".")[1]), "value": v}
        for n, v in fig15.items()
    ]
    path.write_text(
        json.dumps(
            {
                "fig3b": {"reading_error": 0.005, "records": records},
                "fig15": {"reading_error": 0.001, "records": bars},
            }
        )
    )
    return path


def _compare(root: Path, effect: np.ndarray, paper: Path, pairs: int = 400) -> dict:
    """The comparison file of a run tree whose heads move the clean logit
    difference by ``effect``, with a little noise per pair."""
    rng = np.random.default_rng(5)
    clean = rng.normal(3.5, 1.0, (pairs, 1, 1)) * np.ones((1, LAYERS, HEADS))
    patched = clean * (1 + effect) + rng.normal(0, 0.05, (pairs, LAYERS, HEADS))
    for name, values in (("ld_clean", clean), ("ld_patched", patched)):
        _write_step(root / "scan", name, values[:, : LAYERS - 1], layers=True)
        _write_step(root / "last", name, values[:, LAYERS - 1 :], layers=False)
    target = root / "compare.json"
    _script().main(
        {"scan": root / "scan", "last": root / "last", "paper": paper},
        {"compare": target, "plotted": root / "plotted.json"},
    )
    return json.loads(target.read_text())


def test_each_head_is_compared_with_both_figures(tmp_path: Path) -> None:
    effect = _effects()
    paper = _paper_file(tmp_path / "paper.json", effect, FIFTEEN)
    result = _compare(tmp_path, effect, paper)
    plotted = {
        f"{r['layer']}.{r['head']}": r
        for r in json.loads((tmp_path / "plotted.json").read_text())
    }
    assert [h["head"] for h in result["heads"]] == list(FIFTEEN)
    for figure, r in (("fig3b", 0.005), ("fig15", 0.001)):
        total = 0.0
        for head in result["heads"]:
            cell = plotted[head["head"]]
            entry = head[figure]
            spread = math.hypot(cell["sd_n100"], cell["se"])
            assert entry["gap"] == pytest.approx(
                cell["variation"] - FIFTEEN[head["head"]]
            )
            assert entry["bound"] == pytest.approx(r + 2 * spread)
            assert entry["z"] == pytest.approx(entry["gap"] / math.hypot(spread, r))
            assert entry["inside"]
            total += entry["z"] ** 2
        assert result["chi2"][figure] == pytest.approx(total)
    assert result["chi2_limit"] == pytest.approx(24.9958, abs=1e-4)
    assert result["top7"]["ours"] == result["top7"]["paper"] == list(FIFTEEN)[:7]
    assert result["top7"]["in_resamples"] == 1.0
    assert result["tail"]["ours"] == result["tail"]["paper"] == list(FIFTEEN)[7:]
    assert result["tail"]["set_in_resamples"] == 1.0
    # every order pair of the tail is shared by both figures and kept here
    assert len(result["tail"]["pairs"]) == 28
    assert all(pair["margin"] > 0 for pair in result["tail"]["pairs"])
    assert result["signs"]["heads"] == 15 and result["signs"]["mismatches"] == []
    assert result["extremes"] == {"min": "9.9", "max": "10.7"}
    assert result["name_movers"] == pytest.approx(
        sum(plotted[h]["variation"] for h in ("9.9", "9.6", "10.0"))
    )


def test_a_tail_order_both_figures_reverse_is_reported(tmp_path: Path) -> None:
    """Both figures put 10.6 above 8.10, where the run has 8.10 above 10.6
    by 0.008: the pair is listed in the paper's order, with a negative
    margin that no resample of the run reverses. Every other pair keeps
    the paper's order."""
    effect = _effects()
    swapped = {**FIFTEEN, "8.10": -0.056, "10.6": -0.064}
    paper = _paper_file(tmp_path / "paper.json", _effects(swapped), swapped)
    result = _compare(tmp_path, effect, paper)
    pairs = {(p["first"], p["second"]): p for p in result["tail"]["pairs"]}
    reversed_pair = pairs[("10.6", "8.10")]
    assert reversed_pair["margin"] == pytest.approx(-0.008, abs=0.003)
    assert reversed_pair["first_in_resamples"] == 0.0
    assert 0.0 <= reversed_pair["first_in_paper_draws"] < 0.5
    assert [p for p in pairs.values() if p["margin"] < 0] == [reversed_pair]
    assert (
        result["tail"]["all_pairs_in_paper_draws"]
        <= (reversed_pair["first_in_paper_draws"])
    )


def test_a_sign_is_compared_where_the_paper_colours_it(tmp_path: Path) -> None:
    """A head the paper colours at 0.01 or more with the other sign is a
    mismatch; a fainter one is listed apart."""
    effect = _effects({"5.5": -0.03, "6.6": 0.02})
    shown = effect.copy()
    shown[5, 5], shown[6, 6] = 0.02, -0.004
    paper = _paper_file(tmp_path / "paper.json", shown, FIFTEEN)
    signs = _compare(tmp_path, effect, paper)["signs"]
    assert [m["head"] for m in signs["mismatches"]] == ["5.5"]
    assert [m["head"] for m in signs["fainter_mismatches"]] == ["6.6"]


def test_the_colour_scale_ends_at_the_largest_head() -> None:
    """As the authors' plotly heatmap with its midpoint at 0, the scale is
    symmetric and the strongest head saturates it."""
    import pandas as pd

    table = pd.DataFrame({"variation": [0.3, -0.672, 0.41, float("nan")]})
    assert _script().colour_limit(table) == pytest.approx(0.672)
