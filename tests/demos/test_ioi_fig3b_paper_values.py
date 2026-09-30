"""The IOI Figure 3b paper values, and the committed comparison built on them.

``demos/papers/artifacts/data/ioi_fig3b/fig3b_wang2022_values.json`` holds
what ``workflows/scripts/ioi_fig3b/paper_values.py`` read off the paper's PDF
(arXiv 2211.00593v1): the colour of every cell of Figure 3b and the end of
every bar of Figure 15. The PDF is not in the repository, so these checks
tie the values to the file's own record of the drawing:

* each Figure 3b value follows from the cell's colour, the colour scale and
  the colour bar's end, by the file's rule;
* each Figure 15 value follows from the bar's end and the grid lines;
* ``artifacts/figures/ioi_fig3b/fig3b_compare.json``, the numbers the page
  quotes, follows from the committed ``fig3b_plotted.json`` and these values.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "demos" / "papers"
SCRIPTS = PAPERS / "workflows" / "scripts" / "ioi_fig3b"
VALUES = PAPERS / "artifacts" / "data" / "ioi_fig3b" / "fig3b_wang2022_values.json"
FIGURES = PAPERS / "artifacts" / "figures" / "ioi_fig3b"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"ioi_fig3b_{name}", SCRIPTS / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values() -> dict:
    return json.loads(VALUES.read_text())


def test_every_figure_3b_value_follows_from_its_colour() -> None:
    paper_values = _script("paper_values")
    block = _values()["fig3b"]
    decode = block["decode"]
    assert [tuple(stop) for stop in decode["scale"]] == list(paper_values.RDBU)
    slope, _ = np.polyfit(decode["tick_values"], decode["ticks"], 1)
    zmax = (decode["bar"]["bottom"] - decode["bar"]["top"]) / 2 / abs(slope)
    assert decode["zmax"] == pytest.approx(zmax, abs=1e-6)
    cells = {(r["layer"], r["head"]) for r in block["records"]}
    assert len(cells) == len(block["records"]) == 144
    for record in block["records"]:
        t = paper_values.scale_position(record["rgb"])
        assert record["t"] == pytest.approx(t, abs=1e-6), record
        assert record["value"] == pytest.approx(decode["zmax"] * (2 * t - 1), abs=2e-6)
    # the strongest head sits at the end of the scale, as plotly draws it
    strongest = min(block["records"], key=lambda r: r["value"])
    assert (strongest["layer"], strongest["head"], strongest["t"]) == (9, 9, 0.0)


def test_every_figure_15_value_follows_from_its_bar() -> None:
    block = _values()["fig15"]
    decode = block["decode"]
    slope, _ = np.polyfit(decode["grid_values"], decode["grid"], 1)
    assert decode["per_unit"] == pytest.approx(slope, abs=1e-6)
    assert decode["zero"] == decode["grid"][decode["grid_values"].index(0.0)]
    records = block["records"]
    assert len({(r["layer"], r["head"]) for r in records}) == len(records) == 15
    for record in records:
        expected = (record["end"] - decode["zero"]) / decode["per_unit"]
        assert record["value"] == pytest.approx(expected, abs=1e-6), record
    # the figure orders its bars by decreasing absolute effect
    magnitudes = [abs(r["value"]) for r in records]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_committed_comparison_follows_from_the_committed_values() -> None:
    compare = json.loads((FIGURES / "fig3b_compare.json").read_text())
    plotted = {
        f"{r['layer']}.{r['head']}": r
        for r in json.loads((FIGURES / "fig3b_plotted.json").read_text())
    }
    values = _values()
    for figure in ("fig3b", "fig15"):
        paper = {
            f"{r['layer']}.{r['head']}": r["value"] for r in values[figure]["records"]
        }
        r = values[figure]["reading_error"]
        total = 0.0
        for head in compare["heads"]:
            cell, entry = plotted[head["head"]], head[figure]
            assert head["ours"] == cell["variation"]
            assert (head["se"], head["sd_n100"]) == (cell["se"], cell["sd_n100"])
            assert entry["paper"] == paper[head["head"]]
            spread = math.hypot(cell["sd_n100"], cell["se"])
            assert entry["gap"] == pytest.approx(
                cell["variation"] - paper[head["head"]]
            )
            assert entry["bound"] == pytest.approx(r + 2 * spread)
            assert entry["z"] == pytest.approx(entry["gap"] / math.hypot(spread, r))
            total += entry["z"] ** 2
        assert compare["chi2"][figure] == pytest.approx(total)
    assert compare["pairs"] == max(cell["n"] for cell in plotted.values())
