"""The committed Figure 1 (e, f, g) values against the paper's, read from its PDF.

``demos/papers/artifacts/data/rome_fig1/fig1efg_meng2022_values.json`` holds
the fill colour of every heatmap rectangle of the paper's figure (page 2 of
arXiv 2202.05262v5) and the value its ``decode`` block reads from it. The
replication page compares ``artifacts/figures/rome_fig1/fig1_plotted.json``
with those values, so these checks tie both claims to committed files:

* the values follow from the colours by the file's own rule;
* every plotted value is within the reading error of the paper's value:
  0.005 where the paper's value is above 0.1, and 0.011 below it, where
  neighbouring colour-map bins share an 8-bit colour;
* every plotted value, drawn on the figure script's colour scale, is within
  2/255 per channel of the paper's colour.

The two bounds are the acceptance criteria of the package's audit. The paper
draws a value below its colour-bar floor in the floor's colour, so a plotted
value below the floor is compared as the floor.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.demos.papers._scripts import PAPERS, load_script

pytestmark = pytest.mark.unit

fig1_figure = load_script("rome_fig1", "fig1_figure")

VALUES = PAPERS / "artifacts" / "data" / "rome_fig1" / "fig1efg_meng2022_values.json"
PLOTTED = PAPERS / "artifacts" / "figures" / "rome_fig1" / "fig1_plotted.json"
PANELS = ("e", "f", "g")
#: The prompt's 7 tokens by GPT-2 XL's 48 layers, per panel.
GRID = (7, 48)
#: The reading error of a value above and below ``SPLIT``, and of a colour
#: channel in 8-bit units.
READ_HIGH, READ_LOW, SPLIT = 0.005, 0.011, 0.1
COLOUR = 2.0


def _paper() -> dict:
    return json.loads(VALUES.read_text())


def _grid(records: list[dict], field: str) -> dict[str, np.ndarray]:
    """panel -> (7, 48) array of ``field``, or (7, 48, 3) for a colour."""
    out: dict[str, np.ndarray] = {}
    for panel in PANELS:
        rows = [r for r in records if r["panel"] == panel]
        shape = np.shape(rows[0][field])
        grid = np.full(GRID + shape, np.nan)
        for r in rows:
            grid[r["position"], r["layer"]] = r[field]
        out[panel] = grid
    return out


def _lut(name: str) -> np.ndarray:
    """The 256 RGB entries of a matplotlib colour map, in [0, 1]."""
    import matplotlib

    return matplotlib.colormaps[name](np.arange(256))[:, :3]


def test_every_rectangle_of_the_three_panels_is_read_once() -> None:
    records = _paper()["records"]
    keys = {(r["panel"], r["position"], r["layer"]) for r in records}
    assert len(records) == len(keys) == len(PANELS) * GRID[0] * GRID[1]
    for panel, grid in _grid(records, "value").items():
        assert np.isfinite(grid).all(), panel


def test_each_value_follows_from_its_colour_by_the_files_rule() -> None:
    paper = _paper()
    decode = paper["decode"]
    floor, bins = decode["floor"], decode["bins"]
    for r in paper["records"]:
        lut = _lut(decode["colour_maps"][r["panel"]])
        distance = np.abs(lut - np.asarray(r["rgb"]) / 255).sum(axis=1)
        assert r["bin"] == int(distance.argmin()), r
        top = decode["tops"][r["panel"]]
        expected = floor + (r["bin"] + 0.5) / bins * (top - floor)
        assert r["value"] == pytest.approx(expected, abs=1e-12), r


def test_the_plotted_values_are_the_papers_within_its_reading_error() -> None:
    paper = _paper()
    floor = paper["decode"]["floor"]
    theirs = _grid(paper["records"], "value")
    ours = _grid(json.loads(PLOTTED.read_text()), "p_restored")
    for panel in PANELS:
        gap = np.abs(np.maximum(ours[panel], floor) - theirs[panel])
        high = theirs[panel] > SPLIT
        assert gap[high].max() <= READ_HIGH, (panel, float(gap[high].max()))
        assert gap[~high].max() <= READ_LOW, (panel, float(gap[~high].max()))


def test_the_plotted_values_draw_in_the_papers_colours() -> None:
    """The figure script's colour scale, from the corrupted probability to
    the panel's maximum, turns the plotted values into the published
    colours."""
    import matplotlib
    from matplotlib.colors import Normalize

    paper = _paper()
    table = pd.DataFrame(json.loads(PLOTTED.read_text()))
    ours = _grid(table.to_dict("records"), "p_restored")
    theirs = _grid(paper["records"], "rgb")
    for panel in PANELS:
        vmin, vmax = fig1_figure.colour_scale(table, panel)
        cmap = matplotlib.colormaps[paper["decode"]["colour_maps"][panel]]
        drawn = cmap(Normalize(vmin, vmax)(ours[panel]))[..., :3]
        gap = float(np.abs(drawn * 255 - theirs[panel]).max())
        assert gap <= COLOUR, (panel, gap)
