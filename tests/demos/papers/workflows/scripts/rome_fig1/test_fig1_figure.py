"""``fig1_figure.py``: the colour scale of the three heatmaps.

ROME's ``plot_trace_heatmap`` starts every panel's colour scale at the
corrupted run's probability (``vmin=low_score``) and ends it at the panel's
maximum. The script reads the corrupted probability from panel (e): a
last-layer state restored at a token before the last cannot reach the last
position, so those values are the corrupted run itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalab.io.step_io import StepError

from tests.demos.papers._scripts import load_script

pytestmark = pytest.mark.unit

fig1_figure = load_script("rome_fig1", "fig1_figure")

CORRUPTED = 0.0296
CLEAN = 0.9764


def _table(
    last_layer_earlier: tuple[float, ...] = (CORRUPTED, CORRUPTED),
) -> pd.DataFrame:
    """Three tokens and three layers per panel. Panel (e) holds the corrupted
    probability at the last layer before the last token and the clean one at
    the last token; every other value is between them or below the floor."""
    rows = []
    for panel, peak in (("e", 0.97), ("f", 0.89), ("g", 0.80)):
        for position in range(3):
            for layer in range(3):
                value = 0.01 if panel != "e" else 0.5
                if (position, layer) == (1, 1):
                    value = peak
                rows.append((panel, layer, position, value))
    table = pd.DataFrame(rows, columns=["panel", "layer", "position", "p_restored"])
    last = (table["panel"] == "e") & (table["layer"] == 2)
    for position, value in enumerate(last_layer_earlier):
        table.loc[last & (table["position"] == position), "p_restored"] = value
    table.loc[last & (table["position"] == 2), "p_restored"] = CLEAN
    return table


def test_each_panel_runs_from_the_corrupted_score_to_its_maximum() -> None:
    table = _table()
    assert fig1_figure.corrupted_score(table) == CORRUPTED
    assert fig1_figure.colour_scale(table, "e") == (CORRUPTED, CLEAN)
    assert fig1_figure.colour_scale(table, "f") == (CORRUPTED, 0.89)
    assert fig1_figure.colour_scale(table, "g") == (CORRUPTED, 0.80)


def test_a_panel_whose_values_all_sit_below_the_floor_keeps_a_positive_range() -> None:
    table = _table()
    table.loc[table["panel"] == "g", "p_restored"] = 0.01
    vmin, vmax = fig1_figure.colour_scale(table, "g")
    assert vmin == CORRUPTED and vmax > vmin


def test_last_layer_values_that_disagree_are_refused() -> None:
    """They are one forward each of the same corrupted run, so a spread means
    a restored state reached the last position: a bug, not rounding."""
    with pytest.raises(StepError, match="spread"):
        fig1_figure.corrupted_score(_table((CORRUPTED, CORRUPTED + 1e-3)))


def test_a_table_without_panel_e_is_refused() -> None:
    table = _table()
    with pytest.raises(StepError, match=r"panel \(e\)"):
        fig1_figure.corrupted_score(table[table["panel"] != "e"])


def test_the_drawn_panels_start_at_the_corrupted_score(tmp_path, monkeypatch) -> None:
    """``main`` draws every image through ``draw_panels``. In the figure it
    saves, the corrupted value and every value below it draw in the colour
    map's lightest colour, and the panel's maximum in its darkest, as in
    ROME's ``plot_trace_heatmap``."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    saved: list[Figure] = []
    savefig = Figure.savefig

    def record(self: Figure, *args: Any, **kwargs: Any) -> Any:
        saved.append(self)
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    table = _table()
    target = tmp_path / "panels.png"
    fig1_figure.draw_panels(
        target, table, {0: "a*", 1: "b", 2: "c"}, ["e", "f", "g"], ""
    )
    assert target.is_file()
    (figure,) = saved
    images = [ax.images[0] for ax in figure.axes if ax.images]
    assert len(images) == 3
    for image, panel in zip(images, ("e", "f", "g")):
        cmap = matplotlib.colormaps[fig1_figure.CMAPS[panel]]
        vmin, vmax = fig1_figure.colour_scale(table, panel)
        values = np.asarray(image.get_array())
        drawn = image.to_rgba(values)
        floor = values <= vmin
        assert floor.any(), panel
        assert (drawn[floor] == cmap(0.0)).all(), panel
        assert (drawn[values == vmax] == cmap(1.0)).all(), panel
