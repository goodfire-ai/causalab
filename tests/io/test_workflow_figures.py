"""``causalab.io.plots.workflow_figures`` — both kinds, and the axis-coverage refusal.

The interesting assertion is the last one. v1 refused an uncovered sweep axis at
*load*; with the axes published as data rather than derived in the document
model, the refusal moved to the step. It had to survive the move: an uncovered
axis silently averages over a dimension the reader cannot see, which is a wrong
figure rather than a missing one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalab.io.tables import read_table
from causalab.io.plots import workflow_figures as plot
from causalab.io.step_io import StepError
from tests.step_scripts import put_sidecar, put_table, run_step

pytestmark = pytest.mark.unit

GRID = [
    {"sites.target.layers": 1, "positions.tap": 0, "value": 0.1, "example_id": "0"},
    {"sites.target.layers": 1, "positions.tap": 0, "value": 0.3, "example_id": "1"},
    {"sites.target.layers": 2, "positions.tap": 0, "value": 0.8, "example_id": "0"},
    {"sites.target.layers": 2, "positions.tap": 1, "value": 0.5, "example_id": "0"},
    {"sites.target.layers": 1, "positions.tap": 1, "value": 0.2, "example_id": "0"},
]
AXES = ["sites.target.layers", "positions.tap"]


@pytest.fixture()
def grid(tmp_path: Path) -> Path:
    scan = tmp_path / "scan"
    put_table(scan / "iia.json", GRID)
    put_sidecar(scan, AXES)
    return scan / "iia.json"


def test_heatmap_writes_the_image_and_the_plotted_table(grid, tmp_path):
    """A figure is a declared output now (§2.5) — and declaring the rows beside
    it is what makes the picture checkable."""
    out = tmp_path / "out" / "scan.json"
    run_step(
        plot,
        {
            "table": grid,
            "plot": "heatmap",
            "x": "sites.target.layers",
            "y": "positions.tap",
        },
        {"figure": tmp_path / "out" / "scan.png", "plotted": out},
    )
    assert out.is_file()
    assert (out.parent / "scan.png").is_file()
    rows = read_table(out)
    # one row per (layer, tap) cell, mean over examples
    assert len(rows) == 4
    cell = next(
        r for r in rows if r["sites.target.layers"] == 1 and r["positions.tap"] == 0
    )
    assert cell["value"] == pytest.approx(0.2)


def test_lines_with_a_series(tmp_path):
    fit = tmp_path / "fit"
    put_table(
        fit / "iia.json",
        [
            {"featurizers.rot.k": 2, "train.seed": 0, "value": 0.1},
            {"featurizers.rot.k": 4, "train.seed": 0, "value": 0.4},
            {"featurizers.rot.k": 2, "train.seed": 1, "value": 0.2},
            {"featurizers.rot.k": 4, "train.seed": 1, "value": 0.5},
        ],
    )
    put_sidecar(fit, ["featurizers.rot.k", "train.seed"])
    out = tmp_path / "out" / "curve.json"
    run_step(
        plot,
        {
            "table": fit / "iia.json",
            "plot": "lines",
            "x": "featurizers.rot.k",
            "series": "train.seed",
        },
        {"figure": out.parent / "curve.png", "plotted": out},
    )
    assert (out.parent / "curve.png").is_file()


def test_an_uncovered_axis_is_refused(grid, tmp_path):
    with pytest.raises(StepError) as err:
        run_step(
            plot,
            {"table": grid, "plot": "lines", "x": "sites.target.layers"},
            {"figure": tmp_path / "out.png"},
        )
    assert "positions.tap" in str(err.value)
    assert "uncovered" in str(err.value)


def test_heatmap_needs_y(grid, tmp_path):
    with pytest.raises(StepError):
        run_step(
            plot,
            {"table": grid, "plot": "heatmap", "x": "sites.target.layers"},
            {"figure": tmp_path / "out.png"},
        )


def test_unknown_kind_is_refused(grid, tmp_path):
    with pytest.raises(StepError):
        run_step(
            plot,
            {"table": grid, "plot": "violin", "x": "sites.target.layers"},
            {"figure": tmp_path / "out.png"},
        )


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_the_static_formats_are_accepted(grid, tmp_path, suffix):
    run_step(
        plot,
        {
            "table": grid,
            "plot": "heatmap",
            "x": "sites.target.layers",
            "y": "positions.tap",
        },
        {"figure": tmp_path / f"fig{suffix}"},
    )
    assert (tmp_path / f"fig{suffix}").is_file()


def test_html_is_refused_by_this_renderer(grid, tmp_path):
    """`.html` is a legal document output (§2.5); matplotlib just cannot write
    one, and saying which renderer can beats savefig's format list."""
    with pytest.raises(StepError) as err:
        run_step(
            plot,
            {
                "table": grid,
                "plot": "heatmap",
                "x": "sites.target.layers",
                "y": "positions.tap",
            },
            {"figure": tmp_path / "fig.html"},
        )
    assert "matplotlib" in str(err.value)


def test_a_non_visualization_suffix_is_refused(grid, tmp_path):
    """The renderer validates through `normalize_figure_format`, so the
    png-over-pdf preference and the closed set live in one place."""
    with pytest.raises(ValueError):
        run_step(
            plot,
            {
                "table": grid,
                "plot": "heatmap",
                "x": "sites.target.layers",
                "y": "positions.tap",
            },
            {"figure": tmp_path / "fig.svg"},
        )


def test_a_missing_figure_output_is_refused(grid, tmp_path):
    with pytest.raises(StepError) as err:
        run_step(
            plot,
            {
                "table": grid,
                "plot": "heatmap",
                "x": "sites.target.layers",
                "y": "positions.tap",
            },
            {"plotted": tmp_path / "out.json"},
        )
    assert "figure" in str(err.value)


CLASS_PROBS = [
    {
        "sites.target.layers": layer,
        "positions.tap": tap,
        "value": json.dumps({"Q": q, "Z": round(1 - q, 2)}),
        "example_id": "0",
    }
    for layer, tap, q in ((1, 0, 0.1), (1, 1, 0.9), (2, 0, 0.4), (2, 1, 0.6))
]


@pytest.fixture()
def class_probs(tmp_path: Path) -> Path:
    scan = tmp_path / "scan"
    put_table(scan / "p.json", CLASS_PROBS)
    put_sidecar(scan, AXES)
    return scan / "p.json"


def test_key_draws_one_entry_of_a_structured_value(class_probs, tmp_path):
    """``class_probs`` stores one JSON object per row; ``key`` picks the group
    to draw, and the plotted table carries that group's numbers."""
    out = tmp_path / "out" / "p_q.json"
    run_step(
        plot,
        {
            "table": class_probs,
            "plot": "heatmap",
            "key": "Q",
            "x": "sites.target.layers",
            "y": "positions.tap",
        },
        {"figure": tmp_path / "out" / "p_q.png", "plotted": out},
    )
    rows = {
        (r["sites.target.layers"], r["positions.tap"]): r["value"]
        for r in read_table(out)
    }
    assert rows == {(1, 0): 0.1, (1, 1): 0.9, (2, 0): 0.4, (2, 1): 0.6}


def test_key_that_no_row_has_is_refused_by_name(class_probs, tmp_path):
    with pytest.raises(StepError, match=r"no entry 'M'.*\['Q', 'Z'\]"):
        run_step(
            plot,
            {
                "table": class_probs,
                "plot": "heatmap",
                "key": "M",
                "x": AXES[0],
                "y": AXES[1],
            },
            {"figure": tmp_path / "out" / "p.png"},
        )


def test_key_over_a_plain_number_is_refused(grid, tmp_path):
    with pytest.raises(StepError, match="plain number"):
        run_step(
            plot,
            {"table": grid, "plot": "heatmap", "key": "Q", "x": AXES[0], "y": AXES[1]},
            {"figure": tmp_path / "out" / "p.png"},
        )
