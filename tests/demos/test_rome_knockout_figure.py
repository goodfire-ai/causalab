"""The ROME knockout figure script reads the ``class_probs`` save.

The knockout document (``demos/papers/protocols/rome_fig1_knockout.json``)
saves p(Seattle) as a ``class_probs`` table, one JSON object string per cell.
``knockout_figure.py`` draws that probability as written, with no ``exp(-ce)``
step. These tests build a small run tree in the layout the runner writes
(``_step.json`` with the sweep axes, the metric table, the location ledger)
and check the cells the script plots.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from causalab.io.step_io import StepError
from causalab.protocol.rules.errors import ProtocolError

pytestmark = pytest.mark.unit

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "demos"
    / "papers"
    / "workflows"
    / "scripts"
    / "rome_fig1"
)
AXES = ["axes.center", "positions.tap.index"]
TOKENS = {0: "The", 1: "Ġdowntown"}


def _script():
    """``knockout_figure.py`` as a module; it imports ``fig1_figure`` beside it."""
    spec = importlib.util.spec_from_file_location(
        "rome_knockout_figure", SCRIPTS / "knockout_figure.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _step(
    root: Path,
    name: str,
    cells: dict[tuple[int, int], object],
    excluded: tuple[tuple[int, int], ...] = (),
) -> Path:
    """A step directory with one row per (centre, token) cell, plus one
    excluded row per cell of ``excluded``: ``value: null``, ``eligible:
    false``, as ``causalab.neural.shared.results`` writes an ``Unavailable``
    measurement."""
    step = root / name
    step.mkdir(parents=True)
    (step / "_step.json").write_text(json.dumps({"axes": AXES}))
    rows = [
        {
            "example_id": "0",
            "metric": "p_ablated",
            "value": value,
            "axes.center": center,
            "positions.tap.index": position,
            "unit": "fraction",
            "estimand_version": "class_probs/v1",
            "eligible": True,
        }
        for (center, position), value in cells.items()
    ]
    rows += [
        {
            "example_id": "1",
            "metric": "p_ablated",
            "value": None,
            "axes.center": center,
            "positions.tap.index": position,
            "unit": "fraction",
            "estimand_version": "class_probs/v1",
            "eligible": False,
            "reason_code": "empty_selector",
        }
        for center, position in excluded
    ]
    (step / "p_ablated.json").write_text(json.dumps(rows))
    ledger = [
        {
            "example": 0,
            "constituent": "writes.knockout.pos",
            "side": "base",
            "token_index": index,
            "decoded_token": token,
        }
        for index, token in TOKENS.items()
    ]
    (step / "location_ledger.json").write_text(json.dumps(ledger))
    return step


def _cells(values: dict[tuple[int, int], float]) -> dict[tuple[int, int], object]:
    """The runner's cell format: the group mapping as a JSON object string."""
    return {k: json.dumps({"Seattle": v}) for k, v in values.items()}


def test_a_cell_is_the_seattle_probability_as_saved(tmp_path: Path) -> None:
    values = {(0, 0): 0.0075, (0, 1): 0.25, (1, 0): 0.5, (1, 1): 0.976}
    table, labels = _script().load(_step(tmp_path, "knockout", _cells(values)))
    assert labels == {0: "The", 1: "downtown"}
    got = {(r.layer, r.position): r.p_ablated for r in table.itertuples()}
    assert got == pytest.approx(values, abs=0, rel=0)
    assert set(table["n"]) == {1}
    assert set(table["token"]) == {"The", "downtown"}


def test_the_plotted_table_holds_every_component(tmp_path: Path) -> None:
    script = _script()
    values = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4}
    inputs = {
        component: _step(tmp_path / "run", step, _cells(values))
        for component, (step, _, _) in script.COMPONENTS.items()
    }
    plotted = tmp_path / "figures" / "knockout_all_plotted.json"
    script.main(
        inputs, {"figure": tmp_path / "figures" / "all.png", "plotted": plotted}
    )
    rows = json.loads(plotted.read_text())
    assert {r["component"] for r in rows} == set(script.COMPONENTS)
    assert len(rows) == len(values) * len(script.COMPONENTS)
    assert {(r["layer"], r["position"]): r["p_ablated"] for r in rows} == values
    assert (tmp_path / "figures" / "all.png").is_file()


def test_a_cell_without_the_group_is_refused(tmp_path: Path) -> None:
    cells = {(0, 0): json.dumps({"Portland": 0.3})}
    with pytest.raises(StepError, match="without a 'Seattle' group"):
        _script().load(_step(tmp_path, "knockout", cells))


def test_an_excluded_row_is_skipped_and_not_counted(tmp_path: Path) -> None:
    """An excluded measurement (``value: null``, ``eligible: false``) leaves
    its cell's mean to the eligible rows, and ``n`` counts only those. A cell
    with no eligible row has no value and ``n`` 0."""
    cells = _cells({(0, 0): 0.25, (0, 1): 0.5})
    step = _step(tmp_path, "knockout", cells, excluded=((0, 0), (1, 0)))
    table, _ = _script().load(step)
    got = {(r.layer, r.position): (r.p_ablated, r.n) for r in table.itertuples()}
    assert got[(0, 0)] == (0.25, 1)
    assert got[(0, 1)] == (0.5, 1)
    p_missing, n_missing = got[(1, 0)]
    assert math.isnan(p_missing)
    assert n_missing == 0


def test_a_cross_entropy_run_tree_is_refused(tmp_path: Path) -> None:
    """A run tree from the earlier document holds ``ce_ablated.json`` and no
    ``p_ablated.json``; the script names the missing table instead of drawing
    cross-entropies as probabilities."""
    step = _step(tmp_path, "knockout", _cells({(0, 0): 0.5}))
    (step / "p_ablated.json").rename(step / "ce_ablated.json")
    with pytest.raises(ProtocolError, match="p_ablated.json"):
        _script().load(step)
