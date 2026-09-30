"""Checks specific to the MCQA DAS packages (``demos/papers/``).

``mcqa_symbol`` and ``mcqa_pointer`` each ship
``workflows/scripts/<name>/learned_rank.py``, which turns a DBM-DAS boundary
gate into the ``k`` of the matched random control. The rank it writes must be
the number of columns the gate's own eval mask keeps (``Gate.hard_mask``), or
the control would be matched to a different rank than the one the fit
applied. The figure script ``iia_figure.py`` reads the ``full_patch`` step
per layer beside the DAS steps. Each page also shows the DBM-DAS panel as a markdown table, whose
cells must be the values of the committed ``iia_plotted.json``. A package that
is dropped takes its script and its page with it, so the cases below run over
the packages present.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from causalab.io.step_io import StepError
from causalab.neural.shared.featurizers.gate import Gate

pytestmark = pytest.mark.unit

PAPERS = Path(__file__).resolve().parents[2] / "demos/papers"
SCRIPTS = PAPERS / "workflows/scripts"
PACKAGES = [
    name
    for name in ("mcqa_symbol", "mcqa_pointer")
    if (SCRIPTS / name / "learned_rank.py").is_file()
]


def _load(name: str, script: str = "learned_rank"):
    spec = importlib.util.spec_from_file_location(
        f"{name}_{script}", SCRIPTS / name / f"{script}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", PACKAGES)
@pytest.mark.parametrize(
    "theta", [0.0, 1 / 128, 0.44313088059425354, 0.490583598613739, 0.5, 1.0]
)
def test_learned_rank_is_the_gate_hard_mask_size(name: str, theta: float) -> None:
    """The two recorded thetas of the 2026-09-28 run (57 and 63 of 128), the
    exact column boundaries and both ends agree with the gate's eval mask."""
    gate = Gate(128, parametrization="boundary", init=theta)
    kept = int(gate.hard_mask().sum())
    assert _load(name).learned_rank(theta, 128) == kept


@pytest.mark.parametrize("name", PACKAGES)
def test_learned_rank_refuses_a_theta_outside_the_unit_interval(name: str) -> None:
    with pytest.raises(StepError):
        _load(name).learned_rank(1.5, 128)


def _dbm_table(page: Path) -> dict[str, list[str]]:
    """The rows of the markdown table in the page's DBM-DAS section, keyed by
    the first cell, header and rule excluded."""
    section = page.read_text().split("## DBM-DAS: change these lines", 1)[1]
    section = section.split("\n## ", 1)[0]
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
        if line.startswith("|")
    ]
    assert len(rows) > 2, f"{page.name}: no table in the DBM-DAS section"
    assert all(re.fullmatch(r"-+", cell) for cell in rows[1]), rows[1]
    return {row[0]: row[1:] for row in rows[2:]}


@pytest.mark.parametrize("name", PACKAGES)
def test_the_dbm_table_quotes_the_plotted_values(name: str) -> None:
    """Every cell of the DBM-DAS table is the value ``iia_figure.py`` wrote to
    ``iia_plotted.json``, at the three decimals the page prints: the rank,
    the held-out IIA at the chosen layer and the mean of the random
    subspaces of that rank."""
    drawn = json.loads(
        (PAPERS / f"artifacts/figures/{name}/iia_plotted.json").read_text()
    )
    das = next(r for r in drawn["das"] if r["layer"] == drawn["chosen_layer"])
    dbm = drawn["dbm"]
    assert _dbm_table(PAPERS / f"{name}.md") == {
        "DBM-DAS": [
            f"{dbm['learned_rank']} of {dbm['width']}, learned",
            f"{dbm['test_iia']:.3f}",
            f"{drawn['control_dbm']['test_iia_mean']:.3f}",
        ],
        "DAS": [
            f"{drawn['control']['k']}",
            f"{das['test_iia']:.3f}",
            f"{drawn['control']['test_iia_mean']:.3f}",
        ],
    }


def _table(step: Path, axis: str | None, cells: dict[int, float]) -> None:
    """A step directory with a sidecar naming ``axis`` and a per-example
    ``iia.json``: two pairs per point, 1.0 and ``2 * value - 1``, so that the
    mean over the pairs is ``value``."""
    step.mkdir(parents=True)
    (step / "_step.json").write_text(json.dumps({"axes": [axis] if axis else []}))
    rows = []
    for point, value in cells.items():
        for example, iia in enumerate((1.0, 2 * value - 1.0)):
            rows.append({"example_id": str(example), "value": iia})
            if axis:
                rows[-1][axis] = point
    (step / "iia.json").write_text(json.dumps(rows))


def _run_tree(tmp_path: Path, full_layers: range) -> Path:
    """A fabricated run tree of every step ``iia_figure.values`` reads."""
    run = tmp_path / "run"
    layers, seeds = "sites.target.layers", "featurizers.rot.seed"
    _table(run / "apply", layers, {layer: 0.5 + layer / 100 for layer in range(28)})
    _table(run / "apply_train", layers, {layer: 1.0 for layer in range(28)})
    _table(run / "full_patch", layers, {layer: 0.75 for layer in full_layers})
    _table(run / "dbm_apply", None, {0: 0.75})
    _table(run / "control", seeds, {0: 0.5, 1: 0.5, 2: 0.5})
    _table(run / "control_dbm", seeds, {0: 0.5, 1: 0.5, 2: 0.5})
    for name in ("clean_base", "clean_cf", "unpatched_label"):
        (run / "clean").mkdir(parents=True, exist_ok=True)
        (run / "clean" / f"{name}.json").write_text(
            json.dumps([{"example_id": "0", "value": 1.0}])
        )
    for step, payload in (
        ("best", {"best_layer": 27}),
        ("rank", {"learned_rank": 64, "theta": 0.5, "width": 128}),
    ):
        (run / step).mkdir()
        (run / step / "values.json").write_text(json.dumps(payload))
    return run


@pytest.mark.parametrize("name", PACKAGES)
def test_the_figure_reads_the_full_patch_per_layer(name: str, tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    drawn = _load(name, "iia_figure").values(_run_tree(tmp_path, range(28)))
    assert drawn["full_patch"] == [
        {"layer": layer, "test_iia": 0.75} for layer in range(28)
    ]
    assert drawn["chosen_layer"] == 27


@pytest.mark.parametrize("name", PACKAGES)
def test_the_figure_refuses_a_full_patch_with_missing_layers(
    name: str, tmp_path: Path
) -> None:
    pytest.importorskip("pandas")
    with pytest.raises(StepError, match="full_patch"):
        _load(name, "iia_figure").values(_run_tree(tmp_path, range(27)))
