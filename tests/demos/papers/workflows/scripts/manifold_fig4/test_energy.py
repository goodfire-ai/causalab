"""``energy.py``: the reported statistics and the paper's bands."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from tests.demos.papers._scripts import load_script
from tests.demos.papers.workflows.scripts.manifold_fig4._fixtures import (
    DAYS,
    behavior_tables,
    steer_tables,
)

pytestmark = pytest.mark.numerical_unit

energy = load_script("manifold_fig4", "energy")
behavior = load_script("manifold_fig4", "behavior_manifold")
PAIRS = [f"{a}_{b}" for a, b in itertools.combinations(DAYS, 2)]


def _run(tmp_path: Path, pairs: list[str], seed: int) -> tuple[dict, dict]:
    inputs: dict[str, object] = {
        **steer_tables(tmp_path, pairs, np.random.default_rng(seed)),
        "pair_axis": "axes.pair",
        "step_axis": "axes.path",
    }
    fitted = behavior_tables(tmp_path)
    inputs["behavior_centroids"] = fitted["centroids"]
    inputs["behavior_spline"] = fitted["spline"]
    outputs = {
        "trajectory": tmp_path / "trajectory.json",
        "energy": tmp_path / "energy.json",
        "summary": tmp_path / "summary.json",
    }
    energy.main(inputs, outputs)
    return inputs, outputs


def test_the_statistics_run_over_the_centroid_pairs(tmp_path: Path) -> None:
    """The reported mean, standard error and paired t test run over the 21
    centroid pairs, one value per pair, as App. A.7 reports them."""
    from scipy import stats

    from causalab.io.step_io import read_table, read_values

    _, outputs = _run(tmp_path, PAIRS, seed=2)
    summary = read_values(outputs["summary"])
    rows = read_table(outputs["energy"])
    assert [r["pair"] for r in rows if r["method"] == "manifold"] == PAIRS
    by = {(r["method"], r["pair"]): r["energy_sum"] for r in rows}
    man = np.array([by[("manifold", p)] for p in PAIRS])
    lin = np.array([by[("linear", p)] for p in PAIRS])
    assert summary["n_pairs"] == 21
    assert summary["manifold_energy"] == pytest.approx(man.mean())
    assert summary["manifold_energy_se"] == pytest.approx(man.std(ddof=1) / np.sqrt(21))
    assert summary["linear_energy_se"] == pytest.approx(lin.std(ddof=1) / np.sqrt(21))
    test = stats.ttest_rel(lin, man)
    assert summary["paired_t"] == pytest.approx(test.statistic)
    assert summary["paired_p"] == pytest.approx(test.pvalue)
    dense = np.array([r["energy_sum_dense"] for r in rows if r["method"] == "manifold"])
    assert summary["manifold_energy_dense"] == pytest.approx(dense.mean())


@pytest.mark.parametrize(
    "pairs", [["Tuesday_Friday", "Friday_Tuesday"], ["Tuesday_Friday", "extra_00"]]
)
def test_a_pair_counts_once(tmp_path: Path, pairs: list[str]) -> None:
    """A pair steered in both orientations, or a path that joins no two
    centroids, would enter the statistic twice or as a pair it is not, so
    the step refuses it."""
    from causalab.io.step_io import StepError

    with pytest.raises(StepError):
        _run(tmp_path, pairs, seed=4)


def test_the_bands_are_the_papers(tmp_path: Path) -> None:
    """The trajectory carries the half-width of each band the paper draws
    (path_visualization.py): the standard deviation over prompts (ddof 1)
    per day, and ``sqrt`` of the summed day variances for ``other``."""
    from causalab.io.step_io import read_table

    inputs, outputs = _run(tmp_path, ["Tuesday_Friday"], seed=3)
    probs = behavior.probabilities_by_example(
        [
            read_table(Path(str(inputs[f"manifold_{s}"])))
            for s in ("space", "bare", "lower")
        ],
        key=("example_id", "axes.pair", "axes.path"),
    )
    rows = [r for r in read_table(outputs["trajectory"]) if r["method"] == "manifold"]
    assert len(rows) == 3
    for row in rows:
        cube = np.stack([probs[(e, row["pair"], row["step"])] for e in ("0", "1")])
        std = cube.std(axis=0, ddof=1)
        for j, day in enumerate(DAYS):
            assert row[f"band_{day}"] == pytest.approx(std[j])
        assert row["band_other"] == pytest.approx(np.sqrt((std[:7] ** 2).sum()))
        assert row["other"] == pytest.approx(cube[:, 7].mean())
