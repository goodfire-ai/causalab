"""``causalab.analysis.subspace_angles`` against hand-built bases.

The oracle is geometry in R^4: the xy-plane against itself (0°), against the
xz-plane (one shared direction, angles 0° and 90°), and against the zw-plane
(90° twice); a masked trajectory entry restricted to its kept columns; and the
PCA-block truncation that compares a rank-s span to the top-s block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from causalab.analysis import subspace_angles
from causalab.io.step_io import StepError
from causalab.io.tables import read_table
from tests.step_scripts import run_step

pytestmark = pytest.mark.numerical_unit

E = torch.eye(4, dtype=torch.float32)
XY, XZ, ZW = (E[:, :2].contiguous(), E[:, [0, 2]].contiguous(), E[:, 2:].contiguous())


def _bundle(path: Path, tensors: dict, table: dict | None = None, **meta) -> Path:
    metadata = {**meta}
    if table is not None:
        metadata["entries"] = json.dumps(table, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata
    )
    return path


def _angles(tmp_path: Path, tag: str, **inputs):
    out = tmp_path / tag / "angles.json"
    run_step(subspace_angles, inputs, {"angles": out})
    return read_table(out)


def test_the_three_oracles(tmp_path: Path) -> None:
    a = _bundle(
        tmp_path / "a.safetensors",
        {"weight[k=xy]": XY, "weight[k=xz]": XZ, "weight[k=zw]": ZW},
    )
    b = _bundle(tmp_path / "b.safetensors", {"weight": XY})
    rows = {r["a"]: r for r in _angles(tmp_path, "o", a=a, b=b)}
    assert rows["weight[k=xy]"]["max_deg"] == pytest.approx(0.0, abs=1e-6)
    assert rows["weight[k=xy]"]["overlap"] == pytest.approx(1.0)
    assert rows["weight[k=xz]"]["max_deg"] == pytest.approx(90.0)
    assert rows["weight[k=xz]"]["mean_deg"] == pytest.approx(45.0)
    assert rows["weight[k=xz]"]["overlap"] == pytest.approx(0.5)
    assert rows["weight[k=zw]"]["max_deg"] == pytest.approx(90.0)
    assert rows["weight[k=zw]"]["overlap"] == pytest.approx(0.0, abs=1e-12)
    assert all(r["dim_a"] == 2 and r["dim_b"] == 2 for r in rows.values())


def test_angles_are_a_property_of_the_span_not_the_basis(tmp_path: Path) -> None:
    """A rotation inside the span, or a non-orthonormal basis of it, changes
    nothing — the point of measuring projectors, not coordinates."""
    rotated = XY @ torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    skewed = XY @ torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    a = _bundle(
        tmp_path / "a.safetensors", {"weight[s=r]": rotated, "weight[s=k]": skewed}
    )
    b = _bundle(tmp_path / "b.safetensors", {"weight": XY})
    for row in _angles(tmp_path, "s", a=a, b=b):
        assert row["max_deg"] == pytest.approx(0.0, abs=1e-5)


def test_a_mask_restricts_a_trajectory_entry_to_its_kept_columns(
    tmp_path: Path,
) -> None:
    """The joint fit's ``span(Q_S)``: the rotation at a step, cut to the gate's
    kept units at that step (``θ > 0`` for a sigmoid gate)."""
    traj = _bundle(
        tmp_path / "trajectory.safetensors",
        {
            "weight[featurizer=rot,step=10]": E,
            "theta[featurizer=gate,step=10]": torch.tensor([1.0, -1.0, 2.0, -3.0]),
        },
        table={
            "weight[featurizer=rot,step=10]": {
                "slot": "weight",
                "coords": {"featurizer": "rot", "step": 10},
            },
            "theta[featurizer=gate,step=10]": {
                "slot": "theta",
                "coords": {"featurizer": "gate", "step": 10},
            },
        },
    )
    b = _bundle(tmp_path / "b.safetensors", {"weight": XZ})
    (row,) = _angles(tmp_path, "m", a=traj, b=b, a_mask=traj, match=["step"])
    # kept units {0, 2} = the xz-plane itself
    assert row["dim_a"] == 2 and row["max_deg"] == pytest.approx(0.0, abs=1e-6)
    b2 = _bundle(tmp_path / "b2.safetensors", {"weight": XY})
    (row,) = _angles(tmp_path, "m2", a=traj, b=b2, a_mask=traj, match=["step"])
    assert row["mean_deg"] == pytest.approx(45.0)


def test_truncate_b_compares_to_the_top_block_of_the_second_basis(
    tmp_path: Path,
) -> None:
    a = _bundle(tmp_path / "a.safetensors", {"weight": XZ})
    pca = _bundle(tmp_path / "pca.safetensors", {"weight": E})  # "top-4 PCs" = all
    (full,) = _angles(tmp_path, "f", a=a, b=pca)
    assert full["dim_b"] == 4 and full["max_deg"] == pytest.approx(0.0, abs=1e-6)
    (top,) = _angles(tmp_path, "t", a=a, b=pca, truncate_b=True)
    assert top["dim_b"] == 2 and top["mean_deg"] == pytest.approx(45.0)


def test_refusals(tmp_path: Path) -> None:
    a = _bundle(tmp_path / "a.safetensors", {"weight": XY})
    with pytest.raises(StepError, match="does not exist"):
        _angles(tmp_path, "r1", a=a, b=tmp_path / "missing.safetensors")
    thetas = _bundle(tmp_path / "t.safetensors", {"theta": torch.ones(4)})
    with pytest.raises(StepError, match="no 'weight'"):
        _angles(tmp_path, "r2", a=a, b=thetas)
    other_space = _bundle(tmp_path / "o.safetensors", {"weight": torch.eye(3)[:, :2]})
    with pytest.raises(StepError, match="one ambient space"):
        _angles(tmp_path, "r3", a=a, b=other_space)
    with pytest.raises(StepError, match="'match'"):
        _angles(tmp_path, "r4", a=a, b=a, match="step")
