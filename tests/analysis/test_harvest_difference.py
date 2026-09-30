"""``causalab.analysis.harvest_difference`` — the step type earning its keep.

This closes a real stub: "harvest activations on two contrasting corpora and
subtract the means" is the direction half of every steering experiment, and it
was expressible nowhere before. Not an intervention specification (it touches
no network through the intervention vocabulary), and not a registry op (a
one-off
reduction could not be admitted by pull request).

The oracle is deliberate arithmetic rather than a pinned run: positive rows mean
to (3, 0), negative rows to (1, 0), so the difference is (2, 0) and the
normalized direction is (1, 0).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from causalab.io.tables import read_table
from causalab.analysis import harvest_difference
from causalab.io.step_io import StepError, read_tensor, write_tensor
from tests.step_scripts import run_step

pytestmark = pytest.mark.numerical_unit

POSITIVE = torch.tensor([[2.0, 0.0], [4.0, 0.0]])  # mean (3, 0)
NEGATIVE = torch.tensor([[0.0, 0.0], [2.0, 0.0]])  # mean (1, 0)


def _run(tmp_path: Path, tag: str = "a", **extra):
    out = tmp_path / tag
    out.mkdir(parents=True, exist_ok=True)
    run_step(
        harvest_difference,
        {"positive": POSITIVE, "negative": NEGATIVE, **extra},
        {"weight": out / "direction.safetensors", "stats": out / "stats.json"},
    )
    return read_tensor(out / "direction.safetensors"), read_table(out / "stats.json")


def test_the_difference_of_means_matches_the_oracle(tmp_path):
    direction, stats = _run(tmp_path)
    assert torch.allclose(direction, torch.tensor([2.0, 0.0]), atol=1e-6)
    assert [row["value"] for row in stats] == pytest.approx([2.0, 0.0])
    assert [row["dim"] for row in stats] == [0, 1]


def test_normalize_gives_a_unit_direction(tmp_path):
    direction, _ = _run(tmp_path, tag="norm", normalize=True)
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-6)
    assert float(torch.linalg.vector_norm(direction)) == pytest.approx(1.0)


def test_leading_dimensions_are_flattened(tmp_path):
    """[examples, positions, d] and [rows, d] must mean the same direction."""
    out = tmp_path / "nested"
    out.mkdir()
    run_step(
        harvest_difference,
        {
            "positive": POSITIVE.reshape(2, 1, 2),
            "negative": NEGATIVE.reshape(2, 1, 2),
        },
        {"weight": out / "d.safetensors"},
    )
    assert torch.allclose(
        read_tensor(out / "d.safetensors"), torch.tensor([2.0, 0.0]), atol=1e-6
    )


def test_a_mean_reduced_harvest_is_one_row(tmp_path):
    """A save-time `reduce: mean` writes (d,), which is a single row — not an
    error, and not a width mismatch."""
    out = tmp_path / "reduced"
    out.mkdir()
    run_step(
        harvest_difference,
        {"positive": torch.tensor([3.0, 0.0]), "negative": torch.tensor([1.0, 0.0])},
        {"weight": out / "d.safetensors"},
    )
    assert torch.allclose(
        read_tensor(out / "d.safetensors"), torch.tensor([2.0, 0.0]), atol=1e-6
    )


def test_mismatched_widths_are_refused(tmp_path):
    out = tmp_path / "bad"
    out.mkdir()
    with pytest.raises(StepError) as err:
        run_step(
            harvest_difference,
            {"positive": POSITIVE, "negative": torch.tensor([[1.0, 2.0, 3.0]])},
            {"weight": out / "d.safetensors"},
        )
    assert "different widths" in str(err.value)


def test_normalizing_a_zero_direction_is_refused(tmp_path):
    """A zero direction would steer nothing while looking like it steered."""
    out = tmp_path / "zero"
    out.mkdir()
    with pytest.raises(StepError) as err:
        run_step(
            harvest_difference,
            {"positive": POSITIVE, "negative": POSITIVE, "normalize": True},
            {"weight": out / "d.safetensors"},
        )
    assert "no direction to normalize" in str(err.value)


def test_tensor_inputs_may_arrive_as_paths(tmp_path):
    """The runner hands over a path when the document used no selector."""
    write_tensor(tmp_path / "pos.safetensors", POSITIVE, slot="acts")
    write_tensor(tmp_path / "neg.safetensors", NEGATIVE, slot="acts")
    out = tmp_path / "viapath"
    out.mkdir()
    run_step(
        harvest_difference,
        {
            "positive": tmp_path / "pos.safetensors",
            "negative": tmp_path / "neg.safetensors",
        },
        {"weight": out / "d.safetensors"},
    )
    assert torch.allclose(
        read_tensor(out / "d.safetensors"), torch.tensor([2.0, 0.0]), atol=1e-6
    )
