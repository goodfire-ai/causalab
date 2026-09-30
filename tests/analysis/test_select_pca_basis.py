"""The localizer starts from the intended position's training-only PCA."""

import pytest
import torch

from causalab.analysis import pca_by_position, select_pca_basis
from causalab.io.step_io import StepError, read_tensor, read_tensor_with_identity


@pytest.mark.numerical_unit
def test_selects_exact_position_without_refitting(tmp_path):
    generator = torch.Generator().manual_seed(5)
    acts = torch.randn(90, 2, 72, generator=generator)
    paths = {
        name: tmp_path / f"{name}.safetensors"
        for name in ("weight", "mean", "coordinates")
    }
    paths["spectrum"] = tmp_path / "spectrum.json"
    pca_by_position.main({"acts": acts, "train_rows": list(range(80)), "k": 64}, paths)
    out = tmp_path / "selected.safetensors"
    select_pca_basis.main({"weight": paths["weight"], "position": 1}, {"weight": out})
    selected, identity = read_tensor_with_identity(out)
    torch.testing.assert_close(selected, read_tensor(paths["weight"])[1])
    assert selected.shape == (72, 64)
    assert identity["k"] == "64"


@pytest.mark.unit
@pytest.mark.parametrize("position", [-1, 2, True, 0.5])
def test_refuses_ambiguous_position(tmp_path, position):
    with pytest.raises(StepError, match="valid position"):
        select_pca_basis.main(
            {"weight": torch.eye(3).repeat(2, 1, 1), "position": position},
            {"weight": tmp_path / "out.safetensors"},
        )
