"""Projection retains only the selected component, including for nonzero means."""

import pytest
import torch

from causalab.analysis.project_subspace import main, project
from causalab.io.step_io import StepError, read_tensor, write_tensor


@pytest.mark.numerical_unit
def test_reconstruction_omits_mean_and_complement(tmp_path):
    acts = torch.tensor([[[5.0, 20.0, 8.0], [7.0, 30.0, 9.0]]])
    weight = torch.tensor([[1.0], [0.0], [0.0]])
    act_path, weight_path = (
        tmp_path / "acts.safetensors",
        tmp_path / "weight.safetensors",
    )
    write_tensor(act_path, acts, slot="acts")
    write_tensor(weight_path, weight, slot="weight")
    outputs = {
        name: tmp_path / f"{name}.safetensors"
        for name in ("coordinates", "reconstructed")
    }
    main({"acts": act_path, "weight": weight_path}, outputs)
    torch.testing.assert_close(
        read_tensor(outputs["coordinates"]),
        torch.tensor([[[5.0], [7.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        read_tensor(outputs["reconstructed"]),
        torch.tensor([[[5.0, 0.0, 0.0], [7.0, 0.0, 0.0]]], dtype=torch.float64),
    )


@pytest.mark.property
def test_rotated_basis_projection_is_idempotent_and_matches_featurizer():
    from causalab.neural.shared.featurizers import LoadedLinear

    generator = torch.Generator().manual_seed(8)
    acts = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(5, 2, generator=generator, dtype=torch.float64))
    coords, reconstructed = project(acts, q)
    torch.testing.assert_close(project(reconstructed, q)[1], reconstructed)
    torch.testing.assert_close((acts - reconstructed) @ q, torch.zeros_like(coords))
    stage = LoadedLinear("subspace", q)
    torch.testing.assert_close(
        reconstructed, stage.inverse(stage.featurize(acts)[0], None)
    )
    torch.testing.assert_close(project(acts, torch.eye(5))[1], acts)


@pytest.mark.unit
@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(3, 1),
        torch.ones(2, 1),
        torch.empty(3, 0),
        torch.full((3, 1), float("nan")),
    ],
)
def test_invalid_basis_refused(weight):
    with pytest.raises(StepError):
        project(torch.ones(2, 3), weight)


@pytest.mark.property
def test_mask_selects_columns_and_empty_mask_is_zero():
    generator = torch.Generator().manual_seed(4)
    q, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator, dtype=torch.float64))
    acts = torch.randn(7, 1, 8, generator=generator, dtype=torch.float64)
    coords, reconstructed = project(acts, q, [0, 1, 0, 1])
    torch.testing.assert_close(coords, acts @ q[:, [1, 3]])
    torch.testing.assert_close(
        reconstructed,
        acts
        @ q
        @ torch.diag(torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64))
        @ q.T,
    )
    empty, zero = project(acts, q, [0, 0, 0, 0])
    assert empty.shape == (7, 1, 0)
    torch.testing.assert_close(zero, torch.zeros_like(acts))


@pytest.mark.unit
@pytest.mark.parametrize("mask", [[1], [1, 0.5], [1, float("nan")]])
def test_mask_must_be_binary_and_match_basis(mask):
    with pytest.raises(StepError, match="binary value"):
        project(torch.ones(3, 2), torch.eye(2), mask)
