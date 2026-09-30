import math

import pytest
import torch

from causalab.measurement.analysis.stability import (
    compare_gates,
    compare_subspace_groups,
    compare_subspaces,
    gate_selection_frequency,
    subspace_dispersion,
)

pytestmark = pytest.mark.unit


def test_basis_rotation_and_sign_do_not_change_subspace():
    q = torch.eye(4, dtype=torch.float64)[:, :2]
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    result = compare_subspaces(q, q @ rotation)
    assert result["overlap"] == pytest.approx(1)
    assert result["projector_frobenius_distance"] == pytest.approx(0)
    assert result["principal_angles_radians"] == pytest.approx([0, 0])
    spread = subspace_dispersion([[q], [-q], [q @ rotation]])
    assert spread["mean_squared_projector_distance"] == pytest.approx(0)
    assert spread["independent_units"] == 3
    assert spread["pair_distances_are_independent"] is False


def test_orthogonal_and_partial_overlap_oracles():
    basis = torch.eye(4, dtype=torch.float64)
    before = basis[:, :2]
    orthogonal = compare_subspaces(before, basis[:, 2:])
    assert orthogonal["overlap"] == 0
    assert orthogonal["projector_frobenius_distance"] == pytest.approx(2)
    assert orthogonal["principal_angles_radians"] == pytest.approx([math.pi / 2] * 2)
    partial = compare_subspaces(before, basis[:, 1:3])
    assert partial["overlap"] == pytest.approx(0.5)


def test_small_angles_and_variance_survive_nearly_equal_overlaps():
    angle = 1e-10
    q = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    p = torch.tensor([[math.cos(angle)], [math.sin(angle)]], dtype=torch.float64)
    result = compare_subspaces(q, p)
    assert result["principal_angles_radians"] == pytest.approx(
        [angle], rel=1e-6, abs=1e-20
    )
    assert result["projector_frobenius_distance"] == pytest.approx(
        math.sqrt(2) * math.sin(angle), rel=1e-6, abs=1e-20
    )
    assert result["normalized_projector_distance"] == pytest.approx(
        math.sin(angle), rel=1e-6, abs=1e-20
    )
    spread = subspace_dispersion([[q], [p]])
    assert spread["mean_squared_projector_distance"] == pytest.approx(
        2 * math.sin(angle) ** 2, rel=1e-6, abs=1e-30
    )
    means = compare_subspace_groups([[q, q]], [[q, p]])
    assert means["mean_projector_distance"] == pytest.approx(
        math.sin(angle) / math.sqrt(2), rel=1e-6, abs=1e-20
    )


def test_identical_dense_bases_have_zero_drift_and_dispersion():
    generator = torch.Generator().manual_seed(37)
    q = torch.randn(32, 3, generator=generator, dtype=torch.float64)
    result = compare_subspaces(q, q.clone())
    assert result["projector_frobenius_distance"] == 0
    assert result["normalized_projector_distance"] == 0
    assert result["principal_angles_radians"] == [0, 0, 0]
    assert (
        subspace_dispersion([[q], [q.clone()]])["mean_squared_projector_distance"] == 0
    )


def test_mixed_small_and_large_principal_angles_keep_their_order():
    angles = [1e-10, 1.2]
    q = torch.eye(4, dtype=torch.float64)[:, :2]
    p = torch.zeros_like(q)
    for column, angle in enumerate(angles):
        p[column, column] = math.cos(angle)
        p[column + 2, column] = math.sin(angle)
    result = compare_subspaces(q, p)
    assert result["principal_angles_radians"] == pytest.approx(
        angles, rel=1e-6, abs=1e-20
    )


def test_mean_projection_distance_matches_explicit_dense_operators():
    generator = torch.Generator().manual_seed(51)
    bases = [
        torch.linalg.qr(torch.randn(9, 2, generator=generator, dtype=torch.float64)).Q
        for _ in range(8)
    ]
    before, after = [bases[:2], bases[2:4]], [bases[4:6], bases[6:]]
    result = compare_subspace_groups(before, after)
    operators = [q @ q.T for q in bases]
    expected_shift = (sum(operators[:4]) / 4 - sum(operators[4:]) / 4).norm()
    expected_spread = (
        ((operators[0] + operators[1] - operators[2] - operators[3]) / 2).square().sum()
    )
    assert result["mean_projector_distance"] == pytest.approx(
        float(expected_shift), rel=1e-12
    )
    assert result["before"]["mean_squared_projector_distance"] == pytest.approx(
        float(expected_spread), rel=1e-12
    )


def test_seed_mean_projectors_do_not_average_arbitrary_basis_coordinates():
    q = torch.eye(4, dtype=torch.float64)[:, :2]
    p = torch.eye(4, dtype=torch.float64)[:, 2:]
    before = [[q, -q], [q, -q]]
    after = [[q, -q], [p, -p]]
    result = compare_subspace_groups(before, after)
    assert result["before"]["mean_squared_projector_distance"] == pytest.approx(0)
    assert result["after"]["mean_squared_projector_distance"] == pytest.approx(4)
    assert result["dispersion_ratio"] is None
    assert result["mean_projector_distance"] == pytest.approx(1)


def test_rank_and_orthogonality_are_separate_diagnostics():
    q = torch.eye(4, dtype=torch.float64)[:, :2]
    result = compare_subspaces(q, 2 * q)
    assert result["overlap"] == pytest.approx(1)
    assert result["after"]["orthogonality_max_abs"] == 3
    with pytest.raises(ValueError, match="rank deficient"):
        compare_subspaces(q, torch.zeros_like(q))


def test_temperature_changes_gate_confidence_not_mask_membership():
    theta = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    result = compare_gates(theta, theta, before_temperature=1, after_temperature=0.01)
    assert result["mask_jaccard"] == 1
    assert result["before_mask_size"] == result["after_mask_size"] == 1
    assert result["probability_rms_difference"] > 0
    assert result["before_decisive_fraction"] == 0
    assert result["after_decisive_fraction"] == pytest.approx(2 / 3)
    assert gate_selection_frequency([theta, -theta]) == [0.5, 0, 0.5]


def test_empty_mask_policy_and_invalid_temperature():
    theta = -torch.ones(3)
    assert (
        compare_gates(theta, theta, before_temperature=1, after_temperature=1)[
            "mask_jaccard"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="temperature"):
        compare_gates(theta, theta, before_temperature=0, after_temperature=1)
