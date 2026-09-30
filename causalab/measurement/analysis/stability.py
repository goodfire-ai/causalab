"""Invariant comparisons for learned subspaces and temperature-aware gates.

Pairwise distances are descriptive: the independent sampling unit remains a fit
or fitting seed. No tolerance or mechanism-effectiveness verdict is inferred.
"""

from __future__ import annotations

from typing import Any


class MissingGateObservationSpecError(ValueError):
    """A gate's saved semantic key has no corresponding entry metadata."""

    def __init__(self, observation: str) -> None:
        self.observation = observation
        super().__init__(
            f"exact gate observation specification required for {observation!r}"
        )


class UnsupportedGateObservationError(ValueError):
    """A saved gate cannot use the measurement harness's sigmoid statistics."""

    def __init__(self, observation: str, parametrization: str) -> None:
        self.observation = observation
        self.parametrization = parametrization
        super().__init__(
            f"gate observation {observation!r} supports only sigmoid; "
            f"saved parametrization is {parametrization!r}"
        )


def require_sigmoid_gate(observation: str, parametrization: str) -> None:
    if parametrization != "sigmoid":
        raise UnsupportedGateObservationError(observation, parametrization)


def _basis(value: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    value = value.to(torch.float64)
    if value.ndim != 2 or not 0 < value.shape[1] <= value.shape[0]:
        raise ValueError("a subspace must be a nonempty (width, rank) basis")
    if not torch.isfinite(value).all():
        raise ValueError("subspace contains nonfinite values")
    rank = int(torch.linalg.matrix_rank(value))
    if rank != value.shape[1]:
        raise ValueError("subspace basis is rank deficient")
    identity = torch.eye(rank, dtype=value.dtype, device=value.device)
    deviation = float((value.T @ value - identity).abs().max())
    return torch.linalg.qr(value, mode="reduced").Q, {
        "width": value.shape[0],
        "rank": rank,
        "orthogonality_max_abs": deviation,
    }


def compare_subspaces(before: Any, after: Any) -> dict[str, Any]:
    import torch

    qb, info_b = _basis(before)
    qa, info_a = _basis(after)
    if qb.shape != qa.shape:
        raise ValueError("subspace comparison currently requires equal width and rank")
    singular = torch.linalg.svdvals(qb.T @ qa).clamp(0, 1)
    overlap = float(singular.square().mean())
    if torch.equal(before, after) or torch.equal(qb, qa):
        squared, overlap = 0.0, 1.0
        angles = torch.zeros_like(singular)
    else:
        squared = min(2 * qb.shape[1], _factor_distance_squared(qb, qa))
        # acos loses small angles when cos(theta) rounds to one. The residual
        # singular values give sin(theta), in the opposite sorted order.
        sine = torch.linalg.svdvals(qa - qb @ (qb.T @ qa)).flip(0).clamp(0, 1)
        angles = torch.where(
            singular >= 0.5**0.5, torch.asin(sine), torch.acos(singular)
        )
    return {
        "before": info_b,
        "after": info_a,
        "overlap": overlap,
        "principal_angles_radians": angles.tolist(),
        "projector_frobenius_distance": squared**0.5,
        "normalized_projector_distance": (squared / (2 * qb.shape[1])) ** 0.5,
    }


def _projector_factor(bases: list[Any]) -> Any:
    import torch

    return torch.cat([_basis(q)[0] for q in bases], dim=1) / len(bases) ** 0.5


def _factor_distance_squared(left: Any, right: Any) -> float:
    import torch

    # Factors may represent a mean projection operator rather than a subspace.
    if torch.equal(left, right):
        return 0.0
    # Compare operators in the shared thin QR coordinates. Subtracting their
    # squared Gram norms loses distances below sqrt(machine epsilon). Here the
    # small off-diagonal differences survive, without forming wide projectors.
    coordinates = torch.linalg.qr(torch.cat((left, right), dim=1), mode="r").R
    a, b = coordinates[:, : left.shape[1]], coordinates[:, left.shape[1] :]
    return float((a @ a.T - b @ b.T).square().sum())


def subspace_dispersion(groups: list[list[Any]]) -> dict[str, Any]:
    """Spread of projection operators; one group per independent statistical unit.

    With one basis per group this compares repeated fits. With repeated fits per
    seed it compares seed-mean projection operators, avoiding arbitrary basis signs
    and rotations. The average squared pair distance is a dispersion, not n(n-1)/2
    independent measurements and not coordinate-wise sample variance.
    """
    if not groups or any(not group for group in groups):
        raise ValueError("subspace dispersion needs nonempty groups of fits")
    factors = [_projector_factor(group) for group in groups]
    widths = {factor.shape[0] for factor in factors}
    ranks = {basis.shape[1] for group in groups for basis in group}
    if len(widths) != 1 or len(ranks) != 1:
        raise ValueError("subspace dispersion requires matching widths and ranks")
    distances = [
        _factor_distance_squared(factors[i], factors[j])
        for i in range(len(factors))
        for j in range(i + 1, len(factors))
    ]
    return {
        "independent_units": len(groups),
        "fits_per_unit": [len(g) for g in groups],
        "mean_squared_projector_distance": sum(distances) / len(distances)
        if distances
        else None,
        "pair_count": len(distances),
        "pair_distances_are_independent": False,
    }


def compare_subspace_groups(
    before: list[list[Any]], after: list[list[Any]]
) -> dict[str, Any]:
    if len(before) != len(after) or any(
        len(b) != len(a) for b, a in zip(before, after)
    ):
        raise ValueError("subspace groups must be paired by seed/repeat")
    b, a = subspace_dispersion(before), subspace_dispersion(after)
    bv, av = b["mean_squared_projector_distance"], a["mean_squared_projector_distance"]
    return {
        "before": b,
        "after": a,
        "dispersion_change": av - bv if bv is not None and av is not None else None,
        "dispersion_ratio": av / bv if bv not in (0, None) else None,
        "mean_projector_distance": _factor_distance_squared(
            _projector_factor([q for group in before for q in group]),
            _projector_factor([q for group in after for q in group]),
        )
        ** 0.5,
    }


def gate_probabilities(theta: Any, temperature: float) -> Any:
    import math
    import torch

    if (
        isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("gate temperature must be finite and positive")
    value = theta.to(torch.float64)
    if not value.numel() or not torch.isfinite(value).all():
        raise ValueError("gate parameters must be nonempty and finite")
    return torch.sigmoid(value / temperature)


def compare_gates(
    before: Any, after: Any, *, before_temperature: float, after_temperature: float
) -> dict[str, Any]:
    import torch

    if before.shape != after.shape:
        raise ValueError("gate coordinates must be aligned")
    b = gate_probabilities(before, before_temperature)
    a = gate_probabilities(after, after_temperature)
    mb, ma = before > 0, after > 0
    union = int((mb | ma).sum())
    return {
        "units": b.numel(),
        "before_temperature": before_temperature,
        "after_temperature": after_temperature,
        "probability_rms_difference": float((a - b).square().mean().sqrt()),
        "before_mask_size": int(mb.sum()),
        "after_mask_size": int(ma.sum()),
        "mask_jaccard": int((mb & ma).sum()) / union if union else 1.0,
        "empty_mask_policy": "two empty masks have Jaccard 1; effectiveness is separate",
        "before_boundary_margin": (b - 0.5).abs().tolist(),
        "after_boundary_margin": (a - 0.5).abs().tolist(),
        "before_decisive_fraction": float(
            ((b - 0.5).abs() > 0.4).to(torch.float64).mean()
        ),
        "after_decisive_fraction": float(
            ((a - 0.5).abs() > 0.4).to(torch.float64).mean()
        ),
        "decisive_margin": 0.4,
    }


def gate_selection_frequency(fits: list[Any]) -> Any:
    import torch

    if not fits or any(f.shape != fits[0].shape for f in fits):
        raise ValueError("gate fits must be nonempty and coordinate aligned")
    if any(not f.numel() or not torch.isfinite(f).all() for f in fits):
        raise ValueError("gate fits must be nonempty and finite")
    return (
        torch.stack([(fit > 0).to(torch.float64) for fit in fits]).mean(dim=0).tolist()
    )


def summarize_subspaces(
    before: dict[tuple[int, int], Any], after: dict[tuple[int, int], Any]
) -> dict[str, Any]:
    """Paired fit drift and invariant within/across-seed spread."""
    if not before or set(before) != set(after):
        raise ValueError("subspace fit identities must match")
    identities = sorted(before)
    seeds = sorted({seed for seed, _ in identities})
    groups_b = [[before[i] for i in identities if i[0] == seed] for seed in seeds]
    groups_a = [[after[i] for i in identities if i[0] == seed] for seed in seeds]
    return {
        "kind": "subspace",
        "paired_drift": [
            {
                "seed": seed,
                "repeat": repeat,
                **compare_subspaces(before[seed, repeat], after[seed, repeat]),
            }
            for seed, repeat in identities
        ],
        "within_seed": [
            {
                "seed": seed,
                **compare_subspace_groups([[q] for q in b], [[q] for q in a]),
            }
            for seed, b, a in zip(seeds, groups_b, groups_a)
        ],
        "across_seed_means": compare_subspace_groups(groups_b, groups_a),
        "units": "projection operators, invariant to basis sign and rotation",
        "effectiveness": "not inferred; inspect the separately declared task metrics",
    }
