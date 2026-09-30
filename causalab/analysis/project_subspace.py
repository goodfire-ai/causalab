"""Project activations into an orthonormal subspace and reconstruct its component.

Inputs: acts (..., d), weight (d, k), as tensors or single-entry safetensors
paths. Select swept inputs with slot/entry on the workflow reference.
An optional binary mask (k,) selects columns, including an empty selection.
Outputs: coordinates (..., retained), reconstructed (..., d). No centering or residual
addition. The caller chooses the population and checks model/site provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_tensor, write_tensor


def project(acts: Any, weight: Any, mask: Any = None) -> tuple[Any, Any]:
    """Return h @ Q and (h @ Q) @ Q.T for orthonormal columns Q, in float64."""
    import torch

    from causalab.neural.shared.featurizers import (
        ORTHONORMAL_TOLERANCE,
        orthonormality_deviation,
    )

    if (
        acts.ndim < 2
        or weight.ndim != 2
        or acts.shape[-1] != weight.shape[0]
        or not 1 <= weight.shape[1] <= weight.shape[0]
    ):
        raise StepError(
            "project_subspace needs acts (..., d) and weight (d, k), 1 <= k <= d"
        )
    if not torch.isfinite(acts).all() or not torch.isfinite(weight).all():
        raise StepError("project_subspace inputs must be finite")
    rows = acts.detach().cpu().to(torch.float64)
    basis = weight.detach().cpu().to(torch.float64)
    if orthonormality_deviation(basis) > ORTHONORMAL_TOLERANCE:
        raise StepError("project_subspace weight columns must be orthonormal")
    if mask is not None:
        selection = torch.as_tensor(mask, device="cpu")
        if (
            selection.shape != (basis.shape[1],)
            or not ((selection == 0) | (selection == 1)).all()
        ):
            raise StepError(
                "project_subspace mask must contain one binary value per basis column"
            )
        basis = basis[:, selection.bool()]
    coordinates = rows @ basis
    return coordinates, coordinates @ basis.T


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    def tensor(name: str) -> Any:
        value = inputs[name]
        return read_tensor(Path(value)) if isinstance(value, (str, Path)) else value

    mask = tensor("mask") if "mask" in inputs else None
    coordinates, reconstructed = project(tensor("acts"), tensor("weight"), mask)
    write_tensor(outputs["coordinates"], coordinates, slot="coordinates")
    write_tensor(outputs["reconstructed"], reconstructed, slot="reconstructed")
