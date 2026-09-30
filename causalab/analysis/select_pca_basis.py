"""Select one position's PCA basis for a subspace initializer.

Inputs: weight (positions, features, components), position (zero-based integer).
Output: weight (features, components). Use a workflow tensor reference so the
runner inherits the source model, site and training population identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.io.step_io import StepError, read_tensor, write_tensor


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    from causalab.neural.shared.featurizers import (
        ORTHONORMAL_TOLERANCE,
        orthonormality_deviation,
    )

    weight = inputs["weight"]
    if isinstance(weight, (str, Path)):
        weight = read_tensor(Path(weight))
    position = inputs["position"]
    if (
        weight.ndim != 3
        or type(position) is not int
        or not 0 <= position < weight.shape[0]
        or not 1 <= weight.shape[2] <= weight.shape[1]
    ):
        raise StepError(
            "select_pca_basis needs a valid position in a (positions, d, k) basis"
        )
    selected = weight[position]
    if not torch.isfinite(selected).all() or (
        orthonormality_deviation(selected.double()) > ORTHONORMAL_TOLERANCE
    ):
        raise StepError("selected PCA columns must be finite and orthonormal")
    write_tensor(outputs["weight"], selected, identity={"k": selected.shape[1]})
