"""Apply saved Fourier weights and biases to a matching activation population."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from causalab.analysis.fit_fourier_probe import activations, array
from causalab.io.step_io import StepError, write_tensor

_ROW_CHUNK = 512
_FREQUENCY_CHUNK = 128


def apply(acts: Any, weight: Any, bias: Any) -> dict[str, Any]:
    """Return cosine/sine predictions, radius and phase from a frozen fit."""
    import numpy as np

    x, w, b = activations(acts), array(weight), array(bias)
    if (
        w.ndim != 4
        or w.shape[0] != x.shape[1]
        or w.shape[2:] != (x.shape[2], 2)
        or b.shape != (w.shape[0], w.shape[1], 2)
    ):
        raise StepError("Fourier weights must match the position and feature axes")
    n, p, d = x.shape
    f = w.shape[1]
    prediction = np.empty((n, p, f, 2), dtype=np.float64)
    # Batch the two output coordinates through BLAS. Bound temporary weights
    # and predictions while retaining the complete output population.
    for position in range(p):
        for first in range(0, f, _FREQUENCY_CHUNK):
            last = min(first + _FREQUENCY_CHUNK, f)
            weights = np.ascontiguousarray(
                w[position, first:last].transpose(1, 0, 2)
            ).reshape(d, -1)
            for start in range(0, n, _ROW_CHUNK):
                stop = min(start + _ROW_CHUNK, n)
                block = (x[start:stop, position] @ weights).reshape(
                    stop - start, last - first, 2
                )
                block += b[position, first:last]
                prediction[start:stop, position, first:last] = block
    radius = np.linalg.norm(prediction, axis=-1)
    phase = np.remainder(np.arctan2(prediction[..., 1], prediction[..., 0]), 2 * np.pi)
    defined = radius > 1e-8
    phase[~defined] = 0
    if not np.isfinite(prediction).all():
        raise StepError("Fourier readout produced nonfinite values")
    return {
        "predictions": prediction,
        "radius": radius,
        "phase": phase,
        "phase_defined": defined,
    }


def main(inputs: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    import torch

    result = apply(inputs["acts"], inputs["weight"], inputs["bias"])
    for name, value in result.items():
        write_tensor(outputs[name], torch.from_numpy(value.copy()), slot=name)
