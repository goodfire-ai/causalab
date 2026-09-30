"""Compare blocked Fourier application with the original tensor contraction."""

import numpy as np
import pytest

from causalab.analysis.apply_fourier_probe import apply
from causalab.io.step_io import StepError

pytestmark = pytest.mark.numerical_unit


@pytest.mark.parametrize("shape", [(19, 1, 7, 1), (513, 3, 5, 129), (3, 2, 1, 0)])
@pytest.mark.parametrize("strided", [False, True])
def test_predictions_and_phase_match_contraction(shape, strided):
    n, p, d, f = shape
    rng = np.random.default_rng(51)
    if strided:
        x = rng.normal(size=(n * 2, p, d * 2))[::2, :, ::2]
        w = rng.normal(size=(p, f * 2, d * 2, 4))[:, ::2, ::2, ::2]
        b = rng.normal(size=(p, f * 2, 4))[:, ::2, ::2]
    else:
        x = rng.normal(size=(n, p, d))
        w = rng.normal(size=(p, f, d, 2))
        b = rng.normal(size=(p, f, 2))
    before = [value.copy() for value in (x, w, b)]
    expected = np.einsum("npd,pfdc->npfc", x, w) + b
    result = apply(x, w, b)
    assert result["predictions"].shape == (n, p, f, 2)
    assert result["predictions"].dtype == np.float64
    # Float64 reduction order changes; this is tighter than load_fit's tolerance.
    np.testing.assert_allclose(result["predictions"], expected, atol=1e-12, rtol=1e-12)
    radius = np.linalg.norm(expected, axis=-1)
    phase = np.remainder(np.arctan2(expected[..., 1], expected[..., 0]), 2 * np.pi)
    defined = radius > 1e-8
    phase[~defined] = 0
    np.testing.assert_allclose(result["radius"], radius, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(result["phase"], phase, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(result["phase_defined"], defined)
    for value, original in zip((x, w, b), before):
        np.testing.assert_array_equal(value, original)


def test_zero_and_threshold_radius_with_two_dimensional_activations():
    x = np.ones((4, 3), dtype=np.float32)
    w = np.zeros((1, 4, 3, 2), dtype=np.float32)
    b = np.array([[[0, 0], [0, 1e-9], [0, 1e-8], [0, 1.1e-8]]])
    result = apply(x, w, b)
    np.testing.assert_array_equal(
        result["phase_defined"][0, 0], [False, False, False, True]
    )
    np.testing.assert_array_equal(result["phase"][0, 0], [0, 0, 0, np.pi / 2])


def test_nonfinite_prediction_still_fails():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(StepError, match="nonfinite"):
            apply(
                np.full((2, 1), 1e308), np.full((1, 1, 1, 2), 2.0), np.zeros((1, 1, 2))
            )
