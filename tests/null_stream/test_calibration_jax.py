"""JAX compatibility and derivative checks for the calibration fold."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nullcal.null_stream.calibration import compute_calibrated_whitened_antenna_response


@pytest.mark.unit
def test_calibration_fold_is_jittable_and_preserves_masked_values():
    """The public fold remains value-compatible when traced by JAX."""
    response = np.arange(24, dtype=np.float64).reshape(4, 3, 2) / 10.0
    factors = np.array(
        [
            [1.0 + 0.1j, 1.1 + 0.2j, 1.2 + 0.3j, 1.3 + 0.4j],
            [0.9 - 0.1j, 0.8 - 0.2j, 0.7 - 0.3j, 0.6 - 0.4j],
            [1.2 + 0.0j, 1.1 + 0.0j, 1.0 + 0.0j, 0.9 + 0.0j],
        ],
        dtype=np.complex128,
    )
    mask = np.array([True, False, True, False])

    actual = jax.jit(compute_calibrated_whitened_antenna_response)(response, factors, mask)
    expected = np.zeros(response.shape, dtype=np.complex128)
    expected[mask] = response[mask] * factors.T[mask, :, None]

    assert actual.dtype == jnp.complex128
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_calibration_fold_gradient_matches_central_difference():
    """JAX autodiff agrees with an independent finite-difference derivative."""
    response = jnp.arange(24, dtype=jnp.float64).reshape(4, 3, 2) / 10.0
    mask = jnp.array([True, False, True, True])

    def objective(scale):
        factors = jnp.ones((3, 4), dtype=jnp.complex128).at[1, 2].set(1.0 + 1j * scale)
        folded = compute_calibrated_whitened_antenna_response(response, factors, mask)
        return jnp.sum(jnp.abs(folded) ** 2)

    point = 0.17
    step = 1e-5
    autodiff = float(jax.grad(objective)(point))
    finite_difference = float((objective(point + step) - objective(point - step)) / (2.0 * step))
    relative_error = abs(autodiff - finite_difference) / abs(finite_difference)

    assert relative_error <= 1e-9
