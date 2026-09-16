"""Tests for the JAX spline calibration model and Gaussian knot prior."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bilby.core.prior import Gaussian
from bilby.gw.detector.calibration import CubicSpline
from scipy.interpolate import CubicSpline as ScipyCubicSpline

from nullcal.calibration import (
    calibration_factor,
    calibration_log_prior,
    calibration_parameters_to_unconstrained,
    unconstrained_to_calibration_parameters,
)

# Fixed before the first bilby comparison is run. Both implementations solve the
# same at-most-19-dimensional float64 linear system, but NumPy and JAX use
# different LAPACK/XLA reduction paths. 1e-11 peak-relative is over 100 times
# n**2 * float64 epsilon at n=19 while remaining far below any physical scale.
ANCHOR_PEAK_RELATIVE_TOLERANCE = 1e-11

# Fixed before the first nonuniform comparison. SciPy uses the same not-a-knot
# definition that is transitively validated against bilby on uniform log grids;
# the float64 solve has the same at-most-19-dimensional round-off basis above.
NONUNIFORM_PEAK_RELATIVE_TOLERANCE = 1e-11

# Fixed before the first gradient comparison is run. A centred finite difference
# with a 1e-6 step has O(h**2) truncation and O(eps/h) round-off; these tolerances
# leave margin for both without accepting an order-percent derivative error.
FINITE_DIFFERENCE_STEP = 1e-6
GRADIENT_RTOL = 5e-7
GRADIENT_ATOL = 5e-9


def _bilby_factor(frequencies, knots, amplitude, phase):
    model = CubicSpline(
        prefix="recalib_",
        minimum_frequency=knots[0],
        maximum_frequency=knots[-1],
        n_points=knots.size,
    )
    parameters = {f"recalib_amplitude_{index}": value for index, value in enumerate(amplitude)}
    parameters.update({f"recalib_phase_{index}": value for index, value in enumerate(phase)})
    return model.get_calibration_factor(frequencies, **parameters)


@pytest.mark.parametrize(
    ("minimum", "maximum", "count"),
    [(8.0, 512.0, 4), (20.0, 2000.0, 7), (8.0, 2048.0, 10), (8.0, 2048.0, 19)],
)
@pytest.mark.parametrize("node_scale", [0.0, 0.01, 0.2])
def test_uniform_log_spline_matches_bilby_over_knot_grid(minimum, maximum, count, node_scale):
    """Knot counts, placements, and node values must preserve bilby's model."""
    knots = np.geomspace(minimum, maximum, count)
    frequencies = np.geomspace(minimum, maximum, 257)
    coordinate = np.linspace(-1.0, 1.0, count)
    amplitude = node_scale * (coordinate**3 - 0.2 * coordinate)
    phase = node_scale * (coordinate**2 - 0.4)

    expected = _bilby_factor(frequencies, knots, amplitude, phase)
    actual = np.asarray(calibration_factor(frequencies, knots, amplitude, phase))
    peak_relative = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))

    assert peak_relative < ANCHOR_PEAK_RELATIVE_TOLERANCE


def test_nonuniform_not_a_knot_spline_matches_scipy_off_knots():
    """The nonuniform spectroscopy grid must match an independent off-knot reference."""
    knots = np.array(
        [
            8.000,
            14.814,
            27.432,
            50.797,
            94.063,
            174.181,
            199.430,
            211.930,
            224.430,
            236.930,
            249.430,
            261.930,
            274.430,
            286.930,
            299.430,
            322.540,
            597.263,
            1105.981,
            2048.000,
        ]
    )
    amplitude = 0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, knots.size))
    phase = 0.05 * np.cos(np.linspace(0.0, 2.0 * np.pi, knots.size))
    frequencies = np.geomspace(20.0, 2000.0, 1001)
    assert not np.any(np.isclose(frequencies[:, None], knots[None, :], rtol=0.0, atol=1e-12))

    log_knots = np.log10(knots)
    log_frequencies = np.log10(frequencies)
    expected_amplitude = ScipyCubicSpline(log_knots, amplitude, bc_type="not-a-knot")(log_frequencies)
    expected_phase = ScipyCubicSpline(log_knots, phase, bc_type="not-a-knot")(log_frequencies)
    expected = (1.0 + expected_amplitude) * (2.0 + 1j * expected_phase) / (2.0 - 1j * expected_phase)
    actual = np.asarray(calibration_factor(frequencies, knots, amplitude, phase))
    peak_relative = np.max(np.abs(actual - expected)) / np.max(np.abs(expected))

    assert peak_relative < NONUNIFORM_PEAK_RELATIVE_TOLERANCE


def test_calibration_factor_is_jitted_and_explicitly_float64_complex128():
    """JIT execution must retain the precision required by downstream inference."""
    frequencies = jnp.geomspace(8.0, 2048.0, 64)
    knots = jnp.geomspace(8.0, 2048.0, 10)
    amplitude = jnp.linspace(-0.02, 0.02, 10)
    phase = jnp.linspace(0.03, -0.03, 10)

    result = jax.jit(calibration_factor)(frequencies, knots, amplitude, phase)

    assert jax.config.x64_enabled
    assert frequencies.dtype == jnp.float64
    assert result.dtype == jnp.complex128


@pytest.mark.parametrize(
    ("knots", "amplitude", "phase", "match"),
    [
        (np.geomspace(8.0, 2048.0, 3), np.zeros(3), np.zeros(3), "at least four"),
        (np.geomspace(8.0, 2048.0, 5), np.zeros(4), np.zeros(5), "one value per knot"),
        (np.geomspace(8.0, 2048.0, 5), np.zeros(5), np.zeros(4), "one value per knot"),
    ],
)
def test_calibration_factor_rejects_invalid_static_shapes(knots, amplitude, phase, match):
    """Bad knot/value shapes must fail at tracing instead of broadcasting silently."""
    with pytest.raises(ValueError, match=match):
        jax.jit(calibration_factor)(np.geomspace(8.0, 2048.0, 16), knots, amplitude, phase)


def test_calibration_factor_gradient_matches_centred_finite_difference():
    """Node gradients must be numerically correct, not merely traceable by JAX."""
    frequencies = jnp.geomspace(8.0, 2048.0, 73)
    knots = jnp.geomspace(8.0, 2048.0, 7)
    amplitude = jnp.linspace(-0.04, 0.03, knots.size)
    phase = jnp.linspace(0.05, -0.02, knots.size)
    weights = jnp.linspace(0.5, 1.5, frequencies.size) + 1j * jnp.linspace(-0.7, 0.4, frequencies.size)

    def objective(node_amplitude, node_phase):
        factor = calibration_factor(frequencies, knots, node_amplitude, node_phase)
        return jnp.real(jnp.vdot(weights, factor))

    automatic = jax.grad(objective, argnums=(0, 1))(amplitude, phase)
    for argument_index, values in enumerate((amplitude, phase)):
        finite_difference = np.empty(values.size)
        values_np = np.asarray(values)
        for index in range(values.size):
            step = np.zeros(values.size)
            step[index] = FINITE_DIFFERENCE_STEP
            arguments_above = [amplitude, phase]
            arguments_below = [amplitude, phase]
            arguments_above[argument_index] = values_np + step
            arguments_below[argument_index] = values_np - step
            finite_difference[index] = (float(objective(*arguments_above)) - float(objective(*arguments_below))) / (
                2.0 * FINITE_DIFFERENCE_STEP
            )

        np.testing.assert_allclose(
            np.asarray(automatic[argument_index]),
            finite_difference,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        )


def test_log_prior_matches_bilby_gaussian_prior_dict_surface():
    """The pure log-prior must include bilby's Gaussian normalization terms."""
    amplitude = np.array([-0.03, 0.04])
    phase = np.array([0.02, -0.01])
    amplitude_mean = np.array([-0.01, 0.02])
    phase_mean = np.array([0.0, 0.01])
    amplitude_sigma = np.array([0.05, 0.08])
    phase_sigma = np.array([0.03, 0.07])
    values = np.concatenate([amplitude, phase])
    means = np.concatenate([amplitude_mean, phase_mean])
    sigmas = np.concatenate([amplitude_sigma, phase_sigma])
    expected = sum(
        Gaussian(mu=mean, sigma=sigma).ln_prob(value) for value, mean, sigma in zip(values, means, sigmas, strict=True)
    )

    actual = calibration_log_prior(
        amplitude,
        phase,
        amplitude_mean,
        amplitude_sigma,
        phase_mean,
        phase_sigma,
    )

    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(float(actual), expected, rtol=2e-15, atol=0.0)


def test_prior_unconstraining_transform_is_invertible_and_jittable():
    """Physical Gaussian knot values must round-trip through standard-normal space."""
    amplitude = jnp.array([-0.03, 0.04])
    phase = jnp.array([0.02, -0.01])
    amplitude_mean = jnp.array([-0.01, 0.02])
    phase_mean = jnp.array([0.0, 0.01])
    amplitude_sigma = jnp.array([0.05, 0.08])
    phase_sigma = jnp.array([0.03, 0.07])

    unconstrained = jax.jit(calibration_parameters_to_unconstrained)(
        amplitude,
        phase,
        amplitude_mean,
        amplitude_sigma,
        phase_mean,
        phase_sigma,
    )
    restored = jax.jit(unconstrained_to_calibration_parameters)(
        *unconstrained,
        amplitude_mean,
        amplitude_sigma,
        phase_mean,
        phase_sigma,
    )

    assert all(values.dtype == jnp.float64 for values in (*unconstrained, *restored))
    roundoff = np.finfo(np.float64).eps
    np.testing.assert_allclose(np.asarray(restored[0]), np.asarray(amplitude), rtol=roundoff, atol=roundoff)
    np.testing.assert_allclose(np.asarray(restored[1]), np.asarray(phase), rtol=roundoff, atol=roundoff)
