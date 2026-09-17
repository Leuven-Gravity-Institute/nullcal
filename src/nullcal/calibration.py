"""Pure JAX calibration model and Gaussian knot-prior functions.

The spline interpolates amplitude and latent phase in log10 frequency using
not-a-knot boundary conditions.  This is bilby's ``CubicSpline`` model for a
uniform log-frequency grid and also permits nonuniform knot placement.

Importing this module enables JAX's process-wide ``jax_enable_x64`` setting.
Calibration inference requires float64 precision, so this side effect is an
intentional part of the module's public behavior.
"""

from __future__ import annotations

import jax

# Calibration inference needs substantially more precision than JAX's default
# float32 mode.  Enable x64 before importing jax.numpy so every public function
# can explicitly construct float64 inputs and complex128 outputs.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

MINIMUM_CUBIC_SPLINE_KNOTS = 4


def _float64(values):
    """Return ``values`` as a JAX float64 array."""
    return jnp.asarray(values, dtype=jnp.float64)


def _validate_prior_shapes(values, mean, sigma, parameter):
    """Require array hyperparameters to match their knot-value shape."""
    if (mean.ndim != 0 and mean.shape != values.shape) or (sigma.ndim != 0 and sigma.shape != values.shape):
        raise ValueError(f"{parameter} mean and sigma must be scalar or match the value shape")


def _not_a_knot_second_derivatives(log_knots, node_values):
    """Solve the nonuniform not-a-knot cubic-spline continuity equations."""
    knot_count = log_knots.shape[0]
    widths = jnp.diff(log_knots)
    system = jnp.zeros((knot_count, knot_count), dtype=jnp.float64)
    right_hand_side = jnp.zeros(knot_count, dtype=jnp.float64)

    system = system.at[0, :3].set(jnp.array((-widths[1], widths[0] + widths[1], -widths[0])))
    system = system.at[-1, -3:].set(jnp.array((widths[-1], -(widths[-2] + widths[-1]), widths[-2])))

    interior = jnp.arange(1, knot_count - 1)
    left_widths = widths[:-1]
    right_widths = widths[1:]
    system = system.at[interior, interior - 1].set(left_widths)
    system = system.at[interior, interior].set(2.0 * (left_widths + right_widths))
    system = system.at[interior, interior + 1].set(right_widths)
    right_hand_side = right_hand_side.at[interior].set(
        6.0
        * ((node_values[2:] - node_values[1:-1]) / right_widths - (node_values[1:-1] - node_values[:-2]) / left_widths)
    )
    return jnp.linalg.solve(system, right_hand_side)


def _evaluate_spline(frequencies, knot_frequencies, node_values):
    """Evaluate a not-a-knot cubic spline in log10 frequency."""
    log_knots = jnp.log10(knot_frequencies)
    log_frequencies = jnp.log10(frequencies)
    second_derivatives = _not_a_knot_second_derivatives(log_knots, node_values)
    intervals = jnp.searchsorted(log_knots, log_frequencies, side="right") - 1
    intervals = jnp.clip(intervals, 0, log_knots.size - 2)

    left = log_knots[intervals]
    right = log_knots[intervals + 1]
    widths = right - left
    left_distance = right - log_frequencies
    right_distance = log_frequencies - left
    return (
        second_derivatives[intervals] * left_distance**3 / (6.0 * widths)
        + second_derivatives[intervals + 1] * right_distance**3 / (6.0 * widths)
        + (node_values[intervals] - second_derivatives[intervals] * widths**2 / 6.0) * left_distance / widths
        + (node_values[intervals + 1] - second_derivatives[intervals + 1] * widths**2 / 6.0) * right_distance / widths
    )


def calibration_factor(frequencies, knot_frequencies, amplitude, phase):
    """Evaluate bilby's cubic-spline calibration factor in explicit float64.

    Args:
        frequencies: Positive frequencies at which to evaluate the factor.
        knot_frequencies: At least four positive, strictly increasing knots.
            Knot placement and count are function parameters; nonuniform grids
            such as the 19-knot spectroscopy grid are supported.
        amplitude: Fractional amplitude error at each knot.
        phase: Bilby's latent phase parameter at each knot, in radians.

    Returns:
        A complex128 array equal to ``(1 + amplitude_spline) *
        (2 + 1j * phase_spline) / (2 - 1j * phase_spline)``.
    """
    frequencies = _float64(frequencies)
    knot_frequencies = _float64(knot_frequencies)
    amplitude = _float64(amplitude)
    phase = _float64(phase)
    if frequencies.ndim != 1 or knot_frequencies.ndim != 1 or amplitude.ndim != 1 or phase.ndim != 1:
        raise ValueError("frequencies, knots, amplitude, and phase must be one-dimensional")
    if knot_frequencies.size < MINIMUM_CUBIC_SPLINE_KNOTS:
        raise ValueError("a cubic spline requires at least four knots")
    if amplitude.shape != knot_frequencies.shape or phase.shape != knot_frequencies.shape:
        raise ValueError("amplitude and phase must contain one value per knot")
    delta_amplitude = _evaluate_spline(frequencies, knot_frequencies, amplitude)
    delta_phase = _evaluate_spline(frequencies, knot_frequencies, phase)
    imaginary_phase = jnp.asarray(1j, dtype=jnp.complex128) * delta_phase
    return (1.0 + delta_amplitude) * (2.0 + imaginary_phase) / (2.0 - imaginary_phase)


def calibration_log_prior(
    amplitude,
    phase,
    amplitude_mean,
    amplitude_sigma,
    phase_mean,
    phase_sigma,
):
    """Return the normalized independent-Gaussian log-prior for knot values."""
    amplitude = _float64(amplitude)
    phase = _float64(phase)
    amplitude_mean = _float64(amplitude_mean)
    amplitude_sigma = _float64(amplitude_sigma)
    phase_mean = _float64(phase_mean)
    phase_sigma = _float64(phase_sigma)
    _validate_prior_shapes(amplitude, amplitude_mean, amplitude_sigma, "amplitude")
    _validate_prior_shapes(phase, phase_mean, phase_sigma, "phase")

    def gaussian_log_prob(values, means, sigmas):
        return -0.5 * jnp.sum(((values - means) / sigmas) ** 2 + jnp.log(2.0 * jnp.pi * sigmas**2))

    return gaussian_log_prob(amplitude, amplitude_mean, amplitude_sigma) + gaussian_log_prob(
        phase, phase_mean, phase_sigma
    )


def calibration_parameters_to_unconstrained(
    amplitude,
    phase,
    amplitude_mean,
    amplitude_sigma,
    phase_mean,
    phase_sigma,
):
    """Map physical Gaussian knot values to standard-normal coordinates."""
    amplitude = _float64(amplitude)
    phase = _float64(phase)
    amplitude_mean = _float64(amplitude_mean)
    amplitude_sigma = _float64(amplitude_sigma)
    phase_mean = _float64(phase_mean)
    phase_sigma = _float64(phase_sigma)
    _validate_prior_shapes(amplitude, amplitude_mean, amplitude_sigma, "amplitude")
    _validate_prior_shapes(phase, phase_mean, phase_sigma, "phase")
    return (
        (amplitude - amplitude_mean) / amplitude_sigma,
        (phase - phase_mean) / phase_sigma,
    )


def unconstrained_to_calibration_parameters(
    unconstrained_amplitude,
    unconstrained_phase,
    amplitude_mean,
    amplitude_sigma,
    phase_mean,
    phase_sigma,
):
    """Map standard-normal coordinates to physical Gaussian knot values."""
    unconstrained_amplitude = _float64(unconstrained_amplitude)
    unconstrained_phase = _float64(unconstrained_phase)
    amplitude_mean = _float64(amplitude_mean)
    amplitude_sigma = _float64(amplitude_sigma)
    phase_mean = _float64(phase_mean)
    phase_sigma = _float64(phase_sigma)
    _validate_prior_shapes(unconstrained_amplitude, amplitude_mean, amplitude_sigma, "amplitude")
    _validate_prior_shapes(unconstrained_phase, phase_mean, phase_sigma, "phase")
    return (
        amplitude_mean + amplitude_sigma * unconstrained_amplitude,
        phase_mean + phase_sigma * unconstrained_phase,
    )


__all__ = [
    "calibration_factor",
    "calibration_log_prior",
    "calibration_parameters_to_unconstrained",
    "unconstrained_to_calibration_parameters",
]
