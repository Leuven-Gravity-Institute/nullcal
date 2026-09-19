"""Measure the best uniform-error approximation available to a spline basis."""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog

MINIMUM_CUBIC_SPLINE_KNOTS = 4


@dataclass(frozen=True)
class MinimaxSplineFit:
    """Result of fitting a cubic spline by its maximum absolute residual."""

    node_values: np.ndarray
    values: np.ndarray
    residuals: np.ndarray
    max_abs_residual: float


def augmented_log_knots(
    *,
    minimum_frequency: float,
    maximum_frequency: float,
    broadband_count: int,
    peak_frequency: float,
    local_half_width: float,
    local_count: int,
) -> np.ndarray:
    """Combine broadband log-spaced knots with linear knots around a feature."""
    broadband = np.geomspace(minimum_frequency, maximum_frequency, broadband_count)
    local = np.linspace(peak_frequency - local_half_width, peak_frequency + local_half_width, local_count)
    if local[0] <= minimum_frequency or local[-1] >= maximum_frequency:
        raise ValueError("local knots must lie strictly inside the broadband knot band")
    knots = np.sort(np.concatenate([broadband, local]))
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("local and broadband knots must not coincide")
    return knots


def gaussian_bump(
    frequencies: np.ndarray,
    *,
    peak_value: float,
    peak_frequency: float,
    frequency_width: float,
) -> np.ndarray:
    """Evaluate a Gaussian bump whose standard deviation is one quarter-width."""
    frequencies = np.asarray(frequencies, dtype=float)
    sigma = frequency_width / 4.0
    return peak_value * np.exp(-0.5 * ((frequencies - peak_frequency) / sigma) ** 2)


def spline_design_matrix(frequencies: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return the not-a-knot cubic-spline basis evaluated in log-frequency."""
    frequencies = np.asarray(frequencies, dtype=float)
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or knots.size < MINIMUM_CUBIC_SPLINE_KNOTS:
        raise ValueError("a cubic spline requires at least four knots")
    if np.any(~np.isfinite(knots)) or np.any(knots <= 0.0) or np.any(np.diff(knots) <= 0.0):
        raise ValueError("knots must be finite, positive, and strictly increasing")
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(~np.isfinite(frequencies)):
        raise ValueError("frequencies must be a non-empty, finite one-dimensional array")
    if np.any(frequencies < knots[0]) or np.any(frequencies > knots[-1]):
        raise ValueError("frequencies must lie within the knot band")

    identity = np.eye(knots.size)
    interpolator = CubicSpline(np.log10(knots), identity, axis=0, extrapolate=False)
    return interpolator(np.log10(frequencies))


def minimax_spline_fit(frequencies: np.ndarray, target: np.ndarray, knots: np.ndarray) -> MinimaxSplineFit:
    """Find spline node values minimizing max absolute residual on a grid."""
    target = np.asarray(target, dtype=float)
    design = spline_design_matrix(frequencies, knots)
    if target.shape != (design.shape[0],) or np.any(~np.isfinite(target)):
        raise ValueError("target must be finite and have one value per frequency")

    constraint_matrix = np.block([[design, -np.ones((design.shape[0], 1))], [-design, -np.ones((design.shape[0], 1))]])
    constraint_values = np.concatenate([target, -target])
    objective = np.zeros(design.shape[1] + 1)
    objective[-1] = 1.0
    solution = linprog(
        objective,
        A_ub=constraint_matrix,
        b_ub=constraint_values,
        bounds=[(None, None)] * design.shape[1] + [(0.0, None)],
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"minimax spline fit failed: {solution.message}")

    node_values = solution.x[:-1]
    values = np.einsum("ij,j->i", design, node_values)
    residuals = values - target
    return MinimaxSplineFit(
        node_values=node_values,
        values=values,
        residuals=residuals,
        max_abs_residual=float(np.max(np.abs(residuals))),
    )


def minimax_phase_spline_fit(
    frequencies: np.ndarray,
    target_phase: np.ndarray,
    knots: np.ndarray,
    *,
    bisection_steps: int = 48,
) -> MinimaxSplineFit:
    """Minimize physical phase residual for the rational phase mapping."""
    target_phase = np.asarray(target_phase, dtype=float)
    design = spline_design_matrix(frequencies, knots)
    if target_phase.shape != (design.shape[0],) or np.any(~np.isfinite(target_phase)):
        raise ValueError("target_phase must be finite and have one value per frequency")

    lower_error = 0.0
    upper_error = float(np.max(np.abs(target_phase)))
    if np.any(np.abs(target_phase) + upper_error >= np.pi):
        raise ValueError("target_phase +/- bisection bounds must lie strictly between -pi and pi")
    node_values = np.zeros(design.shape[1])
    for _ in range(bisection_steps):
        trial_error = (lower_error + upper_error) / 2.0
        upper_latent = 2.0 * np.tan((target_phase + trial_error) / 2.0)
        lower_latent = 2.0 * np.tan((target_phase - trial_error) / 2.0)
        solution = linprog(
            np.zeros(design.shape[1]),
            A_ub=np.concatenate([design, -design]),
            b_ub=np.concatenate([upper_latent, -lower_latent]),
            bounds=[(None, None)] * design.shape[1],
            method="highs",
        )
        if solution.success:
            upper_error = trial_error
            node_values = solution.x
        else:
            lower_error = trial_error

    latent_values = np.einsum("ij,j->i", design, node_values)
    values = 2.0 * np.arctan(latent_values / 2.0)
    residuals = values - target_phase
    return MinimaxSplineFit(
        node_values=node_values,
        values=values,
        residuals=residuals,
        max_abs_residual=float(np.max(np.abs(residuals))),
    )
