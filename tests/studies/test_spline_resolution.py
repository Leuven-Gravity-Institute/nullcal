import numpy as np
import pytest
from bilby.gw.detector import CubicSpline

from nullcal.studies.spline_resolution import (
    augmented_log_knots,
    gaussian_bump,
    minimax_phase_spline_fit,
    minimax_spline_fit,
    spline_design_matrix,
)


def test_gaussian_bump_uses_quarter_width_as_sigma():
    frequencies = np.array([236.93, 249.43, 261.93])

    bump = gaussian_bump(
        frequencies,
        peak_value=-0.1,
        peak_frequency=249.43,
        frequency_width=50.0,
    )

    np.testing.assert_allclose(bump, -0.1 * np.array([np.exp(-0.5), 1.0, np.exp(-0.5)]), atol=0.0)


def test_minimax_spline_exactly_recovers_cubic_in_log_frequency():
    knots = np.geomspace(20.0, 2000.0, 8)
    frequencies = np.geomspace(20.0, 2000.0, 2001)
    log_frequency = np.log10(frequencies)
    target = 0.2 - 0.3 * log_frequency + 0.1 * log_frequency**2 - 0.01 * log_frequency**3

    fit = minimax_spline_fit(frequencies, target, knots)

    assert fit.max_abs_residual < 1e-10
    np.testing.assert_allclose(fit.values, target, rtol=0.0, atol=1e-10)


def test_minimax_phase_spline_exactly_recovers_constant_physical_phase():
    knots = np.geomspace(8.0, 2048.0, 10)
    frequencies = np.geomspace(20.0, 2000.0, 1001)
    target = np.full(frequencies.size, np.deg2rad(10.0))

    fit = minimax_phase_spline_fit(frequencies, target, knots)

    assert fit.max_abs_residual < 1e-10
    np.testing.assert_allclose(fit.values, target, rtol=0.0, atol=1e-10)


def test_augmented_log_knots_preserves_broadband_grid_and_resolves_peak():
    knots = augmented_log_knots(
        minimum_frequency=8.0,
        maximum_frequency=2048.0,
        broadband_count=10,
        peak_frequency=249.43,
        local_half_width=50.0,
        local_count=9,
    )

    assert knots.size == 19
    assert knots[0] == 8.0
    assert knots[-1] == 2048.0
    assert 249.43 in knots
    assert np.all(np.diff(knots) > 0.0)


def test_uniform_log_design_matrix_matches_bilby_basis():
    knots = np.geomspace(20.0, 2000.0, 10)
    frequencies = np.geomspace(20.0, 2000.0, 1001)
    node_values = np.linspace(-0.1, 0.1, knots.size) ** 3
    model = CubicSpline(
        prefix="recalib_",
        minimum_frequency=knots[0],
        maximum_frequency=knots[-1],
        n_points=knots.size,
    )
    parameters = {f"recalib_amplitude_{index}": value for index, value in enumerate(node_values)}
    parameters.update({f"recalib_phase_{index}": 0.0 for index in range(knots.size)})

    expected = np.abs(model.get_calibration_factor(frequencies, **parameters)) - 1.0
    actual = np.einsum("ij,j->i", spline_design_matrix(frequencies, knots), node_values)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("frequencies", "knots", "match"),
    [
        (np.array([20.0, 30.0]), np.array([20.0, 10.0, 30.0, 40.0]), "strictly increasing"),
        (np.array([19.0, 30.0]), np.array([20.0, 30.0, 40.0, 50.0]), "within the knot band"),
        (np.array([20.0, 30.0]), np.array([20.0, 30.0, 40.0]), "at least four"),
    ],
)
def test_spline_design_matrix_rejects_invalid_domain(frequencies, knots, match):
    with pytest.raises(ValueError, match=match):
        spline_design_matrix(frequencies, knots)
