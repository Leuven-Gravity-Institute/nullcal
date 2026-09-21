import runpy
from pathlib import Path

import numpy as np

import nullcal.studies.lwa_leakage as leakage_module
from nullcal.studies.lwa_leakage import (
    ARM_LENGTH_METRES,
    SPEED_OF_LIGHT_METRES_PER_SECOND,
    arm_transfer,
    et_d_psd,
    et_response_matrix,
    evaluate_frequency,
    generate_population,
    lwa_null_residual_energy,
    null_p_value,
    recover_lwa_map,
    simulate_identical_noise_pair,
)


def test_arm_transfer_matches_transverse_closed_form():
    frequency = np.array([0.0, 200.0, 1000.0])
    arm = np.array([1.0, 0.0, 0.0])
    source_direction = np.array([0.0, 0.0, 1.0])
    light_travel_time = ARM_LENGTH_METRES / SPEED_OF_LIGHT_METRES_PER_SECOND

    actual = arm_transfer(frequency, arm, source_direction)
    expected = np.exp(-2j * np.pi * frequency * light_travel_time) * np.sinc(2.0 * frequency * light_travel_time)

    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_et_lwa_response_closes_for_every_polarization():
    population = generate_population(32, seed=397)

    for direction, polarization in zip(population.direction, population.polarization, strict=True):
        response = et_response_matrix(np.array([0.0]), direction, polarization)[0]
        np.testing.assert_allclose(np.sum(response, axis=0), 0.0, rtol=0.0, atol=2e-16)


def test_et_d_psd_reproduces_pinned_tabulation_nodes():
    frequencies = np.array([20.0, 100.0, 250.0, 1000.0, 2000.0])
    expected = np.array(
        [
            8.83385322431945442e-49,
            1.54691772612811412e-49,
            1.02194370960870872e-49,
            3.33030480074998683e-49,
            1.18267671514987749e-48,
        ]
    )

    np.testing.assert_allclose(et_d_psd(frequencies), expected, rtol=2e-15, atol=0.0)


def test_identical_noise_pair_differs_only_by_the_exact_signal():
    lwa_response = et_response_matrix(np.array([0.0]), np.array([0.0, 0.0, 1.0]), 0.3)[0]
    exact_response = et_response_matrix(np.array([500.0]), np.array([0.0, 0.0, 1.0]), 0.3)[0]
    polarization = np.array([0.8, -0.4j])

    pair = simulate_identical_noise_pair(
        lwa_response,
        exact_response,
        polarization,
        network_snr=120.0,
        psd=float(et_d_psd(np.array([500.0]))[0]),
        delta_f=1.0,
        rng=np.random.default_rng(12),
    )

    np.testing.assert_allclose(pair.signal_present - pair.signal_absent, pair.exact_signal, rtol=0.0, atol=1e-35)
    np.testing.assert_allclose(pair.whitened_present - pair.whitened_absent, pair.whitened_signal, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(np.linalg.norm(pair.whitened_lwa_signal), 120.0, rtol=1e-14, atol=0.0)


def test_lwa_map_recovers_zero_bias_for_an_exact_lwa_signal():
    response = et_response_matrix(np.array([0.0]), np.array([0.2, -0.3, np.sqrt(0.87)]), 1.1)[0]
    signal = response @ np.array([0.7, 0.2j])
    signal *= 120.0 / np.linalg.norm(signal)

    recovered = recover_lwa_map(response, signal, prior_sigma=0.05)

    assert recovered.success
    np.testing.assert_allclose(recovered.amplitude, 0.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(recovered.physical_phase, 0.0, rtol=0.0, atol=1e-9)


def test_exact_and_lwa_signal_controls_recover_identical_maps_when_responses_match():
    response = et_response_matrix(np.array([0.0]), np.array([0.2, 0.5, np.sqrt(0.71)]), 0.4)[0]
    pair = simulate_identical_noise_pair(
        response,
        response,
        np.array([0.9, -0.2j]),
        network_snr=120.0,
        psd=float(et_d_psd(np.array([250.0]))[0]),
        delta_f=1.0,
        rng=np.random.default_rng(91),
    )

    exact_map = recover_lwa_map(response, pair.whitened_present, prior_sigma=0.05)
    lwa_map = recover_lwa_map(
        response,
        pair.whitened_absent + pair.whitened_lwa_signal,
        prior_sigma=0.05,
    )

    np.testing.assert_allclose(exact_map.amplitude, lwa_map.amplitude, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(exact_map.physical_phase, lwa_map.physical_phase, rtol=0.0, atol=1e-9)


def test_closed_form_lwa_null_energy_matches_explicit_projector():
    response = et_response_matrix(np.array([0.0]), np.array([-0.4, 0.1, np.sqrt(0.83)]), 0.7)[0]
    calibration = np.array([0.97 * np.exp(0.03j), 1.02 * np.exp(-0.04j), 1.01 * np.exp(0.02j)])
    data = np.array([1.2 + 0.4j, -0.3 + 1.1j, 0.7 - 0.2j])
    calibrated_response = calibration[:, None] * response
    projector = np.eye(3) - calibrated_response @ np.linalg.solve(
        calibrated_response.conj().T @ calibrated_response,
        calibrated_response.conj().T,
    )
    projected = projector @ data

    actual = lwa_null_residual_energy(data, calibration)

    np.testing.assert_allclose(actual, np.vdot(projected, projected).real, rtol=2e-14, atol=2e-15)


def test_null_p_value_uses_the_complex_gaussian_noise_null():
    null_samples = np.array([0.0j, 1.0 + 0.0j, 1.0 + 1.0j])

    actual = null_p_value(null_samples)

    np.testing.assert_allclose(actual, np.exp(-(np.abs(null_samples) ** 2)), rtol=0.0, atol=0.0)


def test_frequency_evaluation_records_population_noise_and_paired_bias():
    population = generate_population(3, seed=397)

    result = evaluate_frequency(
        250.0,
        population,
        noise_realizations=2,
        noise_seed=9381,
        network_snr=120.0,
        prior_sigma=0.05,
        significance_alpha=0.002699796063260207,
        bootstrap_resamples=32,
        bootstrap_seed=2718,
    )

    assert result["population_size"] == 3
    assert result["noise_realizations"] == 2
    assert result["bias_sample_count"] == 6
    assert result["map_failure_count"] == 0
    assert result["sky_factor_p05"] <= result["sky_factor_median"] <= result["sky_factor_p95"]
    assert result["amplitude_bias_percent_p95"] >= result["amplitude_bias_percent_median"]
    assert result["phase_bias_degrees_p95"] >= result["phase_bias_degrees_median"]
    assert result["amplitude_bias_percent_p95_upper95"] >= result["amplitude_bias_percent_p95"]
    assert result["phase_bias_degrees_p95_upper95"] >= result["phase_bias_degrees_p95"]


def test_study_driver_fixes_bounded_design_and_frequency_grid():
    script = Path(__file__).parents[2] / "scripts" / "lwa_leakage_study.py"
    driver = runpy.run_path(script)

    np.testing.assert_array_equal(
        driver["FREQUENCIES_HZ"],
        np.array(
            [20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
        ),
    )
    assert driver["POPULATION_SIZE"] == 256
    assert driver["NOISE_REALIZATIONS"] == 16
    assert driver["NETWORK_SNR"] == 120.0
    assert driver["PRIOR_SIGMA"] == 0.05
    assert driver["SIGNIFICANCE_ALPHA"] == 0.002699796063260207
    assert driver["BOOTSTRAP_RESAMPLES"] == 2000


def test_validity_band_stops_at_first_p95_bias_failure():
    script = Path(__file__).parents[2] / "scripts" / "lwa_leakage_study.py"
    driver = runpy.run_path(script)
    results = [
        {
            "frequency_hz": 100.0,
            "amplitude_bias_percent_p95_upper95": 1.0,
            "phase_bias_degrees_p95_upper95": 1.0,
        },
        {
            "frequency_hz": 200.0,
            "amplitude_bias_percent_p95_upper95": 3.9,
            "phase_bias_degrees_p95_upper95": 2.0,
        },
        {
            "frequency_hz": 300.0,
            "amplitude_bias_percent_p95_upper95": 4.1,
            "phase_bias_degrees_p95_upper95": 2.0,
        },
        {
            "frequency_hz": 400.0,
            "amplitude_bias_percent_p95_upper95": 3.0,
            "phase_bias_degrees_p95_upper95": 3.0,
        },
    ]

    assert driver["validity_band_maximum"](results) == 200.0


def test_lwa_map_retries_a_failed_line_search_with_powell(monkeypatch):
    response = et_response_matrix(np.array([0.0]), np.array([0.3, 0.4, np.sqrt(0.75)]), 0.2)[0]
    data = np.array([1.0 + 0.2j, -0.4 + 0.1j, 0.7 - 0.5j])
    actual_minimize = leakage_module.minimize
    methods = []

    def fail_first(*args, **kwargs):
        result = actual_minimize(*args, **kwargs)
        methods.append(kwargs["method"])
        if len(methods) == 1:
            result.success = False
        return result

    monkeypatch.setattr(leakage_module, "minimize", fail_first)

    recovered = recover_lwa_map(response, data, prior_sigma=0.05)

    assert recovered.success
    assert methods == ["L-BFGS-B", "Powell"]
