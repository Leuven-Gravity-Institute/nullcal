"""Finite-arm ET response and one-bin LWA calibration-bias utilities.

The arm transfer follows Eq. (6) of Virtuoso and Milotti,
arXiv:2412.01693.  The three directed Michelsons share the physical arms of
an equilateral triangle; their zero-frequency detector tensors therefore sum
to zero exactly.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

SPEED_OF_LIGHT_METRES_PER_SECOND = 299_792_458.0
ARM_LENGTH_METRES = 10_000.0
POLARIZATION_REFERENCE_ALIGNMENT_LIMIT = 0.9
EXACT_ZERO_OBJECTIVE_TOLERANCE = 1e-24

# Pinned samples from bilby's ET_D_psd.txt at commit
# 75f834b14c0f2d8314d61b7276c39ed0ad1c00c9.  Values between these study
# nodes are interpolated in log(f)-log(PSD), matching the positive scale of a
# tabulated noise curve without claiming an analytic ET-D model.
ET_D_FREQUENCIES_HZ = np.array(
    [20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
)
ET_D_PSD_PER_HZ = np.array(
    [
        8.83385322431945442e-49,
        8.20501290795612654e-49,
        3.58369237696470193e-49,
        1.97862355828117043e-49,
        1.54691772612811412e-49,
        1.20395090018587473e-49,
        1.06971126111160879e-49,
        1.02194370960870872e-49,
        1.02432923291918803e-49,
        1.12792810871159091e-49,
        1.32202682728210237e-49,
        2.12804321065693099e-49,
        3.33030480074998683e-49,
        6.84831434812583600e-49,
        1.18267671514987749e-48,
    ]
)

_SQRT_THREE = np.sqrt(3.0)
_TRIANGLE_DIRECTIONS = np.array([[1.0, 0.0, 0.0], [-0.5, _SQRT_THREE / 2.0, 0.0], [-0.5, -_SQRT_THREE / 2.0, 0.0]])
_MICHELSON_ARMS = (
    (_TRIANGLE_DIRECTIONS[0], -_TRIANGLE_DIRECTIONS[2]),
    (_TRIANGLE_DIRECTIONS[1], -_TRIANGLE_DIRECTIONS[0]),
    (_TRIANGLE_DIRECTIONS[2], -_TRIANGLE_DIRECTIONS[1]),
)


@dataclass(frozen=True)
class Population:
    """Isotropic source directions, polarizations, and inclinations."""

    direction: np.ndarray
    polarization: np.ndarray
    cosine_inclination: np.ndarray


@dataclass(frozen=True)
class NoisePair:
    """Signal-present and signal-absent data sharing one noise draw."""

    signal_present: np.ndarray
    signal_absent: np.ndarray
    exact_signal: np.ndarray
    whitened_present: np.ndarray
    whitened_absent: np.ndarray
    whitened_signal: np.ndarray
    whitened_lwa_signal: np.ndarray


@dataclass(frozen=True)
class MapRecovery:
    """One-frequency maximum-a-posteriori LWA calibration recovery."""

    amplitude: np.ndarray
    latent_phase: np.ndarray
    physical_phase: np.ndarray
    objective: float
    iterations: int
    success: bool


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def arm_transfer(
    frequencies_hz: np.ndarray,
    arm_direction: np.ndarray,
    source_direction: np.ndarray,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
) -> np.ndarray:
    """Evaluate the round-trip finite-arm transfer of one directed arm."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or np.any(~np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequencies_hz must be a finite, nonnegative one-dimensional array")
    if not np.isfinite(arm_length_metres) or arm_length_metres <= 0.0:
        raise ValueError("arm_length_metres must be positive and finite")
    arm = _unit_vector(arm_direction, "arm_direction")
    direction = _unit_vector(source_direction, "source_direction")
    travel_time = arm_length_metres / SPEED_OF_LIGHT_METRES_PER_SECOND
    frequency_time = frequencies * travel_time
    projection = float(np.dot(arm, direction))
    return (
        0.5
        * np.exp(-2j * np.pi * frequency_time)
        * (
            np.exp(1j * np.pi * frequency_time * (1.0 - projection)) * np.sinc(frequency_time * (1.0 + projection))
            + np.exp(-1j * np.pi * frequency_time * (1.0 + projection)) * np.sinc(frequency_time * (1.0 - projection))
        )
    )


def _polarization_tensors(source_direction: np.ndarray, polarization: float) -> tuple[np.ndarray, np.ndarray]:
    direction = _unit_vector(source_direction, "source_direction")
    if not np.isfinite(polarization):
        raise ValueError("polarization must be finite")
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, direction)) > POLARIZATION_REFERENCE_ALIGNMENT_LIMIT:
        reference = np.array([0.0, 1.0, 0.0])
    first = _unit_vector(np.cross(reference, direction), "polarization basis")
    second = np.cross(direction, first)
    cosine = np.cos(polarization)
    sine = np.sin(polarization)
    rotated_first = cosine * first + sine * second
    rotated_second = -sine * first + cosine * second
    plus = np.outer(rotated_first, rotated_first) - np.outer(rotated_second, rotated_second)
    cross = np.outer(rotated_first, rotated_second) + np.outer(rotated_second, rotated_first)
    return plus, cross


def et_response_matrix(
    frequencies_hz: np.ndarray,
    source_direction: np.ndarray,
    polarization: float,
    *,
    arm_length_metres: float = ARM_LENGTH_METRES,
) -> np.ndarray:
    """Return the finite-arm ET response with shape (frequency, detector, polarization)."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    plus, cross = _polarization_tensors(source_direction, polarization)
    tensors = (plus, cross)
    response = np.empty((frequencies.size, len(_MICHELSON_ARMS), len(tensors)), dtype=complex)
    for detector, (first_arm, second_arm) in enumerate(_MICHELSON_ARMS):
        first_transfer = arm_transfer(frequencies, first_arm, source_direction, arm_length_metres=arm_length_metres)
        second_transfer = arm_transfer(frequencies, second_arm, source_direction, arm_length_metres=arm_length_metres)
        for mode, tensor in enumerate(tensors):
            first_pattern = np.einsum("i,ij,j->", first_arm, tensor, first_arm)
            second_pattern = np.einsum("i,ij,j->", second_arm, tensor, second_arm)
            response[:, detector, mode] = 0.5 * (first_transfer * first_pattern - second_transfer * second_pattern)
    return response


def et_d_psd(frequencies_hz: np.ndarray) -> np.ndarray:
    """Interpolate the pinned ET-D one-sided PSD samples in log-log space."""
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or np.any(~np.isfinite(frequencies)):
        raise ValueError("frequencies_hz must be a finite one-dimensional array")
    if np.any(frequencies < ET_D_FREQUENCIES_HZ[0]) or np.any(frequencies > ET_D_FREQUENCIES_HZ[-1]):
        raise ValueError("ET-D interpolation is restricted to 20-2000 Hz")
    interpolated = np.exp(
        np.interp(
            np.log(frequencies),
            np.log(ET_D_FREQUENCIES_HZ),
            np.log(ET_D_PSD_PER_HZ),
        )
    )
    for frequency, psd in zip(ET_D_FREQUENCIES_HZ, ET_D_PSD_PER_HZ, strict=True):
        interpolated = np.where(frequencies == frequency, psd, interpolated)
    return interpolated


def generate_population(size: int, *, seed: int) -> Population:
    """Draw an isotropic source population from a recorded random seed."""
    if size <= 0:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    cosine_theta = rng.uniform(-1.0, 1.0, size)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size)
    sine_theta = np.sqrt(1.0 - cosine_theta**2)
    direction = np.column_stack((sine_theta * np.cos(azimuth), sine_theta * np.sin(azimuth), cosine_theta))
    return Population(
        direction=direction,
        polarization=rng.uniform(0.0, np.pi, size),
        cosine_inclination=rng.uniform(-1.0, 1.0, size),
    )


def null_p_value(whitened_null_sample: np.ndarray) -> np.ndarray:
    """Return the single-bin p-value under unit complex Gaussian null noise.

    Under the stated null, the normalized ET sum has independent real and
    imaginary components with variance one half, so twice its squared
    magnitude follows chi-square with two degrees of freedom.
    """
    sample = np.asarray(whitened_null_sample, dtype=complex)
    return np.exp(-(np.abs(sample) ** 2))


def simulate_identical_noise_pair(
    lwa_response: np.ndarray,
    exact_response: np.ndarray,
    polarization: np.ndarray,
    *,
    network_snr: float,
    psd: float,
    delta_f: float,
    rng: np.random.Generator,
) -> NoisePair:
    """Generate exact-response signal data and its identical-noise null pair."""
    lwa_response = np.asarray(lwa_response, dtype=complex)
    exact_response = np.asarray(exact_response, dtype=complex)
    polarization = np.asarray(polarization, dtype=complex)
    if lwa_response.shape != (3, 2) or exact_response.shape != (3, 2) or polarization.shape != (2,):
        raise ValueError("responses must be (3, 2) and polarization must be (2,)")
    if network_snr <= 0.0 or psd <= 0.0 or delta_f <= 0.0:
        raise ValueError("network_snr, psd, and delta_f must be positive")
    raw_lwa_signal = lwa_response @ polarization
    raw_norm = np.linalg.norm(raw_lwa_signal)
    if raw_norm == 0.0:
        raise ValueError("the LWA network response must be nonzero")
    signal_scale = network_snr / raw_norm
    whitened_lwa_signal = signal_scale * raw_lwa_signal
    whitened_signal = signal_scale * (exact_response @ polarization)
    whitening_scale = np.sqrt(psd / (2.0 * delta_f))
    whitened_noise = (rng.normal(size=3) + 1j * rng.normal(size=3)) / np.sqrt(2.0)
    physical_noise = whitening_scale * whitened_noise
    exact_signal = whitening_scale * whitened_signal
    signal_absent = physical_noise
    signal_present = physical_noise + exact_signal
    return NoisePair(
        signal_present=signal_present,
        signal_absent=signal_absent,
        exact_signal=exact_signal,
        whitened_present=signal_present / whitening_scale,
        whitened_absent=signal_absent / whitening_scale,
        whitened_signal=whitened_signal,
        whitened_lwa_signal=whitened_lwa_signal,
    )


def _calibration_factors(parameters: np.ndarray) -> np.ndarray:
    amplitude = parameters[:3]
    latent_phase = parameters[3:]
    imaginary_phase = 1j * latent_phase
    return (1.0 + amplitude) * (2.0 + imaginary_phase) / (2.0 - imaginary_phase)


def lwa_null_residual_energy(whitened_data: np.ndarray, calibration_factors: np.ndarray) -> float:
    """Return the calibrated ET-LWA null-projection energy.

    The uncalibrated ET response has left-null vector ``(1, 1, 1)``.
    Multiplying detector rows by factors ``c_i`` therefore changes the
    left-null vector to ``1 / conj(c_i)``.  Normalizing that vector gives the
    rank-one projector energy without a potentially ill-conditioned Gram
    solve.
    """
    data = np.asarray(whitened_data, dtype=complex)
    factors = np.asarray(calibration_factors, dtype=complex)
    if data.shape != (3,) or factors.shape != (3,):
        raise ValueError("whitened_data and calibration_factors must each contain three detectors")
    if np.any(factors == 0.0):
        raise ValueError("calibration_factors must be nonzero")
    inverse_factors = 1.0 / factors
    residual = np.sum(data * inverse_factors)
    return float(np.abs(residual) ** 2 / np.sum(np.abs(inverse_factors) ** 2))


def recover_lwa_map(lwa_response: np.ndarray, whitened_data: np.ndarray, *, prior_sigma: float) -> MapRecovery:
    """Recover the exact one-bin MAP of the LWA nullcal model.

    This is the frequency-domain specialization of nullcal's projector
    likelihood with equal detector PSDs and independent zero-mean Gaussian
    amplitude and latent-phase priors.
    """
    response = np.asarray(lwa_response, dtype=complex)
    data = np.asarray(whitened_data, dtype=complex)
    if response.shape != (3, 2) or data.shape != (3,):
        raise ValueError("lwa_response must be (3, 2) and whitened_data must be (3,)")
    if not np.isfinite(prior_sigma) or prior_sigma <= 0.0:
        raise ValueError("prior_sigma must be positive and finite")

    def objective(parameters):
        residual_energy = lwa_null_residual_energy(data, _calibration_factors(parameters))
        return float(residual_energy + np.dot(parameters, parameters) / prior_sigma**2)

    zero_parameters = np.zeros(6)
    zero_objective = objective(zero_parameters)
    if zero_objective < EXACT_ZERO_OBJECTIVE_TOLERANCE:
        return MapRecovery(
            amplitude=zero_parameters[:3],
            latent_phase=zero_parameters[3:],
            physical_phase=zero_parameters[3:],
            objective=zero_objective,
            iterations=0,
            success=True,
        )
    solution = minimize(
        objective,
        zero_parameters,
        method="L-BFGS-B",
        bounds=[(-0.5, 0.5)] * 3 + [(-1.0, 1.0)] * 3,
        options={"maxiter": 200},
    )
    if not solution.success:
        solution = minimize(
            objective,
            solution.x,
            method="Powell",
            bounds=[(-0.5, 0.5)] * 3 + [(-1.0, 1.0)] * 3,
            options={"maxiter": 400},
        )
    amplitude = solution.x[:3]
    latent_phase = solution.x[3:]
    return MapRecovery(
        amplitude=amplitude,
        latent_phase=latent_phase,
        physical_phase=2.0 * np.arctan(latent_phase / 2.0),
        objective=float(solution.fun),
        iterations=int(solution.nit),
        success=bool(solution.success),
    )


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def _source_block_bootstrap_p95_upper(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    """Return a one-sided 95% bootstrap bound for a population p95.

    Rows are independent source draws and columns are repeated noise draws for
    that source. Resampling whole rows preserves that clustered design.
    """
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    source_indices = rng.integers(0, values.shape[0], size=(resamples, values.shape[0]))
    resampled = values[source_indices].reshape(resamples, -1)
    estimates = np.quantile(resampled, 0.95, axis=1)
    return max(float(np.quantile(values, 0.95)), float(np.quantile(estimates, 0.95)))


def evaluate_frequency(  # noqa: PLR0915 - the study keeps one explicit paired-data loop
    frequency_hz: float,
    population: Population,
    *,
    noise_realizations: int,
    noise_seed: int,
    network_snr: float,
    prior_sigma: float,
    significance_alpha: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Evaluate leakage significance and paired LWA-MAP bias at one frequency."""
    if noise_realizations <= 0:
        raise ValueError("noise_realizations must be positive")
    if not 0.0 < significance_alpha < 1.0:
        raise ValueError("significance_alpha must lie strictly between zero and one")
    population_size = population.direction.shape[0]
    if population.polarization.shape != (population_size,) or population.cosine_inclination.shape != (population_size,):
        raise ValueError("population arrays must have a common leading size")

    psd = float(et_d_psd(np.array([frequency_hz]))[0])
    angular_frequency_arm = 2.0 * np.pi * frequency_hz * ARM_LENGTH_METRES / SPEED_OF_LIGHT_METRES_PER_SECOND
    rng = np.random.default_rng(noise_seed)
    sky_factors = []
    leakage_snrs = []
    physical_signal_norms = []
    present_statistics = []
    absent_statistics = []
    paired_statistic_increases = []
    amplitude_biases_percent = []
    phase_biases_degrees = []
    map_failure_count = 0

    for direction, polarization_angle, cosine_inclination in zip(
        population.direction,
        population.polarization,
        population.cosine_inclination,
        strict=True,
    ):
        lwa_response = et_response_matrix(np.array([0.0]), direction, polarization_angle)[0]
        exact_response = et_response_matrix(np.array([frequency_hz]), direction, polarization_angle)[0]
        source_polarization = np.array([(1.0 + cosine_inclination**2) / 2.0, -1j * cosine_inclination], dtype=complex)
        raw_lwa_signal = lwa_response @ source_polarization
        signal_scale = network_snr / np.linalg.norm(raw_lwa_signal)
        whitened_signal = signal_scale * (exact_response @ source_polarization)
        leakage_snr = float(np.abs(np.sum(whitened_signal) / np.sqrt(3.0)))
        leakage_snrs.append(leakage_snr)
        sky_factors.append(leakage_snr / (network_snr * angular_frequency_arm))
        physical_signal_norms.append(np.linalg.norm(whitened_signal) * np.sqrt(psd / 2.0))

        for _ in range(noise_realizations):
            pair = simulate_identical_noise_pair(
                lwa_response,
                exact_response,
                source_polarization,
                network_snr=network_snr,
                psd=psd,
                delta_f=1.0,
                rng=rng,
            )
            absent_null = np.sum(pair.whitened_absent) / np.sqrt(3.0)
            present_null = np.sum(pair.whitened_present) / np.sqrt(3.0)
            absent_statistic = float(np.abs(absent_null) ** 2)
            present_statistic = float(np.abs(present_null) ** 2)
            absent_statistics.append(absent_statistic)
            present_statistics.append(present_statistic)
            paired_statistic_increases.append(present_statistic - absent_statistic)

            lwa_control_map = recover_lwa_map(
                lwa_response,
                pair.whitened_absent + pair.whitened_lwa_signal,
                prior_sigma=prior_sigma,
            )
            exact_response_map = recover_lwa_map(lwa_response, pair.whitened_present, prior_sigma=prior_sigma)
            map_failure_count += int(not lwa_control_map.success) + int(not exact_response_map.success)
            amplitude_biases_percent.append(
                100.0 * np.max(np.abs(exact_response_map.amplitude - lwa_control_map.amplitude))
            )
            phase_biases_degrees.append(
                np.rad2deg(np.max(np.abs(exact_response_map.physical_phase - lwa_control_map.physical_phase)))
            )

    sky_factors = np.asarray(sky_factors)
    leakage_snrs = np.asarray(leakage_snrs)
    physical_signal_norms = np.asarray(physical_signal_norms)
    present_statistics = np.asarray(present_statistics)
    absent_statistics = np.asarray(absent_statistics)
    paired_statistic_increases = np.asarray(paired_statistic_increases)
    amplitude_biases_percent = np.asarray(amplitude_biases_percent)
    phase_biases_degrees = np.asarray(phase_biases_degrees)
    amplitude_biases_by_source = amplitude_biases_percent.reshape(population_size, noise_realizations)
    phase_biases_by_source = phase_biases_degrees.reshape(population_size, noise_realizations)
    return {
        "frequency_hz": float(frequency_hz),
        "two_pi_f_l_over_c": angular_frequency_arm,
        "et_d_psd_per_hz": psd,
        "population_size": population_size,
        "noise_realizations": noise_realizations,
        "bias_sample_count": int(amplitude_biases_percent.size),
        "sky_factor_p05": _quantile(sky_factors, 0.05),
        "sky_factor_median": _quantile(sky_factors, 0.50),
        "sky_factor_p95": _quantile(sky_factors, 0.95),
        "sky_factor_p99": _quantile(sky_factors, 0.99),
        "sky_factor_maximum": float(np.max(sky_factors)),
        "leakage_snr_p05": _quantile(leakage_snrs, 0.05),
        "leakage_snr_median": _quantile(leakage_snrs, 0.50),
        "leakage_snr_p95": _quantile(leakage_snrs, 0.95),
        "physical_signal_norm_median": _quantile(physical_signal_norms, 0.50),
        "absent_false_alarm_fraction": float(np.mean(null_p_value(np.sqrt(absent_statistics)) < significance_alpha)),
        "present_detection_fraction": float(np.mean(null_p_value(np.sqrt(present_statistics)) < significance_alpha)),
        "paired_statistic_increase_median": _quantile(paired_statistic_increases, 0.50),
        "paired_statistic_increase_p05": _quantile(paired_statistic_increases, 0.05),
        "paired_statistic_increase_p95": _quantile(paired_statistic_increases, 0.95),
        "amplitude_bias_percent_median": _quantile(amplitude_biases_percent, 0.50),
        "amplitude_bias_percent_p95": _quantile(amplitude_biases_percent, 0.95),
        "amplitude_bias_percent_p95_upper95": _source_block_bootstrap_p95_upper(
            amplitude_biases_by_source,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        "amplitude_bias_percent_p99": _quantile(amplitude_biases_percent, 0.99),
        "amplitude_bias_percent_maximum": float(np.max(amplitude_biases_percent)),
        "phase_bias_degrees_median": _quantile(phase_biases_degrees, 0.50),
        "phase_bias_degrees_p95": _quantile(phase_biases_degrees, 0.95),
        "phase_bias_degrees_p95_upper95": _source_block_bootstrap_p95_upper(
            phase_biases_by_source,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 1,
        ),
        "phase_bias_degrees_p99": _quantile(phase_biases_degrees, 0.99),
        "phase_bias_degrees_maximum": float(np.max(phase_biases_degrees)),
        "map_failure_count": map_failure_count,
    }


__all__ = [
    "ARM_LENGTH_METRES",
    "SPEED_OF_LIGHT_METRES_PER_SECOND",
    "MapRecovery",
    "NoisePair",
    "Population",
    "arm_transfer",
    "et_d_psd",
    "et_response_matrix",
    "evaluate_frequency",
    "generate_population",
    "lwa_null_residual_energy",
    "null_p_value",
    "recover_lwa_map",
    "simulate_identical_noise_pair",
]
