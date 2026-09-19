"""Build the live JAX likelihood only from immutable reference inputs.

The waveform-generating bilby path was intentionally retired with the runtime
migration. This module never regenerates or overwrites the frozen artifacts.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nullcal.data import InterferometerData
from nullcal.likelihood import RecalibrationLikelihood
from nullcal.null_stream.projector import compute_projector
from nullcal.null_stream.whiten import compute_whitened_antenna_response

from . import config

REFERENCE_DIR = Path(__file__).parent / "reference"
OUTPUT_ARTIFACT_PATH = REFERENCE_DIR / "artifacts.npz"
INPUT_ARTIFACT_PATH = REFERENCE_DIR / "inputs.npz"
FED_BACK_INPUT_KEYS = frozenset({"time_frequency_filter"})

ET_BEAM_PATTERN = np.array(
    [[-1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)], [np.sqrt(6.0) / 3.0, 0.0], [-1.0 / np.sqrt(6.0), 1.0 / np.sqrt(2.0)]]
)


def _reference_data(reference_inputs: dict[str, np.ndarray]) -> InterferometerData:
    psd = reference_inputs["power_spectral_density"]
    frequency_array = np.fft.rfftfreq(
        int(config.DURATION * config.SAMPLING_FREQUENCY),
        1.0 / config.SAMPLING_FREQUENCY,
    )
    shared_mask = (
        (frequency_array >= config.MINIMUM_FREQUENCY)
        & (frequency_array <= config.MAXIMUM_FREQUENCY)
        & np.all(np.isfinite(psd) & (psd > 0.0), axis=0)
    )
    return InterferometerData(
        psd=psd,
        strain=np.zeros_like(reference_inputs["whitened_frequency_domain_strain"]),
        mask=np.broadcast_to(shared_mask, psd.shape).copy(),
        frequency_array=frequency_array,
        duration=float(config.DURATION),
        sampling_frequency=float(config.SAMPLING_FREQUENCY),
        start_time=float(config.start_time()),
        name=config.DETECTOR_NAMES,
    )


def parameter_arrays() -> dict[str, jnp.ndarray]:
    """Return the frozen named parameters as detector-by-knot arrays."""
    named = config.calibration_parameters()
    amplitude = np.empty((len(config.DETECTOR_NAMES), config.N_POINTS))
    phase = np.empty_like(amplitude)
    for detector_index, detector_name in enumerate(config.DETECTOR_NAMES):
        for knot_index in range(config.N_POINTS):
            amplitude[detector_index, knot_index] = named[f"recalib_{detector_name}_amplitude_{knot_index}"]
            phase[detector_index, knot_index] = named[f"recalib_{detector_name}_phase_{knot_index}"]
    return {"amplitude": jnp.asarray(amplitude), "phase": jnp.asarray(phase)}


def build_likelihood_from_reference_inputs() -> RecalibrationLikelihood:
    """Build the comparison path without importing or invoking bilby."""
    with np.load(INPUT_ARTIFACT_PATH) as stored_inputs:
        reference_inputs = {key: stored_inputs[key] for key in stored_inputs.files}
    with np.load(OUTPUT_ARTIFACT_PATH) as stored_outputs:
        time_frequency_filter = stored_outputs["time_frequency_filter"]

    return RecalibrationLikelihood(
        interferometers=_reference_data(reference_inputs),
        knot_frequencies=np.geomspace(config.MINIMUM_FREQUENCY, config.MAXIMUM_FREQUENCY, config.N_POINTS),
        wavelet_transform_frequency_resolution=config.FREQUENCY_RESOLUTION,
        wavelet_transform_nx=config.NX,
        time_frequency_filter=time_frequency_filter,
        amplitude_prior_sigma=0.05,
        phase_prior_sigma=0.05,
        whitened_frequency_domain_strain=reference_inputs["whitened_frequency_domain_strain"],
    )


def compute_artifacts(likelihood: RecalibrationLikelihood) -> dict[str, np.ndarray | float]:
    """Compute quantities anchored by the frozen pre-migration artifacts."""
    params = parameter_arrays()
    zeros = jnp.zeros(likelihood.parameter_shape, dtype=jnp.float64)
    frequency_mask = np.all(np.asarray(likelihood.interferometers.mask), axis=0)
    whitened_response = compute_whitened_antenna_response(
        ET_BEAM_PATTERN,
        np.asarray(likelihood.interferometers.psd),
        1.0 / float(likelihood.interferometers.duration),
        frequency_mask,
    )
    projector = compute_projector(whitened_response, frequency_mask=frequency_mask)

    uncalibrated_frequency_domain = likelihood._frequency_domain_null_stream(zeros, zeros)
    calibrated_frequency_domain = likelihood._frequency_domain_null_stream(params["amplitude"], params["phase"])
    uncalibrated_time_frequency = likelihood._time_frequency_null_stream(zeros, zeros)
    calibrated_time_frequency = likelihood._time_frequency_null_stream(params["amplitude"], params["phase"])

    calibration_factor = np.zeros_like(np.asarray(likelihood.interferometers.strain))
    calibration_factor[:, frequency_mask] = np.asarray(
        likelihood._calibration_factor(params["amplitude"], params["phase"])
    )
    probe = config.wavelet_probe_input(n_frequencies=uncalibrated_frequency_domain.shape[1])
    wavelet_probe_output = likelihood.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=probe)

    return {
        "frequency_mask": frequency_mask,
        "time_frequency_filter": np.asarray(likelihood.time_frequency_filter),
        "whitened_antenna_response": whitened_response,
        "projector": projector,
        "calibration_factor": calibration_factor,
        "uncalibrated_frequency_domain_null_stream": uncalibrated_frequency_domain,
        "calibrated_frequency_domain_null_stream": calibrated_frequency_domain,
        "uncalibrated_time_frequency_domain_null_stream": uncalibrated_time_frequency,
        "calibrated_time_frequency_domain_null_stream": calibrated_time_frequency,
        "wavelet_probe_output": wavelet_probe_output,
        "log_likelihood": likelihood.log_likelihood_fn(params),
        "noise_log_likelihood": likelihood.noise_log_likelihood(),
    }


__all__ = ["FED_BACK_INPUT_KEYS", "build_likelihood_from_reference_inputs", "compute_artifacts", "parameter_arrays"]
