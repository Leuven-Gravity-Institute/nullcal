"""Builds the reference pipeline objects from the frozen configuration.

Kept separate from both the generator and the tests so that exactly one construction path
exists. If the generator and the test built the interferometers separately, a divergence
between them would look like a numerical regression.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

import bilby.core.utils.random
import numpy as np
import pandas as pd
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import CubicSpline, InterferometerList, PowerSpectralDensity
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator

from nullcal.likelihood import RecalibrationLikelihood

from . import config

REFERENCE_DIR = Path(__file__).parent / "reference"
OUTPUT_ARTIFACT_PATH = REFERENCE_DIR / "artifacts.npz"
INPUT_ARTIFACT_PATH = REFERENCE_DIR / "inputs.npz"

#: Stored outputs deliberately reused as inputs by the frozen-input comparison path. These
#: values are anchored through the full pipeline instead of counted as reproduced here.
FED_BACK_INPUT_KEYS = frozenset({"time_frequency_filter"})


@contextlib.contextmanager
def quiet_loggers() -> Iterator[None]:
    """Silence bilby's and nullcal's chatter for the duration of a build, then restore it.

    Setting these levels at import time instead would apply during pytest collection, which the
    ``e2e`` marker does not prevent — a normal run would then lose diagnostics from unrelated
    tests sharing the worker.
    """
    loggers = [logging.getLogger("bilby"), logging.getLogger("nullcal")]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in zip(loggers, previous, strict=True):
            logger.setLevel(level)


def build_waveform_generator() -> WaveformGenerator:
    """Waveform generator matching the frozen configuration."""
    return WaveformGenerator(
        duration=config.DURATION,
        sampling_frequency=config.SAMPLING_FREQUENCY,
        start_time=config.start_time(),
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(config.WAVEFORM_ARGUMENTS),
    )


def build_interferometers() -> InterferometerList:
    """ET triangle with the frozen noise realisation and the injected signal."""
    with quiet_loggers():
        bilby.core.utils.random.seed(config.SEED)
        interferometers = InterferometerList(["ET"])
        for interferometer in interferometers:
            interferometer.minimum_frequency = config.MINIMUM_FREQUENCY
            interferometer.maximum_frequency = config.MAXIMUM_FREQUENCY
            interferometer.calibration_model = CubicSpline(
                prefix=f"recalib_{interferometer.name}_",
                minimum_frequency=config.MINIMUM_FREQUENCY,
                maximum_frequency=config.MAXIMUM_FREQUENCY,
                n_points=config.N_POINTS,
            )
        interferometers.set_strain_data_from_power_spectral_densities(
            sampling_frequency=config.SAMPLING_FREQUENCY,
            duration=config.DURATION,
            start_time=config.start_time(),
        )
        injected = dict(config.SOURCE_PARAMETERS)
        injected.update(config.calibration_parameters())
        interferometers.inject_signal(
            waveform_generator=build_waveform_generator(),
            parameters=injected,
        )
        return interferometers


def build_interferometers_from_reference_inputs(reference_inputs: dict[str, np.ndarray]) -> InterferometerList:
    """ET triangle carrying the frozen PSD and metadata, without generating strain."""
    power_spectral_density = reference_inputs["power_spectral_density"]
    whitened_strain = reference_inputs["whitened_frequency_domain_strain"]
    interferometers = InterferometerList(["ET"])

    for index, interferometer in enumerate(interferometers):
        interferometer.minimum_frequency = config.MINIMUM_FREQUENCY
        interferometer.maximum_frequency = config.MAXIMUM_FREQUENCY
        interferometer.calibration_model = CubicSpline(
            prefix=f"recalib_{interferometer.name}_",
            minimum_frequency=config.MINIMUM_FREQUENCY,
            maximum_frequency=config.MAXIMUM_FREQUENCY,
            n_points=config.N_POINTS,
        )
        interferometer.strain_data.set_from_frequency_domain_strain(
            frequency_domain_strain=np.zeros_like(whitened_strain[index]),
            sampling_frequency=config.SAMPLING_FREQUENCY,
            duration=config.DURATION,
            start_time=config.start_time(),
        )
        interferometer.power_spectral_density = PowerSpectralDensity.from_power_spectral_density_array(
            frequency_array=interferometer.frequency_array,
            psd_array=power_spectral_density[index],
        )

    return interferometers


def build_likelihood(tmp_path: Path | None = None) -> RecalibrationLikelihood:
    """The full recalibration likelihood, with an injection-clustering time-frequency filter."""
    interferometers = build_interferometers()
    waveform_generator = build_waveform_generator()

    directory = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    parameter_file = directory / "clustering_parameters.csv"
    pd.DataFrame([config.SOURCE_PARAMETERS]).to_csv(parameter_file, index=False)

    with quiet_loggers():
        return RecalibrationLikelihood(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            wavelet_transform_frequency_resolution=config.FREQUENCY_RESOLUTION,
            wavelet_transform_nx=config.NX,
            clustering_parameter_file=str(parameter_file),
            clustering_threshold=config.CLUSTERING_THRESHOLD,
        )


def build_likelihood_from_reference_inputs() -> RecalibrationLikelihood:
    """Build the comparison pipeline from frozen inputs, without waveform generation."""
    with np.load(INPUT_ARTIFACT_PATH) as stored_inputs:
        reference_inputs = {key: stored_inputs[key] for key in stored_inputs.files}
    with np.load(OUTPUT_ARTIFACT_PATH) as stored_outputs:
        # The filter is fed back to isolate the downstream numerical comparison from waveform
        # generation. It is therefore an input to this path, not an independently reproduced
        # output; a separate full-pipeline test anchors the InjectionClustering-derived value.
        time_frequency_filter = stored_outputs["time_frequency_filter"]

    interferometers = build_interferometers_from_reference_inputs(reference_inputs)
    with quiet_loggers():
        likelihood = RecalibrationLikelihood(
            interferometers=interferometers,
            waveform_generator=None,
            wavelet_transform_frequency_resolution=config.FREQUENCY_RESOLUTION,
            wavelet_transform_nx=config.NX,
            time_frequency_filter=time_frequency_filter,
        )

    # ``NullStream`` computes and caches this input during construction. The frozen-input path
    # replaces that cache directly because the archived quantity is already whitened; round-tripping
    # through an unwhitened strain would add arithmetic and weaken the reference comparison.
    likelihood.null_stream_calculator._whitened_frequency_domain_strain_array = np.array(
        reference_inputs["whitened_frequency_domain_strain"], copy=True
    )
    return likelihood


def compute_artifacts(likelihood: RecalibrationLikelihood) -> dict[str, np.ndarray | float]:
    """Every quantity the reference freezes, computed from one likelihood instance.

    ``noise_log_likelihood`` and the uncalibrated time-frequency null stream were absent from the
    original reference because the code then raised ``IndexError`` and could produce no value. The
    R17 fix makes them computable, and they are frozen here for the reason the harness exists: the
    structural tests in ``test_noise_log_likelihood_filter_domain`` pin the repaired *shape* of the
    result — confined to the filter, finite, sharing the calibrated branch's support — but nothing
    pinned its *values*, so a later change to the uncalibrated branch alone would have passed.
    """
    null_stream = likelihood.null_stream_calculator
    calibration = likelihood.null_stream_calculator
    parameters = config.calibration_parameters()

    frequency_mask = null_stream.frequency_mask
    whitened_response = null_stream._whitened_antenna_response

    from nullcal.null_stream.projector import compute_projector

    projector = compute_projector(whitened_response, frequency_mask=frequency_mask)

    uncalibrated_frequency_domain = null_stream.compute_uncalibrated_frequency_domain_null_stream()
    calibration_factor = calibration.construct_calibration_factor_from_parameters(parameters)
    calibrated_frequency_domain = null_stream.compute_calibrated_frequency_domain_null_stream(
        calibration_factor=calibration_factor
    )
    calibrated_time_frequency = null_stream.compute_calibrated_time_frequency_domain_null_stream_from_parameters(
        parameters=parameters
    )

    uncalibrated_time_frequency = null_stream.compute_uncalibrated_time_frequency_domain_null_stream()

    probe = config.wavelet_probe_input(n_frequencies=uncalibrated_frequency_domain.shape[1])
    wavelet_probe_output = likelihood.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=probe)

    likelihood.parameters = dict(parameters)
    log_likelihood = float(likelihood.log_likelihood())
    noise_log_likelihood = float(likelihood.noise_log_likelihood())

    return {
        "frequency_mask": frequency_mask,
        "time_frequency_filter": null_stream.time_frequency_filter,
        "whitened_antenna_response": whitened_response,
        "projector": projector,
        "calibration_factor": calibration_factor,
        "uncalibrated_frequency_domain_null_stream": uncalibrated_frequency_domain,
        "calibrated_frequency_domain_null_stream": calibrated_frequency_domain,
        "uncalibrated_time_frequency_domain_null_stream": uncalibrated_time_frequency,
        "calibrated_time_frequency_domain_null_stream": calibrated_time_frequency,
        "wavelet_probe_output": wavelet_probe_output,
        "log_likelihood": np.float64(log_likelihood),
        "noise_log_likelihood": np.float64(noise_log_likelihood),
    }
