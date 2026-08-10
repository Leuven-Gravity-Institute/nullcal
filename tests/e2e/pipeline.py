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
from bilby.gw.detector import CubicSpline, InterferometerList
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator

from nullcal.likelihood import RecalibrationLikelihood

from . import config

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
        for logger, level in zip(loggers, previous):
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


def compute_artifacts(likelihood: RecalibrationLikelihood) -> dict[str, np.ndarray | float]:
    """Every quantity the reference freezes, computed from one likelihood instance.

    ``noise_log_likelihood`` is deliberately absent: on the code this reference was taken
    from it raises ``IndexError`` and cannot produce a value. That behaviour is pinned by
    ``test_noise_log_likelihood_filter_domain`` instead, which is a regression test, not a
    reference artifact.
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

    probe = config.wavelet_probe_input(n_frequencies=uncalibrated_frequency_domain.shape[1])
    wavelet_probe_output = likelihood.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=probe)

    likelihood.parameters = dict(parameters)
    log_likelihood = float(likelihood.log_likelihood())

    return {
        "frequency_mask": frequency_mask,
        "time_frequency_filter": null_stream.time_frequency_filter,
        "whitened_antenna_response": whitened_response,
        "projector": projector,
        "calibration_factor": calibration_factor,
        "uncalibrated_frequency_domain_null_stream": uncalibrated_frequency_domain,
        "calibrated_frequency_domain_null_stream": calibrated_frequency_domain,
        "calibrated_time_frequency_domain_null_stream": calibrated_time_frequency,
        "wavelet_probe_output": wavelet_probe_output,
        "log_likelihood": np.float64(log_likelihood),
    }
