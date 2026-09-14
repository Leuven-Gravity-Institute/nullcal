"""Shared fixtures for the clustering tests.

A deterministic Einstein Telescope triangle carrying an analytically known burst: a Gaussian-
enveloped sine at a chosen frequency and time, injected as zero-noise data. Because the burst's
time and frequency are chosen, the time-frequency pixel it must occupy is known in advance, and the
clustering results can be checked against it rather than against a stored map.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullcal.time_frequency_transform.wavelet_transforms import WaveletTransform

logging.getLogger("bilby").setLevel(logging.WARNING)
logging.getLogger("nullcal").setLevel(logging.WARNING)

SAMPLING_FREQUENCY = 512.0
DURATION = 4.0
FREQUENCY_RESOLUTION = 8.0
N_SAMPLES = int(SAMPLING_FREQUENCY * DURATION)

BURST_LAYER = 10
BURST_FREQUENCY = BURST_LAYER * FREQUENCY_RESOLUTION  # 80 Hz, the centre of layer 10
BURST_CENTRE_TIME = 2.0  # seconds from the start of the segment
BURST_WIDTH = 0.05
BURST_AMPLITUDE = 1e-22


@pytest.fixture(scope="module")
def time_frequency_transform():
    return WaveletTransform(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_resolution=FREQUENCY_RESOLUTION,
    )


@pytest.fixture(scope="module")
def burst_time_series():
    """A Gaussian-enveloped sine at ``BURST_FREQUENCY``, centred at ``BURST_CENTRE_TIME``."""
    time_array = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
    envelope = np.exp(-((time_array - BURST_CENTRE_TIME) ** 2) / (2 * BURST_WIDTH**2))
    return BURST_AMPLITUDE * envelope * np.sin(2 * np.pi * BURST_FREQUENCY * time_array)


@pytest.fixture(scope="module")
def interferometers(burst_time_series):
    """An ET triangle whose three detectors all carry the same burst and no noise.

    Identical strain in all three is deliberate: it makes the combined power map exactly three
    times the single-detector map, which is an arithmetic prediction the map test checks.
    """
    ifos = InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=0.0)
    frequency_domain_burst = np.fft.rfft(burst_time_series) / SAMPLING_FREQUENCY
    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_burst,
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=0.0,
        )
    return ifos


@pytest.fixture(scope="module")
def broadband_interferometers():
    """An ET triangle carrying an impulse, so every frequency layer has power.

    Needed where a test has to observe what happens at a chosen band edge: a narrowband burst leaves
    the edge layers empty and the behaviour there unobservable.
    """
    ifos = InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=0.0)
    impulse = np.zeros(N_SAMPLES)
    impulse[N_SAMPLES // 2] = 1e-21
    frequency_domain_impulse = np.fft.rfft(impulse) / SAMPLING_FREQUENCY
    for ifo in ifos:
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_impulse,
            sampling_frequency=SAMPLING_FREQUENCY,
            duration=DURATION,
            start_time=0.0,
        )
    return ifos


@pytest.fixture(scope="module")
def burst_parameters():
    """The burst's known coordinates, so test modules do not restate them.

    ``layer`` and ``centre_time`` are inputs to the injection, not outputs of any transform; the
    tests compare the clustering against them.
    """
    return {
        "layer": BURST_LAYER,
        "frequency": BURST_FREQUENCY,
        "centre_time": BURST_CENTRE_TIME,
        "duration": DURATION,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "frequency_resolution": FREQUENCY_RESOLUTION,
    }
