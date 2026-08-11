"""Frozen end-to-end configuration for the characterisation harness.

This module is the single source of truth for the reference configuration. It is imported
both by ``generate_reference.py`` (which writes the frozen artifacts) and by
``test_reference_artifacts.py`` (which checks the current code against them), so the two
cannot drift.

Nothing here may change without regenerating the reference artifacts in the same commit —
see ``tests/e2e/README.md``.
"""

from __future__ import annotations

import numpy as np

#: Analysis configuration. Small on purpose: 8 s at 2048 Hz keeps the reference run under a
#: minute on the review machine while still exercising the full pipeline.
MINIMUM_FREQUENCY = 20.0
MAXIMUM_FREQUENCY = 1024.0
SAMPLING_FREQUENCY = 2048
DURATION = 8
N_POINTS = 10
FREQUENCY_RESOLUTION = 16
NX = 4.0
CLUSTERING_THRESHOLD = 0.1

#: Seeds both bilby's global RNG (noise realisation) and the calibration draw.
SEED = 20260810

#: A single GW150914-like BBH, face-off-ish, loud enough that the clustering finds pixels.
SOURCE_PARAMETERS = {
    "mass_1": 36.0,
    "mass_2": 29.0,
    "a_1": 0.0,
    "a_2": 0.0,
    "tilt_1": 0.0,
    "tilt_2": 0.0,
    "phi_12": 0.0,
    "phi_jl": 0.0,
    "theta_jn": 0.4,
    "psi": 2.659,
    "phase": 1.3,
    "geocent_time": 1126259462.4,
    "ra": 1.375,
    "dec": -1.2108,
    "luminosity_distance": 800.0,
}

WAVEFORM_ARGUMENTS = {
    "waveform_approximant": "IMRPhenomXPHM",
    "reference_frequency": 50.0,
    "minimum_frequency": MINIMUM_FREQUENCY,
}

DETECTOR_NAMES = ("ET1", "ET2", "ET3")


def start_time() -> int:
    """Segment start time, aligned so the merger sits mid-segment."""
    return int(SOURCE_PARAMETERS["geocent_time"] - DURATION / 2)


def calibration_parameters() -> dict[str, float]:
    """The fixed calibration-error vector the reference is evaluated at.

    Drawn once from a seeded generator rather than hard-coded, so the construction is visible;
    the seed makes it reproducible. Amplitude and phase deviations are 2%-scale, which is
    inside the O3 Livingston envelope the paper uses.
    """
    rng = np.random.default_rng(SEED)
    nodes = np.logspace(np.log10(MINIMUM_FREQUENCY), np.log10(MAXIMUM_FREQUENCY), N_POINTS)
    params: dict[str, float] = {}
    for name in DETECTOR_NAMES:
        amplitudes = rng.normal(0.0, 0.02, N_POINTS)
        phases = rng.normal(0.0, 0.02, N_POINTS)
        for index in range(N_POINTS):
            params[f"recalib_{name}_amplitude_{index}"] = float(amplitudes[index])
            params[f"recalib_{name}_phase_{index}"] = float(phases[index])
            params[f"recalib_{name}_frequency_{index}"] = float(nodes[index])
    return params


def wavelet_probe_input(n_frequencies: int) -> np.ndarray:
    """A deterministic frequency-domain probe for the wavelet transform.

    Isolates the WDM transform from the rest of the pipeline: it does not depend on bilby, the
    detectors, or the injection, so a change in this artifact localises to
    ``time_frequency_transform`` alone.
    """
    rng = np.random.default_rng(SEED + 1)
    real = rng.normal(0.0, 1.0, n_frequencies)
    imag = rng.normal(0.0, 1.0, n_frequencies)
    probe = (real + 1j * imag) / np.sqrt(2.0)
    probe[0] = 0.0
    return probe
