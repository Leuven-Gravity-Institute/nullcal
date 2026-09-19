"""
Helper functions to compute signal-to-noise ratio.
"""

from __future__ import annotations

import numpy as np


def optimal_uncorrelated_null_stream_snr_squared(signals, power_spectral_densities, duration):
    """Compute the optimal uncorrelated null stream
    signal-to-noise ratio. The interferometers are
    assumed to be uncorrelated to each other.

    Args:
        signals (array-like): Frequency domain signals in the interferometers (detector, frequency).
        power_spectral_densities (array-like): Noise power spectral densities of the interferometers
            (detector, frequency).
        duration (float): Duration in second.

    Returns:
        float: Optimal signal-to-noise ratio squared of uncorrelated null stream.
    """
    # Direct sum of the signals
    null_stream = np.sum(signals, axis=0) / np.sqrt(len(signals))

    # Power spectral density
    power_spectral_density = np.sum(power_spectral_densities, axis=0) / len(power_spectral_densities)
    return 4.0 / duration * np.sum(np.conj(null_stream) * null_stream / power_spectral_density)
