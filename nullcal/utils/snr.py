import numpy as np
from bilby.gw.utils import noise_weighted_inner_product as\
    noise_weighted_uncorrelated_inner_product


def optimal_uncorrelated_null_stream_snr_squared(
        signals,
        power_spectral_densities,
        duration):
    """Compute the optimal uncorrelated null stream
    signal-to-noise ratio. The interferometers are
    assumed to be uncorrelated to each other.

    Args:
        signals (array-like): Frequency domain signals in the interferometers (detector, frequency).
        power_spectral_densities (array-like): Noise power spectral densities of the interferometers (detector, frequency).
        duration (float): Duration in second.

    Returns:
        float: Optimal signal-to-noise ratio squared of uncorrelated null stream.
    """
    # Direct sum of the signals
    null_stream = np.sum(signals, axis=0) / np.sqrt(len(signals))

    # Power spectral density
    power_spectral_density = np.sum(power_spectral_densities,
                                    axis=0) / len(power_spectral_densities)
    return noise_weighted_uncorrelated_inner_product(
        aa=null_stream,
        bb=null_stream,
        power_spectral_density=power_spectral_density,
        duration=duration
    )
