"""A submodule for constructing time-frequency maps."""

from __future__ import annotations

import numpy as np

from ..data import InterferometerData
from ..null_stream.whiten import compute_whitened_frequency_domain_strain
from ..time_frequency_transform.wavelet_transforms import WaveletTransform


def construct_time_frequency_map(interferometers: InterferometerData, time_frequency_transform: WaveletTransform):
    """Construct the time-frequency map from the interferometers.

    Args:
        interferometers (InterferometerData): Frozen detector arrays and metadata.
        time_frequency_transform (WaveletTransform): A WaveletTransform instance
            for performing wavelet transforms.
    Returns:
        np.ndarray: The combined time-frequency map.
    """
    n_det = interferometers.strain.shape[0]
    # Compute the whitened time-frequency array
    whitened_frequency_domain_strain = compute_whitened_frequency_domain_strain(
        interferometers.strain,
        interferometers.psd,
        1 / interferometers.duration,
        np.all(interferometers.mask, axis=0),
    )
    combined_power = np.zeros(time_frequency_transform.shape)
    for i in range(n_det):
        whitened_time_frequency_domain_strain_i = time_frequency_transform.frequency_to_wavelet(
            frequency_domain_data=whitened_frequency_domain_strain[i]
        )
        whitened_time_frequency_domain_strain_quadrature_i = time_frequency_transform.frequency_to_wavelet_quadrature(
            frequency_domain_data=whitened_frequency_domain_strain[i]
        )
        combined_power += (
            np.abs(whitened_time_frequency_domain_strain_i) ** 2
            + np.abs(whitened_time_frequency_domain_strain_quadrature_i) ** 2
        ) / 2
    return combined_power
