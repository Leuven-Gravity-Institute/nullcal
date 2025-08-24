"""A submodule for constructing time-frequency maps."""

from __future__ import annotations

import numpy as np
from bilby.gw.detector import InterferometerList

from ..null_stream.whiten import compute_whitened_frequency_domain_strain
from ..time_frequency_transform.wavelet_transforms import WaveletTransform


def construct_time_frequency_map(interferometers: InterferometerList, time_frequency_transform: WaveletTransform):
    """Construct the time-frequency map from the interferometers.

    Args:
        interferometers (InterferometerList): A list of interferometers.
        time_frequency_transform (WaveletTransform): A WaveletTransform instance
            for performing wavelet transforms.
    Returns:
        np.ndarray: The combined time-frequency map.
    """
    n_det = len(interferometers)
    # Compute the whitened time-frequency array
    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in interferometers])
    power_spectral_density_array = np.array([ifo.power_spectral_density_array for ifo in interferometers])
    whitened_frequency_domain_strain = compute_whitened_frequency_domain_strain(
        frequency_domain_strain_array,
        power_spectral_density_array,
        1 / interferometers[0].duration,
        interferometers[0].frequency_mask,
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
