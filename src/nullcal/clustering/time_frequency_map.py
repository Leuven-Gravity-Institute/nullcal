from __future__ import annotations

import numpy as np

from ..null_stream.whiten import compute_whitened_frequency_domain_strain
from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)


def construct_time_frequency_map(interferometers, frequency_resolution, nx, *arg, **kwargs):
    ndet = len(interferometers)
    Nf = int(interferometers[0].sampling_frequency / 2 / frequency_resolution)
    Nt = int(len(interferometers[0].time_array) / Nf)
    # Compute the whitened time-frequency array
    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in interferometers])
    power_spectral_density_array = np.array([ifo.power_spectral_density_array for ifo in interferometers])
    whitened_frequency_domain_strain = compute_whitened_frequency_domain_strain(
            frequency_domain_strain_array,
            power_spectral_density_array,
            1 / interferometers[0].duration,
            interferometers[0].frequency_mask)
    whitened_time_frequency_domain_strain_0 = transform_wavelet_freq(whitened_frequency_domain_strain[0],
                                                                     Nf,
                                                                     Nt,
                                                                     nx)
    whitened_time_frequency_domain_strain_quadrature_0 = transform_wavelet_freq_quadrature(whitened_frequency_domain_strain[0],
                                                                                           Nf,
                                                                                           Nt,
                                                                                           nx)
    combined_power = whitened_time_frequency_domain_strain_0 ** 2 + whitened_time_frequency_domain_strain_quadrature_0 ** 2
    for i in range(1, ndet):
        whitened_time_frequency_domain_strain_i = transform_wavelet_freq(whitened_frequency_domain_strain[i],
                                                                         Nf,
                                                                         Nt,
                                                                         nx)
        whitened_time_frequency_domain_strain_quadrature_i = transform_wavelet_freq_quadrature(whitened_frequency_domain_strain[i],
                                                                                               Nf,
                                                                                               Nt,
                                                                                               nx)
        combined_power += (whitened_time_frequency_domain_strain_i ** 2 + whitened_time_frequency_domain_strain_quadrature_i ** 2)
    return combined_power
