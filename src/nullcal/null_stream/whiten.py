from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_whitened_frequency_domain_strain(frequency_domain_strain_array,
                                             power_spectral_density_array,
                                             delta_f,
                                             frequency_mask):
    ndet, flen = frequency_domain_strain_array.shape
    output = np.zeros_like(frequency_domain_strain_array)
    scaling = delta_f * 2
    for j in range(ndet):
        for i in range(flen):
            if frequency_mask[i]:
                output[j,i] = frequency_domain_strain_array[j,i] / np.sqrt(power_spectral_density_array[j,i] / scaling)
    return output

@njit
def compute_whitened_antenna_response(antenna_response_matrix,
                                      power_spectral_density_array,
                                      delta_f,
                                      frequency_mask):
    ndet, nfreq = power_spectral_density_array.shape
    nmode = antenna_response_matrix.shape[1]
    output = np.zeros((nfreq, ndet, nmode), dtype=antenna_response_matrix.dtype)
    scaling = delta_f * 2
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                for k in range(nmode):
                    output[i,j,k] = antenna_response_matrix[j,k] / np.sqrt(power_spectral_density_array[j,i] / scaling)
    return output

@njit
def compute_whitened_time_frequency_domain_strain(time_frequency_domain_strain_array,
                                                  power_spectral_density_array,
                                                  delta_f,
                                                  time_frequency_mask):
    ndet, tlen, flen = time_frequency_domain_strain_array.shape
    output = np.zeros_like(time_frequency_domain_strain_array)
    scaling = delta_f * 2
    for n in range(ndet):
        for i in range(tlen):
            for j in range(flen):
                if time_frequency_mask[i,j]:
                    output[n,i,j] = time_frequency_domain_strain_array[n,i,j] / np.sqrt(power_spectral_density_array[n,j] / scaling)
    return output
