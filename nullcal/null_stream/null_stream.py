import numpy as np
from numba import njit

@njit
def compute_projected_strain_data(projector, strain_data, frequency_mask):
    nfreq, ndet, _ = projector.shape
    output = np.zeros_like(strain_data)
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                sum = 0.
                for k in range(ndet):
                    sum += projector[i,j,k]*strain_data[k,i]
                output[j,i] = sum
    return output

@njit
def compute_projected_time_frequency_strain_data(projector, whitened_time_frequency_strain_data, time_frequency_mask):
    ndet, ntime, nfreq = whitened_time_frequency_strain_data.shape
    output = np.zeros_like(whitened_time_frequency_strain_data)
    for j in range(nfreq):
        for i in range(ntime):
            if time_frequency_mask[i,j]:
                for k in range(ndet):
                    sum = 0.
                    for l in range(ndet):
                        sum += projector[j,k,l]*whitened_time_frequency_strain_data[l,i,j]
                    output[k,i,j] = sum
    return output    