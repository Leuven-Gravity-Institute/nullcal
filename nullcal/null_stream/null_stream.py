import numpy as np
from numba import njit

@njit
def compute_projected_strain_data(projector, strain_data, frequency_mask):
    nfreq, ndet, _ = projector.shape
    output = np.zeros_like(strain_data)
    for i in range(nfreq):
        if frequency_mask[i] is True:
            for j in range(ndet):
                sum = 0.
                for k in range(ndet):
                    sum += projector[i,j,k]*strain_data[i,k]
                output[i,j] = sum
    return output