from numba import njit
import numpy as np

@njit
def compute_calibrated_whitened_antenna_response(whitened_antenna_response, calibration_factor, frequency_mask):
    output = np.zeros(whitened_antenna_response.shape, dtype=calibration_factor.dtype)
    nfreq, ndet, nmode = whitened_antenna_response.shape
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                for k in range(nmode):
                    output[i,j,k] = whitened_antenna_response[i,j,k] * calibration_factor[j,i]
    return output