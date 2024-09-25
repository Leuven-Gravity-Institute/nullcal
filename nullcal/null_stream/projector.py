from numba import njit
import numpy as np

@njit
def compute_projector(calibrated_whitened_antenna_response_function, frequency_mask):
    """Compute the projector given the calibrated whitened antenna response function.

    Args:
        calibrated_whitened_antenna_response_function (numpy array): Calibrated whitened antenna response function.

    Returns:
        numpy array: Projector.
    """
    # The dimensions of the input is (frequency, detector, mode)
    nfreq, ndet, nmode = calibrated_whitened_antenna_response_function.shape
    FtF = np.zeros((nfreq, nmode, nmode), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i] is True:
            for j in range(nmode):
                for k in range(j, nmode):
                    sum = 0.
                    for l in range(ndet):
                        sum += np.conj(calibrated_whitened_antenna_response_function[i,l,j])*calibrated_whitened_antenna_response_function[i,l,k]
                    FtF[i,j,k] = sum
                    FtF[i,k,j] = np.conj(sum)
    # Compute F @ FtF_inv first
    F_FtF_inv = np.zeros((nfreq, ndet, nmode), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i] is True:
            FtF_inv_i = np.linalg.inv(FtF[i])
            for j in range(ndet):
                for k in range(nmode):
                    sum = 0.
                    for l in range(nmode):
                        sum += calibrated_whitened_antenna_response_function[i,j,l]*FtF_inv_i[l,k]
                    F_FtF_inv[i,j,k] = sum                
    output = np.zeros((nfreq, ndet, ndet), dtype=calibrated_whitened_antenna_response_function.dtype)
    for i in range(nfreq):
        if frequency_mask[i] is True:
            for j in range(ndet):
                for k in range(j, ndet):
                    sum = 0.
                    for l in range(nmode):
                        sum += F_FtF_inv[i,j,l]*np.conj(calibrated_whitened_antenna_response_function[i,k,l])
                    if j != k:
                        output[i,j,k] = -sum
                        output[i,k,j] = -np.conj(sum)
                    else:
                        output[i,j,k] = 1. - sum
    return output