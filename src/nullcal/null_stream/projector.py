from __future__ import annotations

import numpy as np


def compute_projector(calibrated_whitened_antenna_response_function: np.ndarray,
                      frequency_mask: np.ndarray) -> np.ndarray:
    """Compute the projector.

    Args:
        calibrated_whitened_antenna_response_function (np.ndarray): Calibrated whitened antenna response function.
        frequency_mask (np.ndarray): Frequency mask.

    Returns:
        np.ndarray: Projector. Dimensions: (frequency, detector, detector).
    """
    n_freq, n_det, _ = calibrated_whitened_antenna_response_function.shape
    calibrated_whitened_antenna_response_function_masked = calibrated_whitened_antenna_response_function[frequency_mask,:,:]
    output = np.eye(n_det,dtype=calibrated_whitened_antenna_response_function.dtype)[np.newaxis, :, :].repeat(n_freq, axis=0)
    F_dagger = np.transpose(np.conj(calibrated_whitened_antenna_response_function_masked), axes=(0,2,1))
    #I - F@(F'F)^-1 F'
    Pgw = np.matmul(calibrated_whitened_antenna_response_function_masked,np.matmul(np.linalg.inv(np.matmul(F_dagger,calibrated_whitened_antenna_response_function_masked)),F_dagger))
    output[frequency_mask,:,:] = output[frequency_mask,:,:] - Pgw
    return output
