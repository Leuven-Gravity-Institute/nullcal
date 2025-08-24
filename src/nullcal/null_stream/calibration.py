"""
Functions for including calibration factors into
the antenna response function.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_calibrated_whitened_antenna_response(
    whitened_antenna_response: np.ndarray, calibration_factor: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Compute the whitened antenna response function
    with the calibration factor included.

    Args:
        whitened_antenna_response (np.ndarray): Whitened antenna response function.
            Dimensions: (frequency, detector, polarization).
        calibration_factor (np.ndarray): Calibration factor.
            Dimensions: (detector, frequency).
        frequency_mask (np.ndarray): Frequency mask.
            Dimensions: (frequency,)

    Returns:
        np.ndarray: Calibrated antenna response function.
    """
    output = np.zeros(whitened_antenna_response.shape, dtype=calibration_factor.dtype)
    n_freq, n_det, n_mode = whitened_antenna_response.shape
    for i in range(n_freq):
        if frequency_mask[i]:
            for j in range(n_det):
                for k in range(n_mode):
                    output[i, j, k] = whitened_antenna_response[i, j, k] * calibration_factor[j, i]
    return output
