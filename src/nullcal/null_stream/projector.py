"""
Functions for projections.
"""

from __future__ import annotations

import numpy as np


def compute_projector(
    calibrated_whitened_antenna_response_function: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Compute the projector.

    Args:
        calibrated_whitened_antenna_response_function (np.ndarray): Calibrated whitened antenna response function.
            Dimensions: (frequency, detector, mode).
        frequency_mask (np.ndarray): Frequency mask.

    Raises:
        ValueError: Frequency mask shape mismatch. The frequency dimensions must have the same length.

    Returns:
        np.ndarray: Projector. Dimensions: (frequency, detector, detector).
    """
    n_freq, n_det, _ = calibrated_whitened_antenna_response_function.shape
    if n_freq != frequency_mask.shape[0]:
        raise ValueError(
            "Frequency mask shape mismatch."
            "calibrated_whitened_antenna_response_function:"
            f"(frequency={n_freq},detector={n_det},mode={_})"
            f"and frequency_mask: (frequency={frequency_mask.shape[0]})."
            "The frequency dimensions must have the same length."
        )
    calibrated_whitened_antenna_response_function_masked = calibrated_whitened_antenna_response_function[
        frequency_mask, :, :
    ]
    output = np.eye(n_det, dtype=calibrated_whitened_antenna_response_function.dtype)[np.newaxis, :, :].repeat(
        n_freq, axis=0
    )
    f_dagger = np.transpose(np.conj(calibrated_whitened_antenna_response_function_masked), axes=(0, 2, 1))
    # I - F@(F'F)^-1 F'
    projector_gw = np.matmul(
        calibrated_whitened_antenna_response_function_masked,
        np.matmul(np.linalg.inv(np.matmul(f_dagger, calibrated_whitened_antenna_response_function_masked)), f_dagger),
    )
    output[frequency_mask, :, :] = output[frequency_mask, :, :] - projector_gw
    return output
