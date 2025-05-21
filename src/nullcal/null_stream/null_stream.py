from __future__ import annotations

import numpy as np


def compute_projected_strain_data(
    projector: np.ndarray, strain_data: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Compute the projected strain data.

    Args:
        projector (np.ndarray): Projector. Dimensions: (frequency, detector, detector).
        strain_data (np.ndarray): Strain data. Dimensions: (detector, frequency).
        frequency_mask (np.ndarray): Frequency mask. Dimensions: (frequency,).

    Raises:
        ValueError: Projector shape mismatch. project must have the same dimensions in the two last axes.
        ValueError: Shape mismatch. projector and strain_data must have the same detector and frequency dimensions.
        ValueError: Shape mismatch. strain_data and frequency_mask must have the same frequency dimension.

    Returns:
        np.ndarray: Projected strain data. Dimensions: (detector, frequency).
    """
    n_freq_1, n_det_1, n_det_2 = projector.shape
    if n_det_1 != n_det_2:
        raise ValueError(
            "Shape mismatch."
            f"projector: (frequency={n_freq_1},detector={n_det_1}, detector={n_det_2})."
            "project must have the same dimensions in the two last axes."
        )
    n_det_3, n_freq_2 = strain_data.shape
    if n_det_1 != n_det_3 or n_freq_1 != n_freq_2:
        raise ValueError(
            "Shape mismatch."
            f"projector: (frequency={n_freq_1},detector={n_det_1}, detector={n_det_2})."
            f"strain_data: (detector={n_det_3},frequency={n_freq_2})."
            "projector and strain_data must have the same detector and frequency dimensions."
        )
    n_freq_3 = frequency_mask.shape[0]
    if n_freq_2 != n_freq_3:
        raise ValueError(
            "Shape mismatch."
            f"strain_data: (detector={n_det_3},frequency={n_freq_2})."
            f"frequency_mask: (frequency={n_freq_3})."
            "strain_data and frequency_mask must have the same frequency dimension."
        )
    output = np.zeros_like(strain_data)
    output[:, frequency_mask] = np.einsum("fij,jf->if", projector[frequency_mask, :, :], strain_data[:, frequency_mask])
    return output
