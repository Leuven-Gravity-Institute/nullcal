from __future__ import annotations

import numpy as np


def compute_whitened_frequency_domain_strain(
        frequency_domain_strain_array: np.ndarray,
        power_spectral_density_array: np.ndarray,
        delta_f: float,
        frequency_mask: np.ndarray) -> np.ndarray:
    """Compute the whitened frequency domain strain.

    Args:
        frequency_domain_strain_array (np.ndarray): Frequency domain strain array. Dimensions: (detector, frequency).
        power_spectral_density_array (np.ndarray): Power spectral density array. Dimensions: (detector, frequency).
        delta_f (float): Frequency resolution in Hz.
        frequency_mask (np.ndarray): A frequency mask.

    Raises:
        ValueError: delta_f must be positive.
        ValueError: Shape mismatch. frequency_domain_strain_array and power_spectral_density_array must have the same dimensions.
        ValueError: Shape mismatch. frequency_domain_strain_array and frequency_mask must have the same frequency dimension.

    Returns:
        np.ndarray: Whitened frequency domain strain.
    """
    if delta_f <= 0.0:
        raise ValueError(f'delta_f must be positive.')
    if frequency_domain_strain_array.shape != power_spectral_density_array.shape:
        raise ValueError(f'Shape mismatch. frequency_domain_strain_array: {frequency_domain_strain_array.shape} and power_spectral_density_array: {power_spectral_density_array.shape} must have the same dimensions.')
    if frequency_domain_strain_array.shape[1] != frequency_mask.shape[0]:
        raise ValueError(f'Shape mismatch. frequency_domain_strain_array: {frequency_domain_strain_array.shape} and frequency_mask: {frequency_mask.shape} must have the same frequency dimension.')
    return np.divide(frequency_domain_strain_array,
                     np.sqrt(power_spectral_density_array / (2*delta_f)),
                     where=(frequency_mask & np.all(power_spectral_density_array, axis=0)),
                     out=np.zeros_like(frequency_domain_strain_array))

def compute_whitened_antenna_response(antenna_response_matrix: np.ndarray,
                                      power_spectral_density_array: np.ndarray,
                                      delta_f: float,
                                      frequency_mask: np.ndarray) -> np.ndarray:
    """Compute the whitened antenna response function.

    Args:
        antenna_response_matrix (np.ndarray): Antenna response matrix.
        power_spectral_density_array (np.ndarray): Power spectral density array.
        delta_f (float): Frequency resolution in Hz.
        frequency_mask (np.ndarray): Frequency mask.

    Raises:
        ValueError: delta_f must be positive.
        ValueError: Shape mismatch. antenna_response_matrix and power_spectral_density_array must have the same detector dimension.
        ValueError: Shape mismatch. power_spectral_density_array and frequency_mask must have the same frequency dimensions.

    Returns:
        np.ndarray: Whitened antenna response function.
    """
    if delta_f <= 0:
        raise ValueError(f'delta_f must be positive.')
    n_det_1, n_mode_1 = antenna_response_matrix.shape
    n_det_2, n_freq_2 = power_spectral_density_array.shape
    n_freq_3 = frequency_mask.shape[0]
    if n_det_1 != n_det_2:
        raise ValueError(f'Shape mismatch. antenna_response_matrix: (detector={n_det_1}, mode={n_mode_1}) and power_spectral_density_array: (detector={n_det_2}, frequency={n_freq_2}) must have the same detector dimension.')
    if n_freq_2 != n_freq_3:
        raise ValueError(f'Shape mismatch. power_spectral_density_array: (detector={n_det_2}, freq={n_freq_2}) and frequency_mask: (frequency: {n_freq_3}) must have the same frequency dimension.')
    output = np.zeros((n_freq_2, n_det_1, n_mode_1), dtype=antenna_response_matrix.dtype)
    frequency_mask = frequency_mask & np.all(power_spectral_density_array, axis=0)
    print(output[frequency_mask, :, :])
    output[frequency_mask, :, :] = np.einsum('dm,df->fdm', antenna_response_matrix, 1 / np.sqrt(power_spectral_density_array[:, frequency_mask] / (2 * delta_f)))
    return output
