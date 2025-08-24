"""
Utility functions for the time-frequency transform.
"""

from __future__ import annotations


def get_shape_of_wavelet_transform(t_length: int, sampling_frequency: float, frequency_resolution: float) -> tuple:
    """Get the shape of wavelet transform.

    Args:
        t_length (int): Length of the time series.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.

    Returns:
        tuple: n_t, n_f. Number of time points and frequency points.
    """
    n_f = int(sampling_frequency / 2 / frequency_resolution)
    n_t = int(t_length / n_f)
    return n_t, n_f
