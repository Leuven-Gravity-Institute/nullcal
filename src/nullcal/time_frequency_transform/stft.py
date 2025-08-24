"""
Short-time Fourier transform.
"""

from __future__ import annotations

import numpy as np


def stft(data: np.ndarray, sampling_frequency: float, frequency_resolution: float, window: np.ndarray):
    """Short-time Fourier transform.

    Args:
        data (np.ndarray): Data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Jz.
        window (np.ndarray, optional): Window function. Defaults to None.

    Raises:
        ValueError: data must be real-valued.

    Returns:
        np.ndarray: Short-time Fourier transform of data.
    """
    data = np.asarray(data)
    if not np.isrealobj(data):
        raise ValueError("data must be real-valued")

    # Compute number of frames
    n_samples = data.shape[0]
    window_size = int(sampling_frequency / frequency_resolution)
    n_frames = int(n_samples // window_size)

    # Reshape signal into frames
    # Shape: (n_frames, window_size)
    frames = data.reshape(n_frames, window_size)

    # Apply window function
    windowed_frames = frames * window

    # Compute rFFT for each frame
    # Shape: (n_frames, window_size // 2 + 1), complex-valued
    r_stft_matrix = np.fft.rfft(windowed_frames, n=window_size, axis=-1) / sampling_frequency

    return r_stft_matrix
