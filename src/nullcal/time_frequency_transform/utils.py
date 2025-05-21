from __future__ import annotations


def get_shape_of_wavelet_transform(t_length: int, sampling_frequency: float, frequency_resolution: float) -> tuple:
    """Get the shape of wavelet transform.

    Args:
        t_length (int): Length of the time series.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Frequency resolution in Hz.

    Returns:
        tuple: Nt, Nf. Number of time points and frequency points.
    """
    Nf = int(sampling_frequency / 2 / frequency_resolution)
    Nt = int(t_length / Nf)
    return Nt, Nf
