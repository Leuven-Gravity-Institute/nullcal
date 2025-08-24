"""helper functions for transform_time.py"""

from __future__ import annotations

import numpy as np

from .inverse_wavelet_freq_funcs import inverse_wavelet_freq_helper_fast
from .inverse_wavelet_time_funcs import inverse_wavelet_time_helper_fast
from .transform_freq_funcs import (
    phitilde_vec_norm,
    transform_wavelet_freq_helper,
    transform_wavelet_freq_quadrature_helper,
)
from .transform_time_funcs import phi_vec, transform_wavelet_time_helper
from .utils import get_shape_of_wavelet_transform


def inverse_wavelet_time(wave_in, n_f, n_t, nx=4.0, mult=32):
    """Fast inverse wavelet transform to time domain.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of the filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        1D numpy array: Data in time domain.
    """
    mult = min(mult, n_t // 2)  # make sure K isn't bigger than n_d
    phi = phi_vec(n_f, nx=nx, mult=mult) / 2
    output = inverse_wavelet_time_helper_fast(wave_in, phi, n_f, n_t, mult)
    return output / np.sqrt(output.shape[0])


def inverse_wavelet_freq_time(wave_in, n_f, n_t, nx=4.0):
    """Inverse wavelet transform to time domain via Fourier transform of frequency domain.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        1D numpy array: Data in time domain.
    """
    res_f = inverse_wavelet_freq(wave_in, n_f, n_t, nx)
    return np.fft.irfft(res_f)


def inverse_wavelet_freq(wave_in, n_f, n_t, nx=4.0):
    """Inverse wavelet transform to frequency domain signal.

    Args:
        wave_in (2D numpy array): Data in wavelet domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        1D numpy array: Data in time domain.
    """
    phif = phitilde_vec_norm(n_f, n_t, nx)
    output = inverse_wavelet_freq_helper_fast(wave_in, phif, n_f, n_t)
    return output / np.sqrt((output.shape[0] - 1) * 2)


def transform_wavelet_time(data, n_f, n_t, nx=4.0, mult=32):
    """Do the wavelet transform in the time domain,
    note there can be significant leakage if mult is too small and the
    transform is only approximately exact if mult=n_t/2.

    Args:
        data (1D numpy array): Data in time domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 32.

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    mult = min(mult, n_t // 2)  # make sure K isn't bigger than n_d
    phi = phi_vec(n_f, nx, mult)
    wave = transform_wavelet_time_helper(data, n_f, n_t, phi, mult) * np.sqrt(data.shape[0])

    return wave


def transform_wavelet_freq_time(data, n_f, n_t, nx=4.0):
    """Transform time domain data into wavelet domain via FFT and then frequency transform.

    Args:
        data (1D numpy array): Data in time domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    data_fft = np.fft.rfft(data)

    return transform_wavelet_freq(data_fft, n_f, n_t, nx)


def transform_wavelet_freq_time_quadrature(data, n_f, n_t, nx=4.0):
    """Transform time domain data into wavelet quadrature domain via FFT and then frequency transform.

    Args:
        data (1D numpy array): Data in time domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    data_fft = np.fft.rfft(data)

    return transform_wavelet_freq_quadrature(data_fft, n_f, n_t, nx)


def transform_wavelet_freq_quadrature(data, n_f, n_t, nx=4.0):
    """Do the wavelet quadrature transform using the fast wavelet domain transform.

    Args:
        data (1D complex numpy array): Data in frequency domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    phif = 2 / n_f * phitilde_vec_norm(n_f, n_t, nx)
    return transform_wavelet_freq_quadrature_helper(data, n_f, n_t, phif) * np.sqrt((data.shape[0] - 1) * 2)


def transform_wavelet_freq(data, n_f, n_t, nx=4.0):
    """Do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (1D complex numpy array): Data in frequency domain.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float, optional): Steepness of filter. Defaults to 4..

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    phif = 2 / n_f * phitilde_vec_norm(n_f, n_t, nx)
    return transform_wavelet_freq_helper(data, n_f, n_t, phif) * np.sqrt((data.shape[0] - 1) * 2)


class WaveletTransform:
    """A class to handle wavelet transform."""

    def __init__(self, duration: float, sampling_frequency: float, frequency_resolution: float, nx: float = 4.0):
        """Wavelet transform.

        Args:
            duration (float): Duration in second.
            sampling_frequency (float): Sampling frequency in Hz.
            frequency_resolution (float): Frequency resolution in Hz.
            nx (float): Sharpness of wavelet.
                Defaults to 4.0.
        """
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.frequency_resolution = frequency_resolution
        self.nx = nx
        self._t_length = int(duration * sampling_frequency)
        self._f_length = self._t_length // 2 + 1
        self.shape = get_shape_of_wavelet_transform(
            t_length=int(duration * sampling_frequency),
            sampling_frequency=sampling_frequency,
            frequency_resolution=frequency_resolution,
        )

    def frequency_to_wavelet(self, frequency_domain_data: np.ndarray) -> np.ndarray:
        """Transform from frequency domain to wavelet domain.

        Args:
            frequency_domain_data (np.ndarray): Frequency-domain data.

        Raises:
            ValueError: The length of frequency_domain_data does not match
                the expected length.

        Returns:
            np.ndarray: Wavelet-domain data.
        """
        # Check whether the length of frequency-domain data
        # matches the expected length.
        if len(frequency_domain_data) != self._f_length:
            raise ValueError(
                f"The length of frequency_domain_data: {len(frequency_domain_data)}"
                f"does not match the expected length: {self._f_length}."
            )
        return transform_wavelet_freq(data=frequency_domain_data, n_f=self.shape[1], n_t=self.shape[0])

    def wavelet_to_frequency(self, wavelet_domain_data: np.ndarray) -> np.ndarray:
        """Transform from wavelet domain to frequency domain.

        Args:
            wavelet_domain_data (np.ndarray): Wavelet-domain data.

        Raises:
            ValueError: The shape of wavelet_domain_data does not match
                the expected shape.

        Returns:
            np.ndarray: Frequency-domain data.
        """
        # Check whether the shape of wavelet-domain strain
        # matches the expected shape.
        if wavelet_domain_data.shape != self.shape:
            raise ValueError(
                f"The shape of wavelet_domain_data: {wavelet_domain_data.shape}"
                f"does not match the expected shape: {self.shape}."
            )
        return inverse_wavelet_freq(wave_in=wavelet_domain_data, n_f=self.shape[1], n_t=self.shape[0], nx=self.nx)

    def frequency_to_wavelet_quadrature(self, frequency_domain_data: np.ndarray) -> np.ndarray:
        """Transform from frequency domain to wavelet quadrature domain.

        Args:
            frequency_domain_data (np.ndarray): Frequency-domain data.

        Raises:
            ValueError: The length of frequency_domain_data does not match
                the expected length.

        Returns:
            np.ndarray: Wavelet quadrature domain data.
        """
        # Check whether the length of frequency-domain data
        # matches the expected length.
        if len(frequency_domain_data) != self._f_length:
            raise ValueError(
                f"The length of frequency_domain_data: {len(frequency_domain_data)}"
                f"does not match the expected length: {self._f_length}."
            )
        return transform_wavelet_freq_quadrature(data=frequency_domain_data, n_f=self.shape[1], n_t=self.shape[0])
