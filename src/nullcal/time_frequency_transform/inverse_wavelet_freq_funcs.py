"""functions for computing the inverse wavelet transforms"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def inverse_wavelet_freq_helper_fast(wave_in, phif, n_f, n_t):
    """Loop for inverse_wavelet_freq.

    Args:
        wave_in (2D numpy array): Input data in wavelet domain.
        phif (1D numpy array): Wavelet.
        n_f (int): Number of frequency bins.
        n_t (int ): Number of time bins.

    Returns:
        1D complex numpy array: Result.
    """
    n_d = n_f * n_t

    prefactor2s = np.zeros(n_t, np.complex128)
    res = np.zeros(n_d // 2 + 1, dtype=np.complex128)

    for m in range(0, n_f + 1):
        pack_wave_inverse(m, n_t, n_f, prefactor2s, wave_in)
        # with numba.objmode(fft_prefactor2s="complex128[:]"):
        fft_prefactor2s = np.fft.fft(prefactor2s)
        unpack_wave_inverse(m, n_t, n_f, phif, fft_prefactor2s, res)

    return res


@njit
def unpack_wave_inverse(m, n_t, n_f, phif, fft_prefactor2s, res):
    """Helper for unpacking results of frequency domain inverse transform.

    Args:
        m (int): Frequency index.
        n_t (int): Number of time bins.
        n_f (int): Number of frequency bins.
        phif (1D numpy array): Wavelet.
        fft_prefactor2s (1D numpy array): Prefactors of FFT.
        res (1D numpy array): Result.
    """

    if m in (0, n_f):
        for i_ind in range(0, n_t // 2):
            i = np.abs(m * n_t // 2 - i_ind)  # i_off+i_min2
            ind3 = (2 * i) % n_t
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
        if m == n_f:
            i_ind = n_t // 2
            i = np.abs(m * n_t // 2 - i_ind)  # i_off+i_min2
            ind3 = 0
            res[i] += fft_prefactor2s[ind3] * phif[i_ind]
    else:
        ind31 = (n_t // 2 * m) % n_t
        ind32 = (n_t // 2 * m) % n_t
        for i_ind in range(0, n_t // 2):
            i1 = n_t // 2 * m - i_ind
            i2 = n_t // 2 * m + i_ind
            # assert ind31 == i1%n_t
            # assert ind32 == i2%n_t
            res[i1] += fft_prefactor2s[ind31] * phif[i_ind]
            res[i2] += fft_prefactor2s[ind32] * phif[i_ind]
            ind31 -= 1
            ind32 += 1
            if ind31 < 0:
                ind31 = n_t - 1
            if ind32 == n_t:
                ind32 = 0

        res[n_t // 2 * m] = fft_prefactor2s[(n_t // 2 * m) % n_t] * phif[0]


@njit
def pack_wave_inverse(m, n_t, n_f, prefactor2s, wave_in):
    """Helper for fast frequency domain inverse transform to preare for Fourier transform.

    Args:
        m (int): Frequency index.
        n_t (int): Number of time bins.
        n_f (int): Number of frequency bins.
        prefactor2s (1D complex numpy array): Prefactors for the 1D numpy array.
        wave_in (2D numpy array): Input data in wavelet domain.
    """
    if m == 0:
        for n in range(0, n_t):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % n_t, 0]
    elif m == n_f:
        for n in range(0, n_t):
            prefactor2s[n] = 1 / np.sqrt(2) * wave_in[(2 * n) % n_t + 1, 0]
    else:
        for n in range(0, n_t):
            val = wave_in[n, m]
            mult2 = -1j if (n + m) % 2 else 1

            prefactor2s[n] = mult2 * val
