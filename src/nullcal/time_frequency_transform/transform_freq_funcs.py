"""helper functions for transform_freq"""

from __future__ import annotations

import numpy as np
import scipy.special
from numba import njit, prange


def phitilde_vec(om, n_f, nx=4.0):
    """Compute phitilde, om i array, nx is filter steepness, defaults to 4.

    Args:
        om (1D numpy array): om.
        n_f (int): Number of frequency bins.
        nx (float, optional): Filter steepness. Defaults to 4.

    Returns:
        1D numpy array: z.
    """
    omega = np.pi
    d_omega = omega / n_f
    ins_d_omega = 1.0 / np.sqrt(d_omega)
    b = omega / (2 * n_f)
    a = (d_omega - b) / 2
    z = np.zeros(om.size)

    mask = (np.abs(om) >= a) & (np.abs(om) < a + b)

    x = (np.abs(om[mask]) - a) / b
    y = scipy.special.betainc(nx, nx, x)
    z[mask] = ins_d_omega * np.cos(np.pi / 2.0 * y)

    z[np.abs(om) < a] = ins_d_omega
    return z


def phitilde_vec_norm(n_f, n_t, nx):
    """Normalize phitilde as needed for inverse frequency domain transform.

    Args:
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        nx (float): Filter steepness

    Returns:
        phif: Wavelet.
    """
    n_d = n_f * n_t
    oms = 2 * np.pi / n_d * np.arange(0, n_t // 2 + 1)
    phif = phitilde_vec(oms, n_f, nx)
    # nrm should be 1
    nrm = np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / n_d)
    nrm /= np.pi ** (3 / 2) / np.pi
    phif /= nrm
    return phif


@njit
def tukey(data, alpha, n_sample):
    """Apply Tukey window function to data.

    Args:
        data (1D numpy array): Data.
        alpha (float): Rolling parameter.
        n_sample (int): Length of data
    """
    imin = np.int64(alpha * (n_sample - 1) / 2)
    imax = np.int64((n_sample - 1) * (1 - alpha / 2))
    n_win = n_sample - imax

    for i in range(0, n_sample):
        f_mult = 1.0
        if i < imin:
            f_mult = 0.5 * (1.0 + np.cos(np.pi * (i / imin - 1.0)))
        if i > imax:
            f_mult = 0.5 * (1.0 + np.cos(np.pi / n_win * (i - imax)))
        data[i] *= f_mult


def transform_wavelet_freq_helper(data, n_f, n_t, phif):
    """Helper to do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (1D numpy array): Data.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        phif (1D numpy array): Wavelet.

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    wave = np.zeros((n_t, n_f))  # wavelet wavepacket transform of the signal

    # pylint: disable=not-an-iterable
    for m in prange(0, n_f + 1):
        dx = np.zeros(n_t, dtype=np.complex128)
        dx_assign_loop(m, n_t, n_f, dx, data, phif)
        dx_trans = np.fft.ifft(dx, n_t)
        dx_unpack_loop(m, n_t, n_f, dx_trans, wave)
    return wave


def transform_wavelet_freq_partial_helper(data, n_f, n_t, phif, frequency_filter):
    """Helper to do the wavelet transform using the fast wavelet domain transform.

    Args:
        data (1D numpy array): Data.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        phif (1D numpy array): Wavelet.
        frequency_filter (1D numpy numpy): An array to indicate which frequency bins to evaluate.

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    wave = np.zeros((n_t, n_f))  # wavelet wavepacket transform of the signal

    # pylint: disable=not-an-iterable
    for m in prange(0, n_f + 1):
        if frequency_filter[m]:
            dx = np.zeros(n_t, dtype=np.complex128)
            dx_assign_loop(m, n_t, n_f, dx, data, phif)
            dx_trans = np.fft.ifft(dx, n_t)
            dx_unpack_loop(m, n_t, n_f, dx_trans, wave)
    return wave


def transform_wavelet_freq_quadrature_helper(data, n_f, n_t, phif):
    """Helper to do the wavelet transform using the fast wavelet domain quadrature transform.

    Args:
        data (1D numpy array): Data.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        phif (1D numpy array): Wavelet.

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    wave = np.zeros((n_t, n_f))  # wavelet wavepacket transform of the signal

    dx = np.zeros(n_t, dtype=np.complex128)
    for m in range(0, n_f + 1):
        dx_assign_loop(m, n_t, n_f, dx, data, phif)
        dx_trans = np.fft.ifft(dx, n_t)
        dx_unpack_loop_quadrature(m, n_t, n_f, dx_trans, wave)
    return wave


@njit
def dx_assign_loop(m, n_t, n_f, dx, data, phif):
    """Helper for assigning DX in the main loop.

    Args:
        m (int): Frequency index.
        n_t (int): Number of time bins.
        n_f (int): Number of frequency bins.
        dx (1D complex numpy array): DX.
        data (1D complex numpy array): Input data.
        phif (1D numpy array): Wavelet.
    """
    i_base = n_t // 2
    jj_base = m * n_t // 2

    if m in (0, n_f):
        # NOTE this term appears to be needed to recover correct constant (at least for m=0), but was previously missing
        dx[n_t // 2] = phif[0] * data[m * n_t // 2] / 2.0
        dx[n_t // 2] = phif[0] * data[m * n_t // 2] / 2.0
    else:
        dx[n_t // 2] = phif[0] * data[m * n_t // 2]
        dx[n_t // 2] = phif[0] * data[m * n_t // 2]

    for jj in range(jj_base + 1 - n_t // 2, jj_base + n_t // 2):
        j = np.abs(jj - jj_base)
        i = i_base - jj_base + jj
        if (m == n_f and jj > jj_base) or (m == 0 and jj < jj_base):
            dx[i] = 0.0
        elif j == 0:
            continue
        else:
            dx[i] = phif[j] * data[jj]


@njit
def dx_unpack_loop(m, n_t, n_f, dx_trans, wave):
    """Helper for unpacking fftd DX in main loop.

    Args:
        m (int): Frequency index.
        n_t (int): Number of time bins.
        n_f (int): Number of frequency bins.
        dx_trans (1D complex numpy array): DX_trans.
        wave (2D numpy array): Data in wavelet domain.
    """
    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant,
        # so store them in even and odd components of m=0 respectively
        for n in range(0, n_t, 2):
            wave[n, 0] = np.real(dx_trans[n] * np.sqrt(2))
    elif m == n_f:
        for n in range(0, n_t, 2):
            wave[n + 1, 0] = np.real(dx_trans[n] * np.sqrt(2))
    else:
        for n in range(0, n_t):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = -np.imag(dx_trans[n])
                else:
                    wave[n, m] = np.real(dx_trans[n])
            elif (n + m) % 2:
                wave[n, m] = np.imag(dx_trans[n])
            else:
                wave[n, m] = np.real(dx_trans[n])


@njit
def dx_unpack_loop_quadrature(m, n_t, n_f, dx_trans, wave):
    """Helper for unpacking fftd DX in main loop.

    Args:
        m (int): Frequency index.
        n_t (int): Number of time bins.
        n_f (int): Number of frequency bins.
        dx_trans (1D complex numpy array): dx_trans.
        wave (2D numpy array): Data in wavelet domain.
    """
    if m == 0:
        # half of lowest and highest frequency bin pixels are redundant,
        # so store them in even and odd components of m=0 respectively
        for n in range(0, n_t, 2):
            wave[n, 0] = np.real(dx_trans[n + 1] * np.sqrt(2))
    elif m == n_f:
        for n in range(0, n_t, 2):
            wave[n + 1, 0] = np.real(dx_trans[n + 1] * np.sqrt(2))
    else:
        for n in range(0, n_t):
            if m % 2:
                if (n + m) % 2:
                    wave[n, m] = np.real(dx_trans[n])
                else:
                    wave[n, m] = -np.imag(dx_trans[n])
            elif (n + m) % 2:
                wave[n, m] = np.real(dx_trans[n])
            else:
                wave[n, m] = np.imag(dx_trans[n])
