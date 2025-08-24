"""helper functions for transform_time.py"""

from __future__ import annotations

import numpy as np
from numba import njit

from .transform_freq_funcs import phitilde_vec


@njit
def transform_wavelet_time_helper(data, n_f, n_t, phi, mult):
    """Helper function to do the wavelet transform in the time domain.

    Args:
        data (1D numpy array): Data.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        phi (1D numpy array): Wavelet.
        mult (int): mult

    Returns:
        2D numpy array: Data in wavelet domain.
    """
    # the time domain data stream
    n_d = n_f * n_t

    # mult, can cause bad leakage if it is too small but may be possible to mitigate
    # Filter is mult times pixel with in time

    n_k = mult * 2 * n_f

    # windowed data packets
    wdata = np.zeros(n_k)

    wave = np.zeros((n_t, n_f))  # wavelet wavepacket transform of the signal
    data_pad = np.zeros(n_d + n_k)
    data_pad[:n_d] = data
    data_pad[n_d : n_d + n_k] = data[:n_k]

    for i in range(0, n_t):
        assign_wdata(i, n_k, n_d, n_f, wdata, data_pad, phi)
        wdata_trans = np.fft.rfft(wdata, n_k)
        pack_wave(i, mult, n_f, wdata_trans, wave)

    return wave


@njit
def assign_wdata(i, k_cutoff, n_d, n_f, wdata, data_pad, phi):
    """Assign wdata to be fftd in loop, data_pad needs K extra values on the right to loop.

    Args:
        i (int): Time index.
        k_cutoff (int): Frequency cutoff.
        n_d (int): n_d.
        n_f (int): Number of frequency bins.
        wdata (1D numpy array): wdata.
        data_pad (1D numpy array): Padded data.
        phi (1D numpy array): Wavelet.
    """
    # half_K = np.int64(K/2)
    jj = i * n_f - k_cutoff // 2
    if jj < 0:
        jj += n_d  # periodically wrap the data
    if jj >= n_d:
        jj -= n_d  # periodically wrap the data
    for j in range(0, k_cutoff):
        # jj = i*n_f-half_K+j
        wdata[j] = data_pad[jj] * phi[j]  # apply the window
        jj += 1
        # if jj==n_d:
        #    jj -= n_d # periodically wrap the data


@njit
def pack_wave(i, mult, n_f, wdata_trans, wave):
    """Pack fftd wdata into wave array.

    Args:
        i (int): Time index.
        mult (int): mult.
        n_f (int): Number of frequency bins.
        wdata_trans (1D complex numpy array): wdata_trans.
        wave (2D numpy array): wdata.
    """
    if i % 2 == 0 and i < wave.shape[0] - 1:
        # m=0 value at even n_t and
        wave[i, 0] = np.real(wdata_trans[0]) / np.sqrt(2)
        wave[i + 1, 0] = np.real(wdata_trans[n_f * mult]) / np.sqrt(2)

    for j in range(1, n_f):
        if (i + j) % 2:
            wave[i, j] = -np.imag(wdata_trans[j * mult])
        else:
            wave[i, j] = np.real(wdata_trans[j * mult])


@njit
def phi_vec(n_f, nx=4.0, mult=16):
    """Get time domain phi as Fourier transform of phitilde_vec.

    Args:
        n_f (int): Number of frequency bins.
        nx (float, optional): Steepness of filter. Defaults to 4..
        mult (int, optional): mult. Defaults to 16.

    Returns:
        1D numpy array: Time domain phi.
    """

    omega = np.pi
    d_omega = omega / n_f
    ins_d_omega = 1.0 / np.sqrt(d_omega)
    k_cutoff = mult * 2 * n_f
    half_k_cutoff = mult * n_f  # np.int64(K/2)

    dom = 2 * np.pi / k_cutoff  # max frequency is K/2*dom = pi/dt = OM

    dx = np.zeros(k_cutoff, dtype=np.complex128)

    # zero frequency
    dx[0] = ins_d_omega

    dx = dx.copy()
    # positive frequencies
    dx[1 : half_k_cutoff + 1] = phitilde_vec(dom * np.arange(1, half_k_cutoff + 1), n_f, nx)
    # negative frequencies
    dx[half_k_cutoff + 1 :] = phitilde_vec(-dom * np.arange(half_k_cutoff - 1, 0, -1), n_f, nx)
    dx = k_cutoff * np.fft.ifft(dx, k_cutoff)

    phi = np.zeros(k_cutoff)
    phi[0:half_k_cutoff] = np.real(dx[half_k_cutoff:k_cutoff])
    phi[half_k_cutoff:] = np.real(dx[0:half_k_cutoff])

    nrm = np.sqrt(k_cutoff / dom)  # *np.linalg.norm(phi)

    fac = np.sqrt(2.0) / nrm
    phi *= fac
    return phi
