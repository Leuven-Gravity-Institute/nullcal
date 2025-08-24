"""
Functions for computing the inverse wavelet transforms
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def inverse_wavelet_time_helper_fast(wave_in, phi, n_f, n_t, mult):
    """Helper loop fort fast inverse wavelet transform.

    Args:
        wave_in (2D numpy array): Input data in wavelet domain.
        phi (1D numpy array): Wavelet.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        mult (int): mult.

    Returns:
        1D numpy array: Result.
    """
    n_d = n_f * n_t
    k = mult * 2 * n_f
    # res = np.zeros(n_d)

    # extend this array, we can use wrapping boundary conditions at end
    res = np.zeros(n_d + k + n_f)

    afins = np.zeros(2 * n_f, dtype=np.complex128)

    for n in range(0, n_t):
        # old unpacked way, should still work but is necessarily slower,
        # might be more comparable if it could be written as an irfft instead
        # pack_wave_time_helper(n,n_f,n_t,wave_in,afins)
        # ffts_fin_real = np.real(fft.fft(afins))
        # unpack_time_wave_helper(n,n_f,n_t,K,phi,ffts_fin_real,res)

        # we can pack both the sin and cos parts into the real and imaginary parts of the same transform
        # so we only need to do every other one
        # this currently assumes n_t is even
        if n % 2 == 0:
            pack_wave_time_helper_compact(n, n_f, n_t, wave_in, afins)
            ffts_fin = np.fft.fft(afins)
            unpack_time_wave_helper_compact(n, n_f, n_t, k, phi, ffts_fin, res)

    # wrap boundary conditions
    res[: min(k + n_f, n_d)] += res[n_d : min(n_d + k + n_f, 2 * n_d)]
    if k + n_f > n_d:
        res[: k + n_f - n_d] += res[2 * n_d : n_d + k * n_f]

    res = res[:n_d]

    return res


@njit
def unpack_time_wave_helper(n, n_f, n_t, k_cutoff, phis, fft_fin_real, res):
    """Helper for time domain wavelet transform to unpack wavelet domain coefficients.

    Args:
        n (int): Time index.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        k_cutoff (int): Frequency cutoff.
        phis (1D numpy array): Wavelet.
        fft_fin_real (1D numpy array): fft_fin_real.
        res (1D numpy array): Result.
    """
    n_d = n_f * n_t

    idxf = (-k_cutoff // 2 + n * n_f + n_d) % (2 * n_f)
    k = (-k_cutoff // 2 + n * n_f) % n_d

    for k_ind in range(0, k_cutoff):
        res_loc = fft_fin_real[idxf]
        res[k] += phis[k_ind] * res_loc
        idxf += 1
        k += 1

        if idxf == 2 * n_f:
            idxf = 0
        if k == n_d:
            k = 0


@njit
def unpack_time_wave_helper_compact(n, n_f, n_t, k_cutoff, phis, fft_fin, res):  # pylint: disable=too-many-locals
    """Helper for time domain wavelet transform to unpack wavelet domain coefficients
    in compact representation where cosine and sine parts are real and imaginary parts.

    Args:
        n (int): Time index.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        k_cutoff (int): Frequency cutoff.
        phis (1D numpy array): Wavelet.
        fft_fin (1D numpy array): fft_fin.
        res (1D numpy array): Result.
    """
    n_d = n_f * n_t
    fft_fin_real = np.zeros(4 * n_f)
    fft_fin_imag = np.zeros(4 * n_f)
    for itrf in range(0, 2 * n_f):
        fft_fin_real[itrf] = np.real(fft_fin[itrf])
        fft_fin_real[itrf + 2 * n_f] = fft_fin_real[itrf]
        fft_fin_imag[itrf] = np.imag(fft_fin[(itrf + n_f) % (2 * n_f)])
        fft_fin_imag[itrf + 2 * n_f] = fft_fin_imag[itrf]

    idxf1_base = (-k_cutoff // 2 + n * n_f + n_d) % (2 * n_f)
    k1_base = (-k_cutoff // 2 + n * n_f) % n_d
    for k_ind in range(0, k_cutoff, 2 * n_f):
        for idxf1_add in range(0, 2 * n_f):
            idxf1 = idxf1_base + idxf1_add
            k_ind_loc = k_ind + idxf1_add
            k1 = k1_base + k_ind_loc

            res[k1] += phis[k_ind_loc] * fft_fin_real[idxf1]
            res[k1 + n_f] += phis[k_ind_loc] * fft_fin_imag[idxf1]


# @njit()
# def pack_wave_time_helper(n,n_f,n_t,wave_in,afins):
#    """helper for time domain transform to pack wavelet domain coefficients"""
#    if n%2==0:
#        #assign highest and lowest bin correctly
#        afins[0] = 1/np.sqrt(2)*wave_in[n,0]
#        if n+1<n_t:
#            afins[n_f] = 1/np.sqrt(2)*wave_in[n+1,0]
#    else:
#        afins[0] = 0.
#        afins[n_f] = 0.
#
#    for idxm in range(0,n_f//2-1):
#        if n%2:
#            afins[2*idxm+2] = 1j*wave_in[n,2*idxm+2]
#        else:
#            afins[2*idxm+2] = wave_in[n,2*idxm+2]
#
#    for idxm in range(0,n_f//2):
#        if n%2:
#            afins[2*idxm+1] = -wave_in[n,2*idxm+1]
#        else:
#            afins[2*idxm+1] = 1j*wave_in[n,2*idxm+1]


@njit
def pack_wave_time_helper(n, n_f, n_t, wave_in, afins):
    """Helper for time domain transform to pack wavelet domain coefficients.

    Args:
        n (int): Time index.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        wave_in (2D numpy array): Input data in wavelet domain.
        afins (1D complex numpy array): afins.
    """
    if n % 2 == 0:
        # assign highest and lowest bin correctly
        afins[0] = np.sqrt(2) * wave_in[n, 0]
        if n + 1 < n_t:
            afins[n_f] = np.sqrt(2) * wave_in[n + 1, 0]
    else:
        afins[0] = 0.0
        afins[n_f] = 0.0

    for idxm in range(0, n_f // 2 - 1):
        if n % 2:
            afins[2 * idxm + 2] = 1j * wave_in[n, 2 * idxm + 2]
            afins[2 * n_f - 2 * idxm - 2] = -1j * wave_in[n, 2 * idxm + 2]
        else:
            afins[2 * idxm + 2] = 1 * wave_in[n, 2 * idxm + 2]
            afins[2 * n_f - 2 * idxm - 2] = 1 * wave_in[n, 2 * idxm + 2]

    for idxm in range(0, n_f // 2):
        if n % 2:
            afins[2 * idxm + 1] = -1 * wave_in[n, 2 * idxm + 1]
            afins[2 * n_f - 2 * idxm - 1] = -1 * wave_in[n, 2 * idxm + 1]
        else:
            afins[2 * idxm + 1] = 1j * wave_in[n, 2 * idxm + 1]
            afins[2 * n_f - 2 * idxm - 1] = -1j * wave_in[n, 2 * idxm + 1]


@njit
def pack_wave_time_helper_compact(n, n_f, n_t, wave_in, afins):
    """Helper for time domain transform to pack wavelet domain coefficients
    in packed representation with odd and even coefficients in real and imaginary parts.

    Args:
        n (int): Time index.
        n_f (int): Number of frequency bins.
        n_t (int): Number of time bins.
        wave_in (2D numpy array): Input data in wavelet domain.
        afins (1D complex numpy array): afins.
    """
    afins[0] = np.sqrt(2) * wave_in[n, 0]
    if n + 1 < n_t:
        afins[n_f] = np.sqrt(2) * wave_in[n + 1, 0]

    for idxm in range(0, n_f - 2, 2):
        afins[idxm + 2] = wave_in[n, idxm + 2] - wave_in[n + 1, idxm + 2]
        afins[2 * n_f - idxm - 2] = wave_in[n, idxm + 2] + wave_in[n + 1, idxm + 2]

        afins[idxm + 1] = 1j * (wave_in[n, idxm + 1] - wave_in[n + 1, idxm + 1])
        afins[2 * n_f - idxm - 1] = -1j * (wave_in[n, idxm + 1] + wave_in[n + 1, idxm + 1])

    afins[n_f - 1] = 1j * (wave_in[n, n_f - 1] - wave_in[n + 1, n_f - 1])
    afins[n_f + 1] = -1j * (wave_in[n, n_f - 1] + wave_in[n + 1, n_f - 1])
