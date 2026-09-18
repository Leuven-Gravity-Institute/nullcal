"""helper functions for transform_freq"""

from __future__ import annotations

import jax
import numpy as np
import scipy.special

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


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


def tukey(data, alpha, n_sample):
    """Apply Tukey window function to data.

    Args:
        data (1D numpy array): Data.
        alpha (float): Rolling parameter.
        n_sample (int): Length of data
    """
    imin = int(alpha * (n_sample - 1) / 2)
    imax = int((n_sample - 1) * (1 - alpha / 2))
    n_win = n_sample - imax
    values = jnp.asarray(data)
    indices = jnp.arange(n_sample)
    safe_imin = max(imin, 1)
    rising_values = 0.5 * (1.0 + jnp.cos(jnp.pi * (indices / safe_imin - 1.0)))
    falling_values = 0.5 * (1.0 + jnp.cos(jnp.pi / n_win * (indices - imax)))
    window = jnp.where(indices < imin, rising_values, 1.0)
    window = jnp.where(indices > imax, falling_values, window)
    result = values * window
    if isinstance(data, np.ndarray):
        data[...] = np.asarray(result)
        return None
    return result


def _frequency_packets(data, n_f, n_t, phif):
    """Build every windowed frequency packet in one JAX gather."""
    data = jnp.asarray(data)
    phif = jnp.asarray(phif, dtype=data.real.dtype)
    modes = jnp.arange(n_f + 1)[:, None]
    offsets = jnp.arange(1 - n_t // 2, n_t // 2)[None, :]
    frequency_indices = modes * (n_t // 2) + offsets
    safe_indices = jnp.clip(frequency_indices, 0, data.size - 1)
    valid = ~(((modes == 0) & (offsets < 0)) | ((modes == n_f) & (offsets > 0)))
    values = phif[jnp.abs(offsets)] * data[safe_indices]

    packets = jnp.zeros((n_f + 1, n_t), dtype=data.dtype)
    packets = packets.at[:, 1:].set(jnp.where(valid, values, 0.0))
    endpoint_scale = jnp.where((jnp.arange(n_f + 1) == 0) | (jnp.arange(n_f + 1) == n_f), 0.5, 1.0)
    centres = phif[0] * data[jnp.arange(n_f + 1) * (n_t // 2)] * endpoint_scale
    return packets.at[:, n_t // 2].set(centres)


def _unpack_frequency_packets(transformed, n_f, n_t, *, quadrature=False, frequency_filter=None):
    """Map packet FFT components to the real WDM coefficient array."""
    rows = jnp.arange(n_t)[:, None]
    modes = jnp.arange(1, n_f)[None, :]
    interior = transformed[1:n_f].T
    parity = (rows + modes) % 2 == 1
    odd_mode = modes % 2 == 1
    if quadrature:
        values = jnp.where(parity, jnp.real(interior), jnp.where(odd_mode, -jnp.imag(interior), jnp.imag(interior)))
    else:
        values = jnp.where(parity, jnp.where(odd_mode, -jnp.imag(interior), jnp.imag(interior)), jnp.real(interior))

    if frequency_filter is not None:
        selected = jnp.asarray(frequency_filter, dtype=bool)
        values = jnp.where(selected[None, 1:n_f], values, 0.0)
    else:
        selected = jnp.ones(n_f + 1, dtype=bool)

    wave = jnp.zeros((n_t, n_f), dtype=transformed.real.dtype)
    wave = wave.at[:, 1:].set(values)
    packet_indices = jnp.arange(1 if quadrature else 0, n_t, 2)
    wave = wave.at[::2, 0].set(jnp.where(selected[0], jnp.real(transformed[0, packet_indices]) * jnp.sqrt(2.0), 0.0))
    wave = wave.at[1::2, 0].set(
        jnp.where(selected[n_f], jnp.real(transformed[n_f, packet_indices]) * jnp.sqrt(2.0), 0.0)
    )
    return wave


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
    packets = _frequency_packets(data, n_f, n_t, phif)
    return _unpack_frequency_packets(jnp.fft.ifft(packets, axis=1), n_f, n_t)


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
    packets = _frequency_packets(data, n_f, n_t, phif)
    return _unpack_frequency_packets(jnp.fft.ifft(packets, axis=1), n_f, n_t, frequency_filter=frequency_filter)


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
    packets = _frequency_packets(data, n_f, n_t, phif)
    return _unpack_frequency_packets(jnp.fft.ifft(packets, axis=1), n_f, n_t, quadrature=True)
