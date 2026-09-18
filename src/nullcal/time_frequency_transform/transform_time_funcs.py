"""JAX kernels for the direct time-domain WDM transform."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .transform_freq_funcs import phitilde_vec  # noqa: E402


def transform_wavelet_time_helper(data, n_f, n_t, phi, mult):
    """Transform a time series into WDM coefficients with batched JAX FFTs."""
    data = jnp.asarray(data)
    phi = jnp.asarray(phi, dtype=data.dtype)
    n_d = n_f * n_t
    n_k = mult * 2 * n_f
    rows = jnp.arange(n_t)[:, None]
    offsets = jnp.arange(n_k)[None, :]
    sample_indices = (rows * n_f - n_k // 2 + offsets) % n_d
    packets = data[sample_indices] * phi[None, :]
    transformed = jnp.fft.rfft(packets, axis=1)

    modes = jnp.arange(1, n_f)[None, :]
    selected = transformed[:, (jnp.arange(1, n_f) * mult)]
    interior = jnp.where((rows + modes) % 2 == 1, -jnp.imag(selected), jnp.real(selected))

    wave = jnp.zeros((n_t, n_f), dtype=data.dtype)
    wave = wave.at[:, 1:].set(interior)
    wave = wave.at[::2, 0].set(jnp.real(transformed[::2, 0]) / jnp.sqrt(2.0))
    return wave.at[1::2, 0].set(jnp.real(transformed[::2, n_f * mult]) / jnp.sqrt(2.0))


def assign_wdata(i, k_cutoff, n_d, n_f, wdata, data_pad, phi):
    """Fill one legacy packet buffer, or return an immutable JAX packet."""
    indices = (i * n_f - k_cutoff // 2 + jnp.arange(k_cutoff)) % n_d
    result = jnp.asarray(data_pad[:n_d])[indices] * jnp.asarray(phi)
    if isinstance(wdata, np.ndarray):
        wdata[...] = np.asarray(result)
        return None
    return result


def pack_wave(i, mult, n_f, wdata_trans, wave):
    """Pack one transformed time packet into the WDM array."""
    result = jnp.asarray(wave)
    transformed = jnp.asarray(wdata_trans)
    if i % 2 == 0 and i < wave.shape[0] - 1:
        result = result.at[i, 0].set(jnp.real(transformed[0]) / jnp.sqrt(2.0))
        result = result.at[i + 1, 0].set(jnp.real(transformed[n_f * mult]) / jnp.sqrt(2.0))
    modes = jnp.arange(1, n_f)
    selected = transformed[modes * mult]
    values = jnp.where((i + modes) % 2 == 1, -jnp.imag(selected), jnp.real(selected))
    result = result.at[i, 1:].set(values)
    if isinstance(wave, np.ndarray):
        wave[...] = np.asarray(result)
        return None
    return result


def phi_vec(n_f, nx=4.0, mult=16):
    """Return the time-domain WDM window."""
    omega = np.pi
    d_omega = omega / n_f
    inverse_sqrt_d_omega = 1.0 / np.sqrt(d_omega)
    k_cutoff = mult * 2 * n_f
    half_k_cutoff = mult * n_f
    dom = 2 * np.pi / k_cutoff

    spectrum = np.zeros(k_cutoff, dtype=np.complex128)
    spectrum[0] = inverse_sqrt_d_omega
    spectrum[1 : half_k_cutoff + 1] = phitilde_vec(dom * np.arange(1, half_k_cutoff + 1), n_f, nx)
    spectrum[half_k_cutoff + 1 :] = phitilde_vec(-dom * np.arange(half_k_cutoff - 1, 0, -1), n_f, nx)
    transformed = k_cutoff * jnp.fft.ifft(jnp.asarray(spectrum), k_cutoff)
    phi = jnp.concatenate((jnp.real(transformed[half_k_cutoff:]), jnp.real(transformed[:half_k_cutoff])))
    return phi * (jnp.sqrt(2.0) / jnp.sqrt(k_cutoff / dom))
