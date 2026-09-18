"""JAX kernels for inverse WDM transforms in the time domain."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def _compact_packets(wave_in, n_f, n_t):
    """Pack adjacent WDM time rows into complex FFT inputs."""
    wave_in = jnp.asarray(wave_in)
    even_rows = jnp.arange(0, n_t, 2)
    even = wave_in[even_rows]
    odd = wave_in[even_rows + 1]
    packets = jnp.zeros((n_t // 2, 2 * n_f), dtype=jnp.result_type(wave_in, 1j))
    packets = packets.at[:, 0].set(jnp.sqrt(2.0) * even[:, 0])
    packets = packets.at[:, n_f].set(jnp.sqrt(2.0) * odd[:, 0])

    even_modes = jnp.arange(2, n_f - 1, 2)
    packets = packets.at[:, even_modes].set(even[:, even_modes] - odd[:, even_modes])
    packets = packets.at[:, 2 * n_f - even_modes].set(even[:, even_modes] + odd[:, even_modes])

    odd_modes = jnp.arange(1, n_f, 2)
    packets = packets.at[:, odd_modes].set(1j * (even[:, odd_modes] - odd[:, odd_modes]))
    return packets.at[:, 2 * n_f - odd_modes].set(-1j * (even[:, odd_modes] + odd[:, odd_modes]))


def inverse_wavelet_time_helper_fast(wave_in, phi, n_f, n_t, mult):
    """Return the inverse time-domain transform using batched JAX FFTs."""
    wave_in = jnp.asarray(wave_in)
    phi = jnp.asarray(phi, dtype=wave_in.dtype)
    n_d = n_f * n_t
    k_cutoff = mult * 2 * n_f
    transformed = jnp.fft.fft(_compact_packets(wave_in, n_f, n_t), axis=1)

    pair_rows = jnp.arange(0, n_t, 2)[:, None]
    offsets = jnp.arange(k_cutoff)[None, :]
    fft_indices = (-k_cutoff // 2 + pair_rows * n_f + n_d + offsets) % (2 * n_f)
    result_indices = (-k_cutoff // 2 + pair_rows * n_f + offsets) % n_d
    real_values = jnp.take_along_axis(jnp.real(transformed), fft_indices, axis=1)
    imag_indices = (fft_indices + n_f) % (2 * n_f)
    imag_values = jnp.take_along_axis(jnp.imag(transformed), imag_indices, axis=1)

    result = jnp.zeros(n_d, dtype=wave_in.dtype)
    result = result.at[result_indices].add(phi[None, :] * real_values)
    return result.at[(result_indices + n_f) % n_d].add(phi[None, :] * imag_values)
