"""JAX kernels for inverse WDM transforms in the frequency domain."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def inverse_wavelet_freq_helper_fast(wave_in, phif, n_f, n_t):
    """Return the inverse frequency-domain transform."""
    wave_in = jnp.asarray(wave_in)
    phif = jnp.asarray(phif, dtype=wave_in.dtype)
    rows = jnp.arange(n_t)
    modes = jnp.arange(1, n_f)[:, None]
    interior = wave_in[:, 1:n_f].T
    factors = jnp.where((rows[None, :] + modes) % 2 == 1, -1j, 1.0)

    prefactors = jnp.zeros((n_f + 1, n_t), dtype=jnp.result_type(wave_in, 1j))
    prefactors = prefactors.at[0].set(wave_in[(2 * rows) % n_t, 0] / jnp.sqrt(2.0))
    prefactors = prefactors.at[n_f].set(wave_in[(2 * rows) % n_t + 1, 0] / jnp.sqrt(2.0))
    prefactors = prefactors.at[1:n_f].set(interior * factors)
    transformed = jnp.fft.fft(prefactors, axis=1)

    res = jnp.zeros(n_f * n_t // 2 + 1, dtype=transformed.dtype)
    half = n_t // 2

    lower_offsets = jnp.arange(half)
    res = res.at[lower_offsets].add(transformed[0, (2 * lower_offsets) % n_t] * phif[lower_offsets])

    upper_offsets = jnp.arange(half + 1)
    upper_indices = n_f * half - upper_offsets
    res = res.at[upper_indices].add(transformed[n_f, (-2 * upper_offsets) % n_t] * phif[upper_offsets])

    offsets = jnp.arange(1 - half, half)
    centres = jnp.arange(1, n_f)[:, None] * half
    indices = centres + offsets[None, :]
    fft_indices = indices % n_t
    contributions = transformed[1:n_f][jnp.arange(n_f - 1)[:, None], fft_indices] * phif[jnp.abs(offsets)][None, :]
    return res.at[indices].add(contributions)
