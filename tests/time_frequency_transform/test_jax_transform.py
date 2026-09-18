"""JAX tracing, frozen-reference, and gradient checks for the WDM transform."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nullcal.time_frequency_transform import transform_wavelet_freq
from nullcal.time_frequency_transform.wavelet_transforms import (
    WaveletTransform,
    inverse_wavelet_freq,
    inverse_wavelet_freq_time,
    inverse_wavelet_time,
    transform_wavelet_freq_quadrature,
    transform_wavelet_freq_time,
    transform_wavelet_freq_time_quadrature,
    transform_wavelet_time,
)

REFERENCE_PEAK_RTOL = 1e-12
GRADIENT_RTOL = 1e-9
REFERENCE_SEED = 20260810

PUBLIC_TRANSFORMS = (
    inverse_wavelet_freq,
    inverse_wavelet_freq_time,
    inverse_wavelet_time,
    transform_wavelet_freq,
    transform_wavelet_freq_quadrature,
    transform_wavelet_freq_time,
    transform_wavelet_freq_time_quadrature,
    transform_wavelet_time,
    WaveletTransform.frequency_to_wavelet,
    WaveletTransform.frequency_to_wavelet_quadrature,
    WaveletTransform.wavelet_to_frequency,
)


def peak_relative_error(actual, reference):
    """Return max absolute error scaled by the reference peak."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


def frozen_probe(n_frequencies):
    """Recreate the deterministic input recorded by the frozen harness."""
    rng = np.random.default_rng(REFERENCE_SEED + 1)
    probe = (rng.normal(size=n_frequencies) + 1j * rng.normal(size=n_frequencies)) / np.sqrt(2.0)
    probe[0] = 0.0
    return probe


@pytest.mark.unit
def test_public_transforms_declare_and_return_jax_arrays():
    """The public WDM boundary deliberately preserves JAX arrays for tracing."""
    for transform in PUBLIC_TRANSFORMS:
        assert transform.__annotations__["return"] == "jax.Array"

    n_t, n_f = 32, 8
    time_data = np.random.default_rng(595).normal(size=n_t * n_f)
    frequency_data = np.fft.rfft(time_data)
    wavelet_data = transform_wavelet_freq(frequency_data, n_f=n_f, n_t=n_t)
    transformer = WaveletTransform(
        duration=1.0,
        sampling_frequency=n_t * n_f,
        frequency_resolution=n_t * n_f / (2 * n_f),
    )
    runtime_calls = (
        (inverse_wavelet_freq, lambda: inverse_wavelet_freq(wavelet_data, n_f=n_f, n_t=n_t)),
        (inverse_wavelet_freq_time, lambda: inverse_wavelet_freq_time(wavelet_data, n_f=n_f, n_t=n_t)),
        (inverse_wavelet_time, lambda: inverse_wavelet_time(wavelet_data, n_f=n_f, n_t=n_t, mult=n_t // 2)),
        (transform_wavelet_freq, lambda: transform_wavelet_freq(frequency_data, n_f=n_f, n_t=n_t)),
        (
            transform_wavelet_freq_quadrature,
            lambda: transform_wavelet_freq_quadrature(frequency_data, n_f=n_f, n_t=n_t),
        ),
        (transform_wavelet_freq_time, lambda: transform_wavelet_freq_time(time_data, n_f=n_f, n_t=n_t)),
        (
            transform_wavelet_freq_time_quadrature,
            lambda: transform_wavelet_freq_time_quadrature(time_data, n_f=n_f, n_t=n_t),
        ),
        (transform_wavelet_time, lambda: transform_wavelet_time(time_data, n_f=n_f, n_t=n_t, mult=n_t // 2)),
        (WaveletTransform.frequency_to_wavelet, lambda: transformer.frequency_to_wavelet(frequency_data)),
        (
            WaveletTransform.frequency_to_wavelet_quadrature,
            lambda: transformer.frequency_to_wavelet_quadrature(frequency_data),
        ),
        (WaveletTransform.wavelet_to_frequency, lambda: transformer.wavelet_to_frequency(wavelet_data)),
    )

    assert tuple(transform for transform, _call in runtime_calls) == PUBLIC_TRANSFORMS

    for transform, call in runtime_calls:
        output = call()
        assert isinstance(output, jax.Array), transform.__qualname__
        assert not isinstance(output, np.ndarray), transform.__qualname__


@pytest.mark.unit
def test_forward_transform_is_jittable_and_matches_frozen_reference():
    """The JAX path reproduces the pre-port probe with no absolute floor."""
    with np.load("tests/e2e/reference/artifacts.npz") as artifacts:
        reference = artifacts["wavelet_probe_output"]
    n_t, n_f = reference.shape
    probe = frozen_probe(n_frequencies=n_t * n_f // 2 + 1)

    actual = jax.jit(lambda values: transform_wavelet_freq(values, n_f=n_f, n_t=n_t))(probe)

    assert actual.dtype == jnp.float64
    assert peak_relative_error(actual, reference) <= REFERENCE_PEAK_RTOL


@pytest.mark.unit
def test_forward_transform_gradient_matches_central_difference():
    """Autodiff through the FFT path agrees with a central difference."""
    n_t, n_f = 32, 8
    base = jnp.asarray(np.fft.rfft(np.random.default_rng(391).normal(size=n_t * n_f)))
    direction = jnp.linspace(0.2, 1.0, base.size) * (1.0 + 0.25j)

    def objective(scale):
        wave = transform_wavelet_freq(base + scale * direction, n_f=n_f, n_t=n_t)
        return jnp.sum(wave**2) / wave.size

    point = 0.13
    # The objective is exactly quadratic in ``scale``, so central difference has
    # no truncation error here.  A 3e-3 step keeps subtraction round-off small.
    step = 3e-3
    autodiff = float(jax.grad(objective)(point))
    finite_difference = float((objective(point + step) - objective(point - step)) / (2.0 * step))
    relative_error = abs(autodiff - finite_difference) / abs(finite_difference)

    assert relative_error <= GRADIENT_RTOL


@pytest.mark.unit
def test_inverse_frequency_transform_is_jittable_and_round_trips():
    """The JAX inverse recovers its input within the FFT round-off budget."""
    n_t, n_f = 32, 8
    frequency_data = np.fft.rfft(np.random.default_rng(392).normal(size=n_t * n_f))

    recovered = jax.jit(
        lambda values: inverse_wavelet_freq(transform_wavelet_freq(values, n_f=n_f, n_t=n_t), n_f=n_f, n_t=n_t)
    )(frequency_data)

    assert recovered.dtype == jnp.complex128
    assert peak_relative_error(recovered, frequency_data) <= REFERENCE_PEAK_RTOL


@pytest.mark.unit
def test_inverse_frequency_transform_gradient_matches_central_difference():
    """Autodiff through the inverse FFT path agrees with central difference."""
    n_t, n_f = 32, 8
    frequency_data = np.fft.rfft(np.random.default_rng(393).normal(size=n_t * n_f))
    base = transform_wavelet_freq(frequency_data, n_f=n_f, n_t=n_t)
    direction = jnp.reshape(jnp.linspace(-0.4, 0.7, n_t * n_f), (n_t, n_f))

    def objective(scale):
        recovered = inverse_wavelet_freq(base + scale * direction, n_f=n_f, n_t=n_t)
        return jnp.sum(jnp.abs(recovered) ** 2) / recovered.size

    point = -0.21
    step = 3e-3
    autodiff = float(jax.grad(objective)(point))
    finite_difference = float((objective(point + step) - objective(point - step)) / (2.0 * step))
    relative_error = abs(autodiff - finite_difference) / abs(finite_difference)

    assert relative_error <= GRADIENT_RTOL


@pytest.mark.unit
def test_time_domain_transform_and_inverse_are_jittable():
    """The direct time-domain pair traces and recovers the original series."""
    n_t, n_f = 32, 8
    data = np.random.default_rng(394).normal(size=n_t * n_f)

    recovered = jax.jit(
        lambda values: inverse_wavelet_time(
            transform_wavelet_time(values, n_f=n_f, n_t=n_t, mult=n_t // 2),
            n_f=n_f,
            n_t=n_t,
            mult=n_t // 2,
        )
    )(data)

    assert recovered.dtype == jnp.float64
    assert peak_relative_error(recovered, data) <= REFERENCE_PEAK_RTOL


@pytest.mark.unit
def test_frozen_reference_requires_x64_precision():
    """float64 meets the frozen budget while float32 demonstrably does not."""
    with np.load("tests/e2e/reference/artifacts.npz") as artifacts:
        reference = artifacts["wavelet_probe_output"]
    n_t, n_f = reference.shape
    probe = frozen_probe(n_frequencies=n_t * n_f // 2 + 1)

    output64 = transform_wavelet_freq(probe, n_f=n_f, n_t=n_t)
    with jax.enable_x64(False):
        output32 = transform_wavelet_freq(jnp.asarray(probe, dtype=jnp.complex64), n_f=n_f, n_t=n_t)

    error64 = peak_relative_error(output64, reference)
    error32 = peak_relative_error(output32, reference)
    assert output64.dtype == jnp.float64
    assert output32.dtype == jnp.float32
    assert error64 <= REFERENCE_PEAK_RTOL
    assert error32 > REFERENCE_PEAK_RTOL
