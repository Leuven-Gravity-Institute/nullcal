"""Tests for the frequency-domain transform helpers: the Tukey window and the filtered transform."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal.windows

from nullcal.time_frequency_transform import get_shape_of_wavelet_transform
from nullcal.time_frequency_transform.transform_freq_funcs import (
    phitilde_vec_norm,
    transform_wavelet_freq_helper,
    transform_wavelet_freq_partial_helper,
    tukey,
)

SAMPLING_FREQUENCY = 512.0
DURATION = 4.0
FREQUENCY_RESOLUTION = 8.0
N_SAMPLES = int(SAMPLING_FREQUENCY * DURATION)


@pytest.fixture
def shape():
    return get_shape_of_wavelet_transform(
        t_length=N_SAMPLES,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_resolution=FREQUENCY_RESOLUTION,
    )


class TestTukey:
    """``tukey`` applies a tapered cosine window to its argument, in place."""

    @pytest.mark.unit
    @pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
    def test_matches_the_standard_tukey_window(self, alpha):
        """Anchored against ``scipy.signal.windows.tukey``, an independent implementation.

        The two are not bit-identical: this implementation computes the taper boundaries with
        integer truncation (``imin``, ``imax``), which displaces each taper by less than one
        sample. The residual therefore shrinks as the window lengthens -- at ``n = 1024`` it is
        about 0.012 in absolute window value.

        The bound below is 0.05. It is chosen to sit well inside the gap between that residual and
        the nearest plausible wrong answer: the same comparison against a Hann window is 0.56, and
        against a rectangular window 1.0. So the assertion confirms the window is a Tukey window of
        the requested shape while tolerating the boundary convention, and it is nowhere near loose
        enough to accept a different window. The discrimination is asserted below, not assumed.
        """
        n_sample = 1024
        data = np.ones(n_sample)

        tukey(data, alpha, n_sample)

        reference = scipy.signal.windows.tukey(n_sample, alpha, sym=True)
        assert np.max(np.abs(data - reference)) < 0.05
        # The bound discriminates: other standard windows are far outside it.
        assert np.max(np.abs(data - scipy.signal.windows.hann(n_sample, sym=True))) > 0.5
        assert np.max(np.abs(data - np.ones(n_sample))) > 0.5

    @pytest.mark.unit
    def test_alpha_zero_is_the_identity(self):
        """A Tukey window with no taper is rectangular, so the data must come back untouched.

        The limiting case is also the one that divides by ``imin``; at ``alpha = 0`` that index is
        zero, and the test records that the guard ordering keeps it unreachable.
        """
        data = np.arange(1.0, 129.0)
        expected = data.copy()

        tukey(data, 0.0, data.size)

        assert np.array_equal(data, expected)

    @pytest.mark.unit
    def test_windows_in_place_and_returns_nothing(self):
        """The function mutates its argument; a caller expecting a return value would get ``None``."""
        data = np.ones(128)

        result = tukey(data, 0.25, 128)

        assert result is None
        assert data[0] == 0.0  # the window was applied to the caller's array

    @pytest.mark.unit
    def test_taper_rises_from_zero_and_the_body_is_untouched(self):
        """Structure: zero at the first sample, monotone rise, exactly unity across the body.

        The flat unity region is the point of a Tukey window -- it tapers the edges without
        touching the data in the middle. A window that scaled the body would bias every amplitude
        computed from the result.
        """
        n_sample, alpha = 256, 0.25
        data = np.ones(n_sample)

        tukey(data, alpha, n_sample)

        i_min = int(alpha * (n_sample - 1) / 2)
        i_max = int((n_sample - 1) * (1 - alpha / 2))
        assert data[0] == 0.0
        assert np.all(np.diff(data[: i_min + 1]) >= 0.0)
        assert np.all(data[i_min : i_max + 1] == 1.0)
        assert np.all(np.diff(data[i_max:]) <= 0.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
    def test_window_removes_energy(self, alpha):
        """A taper can only take energy out, never add it."""
        data = np.ones(512)

        tukey(data, alpha, data.size)

        assert 0.0 < np.sum(data**2) < 512

    @pytest.mark.unit
    def test_jax_input_is_jittable_and_returns_windowed_values(self):
        """Immutable JAX inputs return the same taper that NumPy receives in place."""
        data = np.linspace(-1.0, 1.0, 128)
        expected = data.copy()
        tukey(expected, 0.25, expected.size)

        actual = jax.jit(lambda values: tukey(values, 0.25, values.size))(jnp.asarray(data))

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)


class TestPartialTransform:
    """``transform_wavelet_freq_partial_helper`` evaluates only the selected frequency layers."""

    @pytest.mark.unit
    def test_selected_layers_match_the_full_transform(self, shape):
        """Restricting the computation must not change the answer where it is computed.

        This is the contract that makes the partial transform an optimisation rather than an
        approximation: layer ``m`` of the partial result is bit-identical to layer ``m`` of the
        full one. ``atol=0.0`` and ``rtol=0.0`` -- anything less than exact equality would mean the
        two paths compute the layer differently, which is the class of defect that makes a
        "fast path" quietly wrong.
        """
        n_t, n_f = shape
        data = np.fft.rfft(np.random.default_rng(20260914).normal(size=N_SAMPLES))
        phif = 2 / n_f * phitilde_vec_norm(n_f, n_t, 4.0)
        selected = [3, 7, 11]
        frequency_filter = np.zeros(n_f + 1, dtype=bool)
        frequency_filter[selected] = True

        full = transform_wavelet_freq_helper(data, n_f, n_t, phif)
        partial = transform_wavelet_freq_partial_helper(data, n_f, n_t, phif, frequency_filter)

        assert np.array_equal(partial[:, selected], full[:, selected])

    @pytest.mark.unit
    def test_unselected_layers_are_left_at_zero(self, shape):
        """Everything not asked for stays zero, so a caller cannot mistake stale data for a result."""
        n_t, n_f = shape
        data = np.fft.rfft(np.random.default_rng(1).normal(size=N_SAMPLES))
        phif = 2 / n_f * phitilde_vec_norm(n_f, n_t, 4.0)
        selected = [3, 7, 11]
        frequency_filter = np.zeros(n_f + 1, dtype=bool)
        frequency_filter[selected] = True

        partial = transform_wavelet_freq_partial_helper(data, n_f, n_t, phif, frequency_filter)

        populated = np.nonzero(np.any(partial != 0.0, axis=0))[0]
        assert list(populated) == selected

    @pytest.mark.unit
    def test_selecting_nothing_returns_zeros(self, shape):
        """The empty selection is a real case: a frequency mask can exclude every layer."""
        n_t, n_f = shape
        data = np.fft.rfft(np.random.default_rng(2).normal(size=N_SAMPLES))
        phif = 2 / n_f * phitilde_vec_norm(n_f, n_t, 4.0)

        partial = transform_wavelet_freq_partial_helper(data, n_f, n_t, phif, np.zeros(n_f + 1, dtype=bool))

        assert partial.shape == (n_t, n_f)
        assert np.all(partial == 0.0)
