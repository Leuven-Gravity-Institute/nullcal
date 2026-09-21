"""Tests for the time-domain WDM path.

There are two independent algorithms in this package for the same mathematical object. One works
from the frequency-domain data (``transform_wavelet_freq``); the other windows the time series
directly (``transform_wavelet_time``). This module tests the second.

The direct algorithm satisfies the same external anchors as the frequency-domain one: Parseval,
an exact round trip, and a tone in the analytically expected layer.

Note on the cross-path comparison in ``test_time_and_frequency_paths_agree``: the agreement between
the two implementations is reported as a *consistency* check only. Two implementations agreeing
bounds neither one's accuracy -- both are anchored separately, above, against properties that do
not come from this package.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.time_frequency_transform import get_shape_of_wavelet_transform, transform_wavelet_freq_time
from nullcal.time_frequency_transform.inverse_wavelet_time_funcs import (
    _compact_packets,
    inverse_wavelet_time_helper_fast,
)
from nullcal.time_frequency_transform.transform_time_funcs import phi_vec, transform_wavelet_time_helper
from nullcal.time_frequency_transform.wavelet_transforms import inverse_wavelet_time, transform_wavelet_time

SAMPLING_FREQUENCY = 512.0
DURATION = 4.0
FREQUENCY_RESOLUTION = 8.0
N_SAMPLES = int(SAMPLING_FREQUENCY * DURATION)
MULT = 32

# Same basis as test_wavelet_transforms.ROUND_TRIP_TOL: ~200x the float64 FFT round-off bound.
ROUND_TRIP_TOL = 1e-12


def peak_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """``max|actual - reference| / max|reference|``."""
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


@pytest.fixture
def shape():
    return get_shape_of_wavelet_transform(
        t_length=N_SAMPLES,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_resolution=FREQUENCY_RESOLUTION,
    )


@pytest.fixture
def phi(shape):
    """The time-domain filter used by the direct transform."""
    _, n_f = shape
    return phi_vec(n_f, 4.0, MULT)


@pytest.fixture
def noise():
    return np.random.default_rng(20260914).normal(size=N_SAMPLES)


@pytest.mark.unit
def test_compact_packets_preserve_odd_row_sign_for_even_modes():
    """Odd-row even modes enter the low and mirrored packets with opposite signs."""
    wave = np.zeros((2, 4))
    wave[0, 2] = 5.0
    wave[1, 2] = 2.0

    packets = np.asarray(_compact_packets(wave, n_f=4, n_t=2))

    np.testing.assert_array_equal(packets[0, [2, 6]], [3.0, 7.0])


@pytest.mark.unit
def test_phi_vec_has_the_length_the_kernels_index(shape, phi):
    """The filter spans ``mult * 2 * n_f`` samples, the window length the kernels loop over.

    A shorter filter would leave the transform without one weight per indexed sample.
    """
    _, n_f = shape

    assert phi.shape == (MULT * 2 * n_f,)
    assert np.isrealobj(phi)
    assert np.all(np.isfinite(phi))


@pytest.mark.unit
def test_time_domain_transform_conserves_energy(shape, phi, noise):
    """Parseval for the time-domain path, with the same constant as the frequency-domain path.

    Anchored on the orthonormality of the WDM basis, not on the other implementation.
    """
    n_t, n_f = shape

    wave = transform_wavelet_time_helper(noise, n_f, n_t, phi, MULT) * np.sqrt(N_SAMPLES)

    assert wave.shape == (n_t, n_f)
    assert np.sum(wave**2) / np.sum(noise**2) == pytest.approx(N_SAMPLES, rel=1e-9)


@pytest.mark.unit
def test_time_domain_transform_places_a_tone_in_its_own_layer(shape, phi):
    """A tone at :math:`f = m \\Delta f` lands in layer :math:`m`, the analytically known answer."""
    n_t, n_f = shape
    layer = 9
    time_array = np.arange(N_SAMPLES) / SAMPLING_FREQUENCY
    data = np.sin(2 * np.pi * layer * FREQUENCY_RESOLUTION * time_array)

    wave = transform_wavelet_time_helper(data, n_f, n_t, phi, MULT) * np.sqrt(N_SAMPLES)

    power_per_layer = np.sum(wave**2, axis=0)
    assert int(np.argmax(power_per_layer)) == layer
    assert power_per_layer[layer] / np.sum(power_per_layer) > 0.999


@pytest.mark.unit
def test_time_domain_inverse_recovers_the_time_series(shape, phi, noise):
    """The time-domain inverse kernel inverts the transform -- an identity, so round-off tolerance.

    ``mult = n_t // 2`` here, the setting at which the docstring of ``transform_wavelet_time`` says
    the transform is exact; at smaller ``mult`` it is only approximate, and that is a documented
    property rather than a defect.
    """
    n_t, n_f = shape
    wave = transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t)

    output = inverse_wavelet_time_helper_fast(wave, phi / 2, n_f, n_t, MULT)
    recovered = output / np.sqrt(output.shape[0])

    assert recovered.shape == noise.shape
    assert peak_relative_error(recovered, noise) <= ROUND_TRIP_TOL


@pytest.mark.unit
def test_time_and_frequency_paths_agree(shape, phi, noise):
    """Consistency check between the two implementations -- not an accuracy claim for either.

    Both are anchored independently above (Parseval, round trip, tone placement). Agreement between
    two implementations bounds neither one's error; it detects that a change touched one path and
    not the other, which is a different and still useful thing. It is recorded here as exactly that.
    """
    n_t, n_f = shape

    from_time = transform_wavelet_time_helper(noise, n_f, n_t, phi, MULT) * np.sqrt(N_SAMPLES)
    from_frequency = transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t)

    assert peak_relative_error(from_time, from_frequency) <= ROUND_TRIP_TOL


@pytest.mark.unit
def test_transform_wavelet_time_is_callable(shape, noise):
    """``transform_wavelet_time`` should transform a time series, as its signature advertises."""
    n_t, n_f = shape

    wave = transform_wavelet_time(noise, n_f=n_f, n_t=n_t)

    assert wave.shape == (n_t, n_f)


@pytest.mark.unit
def test_transform_wavelet_time_promotes_integer_input_to_float():
    """Integer samples use the same floating-point computation as their float64 values."""
    n_t, n_f = 8, 4
    integer_data = np.arange(n_t * n_f) % 7

    integer_wave = np.asarray(transform_wavelet_time(integer_data, n_f=n_f, n_t=n_t, mult=4))
    float_wave = np.asarray(transform_wavelet_time(integer_data.astype(np.float64), n_f=n_f, n_t=n_t, mult=4))

    assert integer_wave.dtype == np.float64
    assert np.count_nonzero(integer_wave) == integer_wave.size
    np.testing.assert_array_equal(integer_wave, float_wave)


@pytest.mark.unit
def test_inverse_wavelet_time_is_callable(shape, noise):
    """``inverse_wavelet_time`` should invert the transform, as its signature advertises."""
    n_t, n_f = shape
    wave = transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t)

    recovered = inverse_wavelet_time(wave, n_f=n_f, n_t=n_t)

    assert peak_relative_error(recovered, noise) <= ROUND_TRIP_TOL
