"""Tests for the short-time Fourier transform.

The anchors here are analytic. For a unit-amplitude tone at exactly the centre of bin :math:`k` and
a rectangular window of ``window_size`` samples, the unnormalised rFFT coefficient has modulus
``window_size / 2`` (the tone's energy splits between the positive and negative frequency, and only
the positive one appears in an rFFT). This implementation divides by the sampling frequency, so the
expected modulus is ``window_size / (2 * sampling_frequency)``. None of that is read off a run.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.time_frequency_transform.stft import stft

SAMPLING_FREQUENCY = 512.0
FREQUENCY_RESOLUTION = 8.0
DURATION = 4.0
N_SAMPLES = int(SAMPLING_FREQUENCY * DURATION)
WINDOW_SIZE = int(SAMPLING_FREQUENCY / FREQUENCY_RESOLUTION)


@pytest.fixture
def time_array():
    return np.arange(N_SAMPLES) / SAMPLING_FREQUENCY


@pytest.fixture
def rectangular_window():
    return np.ones(WINDOW_SIZE)


@pytest.mark.unit
def test_output_shape_follows_the_frame_decomposition(time_array, rectangular_window):
    """``(n_frames, window_size // 2 + 1)``, with the frames tiling the data without overlap."""
    data = np.sin(2 * np.pi * 40.0 * time_array)

    result = stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)

    assert result.shape == (N_SAMPLES // WINDOW_SIZE, WINDOW_SIZE // 2 + 1)
    assert np.iscomplexobj(result)


@pytest.mark.unit
@pytest.mark.parametrize("bin_index", [3, 10, 25])
def test_tone_appears_in_its_own_bin_with_the_analytic_amplitude(time_array, rectangular_window, bin_index):
    """A tone at :math:`k \\Delta f` gives modulus ``window_size / (2 f_s)`` in bin :math:`k`.

    Both the bin and the amplitude are known in closed form, so this pins the frequency axis and
    the normalisation at once. The normalisation matters: the division by ``sampling_frequency``
    makes the result a spectral density rather than a raw FFT sum, and a caller comparing it to a
    power spectral density depends on that.
    """
    data = np.cos(2 * np.pi * bin_index * FREQUENCY_RESOLUTION * time_array)

    result = stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)

    magnitudes = np.abs(result[0])
    assert int(np.argmax(magnitudes)) == bin_index
    assert magnitudes[bin_index] == pytest.approx(WINDOW_SIZE / (2 * SAMPLING_FREQUENCY), rel=1e-12)


@pytest.mark.unit
def test_zero_frequency_bin_holds_the_frame_mean(rectangular_window):
    """Bin 0 of an rFFT is the sum of the frame; divided by ``f_s`` it is the mean times the frame
    duration. A constant offset is the simplest input with a known transform."""
    offset = 3.0
    data = np.full(N_SAMPLES, offset)

    result = stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)

    expected = offset * WINDOW_SIZE / SAMPLING_FREQUENCY
    assert np.all(np.abs(result[:, 0].real - expected) < 1e-12)
    assert np.max(np.abs(result[:, 1:])) < 1e-12  # a constant has no other frequency content


@pytest.mark.unit
def test_window_is_applied_to_every_frame(time_array):
    """A zero window must zero the output; it is the cheapest proof the window is used at all."""
    data = np.sin(2 * np.pi * 40.0 * time_array)

    result = stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, np.zeros(WINDOW_SIZE))

    assert np.all(result == 0.0)


@pytest.mark.unit
def test_frames_are_transformed_independently(rectangular_window):
    """Energy in one frame must not appear in another.

    The frames are laid out by a reshape, so a stride error would smear a transient across the whole
    map -- the kind of fault that leaves the total energy right and the localisation wrong.
    """
    data = np.zeros(N_SAMPLES)
    data[WINDOW_SIZE : 2 * WINDOW_SIZE] = 1.0  # exactly the second frame

    result = stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)

    energy_per_frame = np.sum(np.abs(result) ** 2, axis=1)
    assert int(np.argmax(energy_per_frame)) == 1
    assert np.max(np.delete(energy_per_frame, 1)) == 0.0


@pytest.mark.unit
def test_rejects_complex_input(rectangular_window):
    """The frame count and the rFFT both assume a real time series."""
    data = np.ones(N_SAMPLES, dtype=complex)

    with pytest.raises(ValueError, match="data must be real-valued"):
        stft(data, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)


@pytest.mark.unit
def test_accepts_a_list_as_input(rectangular_window):
    """``np.asarray`` is called on the input, so a plain sequence is a supported argument."""
    result = stft([0.0] * N_SAMPLES, SAMPLING_FREQUENCY, FREQUENCY_RESOLUTION, rectangular_window)

    assert result.shape == (N_SAMPLES // WINDOW_SIZE, WINDOW_SIZE // 2 + 1)
