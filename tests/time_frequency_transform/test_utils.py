"""Tests for the wavelet grid helper.

``get_shape_of_wavelet_transform`` decides the shape of every time-frequency array in the package,
so an error here is a mislabelled frequency axis everywhere downstream rather than a crash.
"""

from __future__ import annotations

import pytest

from nullcal.time_frequency_transform import get_shape_of_wavelet_transform


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_length", "sampling_frequency", "frequency_resolution", "expected"),
    [
        (2048, 512.0, 8.0, (64, 32)),
        (4096, 4096.0, 16.0, (32, 128)),
        (65536, 4096.0, 16.0, (512, 128)),
        (16384, 2048.0, 4.0, (64, 256)),
    ],
)
def test_shape_follows_the_wdm_tiling(t_length, sampling_frequency, frequency_resolution, expected):
    """:math:`n_f = f_s / (2 \\Delta f)` and :math:`n_t = N / n_f`.

    The expected values are computed from the tiling by hand, not read back from the function: the
    number of frequency layers is the Nyquist band divided by the requested resolution, and the
    time rows are whatever is left of the sample count.
    """
    assert get_shape_of_wavelet_transform(t_length, sampling_frequency, frequency_resolution) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("t_length", "sampling_frequency", "frequency_resolution"),
    [(2048, 512.0, 8.0), (4096, 4096.0, 16.0), (16384, 2048.0, 4.0)],
)
def test_grid_is_critically_sampled(t_length, sampling_frequency, frequency_resolution):
    """The grid holds exactly as many pixels as the time series has samples.

    This is the WDM half-density condition, :math:`\\Delta t \\, \\Delta f = 1/2`: the transform is
    critically sampled, so a grid with more or fewer pixels than samples would mean the transform
    is redundant or lossy. See the references in ``test_wavelet_transforms.py``.
    """
    n_t, n_f = get_shape_of_wavelet_transform(t_length, sampling_frequency, frequency_resolution)

    assert n_t * n_f == t_length
    delta_t = (t_length / sampling_frequency) / n_t
    delta_f = sampling_frequency / (2 * n_f)
    assert delta_t * delta_f == pytest.approx(0.5, rel=1e-15)


@pytest.mark.unit
def test_returns_time_rows_before_frequency_layers():
    """Order matters: the result indexes ``wave[time, frequency]``.

    A transposed return would broadcast without error wherever ``n_t == n_f`` and fail obscurely
    elsewhere, so the ordering is pinned with a deliberately non-square grid.
    """
    n_t, n_f = get_shape_of_wavelet_transform(4096, 4096.0, 16.0)

    assert (n_t, n_f) == (32, 128)
    assert n_t != n_f
