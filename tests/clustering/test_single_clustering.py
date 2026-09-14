"""Tests for the two single-signal cluster selectors.

``single_clustering_by_threshold`` and ``single_clustering_by_quantile`` differ only in how they
turn the power map into a boolean filter -- an absolute cut versus a quantile of the non-zero
pixels. Everything else, including the band restriction, is duplicated between them, and the last
test in this module records that the duplication has already drifted.

Several tests replace ``clustering`` with a spy. That is deliberate: the band restriction happens
*before* the connected-component search, and the search keeps only the largest cluster, so the band
edges are not observable in the returned mask. Intercepting the filter is the only way to assert
what band was actually selected rather than what survived the search.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import nullcal.clustering.single as single_module
from nullcal.clustering.single import single_clustering_by_quantile, single_clustering_by_threshold


@pytest.fixture
def captured_filter(monkeypatch):
    """Intercept the boolean filter handed to ``clustering`` and return an all-zero mask."""
    captured = {}

    def spy(tf_filter, dt, df, padding_time=0.1, padding_freq=10):
        captured["tf_filter"] = np.asarray(tf_filter).copy()
        captured["dt"] = dt
        captured["df"] = df
        captured["padding_time"] = padding_time
        captured["padding_freq"] = padding_freq
        return np.zeros(np.shape(tf_filter), dtype=np.uint8)

    monkeypatch.setattr(single_module, "clustering", spy)
    return captured


@pytest.mark.unit
def test_threshold_selector_returns_a_boolean_mask_of_the_transform_shape(interferometers, time_frequency_transform):
    """Callers index the time-frequency arrays with this, so dtype and shape are contractual."""
    mask = single_clustering_by_threshold(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=1.0,
        padding_time=0.0,
        padding_freq=0.0,
    )

    assert mask.shape == time_frequency_transform.shape
    assert mask.dtype == np.bool_


@pytest.mark.unit
def test_threshold_selector_finds_the_injected_burst(interferometers, time_frequency_transform, burst_parameters):
    """The selected pixels sit at the burst's own time and frequency.

    Anchored on the injection parameters, not on a stored mask: the burst was placed at a known
    layer and a known time, and that is where the cluster has to be.

    The selection spreads one layer either side of the burst's own. That is not slop in the test --
    it is the window's compact support, ``|omega| < 3 dOmega / 4``, which
    ``test_wdm_window.py::test_window_support_confines_overlap_to_adjacent_layers`` establishes:
    a layer overlaps its immediate neighbours and nothing further. Selection reaching a
    next-nearest layer would contradict that property and is what this bound rejects.
    """
    n_t, _ = time_frequency_transform.shape
    delta_t = burst_parameters["duration"] / n_t

    mask = single_clustering_by_threshold(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=100.0,
        padding_time=0.0,
        padding_freq=0.0,
    )

    selected = np.argwhere(mask)
    assert len(selected) > 0
    assert burst_parameters["layer"] in set(selected[:, 1])
    assert np.all(np.abs(selected[:, 1] - burst_parameters["layer"]) <= 1)
    assert np.all(np.abs(selected[:, 0] - burst_parameters["centre_time"] / delta_t) <= 2.0)


@pytest.mark.unit
def test_raising_the_threshold_selects_no_more_pixels(interferometers, time_frequency_transform):
    """The selection is monotone in the threshold. A selector that grew with a stricter cut would
    be inverted somewhere, which no single-threshold test can see."""
    low = single_clustering_by_threshold(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=50.0,
        padding_time=0.0,
        padding_freq=0.0,
    )
    high = single_clustering_by_threshold(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=500.0,
        padding_time=0.0,
        padding_freq=0.0,
    )

    assert high.sum() <= low.sum()
    assert np.all(low[high])  # the stricter selection is contained in the looser one


@pytest.mark.unit
def test_quantile_selector_finds_the_injected_burst(interferometers, time_frequency_transform, burst_parameters):
    """The quantile route reaches the same burst without being told an absolute scale."""
    mask = single_clustering_by_quantile(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        quantile=0.999,
        padding_time=0.0,
        padding_freq=0.0,
    )

    selected = np.argwhere(mask)
    assert len(selected) > 0
    assert burst_parameters["layer"] in set(selected[:, 1])
    assert np.all(np.abs(selected[:, 1] - burst_parameters["layer"]) <= 1)


@pytest.mark.unit
def test_quantile_threshold_is_taken_over_non_zero_pixels_only(
    broadband_interferometers, time_frequency_transform, captured_filter
):
    """The quantile is computed over ``map[map > 0]``, so the band-zeroed pixels do not dilute it.

    If the zeros were included, a narrow band would drag the quantile down towards zero and the
    selector would pass almost every in-band pixel. Checked by asking for the 0.5 quantile over a
    restricted band and confirming roughly half the *in-band* pixels are selected, not nearly all
    of them.
    """
    single_clustering_by_quantile(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        quantile=0.5,
        minimum_frequency=80.0,
        maximum_frequency=160.0,
    )

    tf_filter = captured_filter["tf_filter"]
    in_band = tf_filter[:, 10:21]
    fraction = in_band.sum() / in_band.size
    assert 0.2 < fraction < 0.8


@pytest.mark.unit
def test_time_resolution_passed_to_the_search_is_the_pixel_duration(
    interferometers, time_frequency_transform, burst_parameters, captured_filter
):
    """``dt`` must be ``duration / n_t``, since ``clustering`` converts a padding in seconds with it.

    A wrong ``dt`` would silently rescale every time padding the package applies.
    """
    n_t, _ = time_frequency_transform.shape

    single_clustering_by_threshold(
        interferometers=interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=100.0,
    )

    assert captured_filter["dt"] == pytest.approx(burst_parameters["duration"] / n_t)
    assert captured_filter["df"] == burst_parameters["frequency_resolution"]


@pytest.mark.unit
def test_minimum_frequency_excludes_the_layers_below_it(
    broadband_interferometers, time_frequency_transform, captured_filter
):
    """Layers below ``ceil(minimum_frequency / df)`` are zeroed; that layer itself is kept.

    The impulse fixture puts power in every layer, so a surviving in-band layer proves the cut is a
    cut and not a wipe.
    """
    single_clustering_by_threshold(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=1e-6,
        minimum_frequency=32.0,  # layer 4 at 8 Hz resolution
    )

    tf_filter = captured_filter["tf_filter"]
    assert not tf_filter[:, :4].any()
    assert tf_filter[:, 4].any()


@pytest.mark.unit
def test_nyquist_band_edge_is_stepped_down_with_a_warning(
    broadband_interferometers, time_frequency_transform, captured_filter, caplog
):
    """Asking for the whole band drops the top layer rather than including Nyquist.

    The Nyquist layer of the WDM grid holds the redundant half-pixels packed into the ``m = 0``
    column, so treating it as an ordinary layer would double-count. The code steps the edge down
    and says so; this pins both the step and the warning.
    """
    _, n_f = time_frequency_transform.shape

    with caplog.at_level(logging.WARNING, logger="nullcal"):
        single_clustering_by_threshold(
            interferometers=broadband_interferometers,
            time_frequency_transform=time_frequency_transform,
            threshold=1e-6,
            maximum_frequency=250.0,  # floor(250 / 8) = 31 = n_f - 1, the Nyquist layer
        )

    assert int(np.floor(250.0 / 8.0)) == n_f - 1
    assert "contains the Nyquist frequency" in caplog.text
    assert not captured_filter["tf_filter"][:, n_f - 2 :].any()


@pytest.mark.unit
def test_threshold_selector_excludes_the_top_band_layer(
    broadband_interferometers, time_frequency_transform, captured_filter
):
    """Current behaviour of ``single_clustering_by_threshold``: layer ``floor(f_max/df)`` is zeroed.

    Pinned so the divergence recorded in the next test is attributed to the right function.
    """
    single_clustering_by_threshold(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=1e-6,
        maximum_frequency=160.0,  # layer 20
    )

    tf_filter = captured_filter["tf_filter"]
    assert not tf_filter[:, 20].any()
    assert tf_filter[:, 19].any()


@pytest.mark.unit
def test_quantile_selector_includes_the_top_band_layer(
    broadband_interferometers, time_frequency_transform, captured_filter
):
    """Current behaviour of ``single_clustering_by_quantile``: layer ``floor(f_max/df)`` is kept."""
    single_clustering_by_quantile(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        quantile=0.01,
        maximum_frequency=160.0,  # layer 20
    )

    assert captured_filter["tf_filter"][:, 20].any()


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Defect: the two selectors disagree on the top band edge. by_quantile zeroes "
        "[:, freq_high_idx + 1:] and by_threshold zeroes [:, freq_high_idx:], so the same "
        "maximum_frequency gives bands one frequency layer apart. The band restriction is written "
        "out twice and the two copies have drifted. InjectionClustering uses the by_threshold "
        "route, so an injection-derived filter is one layer narrower than a quantile-derived one "
        "for the same configuration. Not fixed here: choosing which convention is correct changes "
        "the analysed band, and that is a numerical decision rather than a cleanup."
    ),
)
def test_both_selectors_use_the_same_band_convention(broadband_interferometers, time_frequency_transform, monkeypatch):
    """The same ``maximum_frequency`` should select the same frequency layers either way."""
    captured = []

    def spy(tf_filter, dt, df, padding_time=0.1, padding_freq=10):
        captured.append(np.asarray(tf_filter).copy())
        return np.zeros(np.shape(tf_filter), dtype=np.uint8)

    monkeypatch.setattr(single_module, "clustering", spy)

    single_clustering_by_threshold(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        threshold=1e-6,
        minimum_frequency=32.0,
        maximum_frequency=160.0,
    )
    single_clustering_by_quantile(
        interferometers=broadband_interferometers,
        time_frequency_transform=time_frequency_transform,
        quantile=0.01,
        minimum_frequency=32.0,
        maximum_frequency=160.0,
    )

    by_threshold_layers = np.nonzero(captured[0].any(axis=0))[0]
    by_quantile_layers = np.nonzero(captured[1].any(axis=0))[0]
    assert by_threshold_layers.max() == by_quantile_layers.max()
