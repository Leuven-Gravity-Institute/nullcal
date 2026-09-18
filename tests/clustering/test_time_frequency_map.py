"""Tests for the combined time-frequency power map.

``construct_time_frequency_map`` whitens each detector's strain, transforms it into the wavelet
domain twice (in phase and in quadrature), and sums the two powers over detectors. The anchors are
that the burst appears in the pixel its time and frequency put it in, and that summing over three
identical detectors multiplies the map by exactly three.
"""

from __future__ import annotations

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullcal.clustering.time_frequency_map import construct_time_frequency_map
from nullcal.data import InterferometerData


@pytest.mark.unit
def test_map_has_the_shape_of_the_transform(interferometers, time_frequency_transform):
    time_frequency_map = construct_time_frequency_map(interferometers, time_frequency_transform)

    assert time_frequency_map.shape == time_frequency_transform.shape


@pytest.mark.unit
def test_map_is_non_negative(interferometers, time_frequency_transform):
    """It is a sum of squares. A negative pixel would mean a sign error in the power combination,
    and the quantile threshold downstream would then be computed over a domain that has no
    physical meaning."""
    time_frequency_map = construct_time_frequency_map(interferometers, time_frequency_transform)

    assert np.all(time_frequency_map >= 0.0)


@pytest.mark.unit
def test_burst_appears_in_the_pixel_its_time_and_frequency_select(
    interferometers, time_frequency_transform, burst_parameters
):
    """The map's peak is at the burst's own time-frequency coordinates.

    Both coordinates are predicted from the burst parameters and the WDM tiling, with no reference
    to a previous run: the frequency layer is ``BURST_FREQUENCY / frequency_resolution`` and the
    time row is ``BURST_CENTRE_TIME / delta_t``. This is the test that would catch a transposed map,
    an off-by-one frequency axis, or a whitening step applied to the wrong array.
    """
    n_t, _ = time_frequency_transform.shape
    delta_t = burst_parameters["duration"] / n_t

    time_frequency_map = construct_time_frequency_map(interferometers, time_frequency_transform)

    row, layer = np.unravel_index(int(np.argmax(time_frequency_map)), time_frequency_map.shape)
    assert layer == burst_parameters["layer"]
    assert abs(row - burst_parameters["centre_time"] / delta_t) <= 1.0


@pytest.mark.unit
def test_power_adds_over_detectors(interferometers, time_frequency_transform):
    """Three identical detectors give exactly three times the one-detector map.

    The detectors carry identical strain and identical noise curves, so the sum over the detector
    axis is a multiplication by three -- arithmetic, not a measurement. A map that averaged instead
    of summing, or that dropped a detector, fails here and nowhere else in this file.
    """
    single = InterferometerData(
        psd=interferometers.psd[:1],
        strain=interferometers.strain[:1],
        mask=interferometers.mask[:1],
        frequency_array=interferometers.frequency_array,
        duration=interferometers.duration,
        sampling_frequency=interferometers.sampling_frequency,
        start_time=interferometers.start_time,
        name=interferometers.name[:1],
    )

    combined_map = construct_time_frequency_map(interferometers, time_frequency_transform)
    single_map = construct_time_frequency_map(single, time_frequency_transform)

    assert len(interferometers) == 3
    # Peak-relative: the map has pixels that are legitimately zero.
    assert np.max(np.abs(combined_map - 3.0 * single_map)) / np.max(np.abs(combined_map)) < 1e-12


@pytest.mark.unit
def test_zero_strain_gives_a_zero_map(time_frequency_transform, burst_parameters):
    """No signal, no power. Guards against a whitening step that adds a constant offset."""
    ifos = InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=burst_parameters["sampling_frequency"],
        duration=burst_parameters["duration"],
        start_time=0.0,
    )

    time_frequency_map = construct_time_frequency_map(
        InterferometerData.from_interferometers(ifos), time_frequency_transform
    )

    assert np.all(time_frequency_map == 0.0)


@pytest.mark.unit
def test_map_scales_quadratically_with_strain_amplitude(interferometers, time_frequency_transform, burst_parameters):
    """Doubling the strain quadruples the map: it is a power, not an amplitude.

    Checked by rescaling the detectors' frequency-domain strain in place and rebuilding the map. A
    map that was linear in strain would still peak in the right pixel, so the tests above do not
    cover this.
    """
    reference = construct_time_frequency_map(interferometers, time_frequency_transform)
    scaled = InterferometerData(
        psd=interferometers.psd,
        strain=2.0 * interferometers.strain,
        mask=interferometers.mask,
        frequency_array=interferometers.frequency_array,
        duration=interferometers.duration,
        sampling_frequency=interferometers.sampling_frequency,
        start_time=interferometers.start_time,
        name=interferometers.name,
    )

    doubled = construct_time_frequency_map(scaled, time_frequency_transform)

    assert np.max(np.abs(doubled - 4.0 * reference)) / np.max(np.abs(doubled)) < 1e-12


@pytest.mark.unit
def test_map_averages_in_phase_and_quadrature_power(interferometers):
    """Each detector contributes ``(in_phase**2 + quadrature**2) / 2``.

    A controlled transform returning the sides 3 and 4 of a Pythagorean triple makes the expected
    per-detector power ``(9 + 16) / 2 = 12.5`` exactly.  This isolates the combination rule from
    the WDM numerics and catches either quadrature being dropped or the factor of two changing.
    """

    class ControlledTransform:
        shape = (2, 3)

        @staticmethod
        def frequency_to_wavelet(frequency_domain_data):
            return np.full((2, 3), 3.0)

        @staticmethod
        def frequency_to_wavelet_quadrature(frequency_domain_data):
            return np.full((2, 3), 4.0)

    result = construct_time_frequency_map(interferometers, ControlledTransform())

    assert np.array_equal(result, np.full((2, 3), len(interferometers) * 12.5))
