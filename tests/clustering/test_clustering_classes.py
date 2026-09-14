"""Tests for the ``Clustering`` base class, ``PrecomputedClustering`` and ``InjectionClustering``.

``InjectionClustering`` is exercised with a sine-Gaussian source model rather than a compact-binary
waveform. The point of the class is the bookkeeping around the clustering -- reading the parameter
file, rebuilding zero-noise interferometers, OR-ing one filter per injection -- and a burst with a
frequency and time chosen by the test makes that bookkeeping checkable against a known answer. A
compact-binary waveform would make the expected cluster depend on a waveform approximant, which is
not what is under test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

from nullcal.clustering.base import Clustering
from nullcal.clustering.injection import InjectionClustering
from nullcal.clustering.precompute import PrecomputedClustering


class TestClusteringBase:
    """The base class holds the transform and lazily exposes a filter."""

    @pytest.mark.unit
    def test_shape_is_delegated_to_the_transform(self, time_frequency_transform):
        """Subclasses validate filters against this, so it has to be the transform's own grid."""
        assert Clustering(time_frequency_transform).shape == time_frequency_transform.shape

    @pytest.mark.unit
    def test_accessing_an_unset_filter_raises(self, time_frequency_transform):
        """The base class has no way to compute a filter, so asking for one is a programming error.

        Returning ``None`` instead would propagate into ``NullStream`` and fail there as a confusing
        indexing error rather than here as a clear one.
        """
        with pytest.raises(ValueError, match="_time_frequency_filter is None"):
            _ = Clustering(time_frequency_transform).time_frequency_filter


class TestPrecomputedClustering:
    """A filter supplied by the caller, validated on construction."""

    @pytest.mark.unit
    def test_returns_the_filter_it_was_given(self, time_frequency_transform):
        time_frequency_filter = np.zeros(time_frequency_transform.shape, dtype=bool)
        time_frequency_filter[3, 4] = True

        clustering = PrecomputedClustering(time_frequency_transform, time_frequency_filter)

        assert np.array_equal(clustering.time_frequency_filter, time_frequency_filter)

    @pytest.mark.unit
    def test_rejects_a_filter_of_the_wrong_shape(self, time_frequency_transform):
        """A mismatched filter would broadcast or index wrongly rather than fail, so it is caught
        at construction, where the caller can still see which array was wrong."""
        n_t, n_f = time_frequency_transform.shape

        with pytest.raises(ValueError, match="does not match the expected shape"):
            PrecomputedClustering(time_frequency_transform, np.zeros((n_t, n_f + 1), dtype=bool))

    @pytest.mark.unit
    def test_rejects_a_transposed_filter(self, time_frequency_transform):
        """The grid is non-square here, so a transposed filter is caught by the same check.

        Worth its own case: transposition is the mistake the shape check exists for, and on a square
        grid it would pass silently.
        """
        n_t, n_f = time_frequency_transform.shape
        assert n_t != n_f

        with pytest.raises(ValueError, match="does not match the expected shape"):
            PrecomputedClustering(time_frequency_transform, np.zeros((n_f, n_t), dtype=bool))


def sine_gaussian_frequency_domain(frequency_array, amplitude, frequency, tau, **kwargs):
    """A sine-Gaussian with an analytic Fourier transform, used as a stand-in source model.

    Chosen so the injected signal's time-frequency footprint is set by ``frequency`` and ``tau``
    alone, with no waveform approximant in the way. Returned as complex because bilby applies a
    time-shift phase to the polarisations.
    """
    envelope = amplitude * tau * np.sqrt(np.pi) * np.exp(-((np.pi * tau) ** 2) * (frequency_array - frequency) ** 2)
    envelope = envelope.astype(complex)
    return {"plus": envelope, "cross": np.zeros_like(envelope)}


@pytest.fixture
def injection_setup(tmp_path, time_frequency_transform, burst_parameters):
    """An ET triangle, a sine-Gaussian generator, and a one-row parameter file."""
    sampling_frequency = burst_parameters["sampling_frequency"]
    duration = burst_parameters["duration"]

    ifos = InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=0.0)
    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=0.0,
        frequency_domain_source_model=sine_gaussian_frequency_domain,
    )
    parameter_file = tmp_path / "injections.csv"
    pd.DataFrame(
        [
            {
                "amplitude": 5e-22,
                "frequency": burst_parameters["frequency"],
                "tau": 0.05,
                "ra": 0.0,
                "dec": 0.0,
                "psi": 0.0,
                "geocent_time": burst_parameters["centre_time"],
            }
        ]
    ).to_csv(parameter_file, index=False)

    return {
        "interferometers": ifos,
        "waveform_generator": waveform_generator,
        "parameter_file": str(parameter_file),
        "time_frequency_transform": time_frequency_transform,
    }


class TestInjectionClustering:
    """The filter is built by injecting each parameter set and OR-ing the resulting clusters."""

    @pytest.mark.unit
    def test_filter_has_the_transform_shape_and_boolean_dtype(self, injection_setup):
        clustering = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=injection_setup["interferometers"],
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=injection_setup["parameter_file"],
            threshold=1.0,
        )

        time_frequency_filter = clustering.time_frequency_filter

        assert time_frequency_filter.shape == injection_setup["time_frequency_transform"].shape
        assert time_frequency_filter.dtype == np.bool_

    @pytest.mark.unit
    def test_filter_selects_the_injected_burst(self, injection_setup, burst_parameters):
        """The filter retains the injected sine-Gaussian's analytically known centre layer.

        The source has Gaussian, and therefore non-compact, frequency support.  Its thresholded
        cluster may consequently extend beyond the WDM window's adjacent-layer overlap; requiring
        every selected pixel to be adjacent would confuse the source spectrum with the transform
        window.  The invariant here is that the bookkeeping does not lose the known centre.
        """
        clustering = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=injection_setup["interferometers"],
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=injection_setup["parameter_file"],
            threshold=1.0,
        )

        selected = np.argwhere(clustering.time_frequency_filter)
        assert len(selected) > 0
        assert burst_parameters["layer"] in set(selected[:, 1])

    @pytest.mark.unit
    def test_filter_is_computed_once_and_cached(self, injection_setup):
        """The property injects waveforms, which is expensive; the second access must not redo it.

        Checked by deleting the parameter file after the first access: a second computation would
        have to read it and would fail.
        """
        import os

        clustering = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=injection_setup["interferometers"],
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=injection_setup["parameter_file"],
            threshold=1.0,
        )

        first = clustering.time_frequency_filter
        os.remove(injection_setup["parameter_file"])
        second = clustering.time_frequency_filter

        assert second is first

    @pytest.mark.unit
    def test_multiple_injections_are_combined_with_or(self, injection_setup, burst_parameters, tmp_path):
        """Two injections at different frequencies give a filter containing both.

        The union is what makes the filter cover every signal the analysis expects, so a filter that
        kept only the last row -- or only the loudest -- would quietly drop signals. Checked against
        the single-injection filter so the growth is attributable.
        """
        single = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=injection_setup["interferometers"],
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=injection_setup["parameter_file"],
            threshold=1.0,
        ).time_frequency_filter

        second_layer = burst_parameters["layer"] + 8
        parameter_file = tmp_path / "two_injections.csv"
        rows = pd.read_csv(injection_setup["parameter_file"]).to_dict("records")
        rows.append({**rows[0], "frequency": second_layer * burst_parameters["frequency_resolution"]})
        pd.DataFrame(rows).to_csv(parameter_file, index=False)

        combined = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=injection_setup["interferometers"],
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=str(parameter_file),
            threshold=1.0,
        ).time_frequency_filter

        assert np.all(combined[single])  # the union contains the first injection's pixels
        assert combined.sum() > single.sum()
        selected_layers = set(np.argwhere(combined)[:, 1])
        assert any(abs(layer - second_layer) <= 1 for layer in selected_layers)

    @pytest.mark.unit
    def test_et_triangle_is_recognised_by_detector_name(self, injection_setup):
        """Three detectors named ET1/ET2/ET3 are rebuilt from the single name ``ET``.

        The special case exists because bilby has no ``ET1`` interferometer to look up on its own --
        the triangle is only constructible as a unit. If the branch stopped matching, the rebuild
        would raise rather than silently mis-cluster, but it would raise deep inside the property.
        """
        ifos = injection_setup["interferometers"]
        assert [ifo.name for ifo in ifos] == ["ET1", "ET2", "ET3"]

        clustering = InjectionClustering(
            time_frequency_transform=injection_setup["time_frequency_transform"],
            interferometers=ifos,
            waveform_generator=injection_setup["waveform_generator"],
            parameter_file=injection_setup["parameter_file"],
            threshold=1.0,
        )

        assert clustering.time_frequency_filter.any()
