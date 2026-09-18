"""Tests for clustering containers and dependency-free injection preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.clustering.base import Clustering
from nullcal.clustering.injection import InjectionClustering
from nullcal.clustering.precompute import PrecomputedClustering


class TestClusteringBase:
    @pytest.mark.unit
    def test_shape_is_delegated_to_the_transform(self, time_frequency_transform):
        assert Clustering(time_frequency_transform).shape == time_frequency_transform.shape

    @pytest.mark.unit
    def test_accessing_an_unset_filter_raises(self, time_frequency_transform):
        with pytest.raises(ValueError, match="_time_frequency_filter is None"):
            _ = Clustering(time_frequency_transform).time_frequency_filter


class TestPrecomputedClustering:
    @pytest.mark.unit
    def test_returns_the_filter_it_was_given(self, time_frequency_transform):
        time_frequency_filter = np.zeros(time_frequency_transform.shape, dtype=bool)
        time_frequency_filter[3, 4] = True
        clustering = PrecomputedClustering(time_frequency_transform, time_frequency_filter)
        assert np.array_equal(clustering.time_frequency_filter, time_frequency_filter)

    @pytest.mark.unit
    def test_rejects_a_filter_of_the_wrong_shape(self, time_frequency_transform):
        n_t, n_f = time_frequency_transform.shape
        with pytest.raises(ValueError, match="does not match the expected shape"):
            PrecomputedClustering(time_frequency_transform, np.zeros((n_t, n_f + 1), dtype=bool))

    @pytest.mark.unit
    def test_rejects_a_transposed_filter(self, time_frequency_transform):
        n_t, n_f = time_frequency_transform.shape
        assert n_t != n_f
        with pytest.raises(ValueError, match="does not match the expected shape"):
            PrecomputedClustering(time_frequency_transform, np.zeros((n_f, n_t), dtype=bool))


class TestInjectionClustering:
    @pytest.mark.unit
    def test_filter_has_the_transform_shape_and_selects_the_prepared_burst(
        self, interferometers, time_frequency_transform, burst_parameters
    ):
        clustering = InjectionClustering(time_frequency_transform, [interferometers], threshold=1.0)
        selected = np.argwhere(clustering.time_frequency_filter)
        assert clustering.time_frequency_filter.shape == time_frequency_transform.shape
        assert clustering.time_frequency_filter.dtype == np.bool_
        assert len(selected) > 0
        assert burst_parameters["layer"] in set(selected[:, 1])

    @pytest.mark.unit
    def test_preprocessing_is_eager_and_runs_once(self, monkeypatch, interferometers, time_frequency_transform):
        calls = []

        def fake_cluster(**kwargs):
            calls.append(kwargs["interferometers"])
            result = np.zeros(time_frequency_transform.shape, dtype=bool)
            result[0, 0] = True
            return result

        monkeypatch.setattr("nullcal.clustering.injection.single_clustering_by_threshold", fake_cluster)
        clustering = InjectionClustering(time_frequency_transform, [interferometers], threshold=1.0)
        first = clustering.time_frequency_filter
        second = clustering.time_frequency_filter
        assert len(calls) == 1
        assert second is first

    @pytest.mark.unit
    def test_multiple_prepared_injections_are_combined_with_or(
        self, monkeypatch, interferometers, broadband_interferometers, time_frequency_transform
    ):
        first = np.zeros(time_frequency_transform.shape, dtype=bool)
        first[1, 2] = True
        second = np.zeros(time_frequency_transform.shape, dtype=bool)
        second[3, 4] = True
        outputs = iter((first, second))
        monkeypatch.setattr(
            "nullcal.clustering.injection.single_clustering_by_threshold", lambda **kwargs: next(outputs)
        )
        result = InjectionClustering(
            time_frequency_transform,
            [interferometers, broadband_interferometers],
            threshold=1.0,
        ).time_frequency_filter
        assert np.array_equal(result, first | second)

    @pytest.mark.unit
    def test_rejects_an_empty_prepared_injection_collection(self, time_frequency_transform):
        with pytest.raises(ValueError, match="at least one"):
            InjectionClustering(time_frequency_transform, [], threshold=1.0)
