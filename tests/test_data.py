"""Tests for the dependency-free detector-data boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from nullcal.data import InterferometerData


def make_interferometers():
    frequencies = np.linspace(0.0, 4.0, 5)
    return [
        SimpleNamespace(
            power_spectral_density_array=np.full(5, index + 1.0),
            frequency_domain_strain=np.full(5, index + 2.0j),
            frequency_mask=np.array([False, True, True, True, False]),
            frequency_array=frequencies,
            duration=4.0,
            sampling_frequency=8.0,
            start_time=100.0,
            name=f"ET{index + 1}",
        )
        for index in range(3)
    ]


def test_loader_stacks_only_the_detector_data_needed_by_the_likelihood():
    data = InterferometerData.from_interferometers(make_interferometers())

    assert data.psd.shape == (3, 5)
    assert data.strain.shape == (3, 5)
    assert data.mask.shape == (3, 5)
    assert data.frequency_array.shape == (5,)
    assert data.duration == 4.0
    assert data.sampling_frequency == 8.0
    assert data.start_time == 100.0
    assert data.name == ("ET1", "ET2", "ET3")


@pytest.mark.parametrize(
    ("attribute", "replacement", "match"),
    [
        ("sampling_frequency", 16.0, "sampling frequency"),
        ("duration", 8.0, "duration"),
        ("start_time", 101.0, "start time"),
    ],
)
def test_loader_rejects_inconsistent_detector_metadata(attribute, replacement, match):
    interferometers = make_interferometers()
    setattr(interferometers[-1], attribute, replacement)

    with pytest.raises(ValueError, match=match):
        InterferometerData.from_interferometers(interferometers)


def test_loader_rejects_an_empty_detector_collection():
    with pytest.raises(ValueError, match="at least one detector"):
        InterferometerData.from_interferometers([])


def test_container_is_frozen():
    data = InterferometerData.from_interferometers(make_interferometers())

    with pytest.raises(FrozenInstanceError):
        data.duration = 8.0


def test_container_is_a_jax_pytree_with_static_detector_names():
    data = InterferometerData.from_interferometers(make_interferometers())

    leaves, structure = jax.tree.flatten(data)
    rebuilt = jax.tree.unflatten(structure, leaves)

    assert rebuilt.name == data.name
    assert len(leaves) == 7
    np.testing.assert_array_equal(rebuilt.psd, data.psd)
    np.testing.assert_array_equal(rebuilt.strain, data.strain)
    np.testing.assert_array_equal(rebuilt.mask, data.mask)
    np.testing.assert_array_equal(rebuilt.frequency_array, data.frequency_array)
