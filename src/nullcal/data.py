"""Immutable detector data consumed by null-stream calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

DETECTOR_ARRAY_NDIM = 2


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InterferometerData:
    """Dependency-free arrays and shared metadata for a detector network.

    Detector names are static pytree metadata. All numerical values are leaves,
    so JAX transformations can move or batch the data without importing the
    package that originally loaded the strain.
    """

    psd: Any
    strain: Any
    mask: Any
    frequency_array: Any
    duration: Any
    sampling_frequency: Any
    start_time: Any
    name: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(array.ndim != DETECTOR_ARRAY_NDIM for array in (self.psd, self.strain, self.mask)):
            raise ValueError("psd, strain, and mask must have (detector, frequency) dimensions")
        if self.psd.shape != self.strain.shape or self.psd.shape != self.mask.shape:
            raise ValueError("psd, strain, and mask must have identical shapes")
        if self.frequency_array.ndim != 1 or self.frequency_array.shape[0] != self.psd.shape[1]:
            raise ValueError("frequency_array must match the frequency dimension")
        if len(self.name) != self.psd.shape[0]:
            raise ValueError("name must contain one label per detector")

    def __len__(self) -> int:
        """Return the number of detectors."""
        return len(self.name)

    @classmethod
    def from_interferometers(cls, interferometers: Sequence[Any]) -> InterferometerData:
        """Load arrays from detector objects without retaining their package type."""
        interferometers = tuple(interferometers)
        if not interferometers:
            raise ValueError("interferometers must contain at least one detector")

        def shared_scalar(attribute: str, label: str) -> float:
            values = np.asarray([getattr(interferometer, attribute) for interferometer in interferometers])
            if not np.allclose(values, values[0]):
                raise ValueError(f"The interferometers do not have the same {label}: {values.tolist()}.")
            return float(values[0])

        frequency_arrays = np.asarray([interferometer.frequency_array for interferometer in interferometers])
        if not np.allclose(frequency_arrays, frequency_arrays[0], rtol=0.0, atol=0.0):
            raise ValueError("The interferometers do not have the same frequency array.")

        return cls(
            psd=np.asarray([interferometer.power_spectral_density_array for interferometer in interferometers]),
            strain=np.asarray([interferometer.frequency_domain_strain for interferometer in interferometers]),
            mask=np.asarray([interferometer.frequency_mask for interferometer in interferometers], dtype=bool),
            frequency_array=np.asarray(frequency_arrays[0]),
            duration=shared_scalar("duration", "duration"),
            sampling_frequency=shared_scalar("sampling_frequency", "sampling frequency"),
            start_time=shared_scalar("start_time", "start time"),
            name=tuple(str(interferometer.name) for interferometer in interferometers),
        )

    def tree_flatten(self):
        children = (
            self.psd,
            self.strain,
            self.mask,
            self.frequency_array,
            self.duration,
            self.sampling_frequency,
            self.start_time,
        )
        return children, self.name

    @classmethod
    def tree_unflatten(cls, name, children):
        return cls(*children, name=name)


__all__ = ["InterferometerData"]
