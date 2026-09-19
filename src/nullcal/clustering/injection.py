"""Clustering over injected strain prepared outside the likelihood."""

import logging
from collections.abc import Iterable

import numpy as np

from ..data import InterferometerData
from ..time_frequency_transform.wavelet_transforms import WaveletTransform
from .base import Clustering
from .single import single_clustering_by_threshold

logger = logging.getLogger("nullcal")


class InjectionClustering(Clustering):
    """Clustering method using waveform injections."""

    def __init__(
        self,
        time_frequency_transform: WaveletTransform,
        injections: Iterable[InterferometerData],
        threshold: float,
        minimum_frequency: float | None = None,
        maximum_frequency: float | None = None,
    ):
        """Clustering method using waveform injections.

        Args:
            time_frequency_transform (WaveletTransform): A WaveletTransform instance.
            injections (Iterable[InterferometerData]): Zero-noise injected
                strain prepared by a loader or waveform package.
            threshold (float): The threshold to select time-frequency pixels.
            minimum_frequency (float, optional): Lowest clustering frequency.
            maximum_frequency (float, optional): Highest clustering frequency.
        """
        super().__init__(time_frequency_transform=time_frequency_transform)
        injections = tuple(injections)
        if not injections:
            raise ValueError("injections must contain at least one prepared data set")
        filters = [
            single_clustering_by_threshold(
                interferometers=injection,
                time_frequency_transform=self.time_frequency_transform,
                threshold=threshold,
                padding_time=0.0,
                padding_freq=0.0,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
            for injection in injections
        ]
        self._time_frequency_filter = np.logical_or.reduce(filters)
        logger.info("Injection clustering preprocessing done.")

    @property
    def time_frequency_filter(self) -> np.ndarray:
        """Get the time-frequency filter.

        Returns:
            np.ndarray: Time-frequency filter.
        """
        return self._time_frequency_filter
