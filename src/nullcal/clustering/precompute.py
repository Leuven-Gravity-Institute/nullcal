"""
A class to handle pre-computed clustering.
"""

import numpy as np

from ..time_frequency_transform.wavelet_transforms import WaveletTransform
from .base import Clustering


class PrecomputedClustering(Clustering):
    """Pre-computed clustering."""

    def __init__(self, time_frequency_transform: WaveletTransform, time_frequency_filter: np.ndarray):
        """Pre-computed clustering.

        Args:
            time_frequency_transform (WaveletTransform): A WaveletTransform instance
                for performing wavelet transforms.
            time_frequency_filter (np.ndarray): A pre-computed time-frequency filter.
        """
        super().__init__(time_frequency_transform=time_frequency_transform)
        # Check the shape of the input time-frequency filter.
        if time_frequency_filter.shape != self.shape:
            raise ValueError(
                f"The shape of time_frequency_filter: {time_frequency_filter.shape}"
                f"does not match the expected shape: {self.shape}."
            )

        self._time_frequency_filter = time_frequency_filter
