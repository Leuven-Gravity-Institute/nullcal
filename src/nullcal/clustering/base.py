"""A submodule for clustering."""

import numpy as np

from ..time_frequency_transform.wavelet_transforms import WaveletTransform


class Clustering:
    """A base class to handle time-frequency clustering."""

    def __init__(self, time_frequency_transform: WaveletTransform):
        """A class to handle time-frequency clustering.

        Args:
            time_frequency_transform (WaveletTransform): Wavelet transform instance.
        """
        self.time_frequency_transform = time_frequency_transform
        self._time_frequency_filter = None

    @property
    def shape(self) -> tuple:
        """Get the shape of the time-frequency transform.

        Returns:
            tuple: Shape of time-frequency transform.
                (n_time, n_freq).
        """
        return self.time_frequency_transform.shape

    @property
    def time_frequency_filter(self) -> np.ndarray:
        """Get the time-frequency filter.

        Returns:
            np.ndarray: Time-frequency filter.
        """
        if self._time_frequency_filter is None:
            raise ValueError("self._time_frequency_filter is None.")
        return self._time_frequency_filter
