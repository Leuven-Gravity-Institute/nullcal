"""Time-frequency recalibration likelihood class."""

from __future__ import annotations

import logging

import numpy as np
from bilby.core.likelihood import Likelihood

from ..clustering.precompute import PrecomputedClustering
from ..data import InterferometerData
from ..null_stream.null_stream import NullStream
from ..time_frequency_transform.wavelet_transforms import WaveletTransform

logger = logging.getLogger("nullcal")


def log_likelihood(params: dict, static_data: NullStream) -> float:
    """Compute the log likelihood from parameters and precomputed static data.

    Args:
        params (dict): Calibration parameters.
        static_data (NullStream): Precomputed null-stream data and transforms.

    Returns:
        float: Log likelihood.
    """
    calibrated_time_frequency_domain_null_stream = (
        static_data.compute_calibrated_time_frequency_domain_null_stream_from_parameters(parameters=params)
    )
    residual_energy = float(np.sum(np.abs(calibrated_time_frequency_domain_null_stream) ** 2))
    return -0.5 * residual_energy


class RecalibrationLikelihood(Likelihood):
    """Time-frequency recalibration likelihood class."""

    def __init__(
        self,
        interferometers: InterferometerData,
        wavelet_transform_frequency_resolution: float = 4,
        wavelet_transform_nx: float = 4,
        time_frequency_filter: np.ndarray | None = None,
    ):
        """Time-frequency likelihood.

        Args:
            interferometers (InterferometerData): Frozen detector arrays and metadata.
            wavelet_transform_frequency_resolution (float, optional): Frequency resolution of wavelet transform.
                Defaults to 4.
            wavelet_transform_nx (float, optional): The sharpness of the wavelet.
                Defaults to 4.
            time_frequency_filter (np.ndarray | None, optional): A time-frequency filter.
                Defaults to None.
        """
        super().__init__({})
        if not isinstance(interferometers, InterferometerData):
            raise TypeError("interferometers must be an InterferometerData instance")
        self.interferometers = interferometers

        duration = self.interferometers.duration
        sampling_frequency = self.interferometers.sampling_frequency

        # Construct the wavelet transform instance
        # for time-frequency transform.
        self.time_frequency_transform = WaveletTransform(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_resolution=wavelet_transform_frequency_resolution,
            nx=wavelet_transform_nx,
        )

        # Construct the time-frequency filter.
        if time_frequency_filter is None:
            raise ValueError("time_frequency_filter must be precomputed before likelihood construction")
        self.clustering = PrecomputedClustering(
            time_frequency_transform=self.time_frequency_transform, time_frequency_filter=time_frequency_filter
        )
        logger.info("Loaded a pre-computed time-frequency filter.")
        # Construct a null stream calculator.
        self.null_stream_calculator = NullStream(
            interferometers=interferometers,
            time_frequency_transform=self.time_frequency_transform,
            time_frequency_filter=self.clustering.time_frequency_filter,
        )

        self._noise_log_likelihood = None

    @property
    def interferometers(self) -> InterferometerData:
        """Frozen detector arrays and metadata.

        Returns:
            InterferometerData: Frozen detector arrays and metadata.
        """

        return self._interferometers

    @interferometers.setter
    def interferometers(self, value: InterferometerData):
        """Set the frozen detector data.

        Args:
            value (InterferometerData): Frozen detector arrays and metadata.
        """
        self._interferometers = value

    def log_likelihood(self) -> float:
        """Compute the log likelihood.

        Returns:
            float: Log likelihood.
        """
        if self.parameters is None:
            raise ValueError("self.parameters is None.")

        return log_likelihood(params=self.parameters, static_data=self.null_stream_calculator)

    def _calculate_noise_log_likelihood(self) -> float:
        """Calculate the noise log-likelihood.

        Returns:
            float: Noise log-likelihood.
        """
        uncalibrated_time_frequency_domain_null_stream = (
            self.null_stream_calculator.compute_uncalibrated_time_frequency_domain_null_stream()
        )
        # Calculate the residual energy in the time-frequency filter
        residual_energy = float(np.sum(np.abs(uncalibrated_time_frequency_domain_null_stream) ** 2))
        # Return the log likelihood

        return -0.5 * residual_energy

    def noise_log_likelihood(self) -> float:
        """Get the noise log-likelihood.

        Returns:
            float: Noise log-likelihood.
        """

        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()

        return self._noise_log_likelihood
