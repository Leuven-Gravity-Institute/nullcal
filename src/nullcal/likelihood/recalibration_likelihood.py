"""Time-frequency recalibration likelihood class."""

from __future__ import annotations

import logging

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

from ..clustering.injection import InjectionClustering
from ..clustering.precompute import PrecomputedClustering
from ..null_stream.null_stream import NullStream
from ..time_frequency_transform.wavelet_transforms import WaveletTransform

logger = logging.getLogger("nullcal")


class RecalibrationLikelihood(Likelihood):
    """Time-frequency recalibration likelihood class."""

    def __init__(
        self,
        interferometers: InterferometerList,
        waveform_generator: WaveformGenerator,
        wavelet_transform_frequency_resolution: float = 4,
        wavelet_transform_nx: float = 4,
        time_frequency_filter: np.ndarray | None = None,
        clustering_parameter_file: str | None = None,
        clustering_threshold: float = 0.1,
        enforce_signal_duration: bool = False,
    ):
        """Time-frequency likelihood.

        Args:
            interferometers (InterferometerList): A list of interferometers.
            waveform_generator (WaveformGenerator): Waveform generator.
            wavelet_transform_frequency_resolution (float, optional): Frequency resolution of wavelet transform.
                Defaults to 4.
            wavelet_transform_nx (float, optional): The sharpness of the wavelet.
                Defaults to 4.
            time_frequency_filter (np.ndarray | None, optional): A time-frequency filter.
                Defaults to None.
            clustering_parameter_file (str | None, optional): A file to the parameters for clustering.
                Defaults to None.
            clustering_threshold (float, optional): The clustering threshold.
                Defaults to 0.1.
            enforce_signal_duration (bool, optional): Enforce signal duration to be smaller than that of data.
                Defaults to False.
        """
        super().__init__({})
        self.interferometers = InterferometerList(interferometers)

        duration = self.interferometers[0].duration
        sampling_frequency = self.interferometers[0].sampling_frequency

        # Construct the wavelet transform instance
        # for time-frequency transform.
        self.time_frequency_transform = WaveletTransform(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_resolution=wavelet_transform_frequency_resolution,
            nx=wavelet_transform_nx,
        )

        # Construct the time-frequency filter.
        if time_frequency_filter is not None:
            self.clustering = PrecomputedClustering(
                time_frequency_transform=self.time_frequency_transform, time_frequency_filter=time_frequency_filter
            )
            logger.info("Loaded a pre-computed time-frequency filter.")
        elif clustering_parameter_file is not None:
            self.clustering = InjectionClustering(
                time_frequency_transform=self.time_frequency_transform,
                interferometers=interferometers,
                waveform_generator=waveform_generator,
                parameter_file=clustering_parameter_file,
                threshold=clustering_threshold,
                enforce_signal_duration=enforce_signal_duration,
            )
            logger.info("Loaded a parameter file: %s", clustering_parameter_file)
            logger.info("Injection clustering will be performed.")
        else:
            raise ValueError("Both time_frequency_filter and clustering_parameter_file are not provided.")
        # Construct a null stream calculator.
        self.null_stream_calculator = NullStream(
            interferometers=interferometers,
            time_frequency_transform=self.time_frequency_transform,
            time_frequency_filter=self.clustering.time_frequency_filter,
        )

        self._noise_log_likelihood = None

    @property
    def interferometers(self) -> InterferometerList:
        """A list of interferometers.

        Returns:
            InterferometerList: A list of interferometers.
        """

        return self._interferometers

    @interferometers.setter
    def interferometers(self, value: InterferometerList):
        """Set the interferometers.

        Args:
            value (InterferometerList): A list of interferometers.

        Raises:
            ValueError: The interferometers do not have the same sampling frequency.
            ValueError: The interferometers do not have the same duration.
            ValueError: The interferometers do not have the same start time.
        """
        # Check whether the sampling frequencies are the same.
        sampling_frequency_array = [ifo.sampling_frequency for ifo in value]

        if not np.allclose(sampling_frequency_array, sampling_frequency_array[0]):
            raise ValueError(
                f"The interferometers do not have the same sampling frequency: {sampling_frequency_array}."
            )
        # Check whether the durations are the same.
        duration_array = [ifo.duration for ifo in value]

        if not np.allclose(duration_array, duration_array[0]):
            raise ValueError(f"The interferometers do not have the same duration: {duration_array}.")
        # Check whether the start times are the same.
        start_time_array = [ifo.start_time for ifo in value]

        if not np.allclose(start_time_array, start_time_array[0]):
            raise ValueError(f"The interferometers do not have the same start time: {start_time_array}.")
        self._interferometers = value

    def log_likelihood(self) -> float:
        """Compute the log likelihood.

        Returns:
            float: Log likelihood.
        """
        if self.parameters is None:
            raise ValueError("self.parameters is None.")

        calibrated_time_frequency_domain_null_stream = (
            self.null_stream_calculator.compute_calibrated_time_frequency_domain_null_stream_from_parameters(
                parameters=self.parameters
            )
        )

        # Calculate the residual energy in the time-frequency filter
        residual_energy = float(np.sum(np.abs(calibrated_time_frequency_domain_null_stream) ** 2))
        # Return the log likelihood

        return -0.5 * residual_energy

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
