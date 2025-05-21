from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

from ..clustering import single_clustering_by_threshold
from ..null_stream import (compute_calibrated_whitened_antenna_response,
                           compute_projector,
                           compute_whitened_antenna_response,
                           compute_whitened_frequency_domain_strain)
from ..time_frequency_transform import (get_shape_of_wavelet_transform,
                                        transform_wavelet_freq)

logger = logging.getLogger("nullcal")


class RecalibrationLikelihood(Likelihood):
    def __init__(
        self,
        interferometers: InterferometerList,
        waveform_generator: WaveformGenerator,
        wavelet_transform_frequency_resolution: float = 4,
        wavelet_transform_nx: float = 4,
        time_frequency_filter: np.ndarray | None = None,
        clustering_parameter_file: str | None = None,
        clustering_threshold: float = 0.1,
    ):
        """Time-frequency likelihood.

        Args:
            interferometers (InterferometerList): A list of interferometers.
            waveform_generator (WaveformGenerator): Waveform generator.
            wavelet_transform_frequency_resolution (float, optional): Frequency resolution of wavelet transform. Defaults to 4.
            wavelet_transform_nx (float, optional): The sharpness of the wavelet. Defaults to 4.
            time_frequency_filter (np.ndarray | None, optional): A time-frequency filter. Defaults to None.
            clustering_parameter_file (str | None, optional): A file to the parameters for clustering. Defaults to None.
            clustering_threshold (float, optional): The clustering threshold. Defaults to 0.1.
        """
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.waveform_generator = waveform_generator
        self.clustering_threshold = clustering_threshold
        self.clustering_parameter_file = clustering_parameter_file
        self.time_frequency_filter = time_frequency_filter
        self._wavelet_transform_frequency_resolution = wavelet_transform_frequency_resolution
        self._wavelet_transform_Nt, self._wavelet_transform_Nf = get_shape_of_wavelet_transform(
            t_length=len(self.interferometers[0].time_array),
            sampling_frequency=self.interferometers[0].sampling_frequency,
            frequency_resolution=wavelet_transform_frequency_resolution,
        )
        self._wavelet_transform_nx = wavelet_transform_nx
        self.frequency_mask = np.all([ifo.frequency_mask for ifo in self.interferometers], axis=0)
        self.masked_frequency_array = interferometers[0].frequency_array[self.frequency_mask]
        self.minimum_frequency = np.max([ifo.minimum_frequency for ifo in self.interferometers])
        self.maximum_frequency = np.min([ifo.maximum_frequency for ifo in self.interferometers])
        # Construct the noise weighed antenna pattern
        F = np.array([[-1.0 / np.sqrt(6), -1 / np.sqrt(2)], [np.sqrt(6) / 3, 0], [-1 / np.sqrt(6), 1 / np.sqrt(2)]])
        self.power_spectral_density_array = np.array(
            [ifo.power_spectral_density_array.copy() for ifo in interferometers]
        )
        self._whitened_antenna_response = compute_whitened_antenna_response(
            F, self.power_spectral_density_array, 1 / self.interferometers[0].duration, self.frequency_mask
        )
        self._whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain(
            frequency_domain_strain_array=np.array([ifo.frequency_domain_strain for ifo in interferometers]),
            power_spectral_density_array=self.power_spectral_density_array,
            delta_f=1.0 / interferometers[0].duration,
            frequency_mask=self.frequency_mask,
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

    @property
    def time_frequency_filter(self) -> np.ndarray | None:
        """Time-frequency filter.

        If time_frequency_filter is not provided, and clustering_parameter_file is provided in __init__(),
        this function reads the clustering_parameter_file to perform a zero-noise injection to obtain the cluster.

        Returns:
            np.ndarray: Time-frequency filter.
        """
        if self._time_frequency_filter is None and self.clustering_parameter_file is not None:
            logger.info("clustering_parameter_file = %s is provided.", self.clustering_parameter_file)
            clustering_parameters = pd.read_csv(self.clustering_parameter_file).iloc[0].to_dict()
            logger.info("Generating zero-noise injection data.")
            if (
                len(self.interferometers) == 3
                and self.interferometers[0].name == "ET1"
                and self.interferometers[1].name == "ET2"
                and self.interferometers[2].name == "ET3"
            ):
                interferometers = InterferometerList(["ET"])
            else:
                interferometers = InterferometerList([ifo.name for ifo in self.interferometers])
            # Copy the power spectral density
            for i in range(len(interferometers)):
                interferometers[i].power_spectral_density = self.interferometers[i].power_spectral_density
                interferometers[i].calibration_model = self.interferometers[i].calibration_model
            interferometers.set_strain_data_from_zero_noise(
                sampling_frequency=self.interferometers[0].sampling_frequency,
                duration=self.interferometers[0].duration,
                start_time=self.interferometers[0].start_time,
            )
            interferometers.inject_signal(
                waveform_generator=self.waveform_generator,
                parameters=clustering_parameters,
            )
            logger.info("Clustering started.")
            self._time_frequency_filter = single_clustering_by_threshold(
                interferometers=interferometers,
                frequency_resolution=self._wavelet_transform_frequency_resolution,
                nx=self._wavelet_transform_nx,
                threshold=self.clustering_threshold,
                padding_time=0.0,
                padding_freq=0.0,
                minimum_frequency=self.interferometers[0].minimum_frequency,
                maximum_frequency=self.interferometers[0].maximum_frequency,
            )
            logger.info("Clustering done.")
        return self._time_frequency_filter

    @time_frequency_filter.setter
    def time_frequency_filter(self, value: np.ndarray | None):
        """Set the time-frequency filter.

        Args:
            value (np.ndarray): A 2D array of time-frequency filter.
        """
        self._time_frequency_filter = value

    def compute_uncalibrated_frequency_domain_null_stream(self) -> np.ndarray:
        """Compute the uncalibrated frequency domain null stream.

        Returns:
            np.ndarray: Uncalibrated frequency domain null stream. Dimensions: (detector, frequency).
        """
        # Dimensions: (frequency, detector, detector)
        projector = compute_projector(self._whitened_antenna_response, frequency_mask=self.frequency_mask)
        # Dimensions: (frequency, detector)
        return np.einsum("ijk,ki->ji", projector, self._whitened_frequency_domain_strain_array)

    def compute_calibrated_frequency_domain_null_stream(self, calibration_factor: np.ndarray) -> np.ndarray:
        """Compute the calibrated frequency domain null stream.

        Args:
            calibration_factor (np.ndarray): Calibration factor. Dimensions: (detector, frequency).

        Returns:
            np.ndarray: Calibrated frequency domain null stream. Dimensions: (detector, frequency).
        """
        calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(
            self._whitened_antenna_response, calibration_factor, self.interferometers[0].frequency_mask
        )
        projector = compute_projector(calibrated_whitened_antenna_response, frequency_mask=self.frequency_mask)
        # Dimensions: (frequency, detector)
        return np.einsum("ijk,ki->ji", projector, self._whitened_frequency_domain_strain_array)

    def construct_calibration_factor_from_parameters(self, parameters: dict) -> np.ndarray:
        """Construct the calibration factor from parameters.

        Args:
            parameters (dict): Calibration parameters.

        Returns:
            np.ndarray: Calibration factor.
        """
        calibration_factor = np.array(
            [
                ifo.calibration_model.get_calibration_factor(
                    frequency_array=self.masked_frequency_array, prefix=f"recalib_{ifo.name}_", **parameters
                )
                for ifo in self.interferometers
            ]
        )
        output = np.zeros_like(self._whitened_frequency_domain_strain_array)
        output[:, self.frequency_mask] = calibration_factor
        return output

    def log_likelihood(self) -> float:
        """Compute the log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor = self.construct_calibration_factor_from_parameters(self.parameters)
        calibrated_frequency_domain_null_stream = self.compute_calibrated_frequency_domain_null_stream(
            calibration_factor=calibration_factor
        )
        # Transform to time-frequency domain
        calibrated_time_frequency_domain_null_stream = np.array(
            [
                transform_wavelet_freq(
                    data=data,
                    Nf=self._wavelet_transform_Nf,
                    Nt=self._wavelet_transform_Nt,
                    nx=self._wavelet_transform_nx,
                )
                for data in calibrated_frequency_domain_null_stream
            ]
        )
        # Calculate the residual energy in the time-frequency filter
        residual_energy = float(
            np.sum(np.abs(calibrated_time_frequency_domain_null_stream[:, self.time_frequency_filter]) ** 2)
        )
        # Return the log likelihood
        return -0.5 * residual_energy

    def _calculate_noise_log_likelihood(self) -> float:
        """Calculate the noise log-likelihood.

        Returns:
            float: Noise log-likelihood.
        """
        uncalibrated_frequency_domain_null_stream = self.compute_uncalibrated_frequency_domain_null_stream()
        # Transform to time-frequency domain
        uncalibrated_time_frequency_domain_null_stream = np.array(
            [
                transform_wavelet_freq(
                    data=data,
                    Nf=self._wavelet_transform_Nf,
                    Nt=self._wavelet_transform_Nt,
                    nx=self._wavelet_transform_nx,
                )
                for data in uncalibrated_frequency_domain_null_stream
            ]
        )
        # Calculate the residual energy in the time-frequency filter
        residual_energy = float(
            np.sum(np.abs(uncalibrated_time_frequency_domain_null_stream[:, self.time_frequency_filter]) ** 2)
        )
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
