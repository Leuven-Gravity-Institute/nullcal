from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList

from ..clustering import single_clustering_by_threshold
from ..null_stream import (compute_calibrated_whitened_antenna_response,
                           compute_projector_numpy,
                           compute_whitened_antenna_response,
                           compute_whitened_frequency_domain_strain)
from ..time_frequency_transform import transform_wavelet_freq

logger = logging.getLogger('nullcal')


class TimeFrequencyLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 waveform_generator,
                 wavelet_transform_frequency_resolution=4,
                 wavelet_transform_nx=4,
                 time_frequency_filter=None,
                 clustering_parameter_file=None,
                 clustering_threshold=0.1):
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.waveform_generator = waveform_generator
        self.clustering_threshold = clustering_threshold
        self.clustering_parameter_file = clustering_parameter_file
        self.time_frequency_filter = time_frequency_filter
        self._wavelet_transform_frequency_resolution = wavelet_transform_frequency_resolution
        self._wavelet_transform_Nf = int(self.interferometers[0].sampling_frequency / 2 / wavelet_transform_frequency_resolution)
        self._wavelet_transform_Nt = int(len(self.interferometers[0].time_array) / self._wavelet_transform_Nf)
        self._wavelet_transform_nx = wavelet_transform_nx
        self.masked_frequency_array = interferometers[0].frequency_array[interferometers[0].frequency_mask]
        # Construct the noise weighed antenna pattern
        F = np.array([[-1./np.sqrt(6), -1/np.sqrt(2)],
                      [np.sqrt(6)/3, 0],
                      [-1/np.sqrt(6), 1/np.sqrt(2)]])
        self.power_spectral_density_array = np.array([ifo.power_spectral_density_array.copy() for ifo in interferometers])
        self._whitened_antenna_response = compute_whitened_antenna_response(F,
                                                                            self.power_spectral_density_array,
                                                                            self._wavelet_transform_frequency_resolution,
                                                                            1/self.interferometers[0].duration)
        self._whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain(
                frequency_domain_strain_array=np.array([ifo.frequency_domain_strain for ifo in interferometers]),
                power_spectral_density_array=self.power_spectral_density_array,
                delta_f=1. / interferometers[0].duration,
                frequency_mask=interferometers[0].frequency_mask)

    @property
    def time_frequency_filter(self):
        return self._time_frequency_filter

    @time_frequency_filter.setter
    def time_frequency_filter(self, value):
        self._time_frequency_filter = value
        if self._time_frequency_filter is None and self.clustering_parameter_file is not None:
            logger.info('clustering_parameter_file = %s is provided.', self.clustering_parameter_file)
            clustering_parameters = pd.read_csv(self.clustering_parameter_file).iloc[0].to_dict()
            logger.info("Generating zero-noise injection data")
            ifos = bilby.gw.detector.InterferometerList([ifo.name for ifo in self.interferometers])
            # Copy the power spectral density
            for i in range(len(ifos)):
                ifos[i].power_spectral_density = self.interferometers[i].power_spectral_density
            ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.interferometers[0].sampling_frequency,
                duration=self.interferometers[0].duration,
                start_time=self.interferometers[0].start_time
            )
            ifos.inject_signal(
                waveform_generator=self.waveform_generator,
                parameters=clustering_parameters,
            )
            logger.info('Start perform clustering.')
            self._time_frequency_filter = single_clustering_by_threshold(interferometers=ifos,
                                                                         frequency_resolution=self._wavelet_transform_frequency_resolution,
                                                                         nx=self._wavelet_transform_nx,
                                                                         threshold=self.clustering_threshold,
                                                                         padding_time=0.0,
                                                                         padding_freq=0.0,
                                                                         minimum_frequency=self.interferometers[0].minimum_frequency,
                                                                         maximum_frequency=self.interferometers[0].maximum_frequency)

    def compute_uncalibrated_frequency_domain_null_stream(self):
        # Dimensions: (frequency, detector, detector)
        projector = compute_projector_numpy(self._whitened_antenna_response,
                                            frequency_mask=self.interferometers[0].frequency_mask)
        # Dimensions: (frequency, detector)
        return np.einsum('ijk,ki->ij', projector, self._whitened_frequency_domain_strain_array)

    def compute_calibrated_frequency_domain_null_stream(self,
                                                        calibration_factor):
        calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(self._whitened_antenna_response,
                                                                                            calibration_factor,
                                                                                            self.interferometers[0].frequency_mask)
        projector = compute_projector_numpy(calibrated_whitened_antenna_response,
                                            frequency_mask=self.interferometers[0].frequency_mask)
        # Dimensions: (frequency, detector)
        return np.einsum('ijk,ki->ij', projector, self._whitened_frequency_domain_strain_array)

    def construct_calibration_factor(self):
        calibration_factor =  np.array([ifo.calibration_model.get_calibration_factor(frequency_array=self.masked_frequency_array,
                                                                                     prefix=f'recalib_{ifo.name}_',
                                                                                     **self.parameters) for ifo in self.interferometers])
        output = np.zeros_like(self._whitened_frequency_domain_strain_array)
        output[:, self.interferometers[0].frequency_mask] = calibration_factor
        return output

    def log_likelihood(self) -> float:
        calibration_factor = self.construct_calibration_factor()
        calibrated_frequency_domain_null_stream = self.compute_calibrated_frequency_domain_null_stream(
                calibration_factor=calibration_factor
        )
        # Transform to time-frequency domain
        calibrated_time_frequency_domain_null_stream = np.array([
                transform_wavelet_freq(data=data,
                                       Nf=self._wavelet_transform_Nf,
                                       Nt=self._wavelet_transform_Nt,
                                       nx=self._wavelet_transform_nx)
                for data in calibrated_frequency_domain_null_stream
        ])
        # Calculate the residual energy in the time-frequency filter
        residual_energy = np.sum(np.abs(calibrated_time_frequency_domain_null_stream[:, self.time_frequency_filter])**2)
        # Return the log likelihood
        return -0.5 * residual_energy

    def _calculate_noise_log_likelihood(self) -> float:
        uncalibrated_frequency_domain_null_stream = self.compute_uncalibrated_frequency_domain_null_stream()
        # Transform to time-frequency domain
        calibrated_time_frequency_domain_null_stream = np.array([
                transform_wavelet_freq(data=data,
                                       Nf=self._wavelet_transform_Nf,
                                       Nt=self._wavelet_transform_Nt,
                                       nx=self._wavelet_transform_nx)
                for data in uncalibrated_frequency_domain_null_stream
        ])
        # Calculate the residual energy in the time-frequency filter
        residual_energy = np.sum(np.abs(calibrated_time_frequency_domain_null_stream[:, self.time_frequency_filter])**2)
        # Return the log likelihood
        return -0.5 * residual_energy

    def noise_log_likelihood(self) -> float:
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood
