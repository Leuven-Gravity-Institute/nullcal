import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList
from .utility import compute_frequency_mask
from ..null_stream import (compute_whitened_antenna_response,
                           compute_whitened_frequency_domain_strain,
                           compute_whitened_time_frequency_domain_strain,
                           compute_calibrated_whitened_antenna_response,
                           compute_projector,
                           compute_projected_strain_data,
                           compute_projected_time_frequency_strain_data)
from ..time_frequency_transform import transform_wavelet_freq

class SelfRecalibrationProjectorTimeFrequencyLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_1,
                 recalibration_generator_2,
                 recalibration_generator_3,
                 time_frequency_filter=None,
                 wavelet_transform_frequency_resolution=16,
                 wavelet_transform_nx=4):
        """Self-calibration likelihood.

        Args:
            interferometers (bilby InterferometerList): An InterferometerList instance.
            recalibration_generator_1 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
            recalibration_generator_2 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
            recalibration_generator_3 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
        """
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.recalibration_generator_1 = recalibration_generator_1
        self.recalibration_generator_2 = recalibration_generator_2
        self.recalibration_generator_3 = recalibration_generator_3
        self.time_frequency_filter = time_frequency_filter.astype(bool)
        self._wavelet_transform_frequency_resolution = wavelet_transform_frequency_resolution
        self._wavelet_transform_Nf = int(self.interferometers[0].sampling_frequency / 2 / wavelet_transform_frequency_resolution)
        self._wavelet_transform_Nt = int(len(self.interferometers[0].time_array) / self._wavelet_transform_Nf)
        self._wavelet_transform_nx = wavelet_transform_nx
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask
        self._frequency_array = self.interferometers[0].frequency_array
        self._masked_frequency_array = self.interferometers[0].frequency_array[self._frequency_mask]
        self._noise_log_likelihood = None
        # Construct the noise weighed antenna pattern        
        F = np.array([[-1./np.sqrt(6), -1/np.sqrt(2)],
                      [np.sqrt(6)/3, 0],
                      [-1/np.sqrt(6), 1/np.sqrt(2)]])
        psd_array = np.array([ifo.power_spectral_density_array for ifo in self.interferometers])
        delta_f = 1. / self.interferometers[0].duration
        self._whitened_antenna_response = compute_whitened_antenna_response(F,
                                                                            psd_array,
                                                                            delta_f,
                                                                            self._frequency_mask)
        # Constructed the noise weighed strain data
        strain_data_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        self._whitened_strain_data_array = compute_whitened_frequency_domain_strain(strain_data_array,
                                                                                    psd_array,
                                                                                    delta_f,
                                                                                    self._frequency_mask)

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor = self._construct_calibration_factor()

        # Absorb the calibration factor into the whitened antenna response
        calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(self._whitened_antenna_response, calibration_factor, self._frequency_mask)
        # Compute the loglikelihood
        logl = self._null_stream_log_likelihood(calibrated_whitened_antenna_response)
        return logl

    def _construct_calibration_factor(self):
        calibration_factor = np.array([self.recalibration_generator_1.get_calibration_factor(self._frequency_array, self.parameters),
                                       self.recalibration_generator_2.get_calibration_factor(self._frequency_array, self.parameters),
                                       self.recalibration_generator_3.get_calibration_factor(self._frequency_array, self.parameters)])
        return calibration_factor

    def _null_stream_log_likelihood(self,
                                   calibrated_whitened_antenna_response):
        """Null stream log likelihood

        Args:
            calibrated_whitened_antenna_response (numpy array): Calibrated whitened antenna response matrix.

        Returns:
            float: Log likelihood.
        """
        projector = compute_projector(calibrated_whitened_antenna_response, self._frequency_mask)
        # Compute the projected strain
        projected_strain_data = compute_projected_strain_data(projector, self._whitened_strain_data_array, self._frequency_mask)
        # Transform the projected strain data to time frequency domain
        projected_time_frequency_strain_data = np.array([transform_wavelet_freq(data, self._wavelet_transform_Nf, self._wavelet_transform_Nt, self._wavelet_transform_nx) for data in projected_strain_data])
        # Apply the time-frequency filter
        filtered_projected_time_frequency_strain_data = projected_time_frequency_strain_data[:,self.time_frequency_filter]
        logl = -np.sum(np.abs(filtered_projected_time_frequency_strain_data) ** 2) * 0.5
        return logl
        
    def _calculate_noise_log_likelihood(self):
        """Log likelihood of the noise model.

        Returns:
            float: Log likelihood.
        """
        logl = self._null_stream_log_likelihood(self._whitened_antenna_response)
        return logl

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood

class TimeFrequencyLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_1,
                 recalibration_generator_2,
                 recalibration_generator_3,
                 psd_1=None,
                 psd_2=None,
                 psd_3=None,
                 time_frequency_filter=None,
                 wavelet_transform_frequency_resolution=4,
                 wavelet_transform_nx=4):
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.recalibration_generator_1 = recalibration_generator_1
        self.recalibration_generator_2 = recalibration_generator_2
        self.recalibration_generator_3 = recalibration_generator_3
        self.time_frequency_filter = time_frequency_filter.astype(bool)
        self._wavelet_transform_frequency_resolution = wavelet_transform_frequency_resolution
        self._wavelet_transform_Nf = int(self.interferometers[0].sampling_frequency / 2 / wavelet_transform_frequency_resolution)
        self._wavelet_transform_Nt = int(len(self.interferometers[0].time_array) / self._wavelet_transform_Nf)
        self._wavelet_transform_nx = wavelet_transform_nx
        self._frequency_mask = compute_frequency_mask(time_frequency_filter)
        self._frequency_array = np.arange(self._wavelet_transform_Nf) * self._wavelet_transform_frequency_resolution
        # Construct the noise weighed antenna pattern        
        F = np.array([[-1./np.sqrt(6), -1/np.sqrt(2)],
                      [np.sqrt(6)/3, 0],
                      [-1/np.sqrt(6), 1/np.sqrt(2)]])
        psd_array = np.array([psd_1, psd_2, psd_3])
        self._whitened_antenna_response = compute_whitened_antenna_response(F,
                                                                            psd_array,
                                                                            self._wavelet_transform_frequency_resolution,
                                                                            self._frequency_mask)
        # Constructed the noise weighed strain data        
        strain_data_array = np.array([transform_wavelet_freq(ifo.frequency_domain_strain,
                                                             self._wavelet_transform_Nf,
                                                             self._wavelet_transform_Nt,
                                                             self._wavelet_transform_nx) for ifo in self.interferometers])
        self._whitened_strain_data_array = compute_whitened_time_frequency_domain_strain(strain_data_array,
                                                                                         psd_array,
                                                                                         self._wavelet_transform_frequency_resolution,
                                                                                         self.time_frequency_filter)
        
    def log_likelihood(self):
        calibration_factor = self._construct_calibration_factor()

        # Absorb the calibration factor into the whitened antenna response
        calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(self._whitened_antenna_response,
                                                                                            calibration_factor,
                                                                                            self._frequency_mask)
        # Compute the loglikelihood
        logl = self._null_stream_log_likelihood(calibrated_whitened_antenna_response)
        return logl
    
    def _construct_calibration_factor(self):
        calibration_factor = np.array([self.recalibration_generator_1.get_calibration_factor(self._frequency_array, self.parameters),
                                       self.recalibration_generator_2.get_calibration_factor(self._frequency_array, self.parameters),
                                       self.recalibration_generator_3.get_calibration_factor(self._frequency_array, self.parameters)])
        return calibration_factor

    def _null_stream_log_likelihood(self,
                                    calibrated_whitened_antenna_response):
        """Null stream log likelihood

        Args:
            calibrated_whitened_antenna_response (numpy array): Calibrated whitened antenna response matrix.

        Returns:
            float: Log likelihood.
        """
        projector = compute_projector(calibrated_whitened_antenna_response, self._frequency_mask)
        # Compute the projected strain
        projected_time_frequency_strain_data = compute_projected_time_frequency_strain_data(projector, self._whitened_strain_data_array, self.time_frequency_filter)
        logl = -np.sum(np.abs(projected_time_frequency_strain_data) ** 2)
        return logl
    
    def _calculate_noise_log_likelihood(self):
        logl = self._null_stream_log_likelihood(self._whitened_antenna_response)
        return logl

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood