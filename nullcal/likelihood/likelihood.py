import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList



class BaseLikelihood(Likelihood)
class SelfCalibrationLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 response_function_generator):
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.response_function_generator = response_function_generator
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask

    def log_likelihood(self):
        response_function_1, response_function_2, response_function_3 = \
            self.response_function_generator.frequency_domain_response_function(self.parameters)
        calibrated_frequency_domain_strain_E1 = self.calculate_calibrated_frequency_domain_strain(self.interferometers[0], response_function_1)
        calibrated_frequency_domain_strain_E2 = self.calculate_calibrated_frequency_domain_strain(self.interferometers[1], response_function_2)
        calibrated_frequency_domain_strain_E3 = self.calculate_calibrated_frequency_domain_strain(self.interferometers[2], response_function_3)
        # FIXME: Here we assume the PSD of the error signal can be measured.
        calibrated_power_spectral_density_array_E1 = self.calculate_calibrated_power_spectral_density_array(self.interferometers[0], response_function_1)
        calibrated_power_spectral_density_array_E2 = self.calculate_calibrated_power_spectral_density_array(self.interferometers[0], response_function_2)
        calibrated_power_spectral_density_array_E3 = self.calculate_calibrated_power_spectral_density_array(self.interferometers[0], response_function_3)

        # Compute the null stream
        null_stream = (calibrated_frequency_domain_strain_E1[self._frequency_mask] + \
                       calibrated_frequency_domain_strain_E2[self._frequency_mask] + \
                       calibrated_frequency_domain_strain_E3[self._frequency_mask]) / np.sqrt(3)
        # Compute the null stream PSD
        null_stream_PSD = (calibrated_power_spectral_density_array_E1[self._frequency_mask] + \
                           calibrated_power_spectral_density_array_E2[self._frequency_mask] + \
                           calibrated_power_spectral_density_array_E3[self._frequency_mask]) / 3
        
        normalization_constant = 4 / self.interferometers[0].duration
        logl = -0.5 * normalization_constant * np.sum(np.abs(null_stream)**2/null_stream_PSD)
        logl -= 0.5 * np.sum(np.log(null_stream_PSD))
        return logl

    def calculate_calibrated_frequency_domain_strain(self, interferometer, response_function):
        return interferometer.frequency_domain_strain * response_function
    
    def calculate_calibrated_power_spectral_density_array(self, interferometer, response_function):
        return interferometer.power_spectral_density_array * np.abs(response_function) ** 2


class SelfRecalibrationLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_1,
                 recalibration_generator_2,
                 recalibration_generator_3):
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
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask
        self._masked_frequency_array = self.interferometers[0].frequency_array[self._frequency_mask]
        self._noise_log_likelihood = None

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor_1 = self.recalibration_generator_1.get_calibration_factor(self._masked_frequency_array, self.parameters)
        calibration_factor_2 = self.recalibration_generator_2.get_calibration_factor(self._masked_frequency_array, self.parameters)
        calibration_factor_3 = self.recalibration_generator_3.get_calibration_factor(self._masked_frequency_array, self.parameters)

        # Recalibrate the strain
        recalibrated_frequency_domain_strain_1 = self.recalibrate_strain(calibration_factor_1,
                                                                         self.interferometers[0].frequency_domain_strain[self._frequency_mask])
        recalibrated_frequency_domain_strain_2 = self.recalibrate_strain(calibration_factor_2,
                                                                         self.interferometers[1].frequency_domain_strain[self._frequency_mask])
        recalibrated_frequency_domain_strain_3 = self.recalibrate_strain(calibration_factor_3,
                                                                         self.interferometers[2].frequency_domain_strain[self._frequency_mask])
        # Recalibrate the power spectral density
        recalibrated_power_spectral_density_1 = self.recalibrate_power_spectral_density(calibration_factor_1,
                                                                                        self.interferometers[0].power_spectral_density_array[self._frequency_mask])
        recalibrated_power_spectral_density_2 = self.recalibrate_power_spectral_density(calibration_factor_2,
                                                                                        self.interferometers[1].power_spectral_density_array[self._frequency_mask])
        recalibrated_power_spectral_density_3 = self.recalibrate_power_spectral_density(calibration_factor_3,
                                                                                        self.interferometers[2].power_spectral_density_array[self._frequency_mask])        
        logl = self._null_stream_log_likelihood(recalibrated_frequency_domain_strain_1,
                                                recalibrated_frequency_domain_strain_2,
                                                recalibrated_frequency_domain_strain_3,
                                                recalibrated_power_spectral_density_1,
                                                recalibrated_power_spectral_density_2,
                                                recalibrated_power_spectral_density_3,
                                                self.interferometers[0].duration)
        return logl

    def recalibrate_strain(self, calibration_factor, frequency_domain_strain):
        """Recalibrate strain.

        Args:
            calibration_factor (numpy array): Calibration factor.
            frequency_domain_strain (numpy array): Frequency domain strain.

        Returns:
            numpy array: Recalibrated frequency domain strain.
        """
        return frequency_domain_strain / calibration_factor

    def recalibrate_power_spectral_density(self, calibration_factor, power_spectral_density_array):
        """Recalibrate power spectral density.

        Args:
            calibration_factor (numpy array): Calibration factor.
            power_spectral_density_array (numpy array): Power spectral density.

        Returns:
            numpy array: Recalibrated power spectral density.
        """
        return power_spectral_density_array / np.abs(calibration_factor) ** 2

    def _null_stream_log_likelihood(self,
                                    frequency_domain_strain_1,
                                    frequency_domain_strain_2,
                                    frequency_domain_strain_3,
                                    power_spectral_density_array_1,
                                    power_spectral_density_array_2,
                                    power_spectral_density_array_3,
                                    duration):
        """Null stream log likelihood

        Args:
            frequency_domain_strain_1 (numpy array): Frequency domain strain of detector 1.
            frequency_domain_strain_2 (numpy array): Frequency domain strain of detector 2.
            frequency_domain_strain_3 (numpy array): Frequency domain strain of detector 3.
            power_spectral_density_array_1 (numpy array): Power spectral density of detector 1.
            power_spectral_density_array_2 (numpy array): Power spectral density of detector 2.
            power_spectral_density_array_3 (numpy array): Power spectral density of detector 3.
            duration (float): Duration of data segment in second.

        Returns:
            float: Log likelihood.
        """
        null_stream = (frequency_domain_strain_1 + \
                       frequency_domain_strain_2 + \
                       frequency_domain_strain_3) / np.sqrt(3)
        null_stream_PSD = (power_spectral_density_array_1 + \
                           power_spectral_density_array_2 + \
                           power_spectral_density_array_3) / 3
        normalization_constant = 4 / duration
        logresidual = -0.5 * normalization_constant * np.sum(np.abs(null_stream)**2/null_stream_PSD)
        lognorm = -0.5 * np.sum(np.log(null_stream_PSD))
        logl = logresidual + lognorm
        #logl = logresidual
        return logl
        
    def _calculate_noise_log_likelihood(self):
        """Log likelihood of the noise model.

        Returns:
            float: Log likelihood.
        """
        uncalibrated_frequency_domain_strain_1 = self.interferometers[0].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_2 = self.interferometers[1].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_3 = self.interferometers[2].frequency_domain_strain[self._frequency_mask]
        uncalibrated_PSD_1 = self.interferometers[0].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_2 = self.interferometers[1].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_3 = self.interferometers[2].power_spectral_density_array[self._frequency_mask]
        # Compute the log likelihood.
        logl = self._null_stream_log_likelihood(uncalibrated_frequency_domain_strain_1,
                                                uncalibrated_frequency_domain_strain_2,
                                                uncalibrated_frequency_domain_strain_3,
                                                uncalibrated_PSD_1,
                                                uncalibrated_PSD_2,
                                                uncalibrated_PSD_3,
                                                self.interferometers[0].duration)
        return logl

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood


class SelfRecalibrationProjectorLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_1,
                 recalibration_generator_2,
                 recalibration_generator_3):
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
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask
        self._masked_frequency_array = self.interferometers[0].frequency_array[self._frequency_mask]
        self._noise_log_likelihood = None
        # Construct the noise weighed antenna pattern
        F = np.array([[ifo.antenna_response(0., 0., 0., 0., 'plus'), ifo.antenna_response(0., 0., 0., 0., 'cross')] for ifo in self.interferometers])
        psd_array = np.array([ifo.power_spectral_density_array[self._frequency_mask] for ifo in self.interferometers])
        whitening_factor = 1. / np.sqrt(psd_array / (2 / self.interferometers[0].duration))
        self._whitened_antenna_response = np.einsum('dm,df->fdm', F, whitening_factor)
        # Constructed the noise weighed strain data
        strain_array = np.array([ifo.frequency_domain_strain[self._frequency_mask] for ifo in self.interferometers])
        self._whitened_strain_data_array = np.einsum('df,df->fd', strain_array, whitening_factor)

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor = np.array([self.recalibration_generator_1.get_calibration_factor(self._masked_frequency_array, self.parameters),
                                       self.recalibration_generator_2.get_calibration_factor(self._masked_frequency_array, self.parameters),
                                       self.recalibration_generator_3.get_calibration_factor(self._masked_frequency_array, self.parameters)])

        # Absorb the calibration factor into the whitened antenna response
        calibrated_whitened_antenna_response = np.einsum('fdm,df->fdm', self._whitened_antenna_response, calibration_factor)
        # Compute the loglikelihood
        logl = self._null_stream_log_likelihood(calibrated_whitened_antenna_response)
        return logl

    def _null_stream_log_likelihood(self,
                                   calibrated_whitened_antenna_response):
        """Null stream log likelihood

        Args:
            calibrated_whitened_antenna_response (numpy array): Calibrated whitened antenna response matrix.

        Returns:
            float: Log likelihood.
        """
        calibrated_whitened_antenna_response_T = np.transpose(calibrated_whitened_antenna_response, [0,2, 1])
        projector = np.array([np.eye(3) for _ in range(len(self._masked_frequency_array))]) - calibrated_whitened_antenna_response @ np.linalg.inv(np.conj(calibrated_whitened_antenna_response_T) @ calibrated_whitened_antenna_response) @ np.conj(calibrated_whitened_antenna_response_T)
        # Compute the projected strain
        projected_strain_data = np.einsum('fdl,fl->fd', projector, self._whitened_strain_data_array)
        logl = -np.sum(np.abs(projected_strain_data) ** 2)
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


class SelfRecalibrationScaleInsensitiveLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_1,
                 recalibration_generator_2,
                 recalibration_generator_3):
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
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask
        self._masked_frequency_array = self.interferometers[0].frequency_array[self._frequency_mask]
        self._noise_log_likelihood = None

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor_1 = self.recalibration_generator_1.get_calibration_factor(self._masked_frequency_array, self.parameters)
        calibration_factor_2 = self.recalibration_generator_2.get_calibration_factor(self._masked_frequency_array, self.parameters)
        calibration_factor_3 = self.recalibration_generator_3.get_calibration_factor(self._masked_frequency_array, self.parameters)

        # Recalibrate the strain
        recalibrated_frequency_domain_strain_1 = self.recalibrate_strain(calibration_factor_1,
                                                                         self.interferometers[0].frequency_domain_strain[self._frequency_mask])
        recalibrated_frequency_domain_strain_2 = self.recalibrate_strain(calibration_factor_2,
                                                                         self.interferometers[1].frequency_domain_strain[self._frequency_mask])
        recalibrated_frequency_domain_strain_3 = self.recalibrate_strain(calibration_factor_3,
                                                                         self.interferometers[2].frequency_domain_strain[self._frequency_mask])
        # Recalibrate the power spectral density
        recalibrated_power_spectral_density_1 = self.recalibrate_power_spectral_density(calibration_factor_1,
                                                                                        self.interferometers[0].power_spectral_density_array[self._frequency_mask])
        recalibrated_power_spectral_density_2 = self.recalibrate_power_spectral_density(calibration_factor_2,
                                                                                        self.interferometers[1].power_spectral_density_array[self._frequency_mask])
        recalibrated_power_spectral_density_3 = self.recalibrate_power_spectral_density(calibration_factor_3,
                                                                                        self.interferometers[2].power_spectral_density_array[self._frequency_mask])        
        logl = self._null_stream_log_likelihood(recalibrated_frequency_domain_strain_1,
                                                recalibrated_frequency_domain_strain_2,
                                                recalibrated_frequency_domain_strain_3,
                                                recalibrated_power_spectral_density_1,
                                                recalibrated_power_spectral_density_2,
                                                recalibrated_power_spectral_density_3,
                                                self.interferometers[0].duration)
        return logl

    def recalibrate_strain(self, calibration_factor, frequency_domain_strain):
        """Recalibrate strain.

        Args:
            calibration_factor (numpy array): Calibration factor.
            frequency_domain_strain (numpy array): Frequency domain strain.

        Returns:
            numpy array: Recalibrated frequency domain strain.
        """
        return frequency_domain_strain / calibration_factor

    def recalibrate_power_spectral_density(self, calibration_factor, power_spectral_density_array):
        """Recalibrate power spectral density.

        Args:
            calibration_factor (numpy array): Calibration factor.
            power_spectral_density_array (numpy array): Power spectral density.

        Returns:
            numpy array: Recalibrated power spectral density.
        """
        return power_spectral_density_array / np.abs(calibration_factor) ** 2

    def _null_stream_log_likelihood(self,
                                    frequency_domain_strain_1,
                                    frequency_domain_strain_2,
                                    frequency_domain_strain_3,
                                    power_spectral_density_array_1,
                                    power_spectral_density_array_2,
                                    power_spectral_density_array_3,
                                    duration):
        """Null stream log likelihood

        Args:
            frequency_domain_strain_1 (numpy array): Frequency domain strain of detector 1.
            frequency_domain_strain_2 (numpy array): Frequency domain strain of detector 2.
            frequency_domain_strain_3 (numpy array): Frequency domain strain of detector 3.
            power_spectral_density_array_1 (numpy array): Power spectral density of detector 1.
            power_spectral_density_array_2 (numpy array): Power spectral density of detector 2.
            power_spectral_density_array_3 (numpy array): Power spectral density of detector 3.
            duration (float): Duration of data segment in second.

        Returns:
            float: Log likelihood.
        """
        null_stream = (frequency_domain_strain_1 + \
                       frequency_domain_strain_2 + \
                       frequency_domain_strain_3) / np.sqrt(3)
        null_stream_PSD = (power_spectral_density_array_1 + \
                           power_spectral_density_array_2 + \
                           power_spectral_density_array_3) / 3
        normalization_constant = 4 / duration
        logl = -0.5 * normalization_constant * np.sum(np.abs(null_stream)**2/null_stream_PSD)
        #lognorm = -0.5 * np.sum(np.log(null_stream_PSD))
        #logl = logresidual + lognorm
        #logl = logresidual
        return logl
        
    def _calculate_noise_log_likelihood(self):
        """Log likelihood of the noise model.

        Returns:
            float: Log likelihood.
        """
        uncalibrated_frequency_domain_strain_1 = self.interferometers[0].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_2 = self.interferometers[1].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_3 = self.interferometers[2].frequency_domain_strain[self._frequency_mask]
        uncalibrated_PSD_1 = self.interferometers[0].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_2 = self.interferometers[1].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_3 = self.interferometers[2].power_spectral_density_array[self._frequency_mask]
        # Compute the log likelihood.
        logl = self._null_stream_log_likelihood(uncalibrated_frequency_domain_strain_1,
                                                uncalibrated_frequency_domain_strain_2,
                                                uncalibrated_frequency_domain_strain_3,
                                                uncalibrated_PSD_1,
                                                uncalibrated_PSD_2,
                                                uncalibrated_PSD_3,
                                                self.interferometers[0].duration)
        return logl

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood        

class RelativeSelfRecalibrationLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 recalibration_generator_21,
                 recalibration_generator_31):
        """Self-calibration likelihood.

        Args:
            interferometers (bilby InterferometerList): An InterferometerList instance.
            recalibration_generator_1 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
            recalibration_generator_2 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
            recalibration_generator_3 (RecalibrationGenerator): A RecalibrationGenerator instance to generate calibration factor.
        """
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.recalibration_generator_21 = recalibration_generator_21
        self.recalibration_generator_31 = recalibration_generator_31
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask
        self._masked_frequency_array = self.interferometers[0].frequency_array[self._frequency_mask]
        self._noise_log_likelihood = None

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        calibration_factor_21 = self.recalibration_generator_21.get_calibration_factor(self._masked_frequency_array, self.parameters)
        calibration_factor_31 = self.recalibration_generator_31.get_calibration_factor(self._masked_frequency_array, self.parameters)

        # Recalibrate the strain
        recalibrated_frequency_domain_strain_1 = self.interferometers[0].frequency_domain_strain[self._frequency_mask]
        recalibrated_frequency_domain_strain_2 = self.recalibrate_strain(calibration_factor_21,
                                                                         self.interferometers[1].frequency_domain_strain[self._frequency_mask])
        recalibrated_frequency_domain_strain_3 = self.recalibrate_strain(calibration_factor_31,
                                                                         self.interferometers[2].frequency_domain_strain[self._frequency_mask])
        # Recalibrate the power spectral density
        recalibrated_power_spectral_density_1 = self.interferometers[0].power_spectral_density_array[self._frequency_mask]
        recalibrated_power_spectral_density_2 = self.recalibrate_power_spectral_density(calibration_factor_21,
                                                                                        self.interferometers[1].power_spectral_density_array[self._frequency_mask])
        recalibrated_power_spectral_density_3 = self.recalibrate_power_spectral_density(calibration_factor_31,
                                                                                        self.interferometers[2].power_spectral_density_array[self._frequency_mask])        
        logl = self._null_stream_log_likelihood(recalibrated_frequency_domain_strain_1,
                                                recalibrated_frequency_domain_strain_2,
                                                recalibrated_frequency_domain_strain_3,
                                                recalibrated_power_spectral_density_1,
                                                recalibrated_power_spectral_density_2,
                                                recalibrated_power_spectral_density_3,
                                                self.interferometers[0].duration)
        return logl

    def recalibrate_strain(self, calibration_factor, frequency_domain_strain):
        """Recalibrate strain.

        Args:
            calibration_factor (numpy array): Calibration factor.
            frequency_domain_strain (numpy array): Frequency domain strain.

        Returns:
            numpy array: Recalibrated frequency domain strain.
        """
        return frequency_domain_strain / calibration_factor

    def recalibrate_power_spectral_density(self, calibration_factor, power_spectral_density_array):
        """Recalibrate power spectral density.

        Args:
            calibration_factor (numpy array): Calibration factor.
            power_spectral_density_array (numpy array): Power spectral density.

        Returns:
            numpy array: Recalibrated power spectral density.
        """
        return power_spectral_density_array / np.abs(calibration_factor) ** 2

    def _null_stream_log_likelihood(self,
                                    frequency_domain_strain_1,
                                    frequency_domain_strain_2,
                                    frequency_domain_strain_3,
                                    power_spectral_density_array_1,
                                    power_spectral_density_array_2,
                                    power_spectral_density_array_3,
                                    duration):
        """Null stream log likelihood

        Args:
            frequency_domain_strain_1 (numpy array): Frequency domain strain of detector 1.
            frequency_domain_strain_2 (numpy array): Frequency domain strain of detector 2.
            frequency_domain_strain_3 (numpy array): Frequency domain strain of detector 3.
            power_spectral_density_array_1 (numpy array): Power spectral density of detector 1.
            power_spectral_density_array_2 (numpy array): Power spectral density of detector 2.
            power_spectral_density_array_3 (numpy array): Power spectral density of detector 3.
            duration (float): Duration of data segment in second.

        Returns:
            float: Log likelihood.
        """
        null_stream = (frequency_domain_strain_1 + \
                       frequency_domain_strain_2 + \
                       frequency_domain_strain_3) / np.sqrt(3)
        null_stream_PSD = (power_spectral_density_array_1 + \
                           power_spectral_density_array_2 + \
                           power_spectral_density_array_3) / 3
        normalization_constant = 4 / duration
        logl = -0.5 * normalization_constant * np.sum(np.abs(null_stream)**2/null_stream_PSD)
        lognorm = -0.5 * np.sum(np.log(null_stream_PSD))
        print(logl, lognorm)
        logl = logl + lognorm
        return logl
        
    def _calculate_noise_log_likelihood(self):
        """Log likelihood of the noise model.

        Returns:
            float: Log likelihood.
        """
        uncalibrated_frequency_domain_strain_1 = self.interferometers[0].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_2 = self.interferometers[1].frequency_domain_strain[self._frequency_mask]
        uncalibrated_frequency_domain_strain_3 = self.interferometers[2].frequency_domain_strain[self._frequency_mask]
        uncalibrated_PSD_1 = self.interferometers[0].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_2 = self.interferometers[1].power_spectral_density_array[self._frequency_mask]
        uncalibrated_PSD_3 = self.interferometers[2].power_spectral_density_array[self._frequency_mask]
        # Compute the log likelihood.
        logl = self._null_stream_log_likelihood(uncalibrated_frequency_domain_strain_1,
                                                uncalibrated_frequency_domain_strain_2,
                                                uncalibrated_frequency_domain_strain_3,
                                                uncalibrated_PSD_1,
                                                uncalibrated_PSD_2,
                                                uncalibrated_PSD_3,
                                                self.interferometers[0].duration)
        return logl

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood