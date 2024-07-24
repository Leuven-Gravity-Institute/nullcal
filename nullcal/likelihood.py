import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList


class SelfCalibrationLikelihood(Likelihood):
    def __init__(self,
                 interferometers,
                 calibration_model):
        super().__init__(dict())
        self.interferometers = InterferometerList(interferometers)
        self.calibration_model = calibration_model
        # Find the frequency mask
        self._frequency_mask = self.interferometers[0].frequency_mask * \
                               self.interferometers[1].frequency_mask * \
                               self.interferometers[2].frequency_mask

    def log_likelihood(self):
        response_function_1, response_function_2, response_function_3 = \
            self.calibration_model.frequency_domain_response_function(self.parameters)
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
