import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList


class RecalibrationLikelihood(Likelihood):
    def __init__(self, interferometers):
        super().__init__(dict())
        self._interferometers = InterferometerList(interferometers)
        self._frequency_array = self._interferometers[0].frequency_array.copy()
        self._frequency_mask = np.logical_and.reduce([ifo.frequency_mask for ifo in self._interferometers])
        self._masked_frequency_array = self._frequency_array[self._frequency_mask]
        self._delta_f = 1. / self._interferometers[0].duration
        self._masked_frequency_domain_strain_array = None
        self._masked_power_spectral_density_array = None
        self._log_normalization = None
        self._noise_log_likelihood = None

    @property
    def interferometers(self):
        return self._interferometers

    @property
    def frequency_array(self):
        return self._frequency_array

    @property
    def frequency_mask(self):
        return self._frequency_mask

    @property
    def masked_frequency_array(self):
        return self._masked_frequency_array

    @property
    def delta_f(self):
        return self._delta_f

    @property
    def masked_frequency_domain_strain_array(self):
        if self._masked_frequency_domain_strain_array is None:
            self._masked_frequency_domain_strain_array = np.array([
                ifo.frequency_domain_strain[self._frequency_mask] for ifo in self._interferometers
            ])
        return self._masked_frequency_domain_strain_array

    @property
    def masked_power_spectral_density_array(self):
        if self._masked_power_spectral_density_array is None:
            self._masked_power_spectral_density_array = np.array([
                ifo.power_spectral_density_array[self._frequency_mask] for ifo in self._interferometers
            ])
        return self._masked_power_spectral_density_array

    @property
    def log_normalization(self):
        if self._log_normalization is None:
            self._log_normalization = -np.log(np.pi / 2 / self.delta_f) * \
                len(self.masked_frequency_array)
        return self._log_normalization

    def log_likelihood(self):
        # Compute the calibration errors
        calibration_errors = np.array([ifo.calibration_model.get_calibration_factor(
            frequency_array=self.masked_frequency_array,
            prefix=f'recalib_{ifo.name}_',
            **self.parameters) for ifo in self.interferometers])

        # Compute the relative calibration errors
        relative_calibration_errors = calibration_errors[0, :] / \
            calibration_errors

        # Multiply the relative calibration errors with the strain data
        recal_frequency_domain_strain_array = np.sum(
            self.masked_frequency_domain_strain_array *
            relative_calibration_errors,
            axis=0) / np.sqrt(3)

        # Compute the corresponding power spectral density
        recal_power_spectral_density = np.sum(
            self.masked_power_spectral_density_array *
            np.abs(relative_calibration_errors)**2, axis=0) / 3

        # Compute the log likelihood
        log_likelihood = -2 * self.delta_f * np.sum(
            np.abs(recal_frequency_domain_strain_array)**2 /
            recal_power_spectral_density)
        log_likelihood += -np.sum(np.log(recal_power_spectral_density)) + \
            self.log_normalization
        return log_likelihood

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            frequency_domain_strain_array = np.sum(
                self.masked_frequency_domain_strain_array,
                axis=0) / np.sqrt(3)
            power_spectral_density = np.sum(
                self.masked_power_spectral_density_array,
                axis=0) / 3

            # Compute the log likelihood
            log_likelihood = -2 * self.delta_f * np.sum(
                np.abs(frequency_domain_strain_array)**2 /
                power_spectral_density)
            log_likelihood += -np.sum(np.log(power_spectral_density)) + \
                self.log_normalization
            self._noise_log_likelihood = log_likelihood
        return self._noise_log_likelihood
