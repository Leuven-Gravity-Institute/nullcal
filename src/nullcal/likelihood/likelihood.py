from __future__ import annotations

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList


class SelfCalibrationLikelihood(Likelihood):
    def __init__(self, interferometers):
        self.interferometers = interferometers
        self._constant_log_normalization = np.log(2 * self.delta_f / np.pi) * np.sum(self.frequency_mask)

    @property
    def interferometers(self) -> InterferometerList:
        """Get the interferometers.

        Returns:
            InterferometerList: Interferometers.
        """
        return self._interferometers

    @interferometers.setter
    def interferometers(self, value: InterferometerList):
        """Set the interferometers.

        Args:
            value (InterferometerList): A list of interferometers.

        Raises:
            ValueError: The number of interferometers has to be 3.
            ValueError: Frequency arrays of the interferometers have to be identical.
        """
        self._interferometers = InterferometerList(value)
        # Check whether there are three detectors.
        if len(self._interferometers) != 3:
            raise ValueError(f'The number of interferometers is {len(self._interferometers)}. Expected 3.')
        self._frequency_array = self._interferometers[0].frequency_array
        for i in range(1, 3):
            if not np.isclose(self._frequency_array, self._interferometers[i].frequency_array).all():
                raise ValueError(f'frequency array of interferometer {self._interferometers[0].name}'
                                  f'is not equal to that of interferometer {self._interferometers[i].name}.')
        # Define the frequency mask.
        self._frequency_mask = (self._interferometers[0].frequency_mask & self._interferometers[1].frequency_mask
                                & self._interferometers[2].frequency_mask)
        # Masked frequency array.
        self._masked_frequency_array = self._frequency_array[self._frequency_mask]
        # Frequency resolution.
        self._delta_f = 1. / self._interferometers[0].duration
        # Arrays of masked frequency domain strain.
        self._masked_frequency_domain_strain_array = np.array([
                ifo.frequency_domain_strain[self._frequency_mask] for ifo in self._interferometers
            ])
        # Arrays of masked power spectral density.
        self._masked_power_spectral_density_array = np.array([
                ifo.power_spectral_density_array[self._frequency_mask] for ifo in self._interferometers
            ])


    @property
    def frequency_array(self) -> np.ndarray:
        """Get the frequency array.

        Returns:
            np.ndarray: Frequency array.
        """
        return self._frequency_array

    @property
    def frequency_mask(self) -> np.ndarray:
        """Get the frequency mask.

        Returns:
            np.ndarray: Frequency mask.
        """
        return self._frequency_mask

    @property
    def masked_frequency_array(self) -> np.ndarray:
        """Get the masked frequency array.

        Returns:
            np.ndarray: Masked frequency array.
        """
        return self._masked_frequency_array

    @property
    def delta_f(self) -> float:
        """Get the frequency resolution.

        Returns:
            float: Frequency resolution in Hz.
        """
        return self._delta_f

    @property
    def masked_frequency_domain_strain_array(self) -> np.ndarray:
        """Get the masked frequency domain strain array.

        Returns:
            np.ndarray: Masked frequency domain strain array.
        """
        return self._masked_frequency_domain_strain_array

    @property
    def masked_power_spectral_density_array(self) -> np.ndarray:
        """Get the masked power spectral density array.

        Returns:
            np.ndarray: Masked power spectral density array.
        """
        return self._masked_power_spectral_density_array

    @property
    def constant_log_normalization(self) -> float:
        """Get the constant part of the log normalization.

        Returns:
            float: Constant part of the log normalization.
        """
        return self._constant_log_normalization

    def _get_calibration_factor_from_parameters(self) -> np.ndarray:
        """Get the calibration factor from parameters.

        Raises:
            ValueError: Calibration factor contains invalid values.

        Returns:
            np.ndarray: Calibration factor (detector, frequency).
        """
        calibration_factor =  np.array([ifo.calibration_model.get_calibration_factor(frequency_array=self.masked_frequency_array,
                                                                                     prefix=f'recalib_{ifo.name}_',
                                                                                     **self.parameters) for ifo in self.interferometers])
        if not np.all(np.isfinite(calibration_factor)):
            raise ValueError("Calibration factor contains invalid values.")
        return calibration_factor

    def log_likelihood(self) -> float:
        """Compute the log likelihood function.

        Returns:
            float: Log likelihood function.
        """
        # Compute the calibration factor.
        calibration_factor = self._get_calibration_factor_from_parameters()
        # Compute the unnormalized calibrated null stream.
        calibrated_null_stream = np.sum(self.masked_frequency_domain_strain_array / calibration_factor, axis=0)
        # Compute the calibrated power spectral density.
        calibration_factor_squared = np.abs(calibration_factor) ** 2
        calibrated_null_stream_psd = np.sum(self.masked_power_spectral_density_array / calibration_factor_squared, axis=0)
        # Compute the residual component of the likelihood function.
        logl = -2 * self.delta_f * np.sum(np.abs(calibrated_null_stream) ** 2 / calibrated_null_stream_psd)
        # Compute the normalization term of the likelihood function.
        logl +=  self.constant_log_normalization - np.sum(np.log(calibrated_null_stream_psd)) - np.sum(np.log(calibration_factor_squared))
        return logl
