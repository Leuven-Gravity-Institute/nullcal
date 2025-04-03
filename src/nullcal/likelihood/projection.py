from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.stats
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList

from ..utils import logger


class ProjectionLikelihood(Likelihood):
    def __init__(self,
                 interferometers: InterferometerList,
                 masking_threshold: float | None=None):
        """Projection likelihood.

        The masking_threshold allows you to add an additional masking on the frequency bins
        to select the high significance frequency bins.

        Args:
            interferometers (InterferometerList): An InterferometerList instance.
            masking_threshold (float, optional): The threshold to mask the frequency array in the unit of sigma. Default to None.
        """
        super().__init__(dict())
        self.masking_threshold = masking_threshold
        self.interferometers = interferometers
        self._frequency_array = None
        self._frequency_mask = None
        self._masked_frequency_array = None
        self._delta_f = None
        self._masked_frequency_domain_strain_array = None
        self._masked_power_spectral_density_array = None
        self._noise_log_likelihood = None

    @property
    def masking_threshold(self) -> float | None:
        """Setting the masking threshold

        Returns:
            Union[float, None]: Masking threshold.
        """
        return self._masking_threshold

    @masking_threshold.setter
    def masking_threshold(self, value: float) -> float:
        """Set the masking threshold.

        Args:
            value (float): Masking threshold.
        """
        self._masking_threshold = value

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

    @property
    def frequency_mask(self) -> np.ndarray:
        if self._frequency_mask is None:
            # Get the frequency mask from the individual interferometers.
            self._frequency_mask = (self._interferometers[0].frequency_mask & self._interferometers[1].frequency_mask
                                    & self._interferometers[2].frequency_mask)
            # Check whether the masking_threshold is provided.
            # If yes, compute the mask.
            if self.masking_threshold is not None:
                logger.info(f'masking_threshold = {self.masking_threshold} is provided.')

                # Get the corresponding significance level.
                significance_level = scipy.stats.norm.sf(self.masking_threshold)

                logger.info(f'The corresponding significance level is {significance_level}.')
                logger.info(f'The original number of frequency bins is {np.sum(self._frequency_mask)}.')

                # Compute the null energy significance
                null_energy_significance = self._compute_uncalibrated_null_energy_significance()
                frequency_mask = null_energy_significance <= significance_level
                self._frequency_mask = (self._frequency_mask & frequency_mask)

                logger.info(f'The updated number of frequency bins is {np.sum(self._frequency_mask)}.')
        return self._frequency_mask

    @property
    def frequency_array(self) -> np.ndarray:
        """Frequency array.

        Returns:
            np.ndarray: Frequency array.
        """
        if self._frequency_array is None:
            self._frequency_array = self.interferometers[0].frequency_array
        return self._frequency_array

    @property
    def masked_frequency_array(self) -> np.ndarray:
        """Masked frequency array.

        Returns:
            np.ndarray: Masked frequency array.
        """
        if self._masked_frequency_array is None:
            # Masked frequency array.
            self._masked_frequency_array = self.frequency_array[self.frequency_mask]
        return self._masked_frequency_array

    @property
    def delta_f(self) -> float:
        """Get the frequency resolution.

        Returns:
            float: Frequency resolution.
        """
        if self._delta_f is None:
            self._delta_f = 1. / self.interferometers[0].duration
        return self._delta_f

    def _compute_uncalibrated_null_energy_significance(self) -> np.ndarray:
        """Compute the uncalibrated null energy significance.

        Returns:
            np.ndarray: An array of null energy significance.
        """
        null_stream = np.sum([ifo.frequency_domain_strain for ifo in self.interferometers], axis=0)
        # Construct the uncalibrated null stream and compute the significance.
        null_stream = np.sum([ifo.frequency_domain_strain for ifo in self.interferometers], axis=0)
        # Construct the uncalibrated null stream noise PSD.
        null_stream_psd = np.sum([ifo.power_spectral_density_array for ifo in self.interferometers], axis=0)
        # Compute the null energy
        null_energy = np.abs(null_stream) ** 2 / null_stream_psd * 4 / self.interferometers[0].duration
        # Compute the significance
        null_energy_significance = scipy.stats.chi2.sf(null_energy, df=2)
        return null_energy_significance

    @property
    def masked_frequency_domain_strain_array(self) -> np.ndarray:
        """Get the masked frequency domain strain array.

        Returns:
            np.ndarray: Masked frequency domain strain array.
        """
        if self._masked_frequency_domain_strain_array is None:
            # Arrays of masked frequency domain strain.
            self._masked_frequency_domain_strain_array = np.array([
                    ifo.frequency_domain_strain[self.frequency_mask] for ifo in self._interferometers
                ])
        return self._masked_frequency_domain_strain_array

    @property
    def masked_power_spectral_density_array(self) -> np.ndarray:
        """Get the masked power spectral density array.

        Returns:
            np.ndarray: Masked power spectral density array.
        """
        if self._masked_power_spectral_density_array is None:
            # Arrays of masked power spectral density.
            self._masked_power_spectral_density_array = np.array([
                    ifo.power_spectral_density_array[self._frequency_mask] for ifo in self._interferometers
                ])
        return self._masked_power_spectral_density_array

    def log_likelihood(self):
        """Log likelihood.

        Returns:
            float: Log likelihood.
        """
        if np.sum(self.frequency_mask) == 0:
            return 0.0
        # Compute the calibration factor.
        calibration_factor = self._get_calibration_factor_from_parameters()
        if not np.all(np.isfinite(calibration_factor)):
            raise ValueError("Calibration factor contains invalid values.")

        # Compute the unnormalized null stream
        unnormalized_null_stream_array = np.sum(self.masked_frequency_domain_strain_array / calibration_factor, axis=0)

        # Compute the unnormalized null stream PSD.
        calibration_factor_squared = np.abs(calibration_factor) ** 2
        unnormalized_null_stream_psd_array = np.sum(self.masked_power_spectral_density_array / calibration_factor_squared, axis=0)

        # Compute the residual term
        residual = -np.sum(np.abs(unnormalized_null_stream_array) ** 2 / unnormalized_null_stream_psd_array) * 2 * self.delta_f

        # Compute the log normalization term
        log_normalization = -np.sum(np.log(unnormalized_null_stream_psd_array / np.sum(1. / calibration_factor_squared, axis=0) * (np.pi / 2 / self.delta_f)))

        # Compute the loglikelihood
        logl = residual + log_normalization
        return logl

    def _get_calibration_factor_from_parameters(self) -> np.ndarray:
        calibration_factor =  np.array([ifo.calibration_model.get_calibration_factor(frequency_array=self.masked_frequency_array,
                                                                                     prefix=f'recalib_{ifo.name}_',
                                                                                     **self.parameters) for ifo in self.interferometers])
        return calibration_factor

    def _calculate_noise_log_likelihood(self) -> float:
        """Log likelihood of the noise model.

        The noise model is defined to be the absence of calibration error here.

        Returns:
            float: Log likelihood.
        """
        if np.sum(self.frequency_mask) == 0:
            return 0.0

        # Compute the unnormalized null stream
        unnormalized_null_stream_array = np.sum(self.masked_frequency_domain_strain_array, axis=0)

        # Compute the unnormalized null stream PSD.
        unnormalized_null_stream_psd_array = np.sum(self.masked_power_spectral_density_array, axis=0)

        # Compute the residual term
        residual = -np.sum(np.abs(unnormalized_null_stream_array) ** 2 / unnormalized_null_stream_psd_array) * 2 * self.delta_f

        # Compute the log normalization term
        log_normalization = -np.sum(np.log(unnormalized_null_stream_psd_array / 3  * (np.pi / 2 / self.delta_f)))

        # Compute the loglikelihood
        logl = residual + log_normalization
        return logl

    def noise_log_likelihood(self) -> float:
        """Get the noise log likelihood.

        The noise model is defined to be the absence of calibration error here.

        Returns:
            float: Noise log likelihood.
        """
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood
