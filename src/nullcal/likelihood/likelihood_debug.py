from __future__ import annotations

import warnings

import numpy as np
from bilby.gw.detector.networks import InterferometerList

from .likelihood import SelfCalibrationLikelihood


class SelfCalibrationLikelihoodDebug(SelfCalibrationLikelihood):
    def __init__(self, interferometers: InterferometerList):
        """The likelihood class for self-calibration using the old implementation.

        Args:
            interferometers (InterferometerList): A list of interferometers.
        """
        super().__init__(interferometers=interferometers)
        warnings.warn("This likelihood class is only for comparing with the old implementation.", DeprecationWarning)

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
        logl +=  self.constant_log_normalization - np.sum(np.log(calibrated_null_stream_psd))
        return logl
