"""A submodule for null stream calculation."""

from __future__ import annotations

import numpy as np
from bilby.gw.detector import InterferometerList

from ..time_frequency_transform.wavelet_transforms import WaveletTransform
from .calibration import compute_calibrated_whitened_antenna_response
from .projector import compute_projector
from .whiten import compute_whitened_antenna_response, compute_whitened_frequency_domain_strain


def compute_projected_strain_data(
    projector: np.ndarray, strain_data: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Compute the projected strain data.

    Args:
        projector (np.ndarray): Projector. Dimensions: (frequency, detector, detector).
        strain_data (np.ndarray): Strain data. Dimensions: (detector, frequency).
        frequency_mask (np.ndarray): Frequency mask. Dimensions: (frequency,).

    Raises:
        ValueError: Projector shape mismatch. project must have the same dimensions in the two last axes.
        ValueError: Shape mismatch. projector and strain_data must have the same detector and frequency dimensions.
        ValueError: Shape mismatch. strain_data and frequency_mask must have the same frequency dimension.

    Returns:
        np.ndarray: Projected strain data. Dimensions: (detector, frequency).
    """
    n_freq_1, n_det_1, n_det_2 = projector.shape
    if n_det_1 != n_det_2:
        raise ValueError(
            "Shape mismatch."
            f"projector: (frequency={n_freq_1},detector={n_det_1}, detector={n_det_2})."
            "project must have the same dimensions in the two last axes."
        )
    n_det_3, n_freq_2 = strain_data.shape
    if n_det_1 != n_det_3 or n_freq_1 != n_freq_2:
        raise ValueError(
            "Shape mismatch."
            f"projector: (frequency={n_freq_1},detector={n_det_1}, detector={n_det_2})."
            f"strain_data: (detector={n_det_3},frequency={n_freq_2})."
            "projector and strain_data must have the same detector and frequency dimensions."
        )
    n_freq_3 = frequency_mask.shape[0]
    if n_freq_2 != n_freq_3:
        raise ValueError(
            "Shape mismatch."
            f"strain_data: (detector={n_det_3},frequency={n_freq_2})."
            f"frequency_mask: (frequency={n_freq_3})."
            "strain_data and frequency_mask must have the same frequency dimension."
        )
    output = np.zeros_like(strain_data)
    output[:, frequency_mask] = np.einsum("fij,jf->if", projector[frequency_mask, :, :], strain_data[:, frequency_mask])
    return output


class NullStream:
    """A class to handle null stream calculation."""

    def __init__(
        self,
        interferometers: InterferometerList,
        time_frequency_transform: WaveletTransform,
        time_frequency_filter: np.ndarray,
    ):
        """A null stream calculator.

        Args:
            interferometers (InterferometerList): An InterferometerList instance.
            time_frequency_transform (WaveletTransform): A WaveletTransform instance.
            time_frequency_filter (np.ndarray): A time-frequency filter.
        """
        self.interferometers = interferometers
        self.time_frequency_transform = time_frequency_transform
        self.time_frequency_filter = time_frequency_filter

        # Pre-compute the whitened quantities.
        self.frequency_mask = np.all([ifo.frequency_mask for ifo in self.interferometers], axis=0)
        self.masked_frequency_array = interferometers[0].frequency_array[self.frequency_mask]
        # Construct the noise weighed antenna pattern
        # This is the orthogonalized beam pattern matrix, correct for ET only,
        # ignoring the small difference in location of the detectors.
        beam_pattern_matrix = np.array(
            [[-1.0 / np.sqrt(6), -1 / np.sqrt(2)], [np.sqrt(6) / 3, 0], [-1 / np.sqrt(6), 1 / np.sqrt(2)]]
        )
        power_spectral_density_array = np.array([ifo.power_spectral_density_array.copy() for ifo in interferometers])
        self._whitened_antenna_response = compute_whitened_antenna_response(
            beam_pattern_matrix,
            power_spectral_density_array,
            1 / self.interferometers[0].duration,
            self.frequency_mask,
        )
        self._whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain(
            frequency_domain_strain_array=np.array([ifo.frequency_domain_strain for ifo in interferometers]),
            power_spectral_density_array=power_spectral_density_array,
            delta_f=1.0 / interferometers[0].duration,
            frequency_mask=self.frequency_mask,
        )

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

    def compute_uncalibrated_time_frequency_domain_null_stream(self) -> np.ndarray:
        """Compute the uncalibrated time-frequency domain null stream.

        Returns:
            np.ndarray: Uncalibrated time-frequency domain null stream.
        """
        uncalibrated_frequency_domain_null_stream = self.compute_uncalibrated_frequency_domain_null_stream()
        # Transform to time-frequency domain
        uncalibrated_time_frequency_domain_null_stream = np.array(
            [
                self.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=data)
                for data in uncalibrated_frequency_domain_null_stream
            ]
        )
        uncalibrated_frequency_domain_null_stream[:, ~self.time_frequency_filter] = 0.0
        return uncalibrated_time_frequency_domain_null_stream

    def compute_calibrated_time_frequency_domain_null_stream(self, calibration_factor: np.ndarray) -> np.ndarray:
        """Compute the calibrated time-frequency domain null stream.

        Args:
            calibration_factor (np.ndarray): Calibration factor.

        Returns:
            np.ndarray: Calibrated time-frequency domain null stream.
        """
        calibrated_frequency_domain_null_stream = self.compute_calibrated_frequency_domain_null_stream(
            calibration_factor=calibration_factor
        )
        # Transform to time-frequency domain
        calibrated_time_frequency_domain_null_stream = np.array(
            [
                self.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=data)
                for data in calibrated_frequency_domain_null_stream
            ]
        )
        return calibrated_time_frequency_domain_null_stream

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

    def compute_calibrated_time_frequency_domain_null_stream_from_parameters(self, parameters: dict) -> np.ndarray:
        """Compute the calibrated time-frequency domain null stream from parameters.

        Args:
            parameters (dict): A dictionary of calibration parameters.

        Returns:
            np.ndarray: Calibrated time-frequency domain null stream.
        """
        calibration_factor = self.construct_calibration_factor_from_parameters(parameters)
        calibrated_time_frequency_domain_null_stream = self.compute_calibrated_time_frequency_domain_null_stream(
            calibration_factor=calibration_factor
        )
        calibrated_time_frequency_domain_null_stream[:, ~self.time_frequency_filter] = 0.0
        return calibrated_time_frequency_domain_null_stream
