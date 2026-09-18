"""A submodule for null stream calculation."""

from __future__ import annotations

import numpy as np

from ..calibration import MINIMUM_CUBIC_SPLINE_KNOTS, calibration_factor
from ..data import InterferometerData
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
        interferometers: InterferometerData,
        time_frequency_transform: WaveletTransform,
        time_frequency_filter: np.ndarray,
    ):
        """A null stream calculator.

        Args:
            interferometers (InterferometerData): Frozen detector arrays and metadata.
            time_frequency_transform (WaveletTransform): A WaveletTransform instance.
            time_frequency_filter (np.ndarray): A time-frequency filter.
        """
        self.interferometers = interferometers
        self.time_frequency_transform = time_frequency_transform
        self.time_frequency_filter = time_frequency_filter

        # Pre-compute the whitened quantities.
        self.frequency_mask = np.all(self.interferometers.mask, axis=0)
        self.masked_frequency_array = self.interferometers.frequency_array[self.frequency_mask]
        # Construct the noise weighed antenna pattern
        # This is the orthogonalized beam pattern matrix, correct for ET only,
        # ignoring the small difference in location of the detectors.
        beam_pattern_matrix = np.array(
            [[-1.0 / np.sqrt(6), -1 / np.sqrt(2)], [np.sqrt(6) / 3, 0], [-1 / np.sqrt(6), 1 / np.sqrt(2)]]
        )
        power_spectral_density_array = np.asarray(interferometers.psd)
        self._whitened_antenna_response = compute_whitened_antenna_response(
            beam_pattern_matrix,
            power_spectral_density_array,
            1 / self.interferometers.duration,
            self.frequency_mask,
        )
        self._whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain(
            frequency_domain_strain_array=np.asarray(interferometers.strain),
            power_spectral_density_array=power_spectral_density_array,
            delta_f=1.0 / interferometers.duration,
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
            self._whitened_antenna_response, calibration_factor, self.frequency_mask
        )
        projector = compute_projector(calibrated_whitened_antenna_response, frequency_mask=self.frequency_mask)
        # Dimensions: (frequency, detector)

        return np.einsum("ijk,ki->ji", projector, self._whitened_frequency_domain_strain_array)

    def compute_uncalibrated_time_frequency_domain_null_stream(self) -> np.ndarray:
        """Compute the uncalibrated time-frequency domain null stream, confined to the filter.

        The returned array is zero outside ``time_frequency_filter``, matching what
        ``compute_calibrated_time_frequency_domain_null_stream_from_parameters`` returns. That
        symmetry is the point: ``noise_log_likelihood`` sums this array's energy and
        ``log_likelihood`` sums the calibrated one's, so if the two were confined to different
        domains their difference — the log Bayes factor, and anything derived from it — would be
        normalised against different sets of pixels.

        Returns:
            np.ndarray: Uncalibrated time-frequency domain null stream, zero outside the filter.
        """
        uncalibrated_frequency_domain_null_stream = self.compute_uncalibrated_frequency_domain_null_stream()
        # Transform to time-frequency domain
        uncalibrated_time_frequency_domain_null_stream = np.array(
            [
                self.time_frequency_transform.frequency_to_wavelet(frequency_domain_data=data)
                for data in uncalibrated_frequency_domain_null_stream
            ]
        )
        uncalibrated_time_frequency_domain_null_stream[:, ~self.time_frequency_filter] = 0.0
        return uncalibrated_time_frequency_domain_null_stream

    def compute_calibrated_time_frequency_domain_null_stream(self, calibration_factor: np.ndarray) -> np.ndarray:
        """Compute the calibrated time-frequency domain null stream, confined to the filter.

        Like its uncalibrated counterpart, the returned array is zero outside
        ``time_frequency_filter``. The two are deliberately symmetric: a caller that used one
        filtered method and one unfiltered one would sum energies over different pixel domains and
        recreate exactly the normalisation defect this pair was fixed for, silently and without an
        exception.

        Args:
            calibration_factor (np.ndarray): Calibration factor.

        Returns:
            np.ndarray: Calibrated time-frequency domain null stream, zero outside the filter.
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
        calibrated_time_frequency_domain_null_stream[:, ~self.time_frequency_filter] = 0.0
        return calibrated_time_frequency_domain_null_stream

    def construct_calibration_factor_from_parameters(self, parameters: dict) -> np.ndarray:
        """Construct the calibration factor from parameters.

        Args:
            parameters (dict): Calibration parameters.

        Returns:
            np.ndarray: Calibration factor.
        """
        detector_nodes = []
        expected_keys = set()
        for name in self.interferometers.name:
            prefix = f"recalib_{name}_"
            frequency_indices = sorted(
                int(key.removeprefix(f"{prefix}frequency_"))
                for key in parameters
                if key.startswith(f"{prefix}frequency_") and key.removeprefix(f"{prefix}frequency_").isdigit()
            )
            if (
                frequency_indices != list(range(len(frequency_indices)))
                or len(frequency_indices) < MINIMUM_CUBIC_SPLINE_KNOTS
            ):
                raise ValueError("calibration parameters do not match the required detector spline nodes")
            keys = {
                f"{prefix}{quantity}_{index}"
                for quantity in ("frequency", "amplitude", "phase")
                for index in frequency_indices
            }
            expected_keys.update(keys)
            detector_nodes.append((prefix, frequency_indices))

        supplied_keys = {key for key in parameters if key.startswith("recalib_")}
        if supplied_keys != expected_keys:
            missing = sorted(expected_keys - supplied_keys)
            unexpected = sorted(supplied_keys - expected_keys)
            raise ValueError(
                f"calibration parameters do not match the required keys; missing={missing}, unexpected={unexpected}"
            )

        calibration_factor_array = np.asarray(
            [
                calibration_factor(
                    self.masked_frequency_array,
                    [parameters[f"{prefix}frequency_{index}"] for index in indices],
                    [parameters[f"{prefix}amplitude_{index}"] for index in indices],
                    [parameters[f"{prefix}phase_{index}"] for index in indices],
                )
                for prefix, indices in detector_nodes
            ]
        )
        output = np.zeros_like(self._whitened_frequency_domain_strain_array)
        output[:, self.frequency_mask] = calibration_factor_array

        return output

    def compute_calibrated_time_frequency_domain_null_stream_from_parameters(self, parameters: dict) -> np.ndarray:
        """Compute the calibrated time-frequency domain null stream from parameters.

        Args:
            parameters (dict): A dictionary of calibration parameters.

        Returns:
            np.ndarray: Calibrated time-frequency domain null stream.
        """
        calibration_factor = self.construct_calibration_factor_from_parameters(parameters)
        # The filtering lives in compute_calibrated_time_frequency_domain_null_stream, which
        # guarantees confinement for every caller rather than only for this wrapper.
        return self.compute_calibrated_time_frequency_domain_null_stream(calibration_factor=calibration_factor)
