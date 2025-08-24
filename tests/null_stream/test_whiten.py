from __future__ import annotations

import bilby.core.utils
import numpy as np
import pytest
import scipy.stats
from bilby.gw.detector import InterferometerList

from nullcal.null_stream.whiten import compute_whitened_antenna_response, compute_whitened_frequency_domain_strain


@pytest.fixture
def mock_data():
    n_det = 3  # 3 ET detectors
    n_freq = 1000  # Frequency bins
    n_mode = 2  # GW polarization modes
    delta_f = 0.1  # Frequency resolution in Hz
    frequencies = np.linspace(1, 1000, n_freq)
    frequency_mask = (frequencies >= 10) & (frequencies <= 500)  # Mask for 10-500 Hz
    strain = np.ones((n_det, n_freq), dtype=np.complex128) * 1e-21  # Mock strain
    psd = np.ones((n_det, n_freq)) * 1e-46  # Mock PSD (~10⁻²³ strain/√Hz squared)
    antenna_response = np.ones((n_det, n_mode), dtype=np.complex128)  # Mock antenna response
    return {
        "n_det": n_det,
        "n_freq": n_freq,
        "n_mode": n_mode,
        "delta_f": delta_f,
        "frequency_mask": frequency_mask,
        "strain": strain,
        "psd": psd,
        "antenna_response": antenna_response,
    }


def test_invalid_strain_psd_shapes(mock_data):
    strain = mock_data["strain"]
    psd_wrong = np.ones((mock_data["n_det"] + 1, mock_data["n_freq"]))  # Wrong n_det
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_whitened_frequency_domain_strain(strain, psd_wrong, mock_data["delta_f"], mock_data["frequency_mask"])


def test_invalid_delta_f(mock_data):
    with pytest.raises(ValueError, match="delta_f must be positive"):
        compute_whitened_frequency_domain_strain(
            mock_data["strain"], mock_data["psd"], 0.0, mock_data["frequency_mask"]
        )


def test_whitening_accuracy(mock_data):
    strain = mock_data["strain"]
    psd = mock_data["psd"]
    delta_f = mock_data["delta_f"]
    mask = mock_data["frequency_mask"]
    expected = np.zeros_like(strain)
    expected[:, mask] = strain[:, mask] / np.sqrt(psd[:, mask] / (2 * delta_f))
    result = compute_whitened_frequency_domain_strain(strain, psd, delta_f, mask)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_zero_psd(mock_data):
    strain = mock_data["strain"]
    psd = mock_data["psd"].copy()
    mask = mock_data["frequency_mask"]
    psd[:, mask] = 0  # Zero PSD in masked region
    result = compute_whitened_frequency_domain_strain(strain, psd, mock_data["delta_f"], mask)
    expected = np.zeros_like(strain)
    np.testing.assert_allclose(result, expected, atol=1e-30)


def test_empty_frequency_mask(mock_data):
    mask = np.zeros(mock_data["n_freq"], dtype=bool)
    result = compute_whitened_frequency_domain_strain(mock_data["strain"], mock_data["psd"], mock_data["delta_f"], mask)
    np.testing.assert_allclose(result, np.zeros_like(mock_data["strain"]), atol=1e-30)


def test_full_frequency_mask(mock_data):
    mask = np.ones(mock_data["n_freq"], dtype=bool)
    strain = mock_data["strain"]
    psd = mock_data["psd"]
    delta_f = mock_data["delta_f"]
    expected = strain / np.sqrt(psd / (2 * delta_f))
    result = compute_whitened_frequency_domain_strain(strain, psd, delta_f, mask)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_invalid_antenna_psd_shapes(mock_data):
    antenna_wrong = np.ones((mock_data["n_det"] + 1, mock_data["n_mode"]))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_whitened_antenna_response(
            antenna_wrong, mock_data["psd"], mock_data["delta_f"], mock_data["frequency_mask"]
        )


def test_invalid_delta_f_antenna(mock_data):
    with pytest.raises(ValueError, match="delta_f must be positive"):
        compute_whitened_antenna_response(
            mock_data["antenna_response"], mock_data["psd"], -0.1, mock_data["frequency_mask"]
        )


def test_invalid_frequency_mask_antenna(mock_data):
    wrong_mask = np.ones(mock_data["n_freq"] - 1, dtype=bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_whitened_antenna_response(
            mock_data["antenna_response"], mock_data["psd"], mock_data["delta_f"], wrong_mask
        )


def test_antenna_response_accuracy(mock_data):
    antenna = mock_data["antenna_response"]
    psd = mock_data["psd"]
    delta_f = mock_data["delta_f"]
    mask = mock_data["frequency_mask"]
    n_freq, n_det, n_mode = mock_data["n_freq"], mock_data["n_det"], mock_data["n_mode"]
    expected = np.zeros((n_freq, n_det, n_mode), dtype=antenna.dtype)
    expected[mask, :, :] = np.einsum("dm,df->fdm", antenna, 1 / np.sqrt(psd[:, mask] / (2 * delta_f)))
    result = compute_whitened_antenna_response(antenna, psd, delta_f, mask)
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_zero_psd_antenna(mock_data):
    psd = mock_data["psd"].copy()
    mask = mock_data["frequency_mask"]
    psd[:, mask] = 0
    result = compute_whitened_antenna_response(mock_data["antenna_response"], psd, mock_data["delta_f"], mask)
    expected = np.zeros((mock_data["n_freq"], mock_data["n_det"], mock_data["n_mode"]))
    np.testing.assert_allclose(result, expected, atol=1e-30)


def test_empty_frequency_mask_antenna(mock_data):
    mask = np.zeros(mock_data["n_freq"], dtype=bool)
    result = compute_whitened_antenna_response(
        mock_data["antenna_response"], mock_data["psd"], mock_data["delta_f"], mask
    )
    expected = np.zeros((mock_data["n_freq"], mock_data["n_det"], mock_data["n_mode"]))
    np.testing.assert_allclose(result, expected, atol=1e-30)


def test_single_mode_antenna(mock_data):
    antenna = mock_data["antenna_response"][:, :1]  # Single mode
    mask = mock_data["frequency_mask"]
    psd = mock_data["psd"]
    delta_f = mock_data["delta_f"]
    result = compute_whitened_antenna_response(antenna, psd, delta_f, mask)
    expected = np.zeros((mock_data["n_freq"], mock_data["n_det"], 1), dtype=antenna.dtype)
    expected[mask, :, :] = np.einsum("dm,df->fdm", antenna, 1 / np.sqrt(psd[:, mask] / (2 * delta_f)))
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.fixture
def mock_colored_noise_data():
    minimum_frequency = 20
    maximum_frequency = 2048
    sampling_frequency = 4096
    duration = 16
    delta_f = 1 / duration
    seed = 12

    bilby.core.utils.random.seed(seed)
    interferometer = InterferometerList(["H1"])[0]
    interferometer.minimum_frequency = minimum_frequency
    interferometer.maximum_frequency = maximum_frequency
    interferometer.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
    )

    return {
        "delta_f": delta_f,
        "strain": np.array([interferometer.frequency_domain_strain]),
        "psd": np.array([interferometer.power_spectral_density_array]),
        "frequency_mask": interferometer.frequency_mask,
    }


def test_whitened_strain_statistical(mock_colored_noise_data):
    """Test that compute_whitened_frequency_domain_strain produces white noise."""
    strain = mock_colored_noise_data["strain"]
    psd = mock_colored_noise_data["psd"]
    delta_f = mock_colored_noise_data["delta_f"]
    mask = mock_colored_noise_data["frequency_mask"]

    # Compute whitened strain
    whitened_strain = compute_whitened_frequency_domain_strain(strain, psd, delta_f, mask)

    # Compute PSD of whitened strain (periodogram: |strain|^2 / delta_f)
    whitened_psd = np.abs(whitened_strain) ** 2 / delta_f

    # Validate unmasked frequencies are zero
    assert np.allclose(whitened_psd[:, ~mask], 0, atol=1e-50), "Unmasked frequencies should be zero"

    samples = np.concatenate((whitened_strain[0, mask].real, whitened_strain[0, mask].imag))
    result = scipy.stats.kstest(samples, cdf="norm", args=(0, np.sqrt(0.5)))
    assert result.pvalue >= 0.05
