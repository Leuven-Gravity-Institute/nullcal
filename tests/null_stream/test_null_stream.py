from __future__ import annotations

import bilby.core.utils
import numpy as np
import pytest
import scipy.stats
from bilby.gw.detector import InterferometerList

from nullcal.null_stream.null_stream import compute_projected_strain_data
from nullcal.null_stream.projector import compute_projector
from nullcal.null_stream.whiten import compute_whitened_antenna_response, compute_whitened_frequency_domain_strain


@pytest.fixture
def mock_projected_strain_data():
    n_det = 3  # 3 ET detectors
    n_mode = 2  # GW polarization modes
    n_freq = 1000  # Frequency bins
    delta_f = 0.1  # Frequency resolution in Hz
    frequencies = np.linspace(0, n_freq * delta_f, n_freq, endpoint=False)
    frequency_mask = (frequencies >= 10) & (frequencies <= 500)
    np.random.seed(12)

    # Mock whitened antenna response for projector
    antenna_response = np.random.randn(n_freq, n_det, n_mode) + np.random.randn(n_freq, n_det, n_mode) * 1.0j

    # Compute projector (simplified from compute_projector)
    projector = compute_projector(antenna_response, frequency_mask)

    # Mock whitened strain (Gaussian noise, mean 0, variance 1)
    np.random.seed(42)
    strain = np.random.normal(0, 1, (n_det, n_freq)) + 1j * np.random.normal(0, 1, (n_det, n_freq))
    strain = strain.astype(np.complex128)

    return {
        "n_det": n_det,
        "n_freq": n_freq,
        "n_mode": n_mode,
        "delta_f": delta_f,
        "frequency_mask": frequency_mask,
        "projector": projector,
        "strain": strain,
        "antenna_response": antenna_response,
        "frequencies": frequencies,
    }


@pytest.fixture
def mock_colored_noise_data():
    minimum_frequency = 20
    maximum_frequency = 2047
    sampling_frequency = 4096
    duration = 4
    delta_f = 1 / duration
    seed = 13

    np.random.randn(seed + 1)
    bilby.core.utils.random.seed(seed)
    interferometers = InterferometerList(["ET"])
    for interferometer in interferometers:
        interferometer.minimum_frequency = minimum_frequency
        interferometer.maximum_frequency = maximum_frequency
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
    )

    n_det = 3
    n_mode = 2

    antenna_response = np.random.randn(n_det, n_mode) + np.random.randn(n_det, n_mode) * 1.0j

    return {
        "delta_f": delta_f,
        "strain": np.array([interferometer.frequency_domain_strain for interferometer in interferometers]),
        "psd": np.array([interferometer.power_spectral_density_array for interferometer in interferometers]),
        "frequency_mask": interferometers[0].frequency_mask,
        "antenna_response": antenna_response,
    }


def test_invalid_projector_shape(mock_projected_strain_data):
    wrong_projector = np.ones(
        (
            mock_projected_strain_data["n_freq"],
            mock_projected_strain_data["n_det"] + 1,
            mock_projected_strain_data["n_det"] + 1,
        ),
        dtype=np.complex128,
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_projected_strain_data(
            wrong_projector, mock_projected_strain_data["strain"], mock_projected_strain_data["frequency_mask"]
        )


def test_invalid_strain_shape(mock_projected_strain_data):
    wrong_strain = np.ones(
        (mock_projected_strain_data["n_det"] + 1, mock_projected_strain_data["n_freq"]), dtype=np.complex128
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_projected_strain_data(
            mock_projected_strain_data["projector"], wrong_strain, mock_projected_strain_data["frequency_mask"]
        )


def test_invalid_frequency_mask(mock_projected_strain_data):
    wrong_mask = np.ones(mock_projected_strain_data["n_freq"] - 1, dtype=np.bool_)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_projected_strain_data(
            mock_projected_strain_data["projector"], mock_projected_strain_data["strain"], wrong_mask
        )


def test_unmasked_frequencies_zero(mock_projected_strain_data):
    projector = mock_projected_strain_data["projector"]
    strain = mock_projected_strain_data["strain"]
    mask = mock_projected_strain_data["frequency_mask"]
    output = compute_projected_strain_data(projector, strain, mask)
    np.testing.assert_allclose(output[:, ~mask], 0, atol=1e-10, err_msg="Unmasked frequencies should be zero")


def test_projection_orthogonality(mock_projected_strain_data):
    projector = mock_projected_strain_data["projector"]
    strain = mock_projected_strain_data["strain"]
    mask = mock_projected_strain_data["frequency_mask"]
    antenna = mock_projected_strain_data["antenna_response"]
    output = compute_projected_strain_data(projector, strain, mask)
    f_dagger = np.transpose(np.conj(antenna[mask]), axes=(0, 2, 1))
    result = np.einsum("fij,jf->if", f_dagger, output[:, mask])
    np.testing.assert_allclose(result, 0, atol=1e-10, err_msg="Projected strain not orthogonal to antenna response")


def test_projector_gaussianity(mock_colored_noise_data):
    strain = mock_colored_noise_data["strain"]
    psd = mock_colored_noise_data["psd"]
    delta_f = mock_colored_noise_data["delta_f"]
    mask = mock_colored_noise_data["frequency_mask"]

    antenna_response = mock_colored_noise_data["antenna_response"]

    whitened_strain = compute_whitened_frequency_domain_strain(strain, psd, delta_f, mask)
    whitened_antenna = compute_whitened_antenna_response(
        antenna_response_matrix=antenna_response, power_spectral_density_array=psd, delta_f=delta_f, frequency_mask=mask
    )
    projector = compute_projector(whitened_antenna, mask)
    projected_strain_data = compute_projected_strain_data(projector, whitened_strain, mask)
    # This is valid only when the PSDs are the same.
    u, _, _ = np.linalg.svd(antenna_response)
    rotated_projected_strain_data = np.einsum("ji,if->jf", np.conj(u.T), projected_strain_data[:, mask])
    samples = np.concatenate((rotated_projected_strain_data[2, :].real, rotated_projected_strain_data[2, :].imag))
    result = scipy.stats.kstest(samples, cdf="norm", args=(0.0, np.sqrt(0.5)))
    assert result.pvalue > 0.05


def test_empty_frequency_mask(mock_projected_strain_data):
    mask = np.zeros(mock_projected_strain_data["n_freq"], dtype=np.bool_)
    output = compute_projected_strain_data(
        mock_projected_strain_data["projector"], mock_projected_strain_data["strain"], mask
    )
    np.testing.assert_allclose(output, 0, atol=1e-10, err_msg="Output should be zero for empty mask")


def test_single_frequency(mock_projected_strain_data):
    mask = np.zeros(mock_projected_strain_data["n_freq"], dtype=np.bool_)
    mask[500] = True
    projector = mock_projected_strain_data["projector"]
    strain = mock_projected_strain_data["strain"]
    output = compute_projected_strain_data(projector, strain, mask)
    expected = np.zeros_like(strain)
    expected[:, 500] = np.matmul(projector[500], strain[:, 500, np.newaxis]).squeeze()
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-10)
