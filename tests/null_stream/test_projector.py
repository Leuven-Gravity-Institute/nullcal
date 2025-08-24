from __future__ import annotations

import numpy as np
import pytest

from nullcal.null_stream.projector import compute_projector


@pytest.fixture
def mock_projector_data():
    np.random.seed(12)
    n_det = 3  # 3 ET detectors
    n_mode = 2  # GW polarization modes
    n_freq = 1000  # Frequency bins
    frequencies = np.linspace(1, 1000, n_freq)
    frequency_mask = (frequencies >= 10) & (frequencies <= 500)  # 10—500 Hz
    # Mock calibrated whitened antenna response (complex-valued)
    antenna_response = np.random.randn(n_freq, n_det, n_mode) + np.random.randn(n_freq, n_det, n_mode) * 1.0j
    return {
        "n_det": n_det,
        "n_freq": n_freq,
        "n_mode": n_mode,
        "frequency_mask": frequency_mask,
        "antenna_response": antenna_response,
    }


def test_invalid_frequency_mask(mock_projector_data):
    wrong_mask = np.ones(mock_projector_data["n_freq"] - 1, dtype=bool)
    with pytest.raises(ValueError, match="Frequency mask shape mismatch"):
        compute_projector(mock_projector_data["antenna_response"], wrong_mask)


def test_projector_idempotence(mock_projector_data):
    antenna = mock_projector_data["antenna_response"]
    mask = mock_projector_data["frequency_mask"]
    projector = compute_projector(antenna, mask)
    projector_squared = np.matmul(projector[mask], projector[mask])
    np.testing.assert_allclose(projector[mask], projector_squared, rtol=1e-5, atol=1e-10)


def test_projector_orthogonality(mock_projector_data):
    antenna = mock_projector_data["antenna_response"]
    mask = mock_projector_data["frequency_mask"]
    projector = compute_projector(antenna, mask)
    result = np.matmul(projector[mask], antenna[mask])
    np.testing.assert_allclose(result, 0, atol=1e-10)


def test_unmasked_identity(mock_projector_data):
    antenna = mock_projector_data["antenna_response"]
    mask = mock_projector_data["frequency_mask"]
    n_det = mock_projector_data["n_det"]
    projector = compute_projector(antenna, mask)
    expected = np.eye(n_det, dtype=antenna.dtype)[np.newaxis, :, :].repeat(mock_projector_data["n_freq"], axis=0)
    np.testing.assert_allclose(projector[~mask], expected[~mask], rtol=1e-5)


def test_empty_frequency_mask(mock_projector_data):
    mask = np.zeros(mock_projector_data["n_freq"], dtype=bool)
    projector = compute_projector(mock_projector_data["antenna_response"], mask)
    expected = np.eye(mock_projector_data["n_det"], dtype=np.complex128)[np.newaxis, :, :].repeat(
        mock_projector_data["n_freq"], axis=0
    )
    np.testing.assert_allclose(projector, expected, rtol=1e-5)


def test_single_frequency(mock_projector_data):
    mask = np.zeros(mock_projector_data["n_freq"], dtype=bool)
    mask[500] = True
    antenna = mock_projector_data["antenna_response"]
    projector = compute_projector(antenna, mask)
    antenna_k = antenna[500:501]
    antenna_k_dagger = np.transpose(np.conj(antenna_k), axes=(0, 2, 1))
    expected = np.eye(mock_projector_data["n_det"], dtype=np.complex128)[np.newaxis, :, :]
    projector_gw = np.matmul(
        antenna_k, np.matmul(np.linalg.inv(np.matmul(antenna_k_dagger, antenna_k)), antenna_k_dagger)
    )
    expected[0] -= projector_gw[0]
    np.testing.assert_allclose(projector[500:501], expected, rtol=1e-5)


def test_singular_matrix(mock_projector_data):
    antenna = mock_projector_data["antenna_response"].copy()
    antenna[:, :, 1] = antenna[:, :, 0]
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        compute_projector(antenna, mock_projector_data["frequency_mask"])
