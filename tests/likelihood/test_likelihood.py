from __future__ import annotations

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullcal.likelihood import SelfCalibrationLikelihood

sampling_frequency = 4096
duration = 16
minimum_frequency = 10

@pytest.fixture
def mock_interferometers():
    """Create a valid InterferometerList with three interferometers."""
    ifos = InterferometerList(['ET'])

    # Set frequency arrays to be identical
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency

    ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                         duration=duration)

    return ifos


def test_initialization(mock_interferometers):
    """Test that SelfCalibrationLikelihood initializes correctly."""
    likelihood = SelfCalibrationLikelihood(interferometers=mock_interferometers)

    assert likelihood.interferometers == mock_interferometers
    assert isinstance(likelihood, SelfCalibrationLikelihood)


def test_invalid_interferometer_count():
    """Test that an error is raised when the number of interferometers is not 3."""
    with pytest.raises(ValueError, match="Expected 3"):
        SelfCalibrationLikelihood(InterferometerList(['H1']))


def test_properties(mock_interferometers):
    """Test the properties of SelfCalibrationLikelihood."""
    likelihood = SelfCalibrationLikelihood(mock_interferometers)

    assert isinstance(likelihood.frequency_array, np.ndarray)
    assert isinstance(likelihood.frequency_mask, np.ndarray)
    assert isinstance(likelihood.masked_frequency_array, np.ndarray)
    assert isinstance(likelihood.delta_f, float)
    assert isinstance(likelihood.masked_frequency_domain_strain_array, np.ndarray)
    assert isinstance(likelihood.masked_power_spectral_density_array, np.ndarray)
    assert isinstance(likelihood.constant_log_normalization, float)


def test_calibration_factor_invalid(mock_interferometers, mocker):
    """Test that invalid calibration factors raise a ValueError."""
    likelihood = SelfCalibrationLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.array([[np.nan]] * 3))

    with pytest.raises(ValueError, match="Calibration factor contains invalid values"):
        likelihood.log_likelihood()


def test_log_likelihood(mock_interferometers, mocker):
    """Test the log_likelihood calculation with mock calibration factors."""
    likelihood = SelfCalibrationLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.ones((3, len(likelihood.masked_frequency_array))))

    logl = likelihood.log_likelihood()
    assert isinstance(logl, float)
    assert not np.isnan(logl)
    assert not np.isinf(logl)
