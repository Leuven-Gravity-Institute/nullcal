from __future__ import annotations

import bilby.core.utils
import numpy as np
import pytest
import scipy.stats
from bilby.gw.detector import InterferometerList

from nullcal.likelihood import ProjectionLikelihood

seed = 12
sampling_frequency = 4096
duration = 16
minimum_frequency = 10
bilby.core.utils.random.seed(seed)


@pytest.fixture
def mock_interferometers():
    """Create a valid InterferometerList with three interferometers."""
    ifos = InterferometerList(['ET'])

    # Set frequency arrays to be identical
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency

    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                       duration=duration)

    return ifos


def test_initialization(mock_interferometers):
    """Test that SelfCalibrationLikelihood initializes correctly."""
    likelihood = ProjectionLikelihood(interferometers=mock_interferometers)

    assert likelihood.interferometers == mock_interferometers
    assert isinstance(likelihood, ProjectionLikelihood)


def test_invalid_interferometer_count():
    """Test that an error is raised when the number of interferometers is not 3."""
    with pytest.raises(ValueError, match="Expected 3"):
        ProjectionLikelihood(InterferometerList(['H1']))


def test_properties(mock_interferometers):
    """Test the properties of SelfCalibrationLikelihood."""
    likelihood = ProjectionLikelihood(mock_interferometers)

    assert isinstance(likelihood.frequency_array, np.ndarray)
    assert isinstance(likelihood.frequency_mask, np.ndarray)
    assert isinstance(likelihood.masked_frequency_array, np.ndarray)
    assert isinstance(likelihood.delta_f, float)
    assert isinstance(likelihood.masked_frequency_domain_strain_array, np.ndarray)
    assert isinstance(likelihood.masked_power_spectral_density_array, np.ndarray)


def test_calibration_factor_invalid(mock_interferometers, mocker):
    """Test that invalid calibration factors raise a ValueError."""
    likelihood = ProjectionLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.array([[np.nan]] * 3))

    with pytest.raises(ValueError, match="Calibration factor contains invalid values"):
        likelihood.log_likelihood()


def test_log_likelihood(mock_interferometers, mocker):
    """Test the log_likelihood calculation with mock calibration factors."""
    likelihood = ProjectionLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.ones((3, len(likelihood.masked_frequency_array))))

    logl = likelihood.log_likelihood()
    assert isinstance(logl, float)
    assert not np.isnan(logl)
    assert not np.isinf(logl)


def test_log_likelihood_scale_invariance(mock_interferometers, mocker):
    """Test the log_likelihood calculation with mock calibration factors."""
    likelihood = ProjectionLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.ones((3, len(likelihood.masked_frequency_array))))
    logl_1 = likelihood.log_likelihood()

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.ones((3, len(likelihood.masked_frequency_array)))*2.)
    logl_2 = likelihood.log_likelihood()

    assert logl_1 == logl_2


def test_noise_log_likelihood(mock_interferometers, mocker):
    """Test the noise_log_likelihood calculation.
    It should return the same value in log_likelihood when the calibration factor is 1.
    """
    likelihood = ProjectionLikelihood(mock_interferometers)

    mocker.patch.object(likelihood, '_get_calibration_factor_from_parameters', return_value=np.ones((3, len(likelihood.masked_frequency_array))))
    logl = likelihood.log_likelihood()
    noise_logl = likelihood.noise_log_likelihood()

    assert logl == noise_logl


def test_null_energy_significance(mock_interferometers):
    """Test the function that computes the null energy significance.
    """
    likelihood = ProjectionLikelihood(mock_interferometers)
    null_energy_significance = likelihood._compute_uncalibrated_null_energy_significance()
    k_low = int(minimum_frequency / likelihood.delta_f)

    # If the computation is correct, the significance should follow the uniform distribution.
    res = scipy.stats.kstest(null_energy_significance[k_low:-1], cdf='uniform')

    assert res.pvalue >= 0.05
