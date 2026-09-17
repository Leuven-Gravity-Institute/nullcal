"""Tests for the temporary bilby-backed ``CalibrationPriorDict`` adapter."""

from __future__ import annotations

import bilby
import pytest
from bilby.core.prior import Uniform
from bilby.gw.prior import CalibrationPriorDict as BilbyCalibrationPriorDict

from nullcal.prior import CalibrationPriorDict


@pytest.fixture
def calibration_prior():
    """A minimal spline calibration prior, in the form bilby names its parameters."""
    return CalibrationPriorDict(
        dictionary={
            "recalib_ET1_amplitude_0": Uniform(-0.2, 0.2, name="recalib_ET1_amplitude_0"),
            "recalib_ET1_phase_0": Uniform(-0.2, 0.2, name="recalib_ET1_phase_0"),
        }
    )


@pytest.mark.unit
def test_is_a_bilby_calibration_prior_dict(calibration_prior):
    """The adapter retains Bilby's calibration-prior and prior-dictionary interfaces."""
    assert isinstance(calibration_prior, BilbyCalibrationPriorDict)
    assert isinstance(calibration_prior, bilby.core.prior.PriorDict)


@pytest.mark.unit
def test_behaves_as_a_prior_dict(calibration_prior):
    """Subclassing must not disturb the sampling and evaluation the dictionary is used for."""
    sample = calibration_prior.sample()

    assert set(sample) == {"recalib_ET1_amplitude_0", "recalib_ET1_phase_0"}
    assert calibration_prior.ln_prob(sample) > -float("inf")


@pytest.mark.unit
def test_validate_prior_bypass_is_absent(calibration_prior):
    """Calibration priors must not advertise an unconditional validation hook."""
    assert not hasattr(BilbyCalibrationPriorDict, "validate_prior")
    assert "validate_prior" not in CalibrationPriorDict.__dict__
    assert not hasattr(calibration_prior, "validate_prior")


@pytest.mark.unit
def test_accepts_an_empty_construction():
    """An empty prior dictionary remains a valid adapter construction."""
    assert len(CalibrationPriorDict()) == 0


@pytest.mark.unit
def test_round_trips_through_a_prior_file(tmp_path, calibration_prior):
    """The inherited file-loading path preserves the adapter type."""
    prior_file = tmp_path / "calibration.prior"
    calibration_prior.to_file(outdir=str(tmp_path), label="calibration")

    reloaded = CalibrationPriorDict(filename=str(prior_file))

    assert set(reloaded) == set(calibration_prior)
    assert isinstance(reloaded, CalibrationPriorDict)
