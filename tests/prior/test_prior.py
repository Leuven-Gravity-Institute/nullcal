"""Tests for ``CalibrationPriorDict``.

The class exists for one reason: to disable ``bilby_pipe``'s calibration-prior validation. That
check asserts the spline prior is fine enough to resolve calibration structure across the analysis
band, which is an assumption about *detector* calibration priors and not about the self-calibration
priors this package samples. The override is therefore deliberate, and the tests below pin it as
deliberate -- including the fact that ``validate_prior`` is an adapter method absent from the bilby
base class, so a future reader can see which integration contract the subclass adds.
"""

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
    """``bilby_pipe`` dispatches on this type, so the inheritance is part of the contract."""
    assert isinstance(calibration_prior, BilbyCalibrationPriorDict)
    assert isinstance(calibration_prior, bilby.core.prior.PriorDict)


@pytest.mark.unit
def test_behaves_as_a_prior_dict(calibration_prior):
    """Subclassing must not disturb the sampling and evaluation the dictionary is used for."""
    sample = calibration_prior.sample()

    assert set(sample) == {"recalib_ET1_amplitude_0", "recalib_ET1_phase_0"}
    assert calibration_prior.ln_prob(sample) > -float("inf")


@pytest.mark.unit
def test_validate_prior_returns_true_without_inspecting_the_prior(calibration_prior):
    """The override accepts anything, by design.

    Every argument is ignored, including ones that would make the base implementation reject the
    prior. Asserting across a spread of arguments -- rather than one convenient set -- is what shows
    the bypass is unconditional rather than accidentally passing for the values tried.
    """
    for duration, minimum_frequency in ((4.0, 20.0), (0.0, 0.0), (-1.0, 1e6)):
        assert calibration_prior.validate_prior(duration, minimum_frequency) is True

    assert calibration_prior.validate_prior(4.0, 20.0, N=1, error=True, warning=True) is True
    assert calibration_prior.validate_prior(4.0, 20.0, N=10**6, error=False, warning=False) is True


@pytest.mark.unit
def test_validate_prior_is_an_adapter_method_absent_from_the_bilby_base(calibration_prior):
    """The subclass adds the validation hook consumed by the surrounding pipeline.

    Bilby's calibration prior dictionary does not define ``validate_prior``. Pinning both sides of
    that boundary prevents this adapter from being mistaken for an override of inherited validation
    logic, while the preceding test specifies the hook's unconditional-acceptance behaviour.
    """
    assert "validate_prior" not in BilbyCalibrationPriorDict.__dict__
    assert "validate_prior" in CalibrationPriorDict.__dict__
    assert calibration_prior.validate_prior(4.0, 20.0) is True


@pytest.mark.unit
def test_accepts_an_empty_construction():
    """``bilby_pipe`` constructs the dictionary before filling it."""
    assert len(CalibrationPriorDict()) == 0


@pytest.mark.unit
def test_round_trips_through_a_prior_file(tmp_path, calibration_prior):
    """Construction from ``filename`` is the path ``bilby_pipe`` uses, so it is exercised here."""
    prior_file = tmp_path / "calibration.prior"
    calibration_prior.to_file(outdir=str(tmp_path), label="calibration")

    reloaded = CalibrationPriorDict(filename=str(prior_file))

    assert set(reloaded) == set(calibration_prior)
    assert isinstance(reloaded, CalibrationPriorDict)
    assert reloaded.validate_prior(4.0, 20.0) is True
