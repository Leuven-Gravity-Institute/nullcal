"""Tests for ``CalibrationPriorDict``.

Bilby's calibration prior dictionary has no ``validate_prior`` method in its MRO. This subclass
adds a named adapter hook that always returns ``True``. That shape is compatible with
``bilby_pipe``'s ``hasattr``-based validation dispatch, but it does not override or disable inherited
validation. The tests pin both sides of that boundary and the adapter's unconditional behaviour.
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
def test_validate_prior_returns_true_without_inspecting_the_prior(calibration_prior):
    """The added hook accepts anything, by design.

    Every argument is ignored. Asserting across a spread of arguments -- rather than one convenient
    set -- is what shows the result is unconditional rather than accidental for one input.
    """
    for duration, minimum_frequency in ((4.0, 20.0), (0.0, 0.0), (-1.0, 1e6)):
        assert calibration_prior.validate_prior(duration, minimum_frequency) is True

    assert calibration_prior.validate_prior(4.0, 20.0, N=1, error=True, warning=True) is True
    assert calibration_prior.validate_prior(4.0, 20.0, N=10**6, error=False, warning=False) is True


@pytest.mark.unit
def test_validate_prior_is_an_adapter_method_absent_from_the_bilby_base(calibration_prior):
    """The subclass adds an optional validation hook that the Bilby base does not provide.

    Bilby's calibration prior dictionary does not define ``validate_prior``. Pinning both sides of
    that boundary prevents this adapter from being mistaken for an override of inherited validation
    logic, while the preceding test specifies the hook's unconditional-acceptance behaviour.
    """
    assert not hasattr(BilbyCalibrationPriorDict, "validate_prior")
    assert "validate_prior" in CalibrationPriorDict.__dict__
    assert calibration_prior.validate_prior(4.0, 20.0) is True


@pytest.mark.unit
def test_accepts_an_empty_construction():
    """An empty prior dictionary remains a valid adapter construction."""
    assert len(CalibrationPriorDict()) == 0


@pytest.mark.unit
def test_round_trips_through_a_prior_file(tmp_path, calibration_prior):
    """The inherited file-loading path preserves the adapter type and hook."""
    prior_file = tmp_path / "calibration.prior"
    calibration_prior.to_file(outdir=str(tmp_path), label="calibration")

    reloaded = CalibrationPriorDict(filename=str(prior_file))

    assert set(reloaded) == set(calibration_prior)
    assert isinstance(reloaded, CalibrationPriorDict)
    assert reloaded.validate_prior(4.0, 20.0) is True
