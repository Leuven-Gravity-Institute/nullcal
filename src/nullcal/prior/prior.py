"""
Prior classes.
"""

from __future__ import annotations

from bilby.gw.prior import CalibrationPriorDict as BilbyCalibrationPriorDict


class CalibrationPriorDict(BilbyCalibrationPriorDict):
    """A temporary bilby-backed prior dictionary for calibration parameters.

    ``bilby_pipe`` conditionally calls ``validate_prior(duration,
    minimum_frequency)`` to check that a compact-binary signal fits inside the
    analysis segment. Bilby's calibration-only prior has no source-duration
    parameters and exposes no such hook. This subclass deliberately follows
    that contract instead of advertising the former unconditional ``True``
    bypass; bilby_pipe therefore skips a check that is inapplicable here.
    """

    def __init__(self, dictionary: dict | None = None, filename: str | None = None):
        """The prior class for self-calibration.

        Args:
            dictionary (dict, optional): See superclass. Defaults to None.
            filename (str, optional): See superclass. Defaults to None.
        """
        super().__init__(dictionary=dictionary, filename=filename)
