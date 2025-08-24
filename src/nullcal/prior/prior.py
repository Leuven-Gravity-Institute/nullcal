"""
Prior classes.
"""

from __future__ import annotations

from bilby.gw.prior import CalibrationPriorDict as BilbyCalibrationPriorDict


class CalibrationPriorDict(BilbyCalibrationPriorDict):
    """A prior dictionary for calibration parameters."""

    def __init__(self, dictionary: dict | None = None, filename: str | None = None):
        """The prior class for self-calibration.

        Args:
            dictionary (dict, optional): See superclass. Defaults to None.
            filename (str, optional): See superclass. Defaults to None.
        """
        super().__init__(dictionary=dictionary, filename=filename)

    # pylint: disable=unused-argument
    def validate_prior(
        self,
        duration: float,
        minimum_frequency: float,
        N: int = 1000,  # noqa: N803, pylint: disable=invalid-name
        error: bool = True,
        warning: bool = False,
    ):
        """This is a placeholder method to bypass the checking in bilby_pipe."""
        return True
