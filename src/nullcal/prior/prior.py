from __future__ import annotations

from bilby.gw.prior import CalibrationPriorDict as BilbyCalibrationPriorDict


class CalibrationPriorDict(BilbyCalibrationPriorDict):
    def __init__(self, dictionary: dict=None, filename: str=None):
        """The prior class for self-calibration.

        Args:
            dictionary (dict, optional): See superclass. Defaults to None.
            filename (str, optional): See superclass. Defaults to None.
        """
        super().__init__(dictionary=dictionary, filename=filename)

    def validate_prior(self, duration, minimum_frequency, N=1000, error=True, warning=False):
        """This is a placeholder method to bypass the checking in bilby_pipe.
        """
        return True
