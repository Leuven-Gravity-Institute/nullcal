import numpy as np
from nullcal.recalibration.base import RecalibrationGenerator


class IdentityRecalibrationGenerator(RecalibrationGenerator):
    def __init__(self, *args, **kwargs):
        pass

    def get_calibration_factor(self, frequency_array, parameters):
        return np.ones(len(frequency_array))