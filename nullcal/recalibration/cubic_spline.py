import numpy as np
from scipy.interpolate import interp1d
from nullcal.recalibration.base import RecalibrationGenerator


class CubicSplineRecalibrationGenerator(RecalibrationGenerator):
    def __init__(self,
                 prefix,
                 minimum_frequency,
                 maximum_frequency,
                 n_points):
        self._prefix = prefix
        self._n_points = n_points
        self._node_frequency_array = 10**np.linspace(
            np.log10(minimum_frequency), np.log10(maximum_frequency), n_points)

    def get_calibration_factor(self, frequency_array, parameters):
        node_amplitude = np.array([parameters[f"{self._prefix}amplitude_{i}"] for i in range(self._n_points)])
        node_phase = np.array([parameters[f"{self._prefix}phase_{i}"] for i in range(self._n_points)])
        node_calibiration = node_amplitude * np.exp(1.j * node_phase)
        return interp1d(self._node_frequency_array, node_calibiration, kind='cubic',
        bounds_error=False, fill_value=1)(frequency_array)