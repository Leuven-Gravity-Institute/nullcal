from bilby.gw.detector.calibration import CubicSpline
from nullcal.response_function_generator.base import ResponseFunctionGenerator


class CubicSplineResponseFunctionGenerator(ResponseFunctionGenerator):
    def __init__(self,
                 duration=None,
                 sampling_frequency=None,
                 start_time=0,
                 minimum_frequency=10,
                 maximum_frequency=2048,
                 n_points=10,
                 prefixes=['ET1_', 'ET2_', 'ET3_']):
        super(CubicSplineResponseFunctionGenerator, self).__init__(duration=duration,
                                                                   sampling_frequency=sampling_frequency,
                                                                   start_time=start_time)
        
        self._model_1 = CubicSpline(prefix=prefixes[0],
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    n_points=n_points)
        self._model_2 = CubicSpline(prefix=prefixes[1],
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    n_points=n_points)
        self._model_3 = CubicSpline(prefix=prefixes[2],
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    n_points=n_points)        
        
    def frequency_domain_response_function(self, parameters):
        return (self._model_1.get_calibration_factor(self.frequency_array, **parameters),
                self._model_2.get_calibration_factor(self.frequency_array, **parameters),
                self._model_3.get_calibration_factor(self.frequency_array, **parameters))