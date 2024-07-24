from nullcal.response_function_generator.base import ResponseFunctionGenerator


class TriangularResponseFunctionGenerator(ResponseFunctionGenerator):
    def __init__(self,
                 duration=None,
                 sampling_frequency=None,
                 start_time=0,
                 frequency_domain_calibration_model=None):
        super(TriangularResponseFunctionGenerator, self).__init__(duration=duration,
                                                         sampling_frequency=sampling_frequency,
                                                         start_time=start_time)
        self.frequency_domain_calibration_model = frequency_domain_calibration_model
        
    def frequency_domain_response_function(self, parameters):
        splitted_parameters = {
            'E1': {},
            'E2': {},
            'E3': {},
        }
        for key, value in parameters.items():
            splitted_parameters[key[:2]][key[3:]] = value
        return (self.frequency_domain_calibration_model(self.frequency_array, **splitted_parameters['E1']),
                self.frequency_domain_calibration_model(self.frequency_array, **splitted_parameters['E2']),
                self.frequency_domain_calibration_model(self.frequency_array, **splitted_parameters['E3']))