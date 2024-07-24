from bilby.core.utils import PropertyAccessor
from bilby.core.series import CoupledTimeAndFrequencySeries


class ResponseFunctionGenerator:
    duration = PropertyAccessor('_times_and_frequencies', 'duration')
    sampling_frequency = PropertyAccessor('_times_and_frequencies', 'sampling_frequency')
    start_time = PropertyAccessor('_times_and_frequencies', 'start_time')
    frequency_array = PropertyAccessor('_times_and_frequencies', 'frequency_array')
    time_array = PropertyAccessor('_times_and_frequencies', 'time_array')
    def __init__(self,
                 duration=None,
                 sampling_frequency=None,
                 start_time=0):
        self._times_and_frequencies = CoupledTimeAndFrequencySeries(duration=duration,
                                                                    sampling_frequency=sampling_frequency,
                                                                    start_time=start_time)

    def frequency_domain_response_function(self, parameters):
        raise NotImplementedError('Not implemented.')