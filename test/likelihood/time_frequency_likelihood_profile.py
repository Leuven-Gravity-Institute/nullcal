
if __name__ == '__main__':
    import bilby
    from nullcal.likelihood.time_frequency_likelihood import SelfRecalibrationProjectorTimeFrequencyLikelihood
    from nullcal.clustering.single import single_clustering_by_quantile
    from nullcal.recalibration.cubic_spline import CubicSplineRecalibrationGenerator
    import numpy as np
    import cProfile


    def get_zero_noise_interferometers(sampling_frequency,
                                    duration,                                   
                                    start_time,
                                    waveform_generator,
                                    injection_parameters,
                                    minimum_frequency,
                                    maximum_frequency):
        interferometers = bilby.gw.detector.InterferometerList(['ET'])
        for interferometer in interferometers:
            interferometer.minimum_frequency = minimum_frequency
            interferometer.maximum_frequency = maximum_frequency
        interferometers.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                        duration=duration,
                                                        start_time=start_time)
        interferometers.inject_signal(parameters=injection_parameters,
                                    waveform_generator=waveform_generator)
        return interferometers

    def get_low_noise_interferometers(sampling_frequency,
                                    duration,                                   
                                    start_time,
                                    waveform_generator,
                                    injection_parameters,
                                    minimum_frequency,
                                    maximum_frequency):
        interferometers = bilby.gw.detector.InterferometerList(['ET'])
        for interferometer in interferometers:
            interferometer.minimum_frequency = minimum_frequency
            interferometer.maximum_frequency = maximum_frequency
        for ifo in interferometers:
            ifo.power_spectral_density.psd_array = np.full((len(ifo.power_spectral_density.psd_array)), 1e-48)
        for ifo in interferometers:
            ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                        duration=duration,
                                                        start_time=start_time)
        interferometers.inject_signal(parameters=injection_parameters,
                                    waveform_generator=waveform_generator)
        return interferometers

    duration = 256.0
    sampling_frequency = 4096
    minimum_frequency = 20
    maximum_frequency = 2048
    wavelet_transform_frequency_resolution = 16
    wavelet_transform_nx = 4
    injection_parameters = dict(
        mass_1=4,
        mass_2=4,
        luminosity_distance=4000,
        psi=0,
        phase=0,
        a_1=0,
        a_2=0,
        tilt_1=0,
        tilt_2=0,
        theta_jn=0,
        geocent_time=1126259642.413,
        ra=2.375,
        dec=-1.2108
    )
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    # Get the zero noise interferometers for clustering.
    low_noise_interferometers = get_low_noise_interferometers(sampling_frequency=sampling_frequency,
                                                                duration=duration,
                                                                start_time=injection_parameters['geocent_time']-2,
                                                                waveform_generator=waveform_generator,
                                                                injection_parameters=injection_parameters,
                                                                minimum_frequency=minimum_frequency,
                                                                maximum_frequency=maximum_frequency)
    # Construct the time-frequency filter
    time_frequency_filter = single_clustering_by_quantile(interferometers=low_noise_interferometers,
                                                            frequency_resolution=wavelet_transform_frequency_resolution,
                                                            nx=wavelet_transform_nx,
                                                            quantile=0.99,
                                                            padding_time=0.05,
                                                            padding_freq=0,
                                                            minimum_frequency=minimum_frequency,
                                                            maximum_frequency=maximum_frequency)
    time_frequency_filter = time_frequency_filter.astype(bool)
    DoF = np.sum(time_frequency_filter)
    # Construct the recalibration generators
    recalibration_generator_1 = CubicSplineRecalibrationGenerator(prefix='E1_', minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, n_points=10)
    recalibration_generator_2 = CubicSplineRecalibrationGenerator(prefix='E2_', minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, n_points=10)
    recalibration_generator_3 = CubicSplineRecalibrationGenerator(prefix='E3_', minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, n_points=10)

    interferometers = get_zero_noise_interferometers(sampling_frequency=sampling_frequency,
                                                                duration=duration,
                                                                start_time=injection_parameters['geocent_time']-2,
                                                                waveform_generator=waveform_generator,
                                                                injection_parameters=injection_parameters,
                                                                minimum_frequency=minimum_frequency,
                                                                maximum_frequency=maximum_frequency)        
    likelihood = SelfRecalibrationProjectorTimeFrequencyLikelihood(interferometers=interferometers,
                                                                recalibration_generator_1=recalibration_generator_1,
                                                                recalibration_generator_2=recalibration_generator_2,
                                                                recalibration_generator_3=recalibration_generator_3,
                                                                time_frequency_filter=time_frequency_filter,
                                                                wavelet_transform_frequency_resolution=wavelet_transform_frequency_resolution,
                                                                wavelet_transform_nx=wavelet_transform_nx)
    likelihood.parameters = {}
    for ifo in ['E1_', 'E2_', 'E3_']:
        for n in range(10):
            likelihood.parameters[f'{ifo}amplitude_{n}'] = 1.
            likelihood.parameters[f'{ifo}phase_{n}'] = 0.


    profiler = cProfile.Profile()
    likelihood.log_likelihood()
    profiler.enable()
    likelihood.log_likelihood()
    profiler.disable()
    profiler.dump_stats('time_frequency_likelihood_profile.prof')