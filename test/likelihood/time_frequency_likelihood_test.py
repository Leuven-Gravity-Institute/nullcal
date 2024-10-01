import unittest
import numpy as np
import bilby
import matplotlib.pyplot as plt
import scipy.stats
from nullcal.likelihood.time_frequency_likelihood import SelfRecalibrationProjectorTimeFrequencyLikelihood
from nullcal.clustering.single import single_clustering_by_quantile
from nullcal.recalibration.identity import IdentityRecalibrationGenerator


INJECTION_PARAMETERS = dict(
                            mass_1=60.0,
                            mass_2=60.0,
                            luminosity_distance=0.1,
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

def get_zero_noise_interferometers(sampling_frequency,
                                   duration,                                   
                                   start_time,
                                   waveform_generator,
                                   injection_parameters):
    interferometers = bilby.gw.detector.InterferometerList(['ET'])
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
                                   injection_parameters):
    interferometers = bilby.gw.detector.InterferometerList(['ET'])
    for ifo in interferometers:
        ifo.power_spectral_density.psd_array = np.full((len(ifo.power_spectral_density.psd_array)), 1e-48)
    for ifo in interferometers:
        ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                    duration=duration,
                                                    start_time=start_time)
    interferometers.inject_signal(parameters=injection_parameters,
                                  waveform_generator=waveform_generator)
    return interferometers

def get_noisy_interferometers(sampling_frequency,
                              duration,                                   
                              start_time,
                              waveform_generator,
                              injection_parameters):
    interferometers = bilby.gw.detector.InterferometerList(['ET'])
    for ifo in interferometers:
        ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                                duration=duration,
                                                                start_time=start_time)
    interferometers.inject_signal(parameters=injection_parameters,
                                     waveform_generator=waveform_generator)
    return interferometers

def get_noise_only_interferometers(sampling_frequency,
                              duration,                                   
                              start_time,
                              waveform_generator,
                              injection_parameters):
    interferometers = bilby.gw.detector.InterferometerList(['ET'])
    for ifo in interferometers:
        ifo.set_strain_data_from_power_spectral_density(sampling_frequency=sampling_frequency,
                                                                duration=duration,
                                                                start_time=start_time)
    return interferometers


class TestSelfRecalibrationProjectorTimeFrequnecyLikelihood(unittest.TestCase):
    def setUp(self):
        seed = 222
        bilby.core.utils.random.seed(seed)

    def test_log_likelihood_distribution_from_noise(self):
        """Test the distribution of the log likelihood function.
        If the implementation is correct, -2\log\mathcal{L} should follow
        the $\chi^{2}$ distribution.        
        """
        # Define the injection parameters and the setup
        duration = 4.0
        sampling_frequency = 1024.0
        minimum_frequency = 20
        maximum_frequency = None
        wavelet_transform_frequency_resolution = 16
        wavelet_transform_nx = 4
        injection_parameters = dict(
            mass_1=60.0,
            mass_2=60.0,
            luminosity_distance=1000,
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
            waveform_approximant="IMRPhenomD",
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
                                                                    injection_parameters=injection_parameters)
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
        recalibration_generator_1 = IdentityRecalibrationGenerator()
        recalibration_generator_2 = IdentityRecalibrationGenerator()
        recalibration_generator_3 = IdentityRecalibrationGenerator()        
        # plt.imshow(time_frequency_filter.T, aspect='auto', origin='lower', extent=[0, low_noise_interferometers[0].duration, 0, low_noise_interferometers[0].frequency_array[-1]], cmap='viridis')
        # plt.savefig('TF_filter.pdf')
        # Construct the normal interferometers
        nsample = 500
        logl_samples = []
        for i in range(nsample):
            interferometers = get_noise_only_interferometers(sampling_frequency=sampling_frequency,
                                                                        duration=duration,
                                                                        start_time=injection_parameters['geocent_time']-2,
                                                                        waveform_generator=waveform_generator,
                                                                        injection_parameters=injection_parameters)

            likelihood = SelfRecalibrationProjectorTimeFrequencyLikelihood(interferometers=interferometers,
                                                                        recalibration_generator_1=recalibration_generator_1,
                                                                        recalibration_generator_2=recalibration_generator_2,
                                                                        recalibration_generator_3=recalibration_generator_3,
                                                                        time_frequency_filter=time_frequency_filter,
                                                                        wavelet_transform_frequency_resolution=wavelet_transform_frequency_resolution,
                                                                        wavelet_transform_nx=wavelet_transform_nx)
            likelihood.parameters = {}
            logl = likelihood.log_likelihood()
            logl_samples.append(logl*-2)
        ks_statistic, p_value = scipy.stats.kstest(logl_samples, 'chi2', args=(DoF,))
        self.assertGreater(p_value, 0.05, "The -2*logL does not follow the chi-squared distribution.")

    def test_log_likelihood_distribution(self):
        """Test the distribution of the log likelihood function.
        If the implementation is correct, -2\log\mathcal{L} should follow
        the $\chi^{2}$ distribution.        
        """
        # Define the injection parameters and the setup
        duration = 4.0
        sampling_frequency = 1024.0
        minimum_frequency = 20
        maximum_frequency = None
        wavelet_transform_frequency_resolution = 16
        wavelet_transform_nx = 4
        injection_parameters = dict(
            mass_1=60.0,
            mass_2=60.0,
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
            waveform_approximant="IMRPhenomD",
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
                                                                    injection_parameters=injection_parameters)
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
        recalibration_generator_1 = IdentityRecalibrationGenerator()
        recalibration_generator_2 = IdentityRecalibrationGenerator()
        recalibration_generator_3 = IdentityRecalibrationGenerator()        
        # plt.imshow(time_frequency_filter.T, aspect='auto', origin='lower', extent=[0, low_noise_interferometers[0].duration, 0, low_noise_interferometers[0].frequency_array[-1]], cmap='viridis')
        # plt.savefig('TF_filter.pdf')
        # Construct the normal interferometers
        nsample = 500
        logl_samples = []
        for i in range(nsample):
            interferometers = get_noisy_interferometers(sampling_frequency=sampling_frequency,
                                                                        duration=duration,
                                                                        start_time=injection_parameters['geocent_time']-2,
                                                                        waveform_generator=waveform_generator,
                                                                        injection_parameters=injection_parameters)

            likelihood = SelfRecalibrationProjectorTimeFrequencyLikelihood(interferometers=interferometers,
                                                                        recalibration_generator_1=recalibration_generator_1,
                                                                        recalibration_generator_2=recalibration_generator_2,
                                                                        recalibration_generator_3=recalibration_generator_3,
                                                                        time_frequency_filter=time_frequency_filter,
                                                                        wavelet_transform_frequency_resolution=wavelet_transform_frequency_resolution,
                                                                        wavelet_transform_nx=wavelet_transform_nx)
            likelihood.parameters = {}
            logl = likelihood.log_likelihood()
            logl_samples.append(logl*-2)
        print('DoF', DoF)
        print('mean', np.mean(logl_samples))
        print('variance', np.var(logl_samples))
        ks_statistic, p_value = scipy.stats.kstest(logl_samples, 'chi2', args=(DoF,))
        self.assertGreater(p_value, 0.05, "The -2*logL does not follow the chi-squared distribution.")

    def test_log_likelihood_distribution_zero_noise(self):
        """Test the distribution of the log likelihood function.
        If the implementation is correct, -2\log\mathcal{L} should follow
        the $\chi^{2}$ distribution.        
        """
        # Define the injection parameters and the setup
        duration = 4.0
        sampling_frequency = 1024.0
        minimum_frequency = 20
        maximum_frequency = None
        wavelet_transform_frequency_resolution = 16
        wavelet_transform_nx = 4
        injection_parameters = dict(
            mass_1=60.0,
            mass_2=60.0,
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
            waveform_approximant="IMRPhenomD",
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
                                                                    injection_parameters=injection_parameters)
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
        recalibration_generator_1 = IdentityRecalibrationGenerator()
        recalibration_generator_2 = IdentityRecalibrationGenerator()
        recalibration_generator_3 = IdentityRecalibrationGenerator()        
        # plt.imshow(time_frequency_filter.T, aspect='auto', origin='lower', extent=[0, low_noise_interferometers[0].duration, 0, low_noise_interferometers[0].frequency_array[-1]], cmap='viridis')
        # plt.savefig('TF_filter.pdf')
        # Construct the normal interferometers
        nsample = 500
        logl_samples = []
        for i in range(nsample):
            interferometers = get_zero_noise_interferometers(sampling_frequency=sampling_frequency,
                                                                        duration=duration,
                                                                        start_time=injection_parameters['geocent_time']-2,
                                                                        waveform_generator=waveform_generator,
                                                                        injection_parameters=injection_parameters)

            likelihood = SelfRecalibrationProjectorTimeFrequencyLikelihood(interferometers=interferometers,
                                                                        recalibration_generator_1=recalibration_generator_1,
                                                                        recalibration_generator_2=recalibration_generator_2,
                                                                        recalibration_generator_3=recalibration_generator_3,
                                                                        time_frequency_filter=time_frequency_filter,
                                                                        wavelet_transform_frequency_resolution=wavelet_transform_frequency_resolution,
                                                                        wavelet_transform_nx=wavelet_transform_nx)
            likelihood.parameters = {}
            logl = likelihood.log_likelihood()
            logl_samples.append(logl*-2)
        print('No noise')
        print('DoF', DoF)
        print('mean', np.mean(logl_samples))
        print('variance', np.var(logl_samples))
        ks_statistic, p_value = scipy.stats.kstest(logl_samples, 'chi2', args=(DoF,))
        self.assertGreater(p_value, 0.05, "The -2*logL does not follow the chi-squared distribution.")

if __name__ == '__main__':
    unittest.main()