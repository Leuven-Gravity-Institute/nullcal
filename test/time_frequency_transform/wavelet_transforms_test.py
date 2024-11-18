from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.noise import noise_from_psd
import numpy as np
import scipy.stats
from nullcal.time_frequency_transform import (transform_wavelet_freq_time,
                                              transform_wavelet_freq_time_partial,
                                              transform_wavelet_freq_time_quadrature,
                                              transform_wavelet_freq,
                                              transform_wavelet_freq_partial,
                                              transform_wavelet_freq_quadrature,
                                              transform_wavelet_time,
                                              transform_wavelet_time_pixel,
                                              inverse_wavelet_time,
                                              inverse_wavelet_freq_time)
import unittest
from unittest import mock
import numpy as np
import tempfile
import json
import os

class TestWaveletTransform(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)
        srate = 128
        self.sine_wave_inj_freq = 32
        seglen = 4
        self.tlen = seglen * srate
        sample_times = np.arange(self.tlen) / srate
        self.df = 4
        self.Nf = int(srate / 2 / self.df)
        self.Nt = int(len(sample_times) / self.Nf)
        self.nx = 4
        self.mult = 32
        self.sine_wave_data = np.sin(2 * np.pi * self.sine_wave_inj_freq * sample_times)

    def test_transform_wavelet_freq_time_of_sine_wave(self):
        data_w = transform_wavelet_freq_time(self.sine_wave_data, self.Nf, self.Nt, self.nx)
        data_q = transform_wavelet_freq_time_quadrature(self.sine_wave_data, self.Nf, self.Nt, self.nx)
        data2 = np.abs(data_w)**2 + np.abs(data_q)**2
        inj_freq_idx = int(self.sine_wave_inj_freq / self.df)
        # Check whether the output peaks at 32Hz for every time bin
        for i in range(self.Nt):
            self.assertEqual(np.argmax(np.abs(data2[i])), inj_freq_idx)

    def test_inverse_wavelet_time(self):
        data_w = transform_wavelet_time(self.sine_wave_data, self.Nf, self.Nt, self.nx, self.mult)
        data_rec = inverse_wavelet_time(data_w, self.Nf, self.Nt, self.nx, self.mult)
        self.assertTrue(np.allclose(self.sine_wave_data, data_rec))

    def test_inverse_wavelet_freq_time(self):
        data_w = transform_wavelet_freq_time(self.sine_wave_data, self.Nf, self.Nt, self.nx)
        data_rec = inverse_wavelet_freq_time(data_w, self.Nf, self.Nt, self.nx)
        self.assertTrue(np.allclose(self.sine_wave_data, data_rec))

    def test_transform_wavelet_freq_standard_gaussian(self):
        seed = 12
        seglen = 16
        srate = 4096
        tlen = seglen * srate
        delta_f = 1 / seglen
        flen = tlen // 2 + 1
        freq_low = 50
        tf_df = 16
        Nf = int(srate / 2 / tf_df)
        Nt = int(tlen / Nf)
        nx = 4
        freq_low_idx = int(np.ceil(freq_low / tf_df))

        psd = EinsteinTelescopeP1600143(flen, delta_f, freq_low)

        noise = noise_from_psd(tlen, 1. / srate, psd, seed=seed)
        # Whiten the noise frequency series.
        whitened_noise_freq = np.divide((np.fft.rfft(noise.numpy()) / srate), np.sqrt(psd.numpy() / 2 / delta_f), where=psd != 0)
        # Transform to time frequency domain
        whitened_noise_time_freq = transform_wavelet_freq(whitened_noise_freq, Nf, Nt, nx)
        ks_statistic, p_value = scipy.stats.kstest(whitened_noise_time_freq[:,freq_low_idx:-1].flatten(), 'norm')
        self.assertGreater(p_value, 0.05, "The output does not follow a standard Gaussian distribution.")

    def test_transform_wavelet_time_standard_gaussian(self):
        seed = 12
        seglen = 16
        srate = 4096
        tlen = seglen * srate
        delta_f = 1 / seglen
        flen = tlen // 2 + 1
        freq_low = 50
        tf_df = 16
        Nf = int(srate / 2 / tf_df)
        Nt = int(tlen / Nf)
        freq_low_idx = int(np.ceil(freq_low / tf_df))

        psd = EinsteinTelescopeP1600143(flen, delta_f, freq_low)

        noise = noise_from_psd(tlen, 1. / srate, psd, seed=seed)
        # Whiten the noise frequency series.
        whitened_noise_freq = np.divide((np.fft.rfft(noise.numpy()) / srate), np.sqrt(psd.numpy() / 2 / delta_f), where=psd != 0)
        # Transform the data back to the frequency domain.
        whitened_noise_time = np.fft.irfft(whitened_noise_freq)
        # Transform to time frequency domain
        whitened_noise_time_freq = transform_wavelet_time(whitened_noise_time, Nf, Nt, self.nx, self.mult)
        ks_statistic, p_value = scipy.stats.kstest(whitened_noise_time_freq[:,freq_low_idx:-1].flatten(), 'norm')
        self.assertGreater(p_value, 0.05, "The output does not follow a standard Gaussian distribution.")

    def test_transform_wavelet_time_pixel(self):
        # from bilby.gw.detector import InterferometerList
        # from bilby.gw.waveform_generator import WaveformGenerator
        # from bilby.gw.source import lal_binary_black_hole
        # from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

        # sampling_frequency = 4096
        # duration = 256
        # interferometers = InterferometerList(['ET'])
        # # Set the minimum frequency
        # for interferometer in interferometers:
        #     interferometer.minimum_frequency = 20
        # waveform_generator = WaveformGenerator(duration=duration,
        #                                        sampling_frequency=sampling_frequency,
        #                                        frequency_domain_source_model=lal_binary_black_hole,
        #                                        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        #                                        waveform_arguments={'approximant': 'IMRPhenomXPHM', 'reference_frequency': 50})
        # interferometers.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
        #                                     duration=duration,
        #                                     start_time=0)        
        # signal_parameters = {"mass_1": 1.7,
        #                      "mass_2": 1.7,
        #                      "a_1": 0.0,
        #                      "a_2": 0.0,
        #                      "tilt_1": 0.0,
        #                      "tilt_2": 0.0,
        #                      "phi_12": 0.0,
        #                      "phi_jl": 0.0,
        #                      "luminosity_distance": 500.0,
        #                      "theta_jn": 0.0,
        #                      "psi": 0.0,
        #                      "phase": 0.0,
        #                      "geocent_time": 240,
        #                      "ra": 0.0,
        #                      "dec": 0.0}
        # interferometers.inject_signal(parameters=signal_parameters,
        #                               waveform_generator=waveform_generator)
        # frequency_domain_strain = interferometers[0].frequency_domain_strain.copy()
        # wavelet_df = 16
        # wavelet_Nf = int(sampling_frequency / 2 / wavelet_df)
        # wavelet_Nt = int(sampling_frequency * duration / wavelet_Nf)
        # whitened_wavelet_domain_strain = transform_wavelet_freq(frequency_domain_strain,
        #                                                             wavelet_Nf,
        #                                                             wavelet_Nt,
        #                                                             nx=4)
        # whitened_wavelet_domain_strain_quadrature = transform_wavelet_freq_quadrature(frequency_domain_strain,
        #                                                                                 wavelet_Nf,
        #                                                                                 wavelet_Nt,
        #                                                                                 nx=4)
        # whitened_wavelet_domain_power = whitened_wavelet_domain_strain ** 2 + whitened_wavelet_domain_strain_quadrature ** 2
        # threshold = np.quantile(whitened_wavelet_domain_power, 0.95)
        # time_frequency_filter = whitened_wavelet_domain_power >= threshold
        # time_domain_strain = interferometers[0].time_domain_strain.copy()
        time_domain_strain = np.random.randn(self.tlen)
        time_frequency_filter = np.random.randint(0,2, size=(self.Nt, self.Nf))
        data_1 = transform_wavelet_time(time_domain_strain, self.Nf, self.Nt, self.nx, self.mult)
        data_2 = transform_wavelet_time_pixel(time_domain_strain, self.Nf, self.Nt, self.nx, self.mult, time_frequency_filter)
        self.assertTrue(np.allclose(data_1[time_frequency_filter.astype(bool)], data_2[time_frequency_filter.astype(bool)]))

    def test_transform_wavelet_freq_partial(self):
        seglen = 128
        srate = 4096
        tlen = seglen * srate
        delta_f = 1 / seglen
        flen = tlen // 2 + 1
        freq_low = 50
        tf_df = 16
        Nf = int(srate / 2 / tf_df)
        Nt = int(tlen / Nf)
        time_domain_strain = np.random.randn(tlen)
        frequency_filter = np.ones(Nf)
        frequency_filter[:8192] = 0
        data_1 = transform_wavelet_freq_time(time_domain_strain, Nf, Nt, self.nx)
        data_2 = transform_wavelet_freq_time_partial(time_domain_strain, Nf, Nt, self.nx, frequency_filter)
        self.assertTrue(np.allclose(data_1[:,frequency_filter.astype(bool)], data_2[:,frequency_filter.astype(bool)]))        
        # import time
        # start = time.time()
        # data_1 = transform_wavelet_freq_time(time_domain_strain, Nf, Nt, self.nx)
        # print(time.time() - start)
        # start = time.time()
        # data_2 = transform_wavelet_freq_time_partial(time_domain_strain, Nf, Nt, self.nx, frequency_filter)
        # print(time.time() - start)


if __name__ == '__main__':
    unittest.main()