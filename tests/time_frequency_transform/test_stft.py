from __future__ import annotations

import bilby
import numpy as np
import scipy.signal
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

# Assuming the stft function is in a module named stft_module
from nullcal.time_frequency_transform.stft import stft


def test_stft_whitening_1():
    """Test that whitened STFT coefficients of ET noise follow a standard Gaussian distribution."""
    # Set random seed for reproducibility
    seed = 12
    bilby.core.utils.random.seed(seed)

    # Parameters
    duration = 16
    sampling_frequency = 2048
    frequency_resolution = 16
    window_size = int(sampling_frequency / frequency_resolution)
    minimum_frequency = 10
    maximum_frequency = sampling_frequency / 2

    # Generate Hann window
    window = np.hanning(window_size)

    # Generate noise using Bilby's ET PSD
    interferometers = bilby.gw.detector.InterferometerList(['ET'])

    for interferometer in interferometers:
        interferometer.minimum_frequency = minimum_frequency / 2
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=0,
    )
    alpha = 0.1
    window = tukey(window_size, alpha=alpha)
    window /= np.sqrt(np.mean(window**2))
    # Compute STFT
    # stft_matrix = stft(
    #     data=interferometers[0].time_domain_strain,
    #     sampling_frequency=sampling_frequency,
    #     frequency_resolution=frequency_resolution,
    #     window=window
    # )
    frequency_mask = (interferometers[0].frequency_array > minimum_frequency) & (interferometers[0].frequency_array < maximum_frequency)
    whitened_frequency_domain_strain = np.divide(interferometers[0].frequency_domain_strain,
                                                 np.sqrt(interferometers[0].power_spectral_density_array / 2 * duration),
                                                 out=np.zeros_like(interferometers[0].frequency_domain_strain))
    print(np.var(whitened_frequency_domain_strain[frequency_mask]))
    # f, t, stft_matrix = scipy.signal.stft(interferometers[0].time_domain_strain,
    #                                 fs=sampling_frequency,
    #                                 nperseg=window_size,
    #                                 noverlap=0,
    #                                 scaling='psd')
    # stft_matrix = stft_matrix.T

    # # Get frequency bins for STFT
    # stft_freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_frequency)

    # # Interpolate PSD to match STFT frequency bins
    # psd_at_freqs = interferometers[0].power_spectral_density.power_spectral_density_interpolated(
    #     stft_freqs)

    # # Whiten the STFT coefficients
    # # PSD is two-sided, STFT is one-sided, so scale by sqrt(PSD/2)
    # whitening_factor = np.sqrt(
    #     psd_at_freqs / 2.0 / frequency_resolution)[None, :]
    # whitened_stft = np.divide(stft_matrix, whitening_factor,
    #                           where=whitening_factor != 0, out=np.zeros_like(stft_matrix))
    # frequency_mask = (stft_freqs > minimum_frequency) & (stft_freqs < maximum_frequency)
    # print(np.var(whitened_stft[:, frequency_mask]))
    # print(0 in whitened_stft[:, frequency_mask])


def test_stft_whitening_2():
    """Test that whitened STFT coefficients of ET noise follow a standard Gaussian distribution."""
    # Set random seed for reproducibility
    seed = 1
    bilby.core.utils.random.seed(seed)

    # Parameters
    duration = 16
    sampling_frequency = 2048
    frequency_resolution = 16
    window_size = int(sampling_frequency / frequency_resolution)
    minimum_frequency = 10
    maximum_frequency = sampling_frequency / 2

    # Generate Hann window
    window = np.hanning(window_size)

    # Generate noise using Bilby's ET PSD
    interferometers = bilby.gw.detector.InterferometerList(['ET'])

    for interferometer in interferometers:
        interferometer.minimum_frequency = minimum_frequency / 2
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=0,
    )
    alpha = 0.1
    window = tukey(window_size, alpha=alpha)
    window /= np.sqrt(np.mean(window**2))
    # Compute STFT
    # stft_matrix = stft(
    #     data=interferometers[0].time_domain_strain,
    #     sampling_frequency=sampling_frequency,
    #     frequency_resolution=frequency_resolution,
    #     window=window
    # )

    f, t, stft_matrix = scipy.signal.stft(interferometers[0].time_domain_strain,
                                    fs=sampling_frequency,
                                    nperseg=window_size,
                                    noverlap=0,
                                    scaling='psd')
    stft_matrix = stft_matrix.T

    # Get frequency bins for STFT
    stft_freqs = np.fft.rfftfreq(window_size, d=1.0 / sampling_frequency)

    # Interpolate PSD to match STFT frequency bins
    psd_at_freqs = interferometers[0].power_spectral_density.power_spectral_density_interpolated(
        stft_freqs)

    # Whiten the STFT coefficients
    # PSD is two-sided, STFT is one-sided, so scale by sqrt(PSD/2)
    whitening_factor = np.sqrt(
        psd_at_freqs / 2.0 / frequency_resolution)[None, :]
    whitened_stft = np.divide(stft_matrix, whitening_factor,
                              where=whitening_factor != 0, out=np.zeros_like(stft_matrix))
    frequency_mask = (stft_freqs > minimum_frequency) & (stft_freqs < maximum_frequency)
