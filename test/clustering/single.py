import bilby
import numpy as np
import matplotlib.pyplot as plt
from nullcal.clustering.single import single_clustering_by_quantile
from nullcal.clustering.time_frequency_map import construct_time_frequency_map

if __name__ == '__main__':
    duration = 4.0
    sampling_frequency = 1024.0

    injection_parameters = dict(
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

    waveform_arguments = dict(
        waveform_approximant="IMRPhenomD",
        reference_frequency=50.0,
        minimum_frequency=20.0,
    )


    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

    for i in range(len(ifos)):
        ifos[i].power_spectral_density.psd_array = np.full((len(ifos[i].power_spectral_density.psd_array)), 1e-40)
    ifos.set_strain_data_from_power_spectral_densities(
    # ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )

    injection_polarizations_fd = waveform_generator.frequency_domain_strain(injection_parameters)
    # injection_polarizations_fd['x'] = injection_polarizations_fd['plus']
    # injection_polarizations_fd['y'] = injection_polarizations_fd['cross']

    ifos.inject_signal(
        parameters=injection_parameters,
        injection_polarizations=injection_polarizations_fd
    )
    time_frequency_map = construct_time_frequency_map(ifos, frequency_resolution=16)
    plt.imshow(time_frequency_map.T, aspect='auto', origin='lower', extent=[0, ifos[0].duration, 0, ifos[0].frequency_array[-1]], cmap='viridis')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Energy')
    plt.savefig('single_time_frequency_map.pdf')
    plt.close()


    filter = single_clustering_by_quantile(ifos, frequency_resolution=16, quantile=0.95, padding_freq=0, padding_time=0)

    plt.imshow(filter.T, aspect='auto', origin='lower', extent=[0, ifos[0].duration, 0, ifos[0].frequency_array[-1]], cmap='viridis')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Energy')
    plt.savefig('single_cluster.pdf')
    plt.close()