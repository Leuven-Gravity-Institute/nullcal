import numpy as np


def whiten_frequency_domain_strain_helper(frequency_domain_strain, power_spectral_density_array, delta_f):
    return np.divide(frequency_domain_strain,
                     np.sqrt(power_spectral_density_array / (2 * delta_f)),
                     np.zeros_like(frequency_domain_strain),
                     where=power_spectral_density_array!=0.)

def whiten_frequency_domain_strain(interferometer):
    delta_f = interferometer.frequency_array[1] - interferometer.frequency_array[0]
    return whiten_frequency_domain_strain_helper(interferometer.frequency_domain_strain,
                                                 interferometer.power_spectral_density_array,
                                                 delta_f)