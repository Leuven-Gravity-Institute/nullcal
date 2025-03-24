import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList


class SelfCalibrationLikelihood(Likelihood):
    def __init__(self, interferometers):
        self._interferometers = InterferometerList(interferometers)
        if len(self.interferometers) != 3:
            raise ValueError(f'The number of interferometers is {len(self.interferometers)}. Expected 3.')
        self._rotation_matrix = np.array([[-np.sqrt(6)/6, np.sqrt(6)/3, -np.sqrt(6)/6],
                                          [-np.sqrt(2)/2, 0, np.sqrt(2)/2],
                                          [np.sqrt(3)/3, np.sqrt(3)/3, np.sqrt(3)/3]])

    @property
    def interferometers(self):
        return self._interferometers

    @property
    def frequency_array(self):
        return self._frequency_array

    @property
    def frequency_mask(self):
        return self._frequency_mask

    @property
    def masked_frequency_array(self):
        return self._masked_frequency_array

    @property
    def delta_f(self):
        return self._delta_f

    @property
    def masked_frequency_domain_strain_array(self):
        if self._masked_frequency_domain_strain_array is None:
            self._masked_frequency_domain_strain_array = np.array([
                ifo.frequency_domain_strain[self._frequency_mask] for ifo in self._interferometers
            ])
        return self._masked_frequency_domain_strain_array

    @property
    def masked_power_spectral_density_array(self):
        if self._masked_power_spectral_density_array is None:
            self._masked_power_spectral_density_array = np.array([
                ifo.power_spectral_density_array[self._frequency_mask] for ifo in self._interferometers
            ])
        return self._masked_power_spectral_density_array