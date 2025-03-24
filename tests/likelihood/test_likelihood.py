from __future__ import annotations

import pytest
from bilby.gw.detector import InterferometerList

from nullcal.likelihood import SelfCalibrationLikelihood


def test_initialization():
    ifos = InterferometerList(['ET'])
    for ifo in ifos:
        ifo.minimum_frequency = 10
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=2048,
                                                       duration=16,
                                                       start_time=0)
    likelihood = SelfCalibrationLikelihood(ifos)
