import numpy as np
from bilby.gw.detector.networks import TriangularInterferometer
from ..utils import optimal_uncorrelated_null_stream_snr_squared


def optimal_null_stream_snr_squared(self, signals):
    return optimal_uncorrelated_null_stream_snr_squared(
        signals=signals,
        power_spectral_densities=np.array(
            ifo.power_spectral_density_array for ifo in self
        )
        duration=self[0].duration
    )


TriangularInterferometer.optimal_null_stream_snr_squared = \
    optimal_null_stream_snr_squared
