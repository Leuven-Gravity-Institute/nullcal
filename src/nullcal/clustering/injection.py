"""A submodule for clustering using waveform injections."""

import logging

import numpy as np
import pandas as pd
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

from ..time_frequency_transform.wavelet_transforms import WaveletTransform
from .base import Clustering
from .single import single_clustering_by_threshold

logger = logging.getLogger("nullcal")


class InjectionClustering(Clustering):
    """Clustering method using waveform injections."""

    def __init__(
        self,
        time_frequency_transform: WaveletTransform,
        interferometers: InterferometerList,
        waveform_generator: WaveformGenerator,
        parameter_file: str,
        threshold: float,
        enforce_signal_duration: bool = False,
    ):
        """Clustering method using waveform injections.

        Args:
            time_frequency_transform (WaveletTransform): A WaveletTransform instance.
            interferometers (InterferometerList): An InterferometerList instance.
            waveform_generator (WaveformGenerator): A WaveformGenerator instance.
            parameter_file (str): Path to a file that contains the list of
                waveform parameters.
            threshold (float): The threshold to select time-frequency pixels.
            enforce_signal_duration (bool, optional): If True, raise error if the duration of signal
                is longer than the data. Defaults to False.
        """
        super().__init__(time_frequency_transform=time_frequency_transform)
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self.parameter_file = parameter_file
        self.threshold = threshold
        self.enforce_signal_duration = enforce_signal_duration

    @property
    def time_frequency_filter(self) -> np.ndarray:
        """Get the time-frequency filter.

        Returns:
            np.ndarray: Time-frequency filter.
        """
        if self._time_frequency_filter is None:
            # Get the shape of the time-frequency domain data.
            n_t, n_f = self.shape

            # Initialize the time-frequency filter to all False.
            time_frequency_filter = np.full((n_t, n_f), False, dtype=bool)

            # Get the name of interferometers.
            if (
                len(self.interferometers) == 3  # noqa: PLR2004
                and self.interferometers[0].name == "ET1"
                and self.interferometers[1].name == "ET2"
                and self.interferometers[2].name == "ET3"
            ):
                interferometers_name = ["ET"]
            else:
                interferometers_name = [ifo.name for ifo in self.interferometers]
            # Read the parameter file into a DataFrame.
            parameters_df = pd.read_csv(self.parameter_file)
            for i in range(len(parameters_df)):
                parameters = parameters_df.iloc[i].to_dict()

                logger.info("Generating zero-noise injection data.")

                # Copy the power spectral density
                interferometers = InterferometerList(interferometers_name)
                for j, ifo in enumerate(interferometers):
                    ifo.power_spectral_density = self.interferometers[j].power_spectral_density
                    ifo.calibration_model = self.interferometers[j].calibration_model
                interferometers.set_strain_data_from_zero_noise(
                    sampling_frequency=self.interferometers[0].sampling_frequency,
                    duration=self.interferometers[0].duration,
                    start_time=self.interferometers[0].start_time,
                )
                interferometers.inject_signal(
                    waveform_generator=self.waveform_generator,
                    parameters=parameters,
                    raise_error=self.enforce_signal_duration,
                )
                logger.info("Clustering started.")
                _time_frequency_filter = single_clustering_by_threshold(
                    interferometers=interferometers,
                    time_frequency_transform=self.time_frequency_transform,
                    threshold=self.threshold,
                    padding_time=0.0,
                    padding_freq=0.0,
                    minimum_frequency=self.interferometers[0].minimum_frequency,
                    maximum_frequency=self.interferometers[0].maximum_frequency,
                )

                # Do the OR operation
                time_frequency_filter |= _time_frequency_filter
            self._time_frequency_filter = time_frequency_filter
            logger.info("Clustering done.")
        return self._time_frequency_filter
