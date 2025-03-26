from __future__ import annotations

from collections.abc import Iterable

import bilby.gw.detector.networks
import numpy as np
from bilby.gw.detector.interferometer import Interferometer
from bilby.gw.detector.networks import \
    InterferometerList as BilbyInterferometerList
from bilby.gw.detector.networks import get_empty_interferometer
from bilby.gw.detector.psd import PowerSpectralDensity
from bilby.gw.utils import get_vertex_position_geocentric
from scipy.spatial.transform import Rotation

from ..utils import (get_vertex_position_ellipsoid, logger,
                     optimal_uncorrelated_null_stream_snr_squared)


class InterferometerList(BilbyInterferometerList):
    """A list of Interferometer objects"""

    def __init__(self, interferometers: Iterable):
        """Instantiate a InterferometerList

        The InterferometerList is a list of Interferometer objects, each
        object has the data used in evaluating the likelihood

        Args:
            interferometers (Iterable): The list of interferometers

        Raises:
            TypeError: Input must not be a string.
            TypeError: Input list of interferometers are not all Interferometer objects.
        """
        if isinstance(interferometers, str):
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_interferometer(ifo)
            if not isinstance(ifo, (Interferometer, TriangularInterferometer)):
                raise TypeError(
                    "Input list of interferometers are not all Interferometer objects"
                    f"{type(ifo)} found."
                )
            else:
                self.append(ifo)
        self._check_interferometers()


class TriangularInterferometer(InterferometerList):
    def __init__(
            self,
            name,
            power_spectral_density,
            minimum_frequency,
            maximum_frequency,
            length,
            latitude,
            longitude,
            elevation,
            xarm_azimuth,
            yarm_azimuth,
            xarm_tilt=0.0,
            yarm_tilt=0.0,
            clockwise=True,
    ):
        super().__init__([])
        self.name = name
        # for attr in ['power_spectral_density', 'minimum_frequency', 'maximum_frequency']:
        if isinstance(power_spectral_density, PowerSpectralDensity):
            power_spectral_density = [power_spectral_density] * 3
        if isinstance(minimum_frequency, float) or isinstance(minimum_frequency, int):
            minimum_frequency = [minimum_frequency] * 3
        if isinstance(maximum_frequency, float) or isinstance(maximum_frequency, int):
            maximum_frequency = [maximum_frequency] * 3

        for ii in range(3):

            self.append(
                Interferometer(
                    f"{name}{ii + 1}",
                    power_spectral_density[ii],
                    minimum_frequency[ii],
                    maximum_frequency[ii],
                    length,
                    latitude,
                    longitude,
                    elevation,
                    xarm_azimuth,
                    yarm_azimuth,
                    xarm_tilt,
                    yarm_tilt,
                )
            )

            unit_vector_x = self[ii].geometry.unit_vector_along_arm("x")
            unit_vector_y = self[ii].geometry.unit_vector_along_arm("y")

            vertex_geocentric = get_vertex_position_geocentric(self[ii].latitude_radians,
                                                               self[ii].longitude_radians,
                                                               self[ii].elevation)
            next_vertex_geocentric = vertex_geocentric + length * 1000 * (unit_vector_y if clockwise else unit_vector_x)
            next_vertex_ellipsoid = get_vertex_position_ellipsoid(next_vertex_geocentric[0],
                                                                next_vertex_geocentric[1],
                                                                next_vertex_geocentric[2])
            next_latitude_rad, next_longitude_rad, next_elevation = next_vertex_ellipsoid

            rotation_vector = np.cross(unit_vector_x, unit_vector_y)
            rotation_vector /= np.linalg.norm(rotation_vector)
            rotation_angle = - 2 / 3 * np.pi if clockwise else 2 / 3 * np.pi
            rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)
            next_unit_vector_x = rotation.apply(unit_vector_x)
            next_unit_vector_y = rotation.apply(unit_vector_y)

            next_local_normal_vector = np.array([np.cos(next_latitude_rad) * np.cos(next_longitude_rad),
                                                np.cos(next_latitude_rad) * np.sin(next_longitude_rad),
                                                np.sin(next_latitude_rad)])
            next_local_north_vector = np.array([-np.sin(next_latitude_rad) * np.cos(next_longitude_rad),
                                                -np.sin(next_latitude_rad) * np.sin(next_longitude_rad),
                                                np.cos(next_latitude_rad)])
            next_local_east_vector = np.array([-np.sin(next_longitude_rad),
                                            np.cos(next_longitude_rad), 0])

            next_xarm_tilt_rad = np.arcsin(np.dot(next_unit_vector_x, next_local_normal_vector))
            next_yarm_tilt_rad = np.arcsin(np.dot(next_unit_vector_y, next_local_normal_vector))
            next_xarm_azimuth_rad = np.arctan2(np.dot(next_unit_vector_x, next_local_north_vector),
                                            np.dot(next_unit_vector_x, next_local_east_vector))
            next_yarm_azimuth_rad = np.arctan2(np.dot(next_unit_vector_y, next_local_north_vector),
                                            np.dot(next_unit_vector_y, next_local_east_vector))

            latitude = np.rad2deg(next_latitude_rad)
            longitude = np.rad2deg(next_longitude_rad)
            elevation = next_elevation
            xarm_azimuth = np.rad2deg(next_xarm_azimuth_rad)
            yarm_azimuth = np.rad2deg(next_yarm_azimuth_rad)
            xarm_tilt = next_xarm_tilt_rad
            yarm_tilt = next_yarm_tilt_rad


    def optimal_null_stream_snr_squared(self, signals):
        """Compute the optimal null stream signal-to-noise ratio
        squared.

        Args:
            signals (_type_): _description_

        Returns:
            _type_: _description_
        """
        return optimal_uncorrelated_null_stream_snr_squared(
            signals=signals,
            power_spectral_densities=np.array(
                [ifo.power_spectral_density_array for ifo in self]
            ),
            duration=self[0].duration
        )

    @property
    def meta_data(self):
        return {'network': self._network_meta_data,
                **{ifo.name: ifo.meta_data for ifo in self}}

    def inject_signal(
            self,
            parameters=None,
            injection_polarizations=None,
            waveform_generator=None,
            raise_error=True,
    ):
        """ Inject a signal into noise in each of the three detectors.

        Args:
            parameters (dict): Parameters of the injection.
            injection_polarizations (dict): Polarizations of waveform to inject, output of
                `waveform_generator.frequency_domain_strain()`. If
                `waveform_generator` is also given, the injection_polarizations will
                be calculated directly and this argument can be ignored.
            waveform_generator (bilby.gw.waveform_generator.WaveformGenerator):
                A WaveformGenerator instance using the source model to inject. If
                `injection_polarizations` is given, this will be ignored.
            raise_error (bool):
                Whether to raise an error if the injected signal does not fit in
                the segment.

        Notes: if your signal takes a substantial amount of time to generate, or
            you experience buggy behaviour. It is preferable to provide the
            injection_polarizations directly.

        Returns:
            dict: injection_polarizations

        """
        if injection_polarizations is None:
            if waveform_generator is not None:
                injection_polarizations = waveform_generator.frequency_domain_strain(
                    parameters
                )
            else:
                raise ValueError(
                    "inject_signal needs one of waveform_generator or "
                    "injection_polarizations."
                )

        all_injection_polarizations = list()
        for interferometer in self:
            all_injection_polarizations.append(
                interferometer.inject_signal(
                    parameters=parameters,
                    injection_polarizations=injection_polarizations,
                    raise_error=raise_error,
                )
            )

        # Compute the null stream optimal SNR.
        signal_ifos = []
        for i in len(self.interferometers):
            polarizations = all_injection_polarizations[i]
            # Calculate the signal.
            signal_ifo = self.interferometers[i].get_detector_response(
                waveform_polarizations=polarizations,
                parameters=self.injection_parameters,
            )
            signal_ifos.append(signal_ifo)

        # Calculate the optimal null stream snr
        self.meta_data['network']['optimal_null_stream_SNR'] = (
                np.sqrt(self.interferometers.optimal_null_stream_snr_squared(signal_ifos).real))
        logger.info("Optimal null stream SNR = {:.2f}".format(self.meta_data['optimal_null_stream_SNR']))
        return all_injection_polarizations


bilby.gw.detector.networks.TriangularInterferometer = TriangularInterferometer
