from __future__ import annotations

import logging
import tempfile
from os import uname

import bilby.core.utils.random
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import CubicSpline, InterferometerList
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.utils import noise_weighted_inner_product
from bilby.gw.waveform_generator import WaveformGenerator

from nullcal.clustering import single_clustering_by_threshold
from nullcal.likelihood import RecalibrationLikelihood
from nullcal.null_stream import compute_calibrated_whitened_antenna_response
from nullcal.time_frequency_transform import (get_shape_of_wavelet_transform,
                                              transform_wavelet_freq)

bilby_logger = logging.getLogger("bilby")
bilby_logger.setLevel(logging.WARNING)
nullcal_logger = logging.getLogger("nullcal")
nullcal_logger.setLevel(logging.WARNING)


def compute_SNR(frequency_domain_strain, power_spectral_density_array, duration):
    return np.sqrt(
        noise_weighted_inner_product(
            aa=frequency_domain_strain,
            bb=frequency_domain_strain,
            power_spectral_density=power_spectral_density_array,
            duration=duration,
        ).real
    )


@pytest.fixture
def mock_data():
    minimum_frequency = 10
    maximum_frequency = 2048
    sampling_frequency = 4096
    duration = 16
    n_points = 10
    wavelet_transform_frequency_resolution = 16
    wavelet_transform_nx = 4.0
    clustering_threshold = 0.1
    seed = 12
    bilby.core.utils.random.seed(seed)

    # calibration parameters
    calibration_parameters = {
        "recalib_ET1_amplitude_0": -0.0015200418959315,
        "recalib_ET1_amplitude_1": 0.0137262334514915,
        "recalib_ET1_amplitude_2": -0.0098674551286916,
        "recalib_ET1_amplitude_3": 0.0496653832340328,
        "recalib_ET1_amplitude_4": 0.0050244018902233,
        "recalib_ET1_amplitude_5": 0.012957149543555,
        "recalib_ET1_amplitude_6": -0.0095396336156851,
        "recalib_ET1_amplitude_7": 0.0047935297196438,
        "recalib_ET1_amplitude_8": 0.0054113884142458,
        "recalib_ET1_amplitude_9": 0.0260352806778183,
        "recalib_ET1_phase_0": -0.0723524157564597,
        "recalib_ET1_phase_1": -0.13634780567121,
        "recalib_ET1_phase_2": 0.0010756491344211,
        "recalib_ET1_phase_3": 0.0377631722932499,
        "recalib_ET1_phase_4": 0.048417333675392,
        "recalib_ET1_phase_5": -0.0154291536722841,
        "recalib_ET1_phase_6": -0.0146625647850522,
        "recalib_ET1_phase_7": 0.0072359178419159,
        "recalib_ET1_phase_8": -0.0088257329442514,
        "recalib_ET1_phase_9": -0.0163642590206929,
        "recalib_ET1_frequency_0": 9.999999999999998,
        "recalib_ET1_frequency_1": 18.06402140273364,
        "recalib_ET1_frequency_2": 32.63088692384192,
        "recalib_ET1_frequency_3": 58.944503978246175,
        "recalib_ET1_frequency_4": 106.47747814365572,
        "recalib_ET1_frequency_5": 192.34114440961008,
        "recalib_ET1_frequency_6": 347.4454549241482,
        "recalib_ET1_frequency_7": 627.6262134032335,
        "recalib_ET1_frequency_8": 1133.7453351832692,
        "recalib_ET1_frequency_9": 2048.0000000000005,
        "recalib_ET2_amplitude_0": -0.1679581792790701,
        "recalib_ET2_amplitude_1": -0.053306364347335,
        "recalib_ET2_amplitude_2": 0.0465697300377795,
        "recalib_ET2_amplitude_3": -0.0150190692996883,
        "recalib_ET2_amplitude_4": -0.0149934881643516,
        "recalib_ET2_amplitude_5": -0.0440038186863086,
        "recalib_ET2_amplitude_6": 0.0157757736488322,
        "recalib_ET2_amplitude_7": -0.0137612829917987,
        "recalib_ET2_amplitude_8": -0.0128744516410139,
        "recalib_ET2_amplitude_9": -0.0006411473283399,
        "recalib_ET2_phase_0": 0.124048613599911,
        "recalib_ET2_phase_1": -0.0065838182619891,
        "recalib_ET2_phase_2": -0.0617960739070807,
        "recalib_ET2_phase_3": -0.0383128685730137,
        "recalib_ET2_phase_4": 0.055367913992918,
        "recalib_ET2_phase_5": -0.0058317483412057,
        "recalib_ET2_phase_6": -0.011593512651481,
        "recalib_ET2_phase_7": -0.0191142988904006,
        "recalib_ET2_phase_8": -0.0079892452078436,
        "recalib_ET2_phase_9": -0.0437037119484562,
        "recalib_ET2_frequency_0": 9.999999999999998,
        "recalib_ET2_frequency_1": 18.06402140273364,
        "recalib_ET2_frequency_2": 32.63088692384192,
        "recalib_ET2_frequency_3": 58.944503978246175,
        "recalib_ET2_frequency_4": 106.47747814365572,
        "recalib_ET2_frequency_5": 192.34114440961008,
        "recalib_ET2_frequency_6": 347.4454549241482,
        "recalib_ET2_frequency_7": 627.6262134032335,
        "recalib_ET2_frequency_8": 1133.7453351832692,
        "recalib_ET2_frequency_9": 2048.0000000000005,
        "recalib_ET3_amplitude_0": 0.1547825340418115,
        "recalib_ET3_amplitude_1": -0.0082457150500027,
        "recalib_ET3_amplitude_2": -0.1516463026295033,
        "recalib_ET3_amplitude_3": -0.0084226452082301,
        "recalib_ET3_amplitude_4": 0.0058476522162691,
        "recalib_ET3_amplitude_5": 0.0304297460749714,
        "recalib_ET3_amplitude_6": -0.0071604357169415,
        "recalib_ET3_amplitude_7": -0.0040239078801284,
        "recalib_ET3_amplitude_8": -0.002781119636238,
        "recalib_ET3_amplitude_9": -0.0020275445786096,
        "recalib_ET3_phase_0": 0.0893723149735473,
        "recalib_ET3_phase_1": 0.0160386697680951,
        "recalib_ET3_phase_2": -0.0807479302784239,
        "recalib_ET3_phase_3": 0.0780242865966646,
        "recalib_ET3_phase_4": -0.0053671983079773,
        "recalib_ET3_phase_5": 0.0230565929975641,
        "recalib_ET3_phase_6": -0.0118184392809595,
        "recalib_ET3_phase_7": -0.0154827938448187,
        "recalib_ET3_phase_8": -0.0020955260491795,
        "recalib_ET3_phase_9": -0.0267874537238445,
        "recalib_ET3_frequency_0": 9.999999999999998,
        "recalib_ET3_frequency_1": 18.06402140273364,
        "recalib_ET3_frequency_2": 32.63088692384192,
        "recalib_ET3_frequency_3": 58.944503978246175,
        "recalib_ET3_frequency_4": 106.47747814365572,
        "recalib_ET3_frequency_5": 192.34114440961008,
        "recalib_ET3_frequency_6": 347.4454549241482,
        "recalib_ET3_frequency_7": 627.6262134032335,
        "recalib_ET3_frequency_8": 1133.7453351832692,
        "recalib_ET3_frequency_9": 2048.0000000000005,
    }

    # The first signal.
    parameters_0 = {
        "mass_1": 35.6,
        "mass_2": 30.6,
        "a_1": 0.3,
        "a_2": 0.36,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "theta_jn": 2.68,
        "psi": 1.6,
        "phase": 0.0,
        "geocent_time": 1126259462.4 + 2,
        "ra": 1.97,
        "dec": -1.21,
        "luminosity_distance": 1500.3719941553823,
    }

    # The second signal.
    parameters_1 = {
        "mass_1": 31.6,
        "mass_2": 30.6,
        "a_1": 0.3,
        "a_2": 0.36,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "theta_jn": 2.68,
        "psi": 1.6,
        "phase": 0.0,
        "geocent_time": 1126259462.4 - 2,
        "ra": 1.97,
        "dec": -1.21,
        "luminosity_distance": 1500.3719941553823,
    }

    # The third signal.
    parameters_2 = {
        "mass_1": 31.6,
        "mass_2": 30.6,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "theta_jn": 0.0,
        "psi": 0.0,
        "phase": 0.0,
        "geocent_time": 1126259462.4 + 6,
        "ra": 1.97,
        "dec": -1.21,
        "luminosity_distance": 1500.3719941553823,
    }

    for key in calibration_parameters:
        parameters_0[key] = calibration_parameters[key]
        parameters_1[key] = calibration_parameters[key]
        parameters_2[key] = calibration_parameters[key]

    # Combine the full set of parameters
    # parameters_list = [parameters_0, parameters_1, parameters_2]
    parameters_list = [parameters_0, parameters_1, parameters_2]

    start_time = int(1126259462.4 - duration / 2)
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXPHM",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    interferometers = InterferometerList(["ET"])
    for interferometer in interferometers:
        interferometer.minimum_frequency = minimum_frequency
        interferometer.maximum_frequency = maximum_frequency
        interferometer.calibration_model = CubicSpline(
            prefix=f"recalib_{interferometer.name}_",
            minimum_frequency=interferometer.minimum_frequency,
            maximum_frequency=maximum_frequency,
            n_points=n_points,
        )
    # Create noise.
    interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
    )

    # Inject signal
    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    for parameters in parameters_list:
        interferometers.inject_signal(waveform_generator=waveform_generator, parameters=parameters)

    # Get the noiseless interferometers
    noiseless_interferometers_list = []
    for parameters in parameters_list:
        noiseless_interferometers = InterferometerList(["ET"])
        for interferometer in noiseless_interferometers:
            interferometer.minimum_frequency = minimum_frequency
            interferometer.maximum_frequency = maximum_frequency
            interferometer.calibration_model = CubicSpline(
                prefix=f"recalib_{interferometer.name}_",
                minimum_frequency=interferometer.minimum_frequency,
                maximum_frequency=maximum_frequency,
                n_points=n_points,
            )
        # Create zero noise.
        noiseless_interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
        )
        noiseless_interferometers.inject_signal(waveform_generator=waveform_generator, parameters=parameters)

        noiseless_interferometers_list.append(noiseless_interferometers)

    # Compute the SNRs
    ET1_frequency_domain_strain = np.sum(
        [ifos[0].frequency_domain_strain for ifos in noiseless_interferometers_list], axis=0
    )
    ET1_power_spectral_density = noiseless_interferometers_list[0][0].power_spectral_density_array
    ET1_SNR = compute_SNR(ET1_frequency_domain_strain, ET1_power_spectral_density, duration)

    ET2_frequency_domain_strain = np.sum(
        [ifos[1].frequency_domain_strain for ifos in noiseless_interferometers_list], axis=0
    )
    ET2_power_spectral_density = noiseless_interferometers_list[0][1].power_spectral_density_array
    ET2_SNR = compute_SNR(ET2_frequency_domain_strain, ET2_power_spectral_density, duration)
    ET3_frequency_domain_strain = np.sum(
        [ifos[2].frequency_domain_strain for ifos in noiseless_interferometers_list], axis=0
    )
    ET3_power_spectral_density = noiseless_interferometers_list[0][2].power_spectral_density_array
    ET3_SNR = compute_SNR(ET3_frequency_domain_strain, ET3_power_spectral_density, duration)

    null_stream = ET1_frequency_domain_strain + ET2_frequency_domain_strain + ET3_frequency_domain_strain
    null_stream_power_spectral_density = (
        ET1_power_spectral_density + ET2_power_spectral_density + ET3_power_spectral_density
    ) / np.sqrt(3)

    null_stream_SNR = compute_SNR(null_stream, null_stream_power_spectral_density, duration)

    return {
        "sampling_frequency": sampling_frequency,
        "minimum_frequency": minimum_frequency,
        "maximum_frequency": maximum_frequency,
        "duration": duration,
        "n_points": n_points,
        "start_time": start_time,
        "waveform_arguments": waveform_arguments,
        "interferometers": interferometers,
        "noiseless_interferometers_list": noiseless_interferometers_list,
        "calibration_parameters": calibration_parameters,
        "injection_parameters": parameters_list,
        "wavelet_transform_frequency_resolution": wavelet_transform_frequency_resolution,
        "wavelet_transform_nx": wavelet_transform_nx,
        "clustering_threshold": clustering_threshold,
        "frequency_mask": np.all([interferometer.frequency_mask for interferometer in interferometers], axis=0),
        "ET1_SNR": ET1_SNR,
        "ET2_SNR": ET2_SNR,
        "ET3_SNR": ET3_SNR,
        "null_stream_SNR": null_stream_SNR,
    }


@pytest.fixture
def recalibration_likelihood(mock_data):
    interferometers = mock_data["interferometers"]
    injection_parameters = mock_data["injection_parameters"]
    duration = mock_data["duration"]
    sampling_frequency = mock_data["sampling_frequency"]
    start_time = mock_data["start_time"]
    waveform_arguments = mock_data["waveform_arguments"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    clustering_threshold = mock_data["clustering_threshold"]

    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
        clustering_parameter_file = f.name
        parameters_df = pd.DataFrame.from_dict(injection_parameters)
        # Save to file.
        parameters_df.to_csv(clustering_parameter_file)
        likelihood = RecalibrationLikelihood(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            wavelet_transform_frequency_resolution=wavelet_transform_frequency_resolution,
            wavelet_transform_nx=wavelet_transform_nx,
            time_frequency_filter=None,
            clustering_parameter_file=clustering_parameter_file,
            clustering_threshold=clustering_threshold,
        )
        # Run the time-frequency clustering before the temp file is deleted.
        _ = likelihood.time_frequency_filter

    return likelihood


def test_initialization(mock_data, recalibration_likelihood):
    duration = mock_data["duration"]
    sampling_frequency = mock_data["sampling_frequency"]
    minimum_frequency = mock_data["minimum_frequency"]
    maximum_frequency = mock_data["maximum_frequency"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    frequency_mask = mock_data["frequency_mask"]
    interferometers = mock_data["interferometers"]
    frequency_array = interferometers[0].frequency_array

    assert isinstance(recalibration_likelihood.interferometers, InterferometerList)
    expected_wavelet_transform_Nt, expected_wavelet_transform_Nf = get_shape_of_wavelet_transform(
        t_length=int(duration * sampling_frequency),
        sampling_frequency=sampling_frequency,
        frequency_resolution=wavelet_transform_frequency_resolution,
    )
    assert recalibration_likelihood._wavelet_transform_Nt == expected_wavelet_transform_Nt
    assert recalibration_likelihood._wavelet_transform_Nf == expected_wavelet_transform_Nf
    assert recalibration_likelihood._wavelet_transform_nx == wavelet_transform_nx
    assert np.array_equal(recalibration_likelihood.frequency_mask, frequency_mask)
    assert np.array_equal(
        recalibration_likelihood.masked_frequency_array, interferometers[0].frequency_array[frequency_mask]
    )
    assert recalibration_likelihood.minimum_frequency == minimum_frequency
    assert recalibration_likelihood.maximum_frequency == maximum_frequency
    assert recalibration_likelihood._whitened_antenna_response.shape == (len(frequency_array), 3, 2)
    assert recalibration_likelihood._whitened_frequency_domain_strain_array.shape == (3, len(frequency_array))


def test_clustering(mock_data, recalibration_likelihood):
    noiseless_interferometers_list = mock_data["noiseless_interferometers_list"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    clustering_threshold = mock_data["clustering_threshold"]
    minimum_frequency = mock_data["minimum_frequency"]
    maximum_frequency = mock_data["maximum_frequency"]
    expected_time_frequency_filter = np.logical_or.reduce(
        [
            single_clustering_by_threshold(
                interferometers=noiseless_interferometers,
                frequency_resolution=wavelet_transform_frequency_resolution,
                nx=wavelet_transform_nx,
                threshold=clustering_threshold,
                padding_time=0.0,
                padding_freq=0.0,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
            for noiseless_interferometers in noiseless_interferometers_list
        ]
    )

    assert np.array_equal(expected_time_frequency_filter, recalibration_likelihood.time_frequency_filter)


def test_uncalibrated_time_frequency_domain_null_stream(mock_data, recalibration_likelihood):
    duration = mock_data["duration"]
    sampling_frequency = mock_data["sampling_frequency"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    frequency_mask = mock_data["frequency_mask"]
    wavelet_transform_Nt, wavelet_transform_Nf = get_shape_of_wavelet_transform(
        t_length=int(duration * sampling_frequency),
        sampling_frequency=sampling_frequency,
        frequency_resolution=wavelet_transform_frequency_resolution,
    )
    uncalibrated_frequency_domain_null_stream = (
        recalibration_likelihood.compute_uncalibrated_frequency_domain_null_stream()
    )
    rotated_uncalibrated_frequency_domain_null_stream = np.zeros_like(uncalibrated_frequency_domain_null_stream)
    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            U, _, _ = np.linalg.svd(recalibration_likelihood._whitened_antenna_response[i, :, :])
            rotated_uncalibrated_frequency_domain_null_stream[:, i] = np.einsum(
                "ij,j->i", np.conj(U).T, uncalibrated_frequency_domain_null_stream[:, i]
            )

    # Perform the time-frequency transform
    rotated_uncalibrated_time_frequency_domain_null_stream = np.array(
        [
            transform_wavelet_freq(
                data=data,
                Nf=wavelet_transform_Nf,
                Nt=wavelet_transform_Nt,
                nx=wavelet_transform_nx,
            )
            for data in rotated_uncalibrated_frequency_domain_null_stream
        ]
    )
    rotated_uncalibrated_time_frequency_domain_null_stream_filtered = (
        rotated_uncalibrated_time_frequency_domain_null_stream[2, recalibration_likelihood.time_frequency_filter]
    )
    result = scipy.stats.kstest(
        rotated_uncalibrated_time_frequency_domain_null_stream_filtered, cdf="norm", args=(0.0, 1.0)
    )
    assert result.pvalue < 0.05


def test_calibrated_time_frequency_domain_null_stream(mock_data, recalibration_likelihood):
    duration = mock_data["duration"]
    sampling_frequency = mock_data["sampling_frequency"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    frequency_mask = mock_data["frequency_mask"]
    calibration_parameters = mock_data["calibration_parameters"]
    wavelet_transform_Nt, wavelet_transform_Nf = get_shape_of_wavelet_transform(
        t_length=int(duration * sampling_frequency),
        sampling_frequency=sampling_frequency,
        frequency_resolution=wavelet_transform_frequency_resolution,
    )
    calibration_factor = recalibration_likelihood.construct_calibration_factor_from_parameters(calibration_parameters)
    calibrated_frequency_domain_null_stream = recalibration_likelihood.compute_calibrated_frequency_domain_null_stream(
        calibration_factor
    )
    # Get the calibrated whitened antenna response.
    calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(
        recalibration_likelihood._whitened_antenna_response, calibration_factor, frequency_mask
    )
    rotated_calibrated_frequency_domain_null_stream = np.zeros_like(calibrated_frequency_domain_null_stream)
    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            U, _, _ = np.linalg.svd(calibrated_whitened_antenna_response[i, :, :])
            rotated_calibrated_frequency_domain_null_stream[:, i] = np.einsum(
                "ij,j->i", np.conj(U).T, calibrated_frequency_domain_null_stream[:, i]
            )
    # Perform the time-frequency transform
    rotated_calibrated_time_frequency_domain_null_stream = np.array(
        [
            transform_wavelet_freq(
                data=data,
                Nf=wavelet_transform_Nf,
                Nt=wavelet_transform_Nt,
                nx=wavelet_transform_nx,
            )
            for data in rotated_calibrated_frequency_domain_null_stream
        ]
    )
    rotated_calibrated_time_frequency_domain_null_stream_filtered = (
        rotated_calibrated_time_frequency_domain_null_stream[2, recalibration_likelihood.time_frequency_filter]
    )
    result = scipy.stats.kstest(
        rotated_calibrated_time_frequency_domain_null_stream_filtered, cdf="norm", args=(0.0, 1.0)
    )
    assert result.pvalue > 0.05


def test_incorrectly_calibrated_time_frequency_domain_null_stream(mock_data, recalibration_likelihood):
    duration = mock_data["duration"]
    sampling_frequency = mock_data["sampling_frequency"]
    wavelet_transform_frequency_resolution = mock_data["wavelet_transform_frequency_resolution"]
    wavelet_transform_nx = mock_data["wavelet_transform_nx"]
    frequency_mask = mock_data["frequency_mask"]
    calibration_parameters = mock_data["calibration_parameters"]
    n_points = mock_data["n_points"]
    wavelet_transform_Nt, wavelet_transform_Nf = get_shape_of_wavelet_transform(
        t_length=int(duration * sampling_frequency),
        sampling_frequency=sampling_frequency,
        frequency_resolution=wavelet_transform_frequency_resolution,
    )
    incorrect_parameters = calibration_parameters.copy()
    np.random.seed(13)
    for i in range(n_points):
        incorrect_parameters[f"recalib_ET1_amplitude_{i}"] = np.random.randn()
        incorrect_parameters[f"recalib_ET2_amplitude_{i}"] = np.random.randn()
        incorrect_parameters[f"recalib_ET3_amplitude_{i}"] = np.random.randn()
        incorrect_parameters[f"recalib_ET1_phase_{i}"] = np.random.randn()
        incorrect_parameters[f"recalib_ET2_phase_{i}"] = np.random.randn()
        incorrect_parameters[f"recalib_ET3_phase_{i}"] = np.random.randn()

    calibration_factor = recalibration_likelihood.construct_calibration_factor_from_parameters(incorrect_parameters)
    calibrated_frequency_domain_null_stream = recalibration_likelihood.compute_calibrated_frequency_domain_null_stream(
        calibration_factor
    )
    # Get the calibrated whitened antenna response.
    calibrated_whitened_antenna_response = compute_calibrated_whitened_antenna_response(
        recalibration_likelihood._whitened_antenna_response, calibration_factor, frequency_mask
    )
    rotated_calibrated_frequency_domain_null_stream = np.zeros_like(calibrated_frequency_domain_null_stream)
    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            U, _, _ = np.linalg.svd(calibrated_whitened_antenna_response[i, :, :])
            rotated_calibrated_frequency_domain_null_stream[:, i] = np.einsum(
                "ij,j->i", np.conj(U).T, calibrated_frequency_domain_null_stream[:, i]
            )
    # Perform the time-frequency transform
    rotated_calibrated_time_frequency_domain_null_stream = np.array(
        [
            transform_wavelet_freq(
                data=data,
                Nf=wavelet_transform_Nf,
                Nt=wavelet_transform_Nt,
                nx=wavelet_transform_nx,
            )
            for data in rotated_calibrated_frequency_domain_null_stream
        ]
    )
    rotated_calibrated_time_frequency_domain_null_stream_filtered = (
        rotated_calibrated_time_frequency_domain_null_stream[2, recalibration_likelihood.time_frequency_filter]
    )
    result = scipy.stats.kstest(
        rotated_calibrated_time_frequency_domain_null_stream_filtered, cdf="norm", args=(0.0, 1.0)
    )
    assert result.pvalue < 0.05
