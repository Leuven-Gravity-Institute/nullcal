from bilby_pipe.input import Input
from bilby_pipe.data_generation import DataGenerationInput as \
    BilbyDataGenerationInput
from bilby_pipe.utils import convert_string_to_dict, DataDump
from bilby_pipe.main import parse_args
import sys
import bilby_pipe.utils
import numpy as np
from .parser import create_nullpol_parser
from ..networks import *
from ..utils import logger
from .. import (__version__,
                log_version_information)
bilby_pipe.utils.logger = logger


class DataGenerationInput(BilbyDataGenerationInput):
    """Handles user-input for the data generation script

    Args:
        args (list): A list of the arguments to parse. Defaults to `sys.argv[1:]`
        unknown_args (list): A list of unknown arguments.
        create_data (bool): If false, no data is generated (used for testing).
    """
    def __init__(self, args, unknown_args, create_data=True):
        Input.__init__(self, args, unknown_args)

        # Generic initialisation
        self.meta_data = dict(
            command_line_args=args.__dict__,
            unknown_command_line_args=unknown_args,
            injection_parameters=None,
            nullpol_version=__version__,
        )
        self.injection_parameters = None

        # Admin arguments
        self.ini = args.ini
        self.transfer_files = args.transfer_files

        # Run index arguments
        self.idx = args.idx
        self.generation_seed = args.generation_seed
        self.trigger_time = args.trigger_time

        # Naming arguments
        self.outdir = args.outdir
        self.label = args.label

        # Prior arguments
        self.reference_frame = args.reference_frame
        self.time_reference = args.time_reference
        self.prior_file = args.prior_file
        self.prior_dict = args.prior_dict
        self.deltaT = args.deltaT
        self.default_prior = args.default_prior

        # Data arguments
        self.ignore_gwpy_data_quality_check = \
            args.ignore_gwpy_data_quality_check
        self.detectors = args.detectors
        self.channel_dict = args.channel_dict
        self.data_dict = args.data_dict
        self.data_format = args.data_format
        self.allow_tape = args.allow_tape
        self.tukey_roll_off = args.tukey_roll_off
        self.gaussian_noise = args.gaussian_noise
        self.zero_noise = args.zero_noise
        self.resampling_method = args.resampling_method

        if args.timeslide_dict is not None:
            self.timeslide_dict = convert_string_to_dict(args.timeslide_dict)
            logger.info(("Read-in timeslide dict directly: "
                         f"{self.timeslide_dict}"))
        elif args.timeslide_file is not None:
            self.gps_file = args.gps_file
            self.timeslide_file = args.timeslide_file
            self.timeslide_dict = self.get_timeslide_dict(self.idx)

        # Data duration arguments
        self.duration = args.duration
        self.post_trigger_duration = args.post_trigger_duration

        # Frequencies
        self.sampling_frequency = args.sampling_frequency
        self.minimum_frequency = args.minimum_frequency
        self.maximum_frequency = args.maximum_frequency

        # PSD
        self.psd_maximum_duration = args.psd_maximum_duration
        self.psd_dict = args.psd_dict
        self.psd_length = args.psd_length
        self.psd_fractional_overlap = args.psd_fractional_overlap
        self.psd_start_time = args.psd_start_time
        self.psd_method = args.psd_method

        # Calibration
        self.calibration_model = args.calibration_model
        self.spline_calibration_envelope_dict = \
            args.spline_calibration_envelope_dict
        self.spline_calibration_amplitude_uncertainty_dict = (
            args.spline_calibration_amplitude_uncertainty_dict
        )
        self.spline_calibration_phase_uncertainty_dict = (
            args.spline_calibration_phase_uncertainty_dict
        )
        self.spline_calibration_nodes = args.spline_calibration_nodes
        self.calibration_prior_boundary = args.calibration_prior_boundary

        # Plotting
        self.plot_data = args.plot_data
        self.plot_spectrogram = args.plot_spectrogram
        self.plot_injection = args.plot_injection

        if create_data:
            self.create_data(args)

    def _set_interferometers_from_injection_in_gaussian_noise(self):
        """Method to generate the interferometers data from an injection in Gaussian noise"""

        self.injection_parameters = self.injection_df.iloc[self.idx].to_dict()
        logger.info("Injecting waveform with ")
        for prop in [
            "minimum_frequency",
            "maximum_frequency",
            "trigger_time",
            "start_time",
            "duration",
        ]:
            logger.info(f"{prop} = {getattr(self, prop)}")

        self._set_interferometers_from_gaussian_noise()

        waveform_arguments = self.get_injection_waveform_arguments()
        logger.info(f"Using waveform arguments: {waveform_arguments}")
        waveform_generator = self.waveform_generator_class(
            duration=self.duration,
            start_time=self.start_time,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=self.injection_bilby_frequency_domain_source_model,
            parameter_conversion=self.parameter_conversion,
            waveform_arguments=waveform_arguments,
        )

        all_injection_polarizations = self.interferometers.inject_signal(
            waveform_generator=waveform_generator,
            parameters=self.injection_parameters,
            raise_error=self.enforce_signal_duration,
        )

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
        self.meta_data['optimal_null_stream_SNR'] = (
            np.sqrt(self.interferometers.optimal_null_stream_snr_squared(signal_ifos).real))
        logger.info("Optimal null stream SNR = {:.2f}".format(self.meta_data['optimal_null_stream_SNR']))

    def save_data_dump(self):
        """Method to dump the saved data to disk for later analysis"""
        self.meta_data["reweighting_configuration"] = self.reweighting_configuration
        data_dump = DataDump(
            outdir=self.data_directory,
            label=self.label,
            idx=self.idx,
            trigger_time=self.trigger_time,
            interferometers=self.interferometers,
            meta_data=self.meta_data,
            likelihood_lookup_table=None,
            likelihood_roq_weights=None,
            likelihood_roq_params=None,
            likelihood_multiband_weights=None,
            priors_dict=dict(self.priors),
            priors_class=self.priors.__class__,
        )
        data_dump.to_pickle()


def create_generation_parser():
    """Data generation parser creation"""
    return create_nullpol_parser(top_level=False)


def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_generation_parser())
    log_version_information()
    data = DataGenerationInput(args, unknown_args)
    data.save_data_dump()
    logger.info("Completed data generation")
