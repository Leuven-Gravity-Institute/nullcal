from configargparse import ArgParser, 
import pkg_resources
import json
from bilby.gw.detector import InterferometerList
from bilby.gw.detector import PowerSpectralDensity
import subprocess
from pathlib import Path
from .utility import (load_config_file,
                      resolve_config_conflicts)
from ..utility import logger
from ..prior import CalibrationPriorDict
from ..likelihood import SelfRecalibrationProjectorTimeFrequencyLikelihood

def json_loads_with_none(value):
    # Replace 'None' with 'null' to make it valid JSON
    value = value.replace('None', 'null')
    return json.loads(value)

def main():
    default_config_file_path = pkg_resources.resource_filename('nullcal.tools', 'default_config_nullcal_pipe.ini')
    parser = ArgParser(default_config_files=[default_config_file_path])
    parser.add('-c', '--config', is_config_file=True, help="Path to a custom config file.")
    parser.add('--outdir', type=str, help="Output directory.")
    parser.add('--label', type=str, help="Label of the run.", default="SIM")
    parser.add('--frame-files', type=json_loads_with_none, help='A dictionary of frame files.')
    parser.add('--channels', type=json_loads_with_none, help='A dictionary of channels.')
    parser.add('--psd-files', type=json_loads_with_none, help='A dictionary of PSD files.')
    parser.add('--prior-file', type=str, help='A prior file.')
    parser.add('--prior-from-envelope-files', type=json_loads_with_none, help='A dictionary of calibration envelope files to construct the priors.')
    parser.add('--prior-from-result-file', type=str, help='A result file from the previous anslysis to construct the prior.')
    parser.add('--start-time', type=int, help='Start GPS time in second.', default=0)
    parser.add('--duration', type=int, help='Duration in second.', default=8)
    parser.add('--sampling-frequency', type=float, help='Sampling frequency in Hz.', default=2048)
    parser.add('--minimum-frequency', type=float, help='Minimum frequency in Hz.', default=20)
    parser.add('--maximum-frequency', type=float, help='Minimum frequency in Hz.', default=1024)
    parser.add('--n-nodes', type=int, help='Number of frequency nodes for the calibration spline model.', default=10)
    parser.add('--injection', action="store_true", help="Simulation data are generated with this flag.")
    parser.add('--nullcal-create-injection', help="Path to the executable nullcal-create-injection.")
    parser.add('--nullcal-create-injection-config', help="Path to the config file of nullcal-create-injection.")

    args = parser.parse_args()    

    outdir = Path(args.outdir)
    if args.maximum_frequency is None:
        args.maximum_frequency = args.sampling_frequency / 2

    # Generate simulation data
    if args.injection:        
        logger.info("--injection is provided. Simulation data will be generated for analysis.")
        if args.nullcal_create_injection is None:
            args.nullcal_create_injection = "nullcal-create-injection"
        # Resolve any conflict
        ## Read the config file for nullcall-create-injection
        nullcall_create_injection_config = load_config_file(args.nullcal_create_injection_config)
        resolve_config_conflicts(args,
                                 {key:val for key,val in nullcall_create_injection_config.items() if key in ['minimum-frequency',
                                                                                                             'duration',
                                                                                                             'start-time',
                                                                                                             'sampling-frequency']})
        if float(nullcall_create_injection_config['minimum-frequency']) != args.minimum_frequency:
            args.minimum_frequency = float(nullcall_create_injection_config['minimum-frequency'])

        subprocess.run([f"{args.nullcal_create_injection}",
                        "--label", args.label],
                        ""
                       capture_output=True)

    # Read the frame files and PSD files.
    interferometers = InterferometerList(['ET']) 
    for i in range(3):
        interferometers[i].minimum_frequency = args.minimum_frequency        
        interferometers[i].maximum_frequency = args.maximum_frequency
        logger.info(f'Reading frame file for ET{i+1}')
        interferometers[i].set_strain_data_from_frame_file(frame_file=args.frame_files[f'ET{i+1}'],
                                                           sampling_frequency=args.sampling_frequency,
                                                           duration=args.duration,
                                                           start_time=args.start_time,
                                                           channel=args.channels[f'ET{i+1}'])
        logger.info(f'Reading PSD file for ET{i+1}')
        interferometers[i].power_spectral_density = PowerSpectralDensity(psd_file=args.psd_files[f'ET{i+1}'])

    # Read the prior file
    if args.prior_file is not None:
        logger.info(f'Reading priors from prior-file={args.prior_file}')
        priors = CalibrationPriorDict(filename=args.prior_file)
    elif args.prior_from_envelope_files is not None:
        logger.info(f'Reading priors from envelope file={args.prior_from_envelope_files[f'ET{i+1}']}')
        priors = CalibrationPriorDict()
        for i in range(3):
            priors.update(CalibrationPriorDict.from_envelope_file(envelope_file=args.prior_from_envelope_files[f'ET{i+1}'],
                                                                  minimum_frequency=args.minimum_frequency,
                                                                  maximum_frequency=args.maximum_frequency,
                                                                  n_nodes=args.n_nodes,
                                                                  label=f'ET{i+1}'))
    elif args.prior_from_result_file is not None:
        logger.info(f'Reading priors from result file={args.prior_from_result_file}')
        priors = CalibrationPriorDict.from_result_file(result_file=args.prior_from_result_file,
                                                       minimum_frequency=args.minimum_frequency,
                                                       maximum_frequency=args.maximum_frequency,
                                                       n_nodes=args.n_nodes,
                                                       boundary='reflective')
    else:
        raise ValueError('Prior information is not found.')

    # Construct the likelihood        

