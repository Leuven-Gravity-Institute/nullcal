from __future__ import annotations

import os
import numpy as np
from bilby.core.result import Result as CoreResult
from bilby.core.utils import logger, safe_save_figure
from bilby.gw.utils import spline_angle_xform
from .utils import plot_spline_pos


class Result(CoreResult):
    """
    Result class specific for calibration error analysis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_calibration_posterior(self, filename=None, minimum_frequency=None, maximum_frequency=None, show_injected_value=False, show_priors=False, quantile=.9):
        """
        Plots the calibration amplitude and phase uncertainty.
        Adapted from the same function in bilby.gw.result

        Parameters
        ==========
        filename: str
            If provided, save the plot to disk, and otherwise show the plot.
        minimum_frequency: float
            Minimum frequency to display.
        maximum_frequency: float
            Maximum frequency to display
        show_injected_value: bool
            If true, display the injected value.
        show_priors: bool
            If true, display the uncertainty band from the prior.
        quantile: float
            Quantile for confidence levels, default=0.9, i.e., 90% interval.
        """

        import matplotlib.pyplot as plt

        # Retrieve posterior
        posterior = self.posterior
        parameters = posterior.keys()

        # Detectors
        ifos = np.unique([param.split('_')[1] for param in parameters if 'recalib_' in param])
        if ifos.size == 0:
            logger.info("No calibration parameters found. Aborting calibration plot.")
            return

        fig, axes = plt.subplots(2 * len(ifos), 1, figsize=(15, 10 * len(ifos)))
        if filename is not None:
            font_size = 32
        else:
            font_size = 10

        for i, ifo in enumerate(ifos):

            # Colors
            if ifo == 'ET1':
                color = 'r'
            elif ifo == 'ET2':
                color = 'g'
            elif ifo == 'ET3':
                color = 'm'
            else:
                color = 'c'

            # Assume spline control frequencies are constant
            freq_params = np.sort(
                [param for param in parameters if 'recalib_{0}_frequency_'.format(ifo) in param])
            logfreqs = np.log([posterior[param].iloc[0] for param in freq_params])

            # Retrieve injected values if requested
            if show_injected_value:
                injected_values = getattr(self, 'injection_parameters', False)[0]
                injected_values_parameters = injected_values.keys()
            if isinstance(injected_values, dict):
                # Amplitude injected parameters
                inj_amp_params = np.sort(
                    [param for param in injected_values_parameters if 'recalib_{0}_amplitude_'.format(ifo) in param])
                injected_amplitude = [injected_values[param] for param in inj_amp_params]
                # Phase injected parameters
                inj_phase_params = np.sort(
                    [param for param in injected_values_parameters if 'recalib_{0}_phase_'.format(ifo) in param])
                injected_phase = [injected_values[param] for param in inj_phase_params]
            elif injected_values in [False, None]:
                injected_amplitude = None
                injected_phase = None

            # Retrieve priors if requested
            if show_priors:
                priors = getattr(self, 'priors', False)
                priors_parameters = priors.keys()
            if isinstance(priors, dict):
                # Amplitude priors
                amp_priors_params = np.sort(
                    [param for param in priors_parameters if 'recalib_{0}_amplitude_'.format(ifo) in param])
                amplitude_priors = [priors[param] for param in amp_priors_params]
                # Phase priors
                phase_priors_params = np.sort(
                    [param for param in priors_parameters if 'recalib_{0}_phase_'.format(ifo) in param])
                phase_priors = [priors[param] for param in phase_priors_params]
            elif priors in [False, None]:
                amplitude_priors = None
                phase_priors = None
            else:
                raise ValueError('Input priors={} not understood'.format(priors))

            # Amplitude calibration model
            ax1 = axes[2*i]
            plt.sca(ax1)
            amp_params = np.sort(
                [param for param in parameters if 'recalib_{0}_amplitude_'.format(ifo) in param])
            if len(amp_params) > 0:
                amplitude = 100 * np.column_stack([posterior[param] for param in amp_params])
                plot_spline_pos(log_freqs=logfreqs,
                                samples=amplitude,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                level=quantile,
                                injected_values=injected_amplitude,
                                priors=amplitude_priors,
                                errorbar=False,
                                color=color,
                                label=r"{0} {1}$\%$C.L.".format(ifo.upper(), int(quantile * 100)))

            # Phase calibration model
            ax2 = axes[2*i+1]
            plt.sca(ax2)
            phase_params = np.sort([param for param in parameters if
                                    'recalib_{0}_phase_'.format(ifo) in param])
            inj_phase_params = np.sort(
                [param for param in injected_values_parameters if 'recalib_{0}_phase_'.format(ifo) in param])
            if len(phase_params) > 0:
                phase = np.column_stack([posterior[param] for param in phase_params])
                plot_spline_pos(log_freqs=logfreqs,
                                samples=phase,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                level=quantile,
                                injected_values=injected_phase,
                                priors=phase_priors,
                                errorbar=False,
                                color=color,
                                label=r"{0} {1}$\%$C.L.".format(ifo.upper(), int(quantile * 100)),
                                xform=spline_angle_xform)

            ax1.tick_params(labelsize=.75 * font_size)
            ax2.tick_params(labelsize=.75 * font_size)
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_ylabel(r'Amplitude [$\%$]', fontsize=font_size)
            ax2.set_ylabel('Phase [deg]', fontsize=font_size)

        ax2.set_xlabel('Frequency [Hz]', fontsize=font_size)

        # Save or show figure
        if filename is not None:
            fig.tight_layout()
            safe_save_figure(fig=fig, filename=filename, dpi=500, bbox_inches='tight')
            logger.debug("Calibration figure saved to {}".format(filename))
            plt.close()
        else:
            for ax in axes:
                ax.legend(fontsize=font_size)
            plt.tight_layout()
            plt.show()
