from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.result import Result as CoreResult
from bilby.core.utils import logger, safe_save_figure
from bilby.gw.utils import spline_angle_xform

from .utils import (plot_spline_pos, plot_spline_pos_relative_amplitude,
                    plot_spline_pos_relative_phase, spline_percentage_xform)


class Result(CoreResult):
    """
    Result class specific for calibration error analysis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def posterior(self) -> pd.DataFrame:
        """Get a pandas DataFrame of the posterior.

        Returns:
            pd.DataFrame: A pandas DataFrame of the posterior.
        """
        if self._posterior is None:
            self.samples_to_posterior(priors=self.priors)
        return self._posterior

    @posterior.setter
    def posterior(self, posterior: pd.DataFrame):
        """Set the posterior.

        Args:
            posterior (pd.DataFrame): A pandas DataFrame of the posterior.
        """
        self._posterior = posterior

    @property
    def detectors(self) -> np.ndarray:
        """Get an array of detector labels.

        Returns:
            np.ndarray: An array of detector labels.
        """
        if not hasattr(self, '_detectors'):
            self._detectors = np.unique([param.split('_')[1] for param in self.search_parameter_keys if 'recalib_' in param])
        return self._detectors

    def get_calibration_frequency_parameters(self, detector: str) -> np.ndarray:
        """Get the calibration frequency parameter keys.

        Args:
            detector (str): Label of the detector.

        Returns:
            np.ndarray: An array of the calibration frequency parameter keys.
        """
        return np.sort([param for param in self.fixed_parameter_keys if f'recalib_{detector}_frequency_' in param])

    def get_calibration_amplitude_parameters(self, detector: str) -> np.ndarray:
        """Get the calibration amplitude parameter keys.

        Args:
            detector (str): Label of the detector.

        Returns:
            np.ndarray: An array of the calibration amplitude parameter keys.
        """
        return np.sort([param for param in self.search_parameter_keys if f'recalib_{detector}_amplitude_' in param])

    def get_calibration_phase_parameters(self, detector: str) -> np.ndarray:
        """Get the calibration phase parameter keys.

        Args:
            detector (str): Label of the detector.

        Returns:
            np.ndarray: An array of the calibration phase parameter keys.
        """
        return np.sort([param for param in self.search_parameter_keys if f'recalib_{detector}_phase_' in param])

    def get_calibration_knot_frequency_array(self) -> np.ndarray:
        """Get the knot frequencies of the spline model used in calibration.

        WARNING: The knot frequencies are assumed to be the same for all detectors.
        Only the data for the first detector are used to retrieve the knot frequencies.

        Returns:
            np.ndarray: _description_
        """
        freq_params = self.get_calibration_frequency_parameters(detector=self.detectors[0])
        return np.array([self.posterior[param].iloc[0] for param in freq_params])

    def get_injection_calibration_amplitude_error(self, detector: str) -> np.ndarray:
        """Get the injection calibration amplitude error.

        Args:
            detector (str): Label of the detector.

        Raises:
            ValueError: Injected values are not found.
            ValueError: Injected values are not understood.

        Returns:
            np.ndarray: Injection calibration amplitude error.
        """
        injected_values = getattr(self, 'injection_parameters', False)[0]
        if isinstance(injected_values, dict):
            # Amplitude injected parameters
            inj_amp_params = self.get_calibration_amplitude_parameters(detector=detector)
            return np.array([injected_values[param] for param in inj_amp_params])
        elif len(injected_values) == 0:
            raise ValueError('Injected values are not found.')
        else:
            raise ValueError(f'Injected values={injected_values} not understood')

    def get_injection_calibration_phase_error(self, detector: str) -> np.ndarray:
        """Get the injection calibration phase error.

        Args:
            detector (str): Label of the detector.

        Raises:
            ValueError: Injected values are not found.
            ValueError: Injected values are not understood.

        Returns:
            np.ndarray: Injection calibration phase error.
        """
        injected_values = getattr(self, 'injection_parameters', False)[0]
        if isinstance(injected_values, dict):
            # Phase injected parameters
            inj_phase_params = self.get_calibration_phase_parameters(detector=detector)
            return np.array([injected_values[param] for param in inj_phase_params])
        elif len(injected_values) == 0:
            raise ValueError('Injected values are not found.')
        else:
            raise ValueError(f'Injected values={injected_values} not understood')

    def get_calibration_amplitude_error_prior_samples(self, detector: str, n_samples: int=10000) -> np.ndarray:
        """Get the prior samples of the calibration amplitude error.

        Args:
            detector (str): Label of the detector.
            n_samples (int, optional): Number of samples. Defaults to 10000.

        Raises:
            ValueError: priors are not found.
            ValueError: Input priors are not understood.

        Returns:
            np.ndarray: Prior samples of the calibration amplitude error.
        """
        priors = getattr(self, 'priors', False)
        if isinstance(priors, dict):
            amp_priors_params = self.get_calibration_amplitude_parameters(detector=detector)
            return np.transpose([priors[param].sample(n_samples)
                                                        for param in amp_priors_params])
        elif priors is False:
            raise ValueError(f'priors are not found.')
        else:
            raise ValueError(f'Input priors={priors} not understood')

    def get_calibration_phase_error_prior_samples(self, detector: str, n_samples: int=10000) -> np.ndarray:
        """Get the prior samples of the calibration phase error.

        Args:
            detector (str): Label of the detector.
            n_samples (int, optional): Number of samples. Defaults to 10000.

        Raises:
            ValueError: priors are not found.
            ValueError: Input priors are not understood.

        Returns:
            np.ndarray: Prior samples of the calibration phase error.
        """
        priors = getattr(self, 'priors', False)
        if isinstance(priors, dict):
            phase_priors_params = self.get_calibration_phase_parameters(detector=detector)
            return np.transpose([priors[param].sample(n_samples)
                                                        for param in phase_priors_params])
        elif priors is False:
            raise ValueError(f'priors are not found.')
        else:
            raise ValueError(f'Input priors={priors} not understood')

    def get_calibration_amplitude_error_posterior_samples(self, detector: str) -> np.ndarray:
        """Get the posterior samples of the calibration amplitude error of the detectors.

        Args:
            detector (str): Label of the detector.

        Returns:
            Dict: A dictionary of the posterior samples of the calibration amplitude error of the detectors.
        """
        parameters = self.get_calibration_amplitude_parameters(detector=detector)
        return np.column_stack([self.posterior[param] for param in parameters])

    def get_calibration_phase_error_posterior_samples(self, detector: str) -> np.ndarray:
        """Get the posterior samples of the calibration phase error of the detectors.

        Args:
            detector (str): Label of the detector.

        Returns:
            np.ndarray: Posterior samples of the calibration phase error.
        """
        parameters = self.get_calibration_phase_parameters(detector=detector)
        return np.column_stack([self.posterior[param] for param in parameters])

    def plot_calibration_posterior(self,
                                   filename: str | None=None,
                                   minimum_frequency: float | None=None,
                                   maximum_frequency: float | None=None,
                                   show_injected_value: bool | None=True,
                                   show_priors: bool | None=True,
                                   show_knots: bool | None=True,
                                   show_errorbar: bool | None=False,
                                   quantile: float | None=.9,
                                   font_size: float | None=32):
        """
        Plots the calibration amplitude and phase uncertainty.
        Adapted from the same function in bilby.gw.result

        Args:
            filename (str, optional): If provided, save the plot to disk, and otherwise show the plot.
            minimum_frequency (float, optional): Minimum frequency to display.
            maximum_frequency (float, optional): Maximum frequency to display
            show_injected_value (bool, optional): If true, display the injected value. Defaults to True.
            show_priors (bool, optional): If true, display the uncertainty band from the prior. Defaults to True.
            show_knots (bool, optional): If true, display the frequency knots. Defaults to True.
            show_errorbar (bool, optional): If true, display the posterior errorbars at the spline control points
            quantile (float, optional): Quantile for confidence levels, default=0.9, i.e., 90% interval.
            font_size (float, optional): Font size.
        """
        # Assume spline control frequencies are constant
        logfreqs = np.log(self.get_calibration_knot_frequency_array())

        # Detectors
        ifos = self.detectors
        if ifos.size == 0:
            logger.error("No calibration parameters found. Aborting calibration plot.")
            return

        fig, axes = plt.subplots(2 * len(ifos), 1, figsize=(15, 10 * len(ifos)))

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

            # Get the injected values.
            if show_injected_value:
                injected_amplitude = self.get_injection_calibration_amplitude_error(detector=ifo)
                injected_phase = self.get_injection_calibration_phase_error(detector=ifo)
            else:
                injected_amplitude = None
                injected_phase = None

            # Get the priors.
            if show_priors:
                prior_amplitude = self.get_calibration_amplitude_error_prior_samples(detector=ifo)
                prior_phase = self.get_calibration_phase_error_prior_samples(detector=ifo)
            else:
                prior_amplitude = None
                prior_phase = None

            # Get the posteriors.
            posterior_amplitude = self.get_calibration_amplitude_error_posterior_samples(detector=ifo)
            posterior_phase = self.get_calibration_phase_error_posterior_samples(detector=ifo)

            # Amplitude calibration model
            ax1 = axes[2*i]
            plt.sca(ax1)
            plot_spline_pos(log_freqs=logfreqs,
                            samples=posterior_amplitude,  # Convert to percentage
                            minimum_frequency=minimum_frequency,
                            maximum_frequency=maximum_frequency,
                            level=quantile,
                            injected_values=injected_amplitude,  # Convert to percentage
                            priors_samples=prior_amplitude,  # Convert to percentage
                            show_knots=show_knots,
                            errorbar=show_errorbar,
                            color=color,
                            label=fr"{ifo.upper()} {int(quantile * 100)}$\%$C.L.",
                            xform=spline_percentage_xform)

            # Phase calibration model
            ax2 = axes[2*i+1]
            plt.sca(ax2)
            plot_spline_pos(log_freqs=logfreqs,
                            samples=posterior_phase,
                            minimum_frequency=minimum_frequency,
                            maximum_frequency=maximum_frequency,
                            level=quantile,
                            injected_values=injected_phase,
                            priors_samples=prior_phase,
                            show_knots=show_knots,
                            errorbar=show_errorbar,
                            color=color,
                            label=fr"{ifo.upper()} {int(quantile * 100)}$\%$C.L.",
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
            logger.debug(f"Calibration figure saved to {filename}")
            plt.close()
        else:
            for ax in axes:
                ax.legend(fontsize=font_size)
            plt.tight_layout()
            plt.show()

    def plot_relative_calibration_posterior(self,
                                            filename: str | None=None,
                                            minimum_frequency: float | None=None,
                                            maximum_frequency: float | None=None,
                                            show_injected_value: bool=True,
                                            show_priors: bool=True,
                                            show_knots: bool=True,
                                            show_errorbar: bool=False,
                                            quantile: float=.9,
                                            font_size: float=32):
        """
        Plots the relative calibration amplitude and phase uncertainty

        Args:
            filename (str, optional): If provided, save the plot to disk, and otherwise show the plot.
            minimum_frequency (float, optional): Minimum frequency to display.
            maximum_frequency (float, optional): Maximum frequency to display
            show_injected_value (bool, optional): If true, display the injected value.
            show_priors (bool, optional): If true, display the uncertainty band from the prior.
            show_knots (bool, optional): If true, display the frequency knots. Defaults to True.
            show_errorbar (bool, optional): If true, display the posterior errorbars at the spline control points
            quantile (float, optional): Quantile for confidence levels, default=0.9, i.e., 90% interval.
            font_size (float, optional): Font size. Defaults to 32.
        """
                # Assume spline control frequencies are constant
        logfreqs = np.log(self.get_calibration_knot_frequency_array())

        # Detectors
        ifos = self.detectors[1:]
        if ifos.size == 0:
            logger.error("No calibration parameters found. Aborting calibration plot.")
            return

        fig, axes = plt.subplots(2 * len(ifos), 1, figsize=(15, 10 * len(ifos)))

        # Get the data from the detector to be compared against
        if show_injected_value:
            injected_amplitude_0 = self.get_injection_calibration_amplitude_error(detector=self.detectors[0])
            injected_phase_0 = self.get_injection_calibration_phase_error(detector=self.detectors[0])
        if show_priors:
            prior_amplitude_0 = self.get_calibration_amplitude_error_prior_samples(detector=self.detectors[0])
            prior_phase_0 = self.get_calibration_phase_error_prior_samples(detector=self.detectors[0])

        posterior_amplitude_0 = self.get_calibration_amplitude_error_posterior_samples(detector=self.detectors[0])
        posterior_phase_0 = self.get_calibration_phase_error_posterior_samples(detector=self.detectors[0])

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

            # Get the injected values.
            if show_injected_value:
                injected_amplitude = self.get_injection_calibration_amplitude_error(detector=ifo)
                injected_phase = self.get_injection_calibration_phase_error(detector=ifo)

                relative_injected_amplitude = (1 + injected_amplitude) / (1 + injected_amplitude_0)
                relative_injected_phase = injected_phase - injected_phase_0
            else:
                relative_injected_amplitude = None
                relative_injected_phase = None

            # Get the priors.
            if show_priors:
                prior_amplitude = self.get_calibration_amplitude_error_prior_samples(detector=ifo)
                prior_phase = self.get_calibration_phase_error_prior_samples(detector=ifo)

                relative_prior_amplitude = (1 + prior_amplitude) / (1 + prior_amplitude_0)
                relative_prior_phase = prior_phase - prior_phase_0
            else:
                relative_prior_amplitude = None
                relative_prior_phase = None

            # Get the posteriors.
            posterior_amplitude = self.get_calibration_amplitude_error_posterior_samples(detector=ifo)
            posterior_phase = self.get_calibration_phase_error_posterior_samples(detector=ifo)

            relative_posterior_amplitude = (1 + posterior_amplitude) / (1 + posterior_amplitude_0)
            relative_posterior_phase = posterior_phase - posterior_phase_0

            # Amplitude calibration model
            ax1 = axes[2*i]
            plt.sca(ax1)
            plot_spline_pos(log_freqs=logfreqs,
                            samples=relative_posterior_amplitude,
                            minimum_frequency=minimum_frequency,
                            maximum_frequency=maximum_frequency,
                            level=quantile,
                            injected_values=relative_injected_amplitude,
                            priors_samples=relative_prior_amplitude,
                            show_knots=show_knots,
                            errorbar=show_errorbar,
                            color=color,
                            label=fr"{ifo.upper()}/{self.detectors[0]} {int(quantile * 100)}$\%$C.L.")

            # Phase calibration model
            ax2 = axes[2*i+1]
            plt.sca(ax2)
            plot_spline_pos(log_freqs=logfreqs,
                            samples=relative_posterior_phase,
                            minimum_frequency=minimum_frequency,
                            maximum_frequency=maximum_frequency,
                            level=quantile,
                            injected_values=relative_injected_phase,
                            priors_samples=relative_prior_phase,
                            show_knots=show_knots,
                            errorbar=show_errorbar,
                            color=color,
                            label=fr"{ifo.upper()}/{self.detectors[0]} {int(quantile * 100)}$\%$C.L.",
                            xform=spline_angle_xform)

            ax1.tick_params(labelsize=.75 * font_size)
            ax2.tick_params(labelsize=.75 * font_size)
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_ylabel(r'Amplitude ratio', fontsize=font_size)
            ax2.set_ylabel('Phase difference [deg]', fontsize=font_size)

        ax2.set_xlabel('Frequency [Hz]', fontsize=font_size)

        # Save or show figure
        if filename is not None:
            fig.tight_layout()
            safe_save_figure(fig=fig, filename=filename, dpi=500, bbox_inches='tight')
            logger.debug(f"Calibration figure saved to {filename}")
            plt.close()
        else:
            for ax in axes:
                ax.legend(fontsize=font_size)
            plt.tight_layout()
            plt.show()
