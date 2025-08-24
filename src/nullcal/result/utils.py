"""
Utility functions for processing results.
"""

from __future__ import annotations

import numpy as np


def spline_percentage_xform(delta_a: float | np.ndarray) -> float | np.ndarray:
    """Returns the error in percentage corresponding to the spline
    calibration parameters delta_A.

    Args:
        delta_A (float | np.ndarray): Calibration amplitude uncertainty.

    Returns:
        float | np.ndarray: delta_A in percentage.
    """
    return delta_a * 100


# def plot_spline_pos(
#     log_freqs: np.ndarray,
#     samples: np.ndarray,
#     minimum_frequency: float,
#     maximum_frequency: float,
#     nfreqs: int = 100,
#     level: float = 0.9,
#     injected_values: np.ndarray | None = None,
#     priors_samples: np.ndarray | None = None,
#     show_knots: bool = True,
#     errorbar: bool | None = False,
#     color: str | None = "k",
#     label: str | None = None,
#     xform: Callable | None = None,
#     font_size: float = 32,
# ):
#     """
#     Plot calibration posterior estimates for a spline model in log space.
#     Adapted from the same function in bilby.gw.utils

#     Args:
#         log_freqs (array-like): The (log) location of spline control points.
#         samples (array-like): List of posterior draws of function at control points ``log_freqs``
#         minimum_frequency (float): Minimum frequency for plotting.
#         maximum_frequency (float): Maximum frequency for plotting.
#         nfreqs (int): Number of points to evaluate spline at for plotting.
#         level (float): Credible level to fill in.
#         injected_values (array-like): List of injected values at control points ``log_freqs``
#         priors_samples (array-like): List of prior draws of function at control points ``log_freqs``
#         errorbar (bool): If True, plot the posterior draws errorbars of function at control points ``log_freqs``
#         color (str): Color to plot with.
#         label (str): Label for plot.
#         xform (Callable): Function to transform the spline into plotted values.
#         font_size (float): Font size. Defaults to 32.
#     """
#     freq_points = np.exp(log_freqs)
#     if minimum_frequency is None:
#         minimum_frequency = min(log_freqs)
#     if maximum_frequency is None:
#         maximum_frequency = max(log_freqs)
#     freqs = np.logspace(minimum_frequency, maximum_frequency, nfreqs, base=np.exp(1))

#     # Retrieve posterior samples
#     if xform is None:
#         scaled_samples = samples
#     else:
#         scaled_samples = xform(samples)
#     scaled_samples_summary = SamplesSummary(scaled_samples, average="median", confidence_level=level)
#     if not isinstance(scaled_samples_summary.average, np.ndarray):
#         raise ValueError("scaled_samples_summary.average is not a numpy array.")

#     # Plot errorbar
#     if errorbar:
#         plt.errorbar(
#             freq_points,
#             scaled_samples_summary.average,
#             yerr=[
#                 -scaled_samples_summary.lower_relative_credible_interval,
#                 scaled_samples_summary.upper_relative_credible_interval,
#             ],
#             fmt=".",
#             color=color,
#             lw=4,
#             alpha=0.5,
#             capsize=0,
#         )

#     # Plot posterior samples
#     data = np.zeros((samples.shape[0], nfreqs))
#     for i, sample in enumerate(samples):
#         temp = interp1d(log_freqs, sample, kind="cubic", fill_value="extrapolate", bounds_error=False)(np.log(freqs))
#         if xform is None:
#             data[i] = temp
#         else:
#             data[i] = xform(temp)
#     data_summary = SamplesSummary(data, average="median", confidence_level=level)
#     plt.fill_between(
#         freqs,
#         data_summary.lower_absolute_credible_interval,
#         data_summary.upper_absolute_credible_interval,
#         color=color,
#         alpha=0.2,
#         linewidth=0.1,
#         label=label,
#     )

#     # Plot injected values
#     if injected_values is None:
#         pass
#     else:
#         injected_values_data = interp1d(
#             log_freqs, injected_values, kind="cubic", fill_value="extrapolate", bounds_error=False
#         )(np.log(freqs))
#         if xform is None:
#             injected_values_data = injected_values_data
#         else:
#             injected_values_data = xform(injected_values_data)
#         plt.plot(freqs, injected_values_data, color="k", ls="-.", lw=1.5)

#     # Plot priors
#     if priors_samples is None:
#         pass
#     else:
#         priors_data = np.zeros((priors_samples.shape[0], nfreqs))
#         for i, sample in enumerate(priors_samples):
#             temp = interp1d(log_freqs, sample, kind="cubic", fill_value="extrapolate", bounds_error=False)(
#                 np.log(freqs)
#             )
#             if xform is None:
#                 priors_data[i] = temp
#             else:
#                 priors_data[i] = xform(temp)
#         priors_data_summary = SamplesSummary(priors_data, average="median", confidence_level=level)
#         plt.plot(freqs, priors_data_summary.lower_absolute_credible_interval, color=color, lw=2.5)
#         plt.plot(freqs, priors_data_summary.upper_absolute_credible_interval, color=color, lw=2.5)

#     if show_knots:
#         for freq in freq_points:
#             plt.axvline(x=freq)

#     plt.xlim(freq_points.min() - 0.5, freq_points.max() + 50)
#     plt.legend(loc="upper right", prop={"size": 0.75 * font_size})


# def plot_spline_pos_relative_amplitude(
#     log_freqs: np.ndarray,
#     samples_1: np.ndarray,
#     samples_2: np.ndarray,
#     minimum_frequency: float,
#     maximum_frequency: float,
#     nfreqs: int = 100,
#     level: float = 0.9,
#     injected_values: tuple | None = None,
#     priors_samples: np.ndarray | None = None,
#     errorbar: bool | None = False,
#     color: str | None = "k",
#     label: str | None = None,
#     font_size: float = 32,
# ):
#     """
#     Plot calibration posterior estimates relative amplitude for a spline model in log space.
#     Adapted from the function plot_spline_pos in bilby.gw.utils

#     Args:
#         log_freqs (array-like): The (log) location of spline control points.
#         samples_1 (array-like): List of amplitude posterior draws of
#           function at control points ``log_freqs`` for detector 1
#         samples_2 (array-like): List of amplitude posterior draws of
#           function at control points ``log_freqs`` for detector 2
#         minimum_frequency (float): Minimum frequency for plotting.
#         maximum_frequency (float): Maximum frequency for plotting.
#         nfreqs (int): Number of points to evaluate spline at for plotting.
#         level (float): Credible level to fill in.
#             injected_values: tuple of array-like
#             Tuple of the list of injected values at control points ``log_freqs``
#             for detectors 1 and detectors 2
#         priors_samples (array-like): List of prior draws of function at control points ``log_freqs``
#         errorbar (bool): If True, plot the posterior draws errorbars of function
#             at control points ``log_freqs``
#         color (str): Color to plot with.
#         label (str): Label for plot.
#         font_size (float): Font size.
#     """
#     freq_points = np.exp(log_freqs)
#     if minimum_frequency is None:
#         minimum_frequency = min(log_freqs)
#     if maximum_frequency is None:
#         maximum_frequency = max(log_freqs)
#     freqs = np.logspace(minimum_frequency, maximum_frequency, nfreqs, base=np.exp(1))

#     # Retrieve posterior samples
#     errorbar_samples_1 = samples_1
#     errorbar_samples_2 = samples_2
#     errorbar_samples_relative = (1 + errorbar_samples_1) / (1 + errorbar_samples_2)

#     errorbar_samples_relative_summary = SamplesSummary(
#         errorbar_samples_relative, average="median", confidence_level=level
#     )

#     if not isinstance(errorbar_samples_relative_summary.average, np.ndarray):
#         raise ValueError("errorbar_samples_relative_summary.average is not a numpy array.")

#     # Plot errorbar
#     if errorbar:
#         plt.errorbar(
#             freq_points,
#             errorbar_samples_relative_summary.average,
#             yerr=[
#                 -errorbar_samples_relative_summary.lower_relative_credible_interval,
#                 errorbar_samples_relative_summary.upper_relative_credible_interval,
#             ],
#             fmt=".",
#             color=color,
#             lw=4,
#             alpha=0.5,
#             capsize=0,
#         )

#     # Plot posterior samples
#     data_relative = np.zeros((samples_1.shape[0], nfreqs))
#     for i, (sample_1, sample_2) in enumerate(zip(samples_1, samples_2)):
#         data_1 = interp1d(log_freqs, sample_1, kind="cubic", fill_value="extrapolate", bounds_error=False)(
#             np.log(freqs)
#         )
#         data_2 = interp1d(log_freqs, sample_2, kind="cubic", fill_value=np.inf, bounds_error=False)(np.log(freqs))
#         data_relative[i] = (1 + data_1) / (1 + data_2)
#     data_relative_summary = SamplesSummary(data_relative, average="median", confidence_level=level)
#     plt.fill_between(
#         freqs,
#         data_relative_summary.lower_absolute_credible_interval,
#         data_relative_summary.upper_absolute_credible_interval,
#         color=color,
#         alpha=0.2,
#         linewidth=0.1,
#         label=label,
#     )

#     # Plot injected values
#     if injected_values is None:
#         pass
#     else:
#         injected_values_1, injected_values_2 = injected_values
#         injected_values_data_1 = interp1d(
#             log_freqs, injected_values_1, kind="cubic", fill_value="extrapolate", bounds_error=False
#         )(np.log(freqs))
#         injected_values_data_2 = interp1d(
#             log_freqs, injected_values_2, kind="cubic", fill_value=np.inf, bounds_error=False
#         )(np.log(freqs))
#         injected_values_data_relative = (1 + injected_values_data_1) / (1 + injected_values_data_2)
#         plt.plot(freqs, injected_values_data_relative, color="k", ls="-.", lw=1.5)

#     # Plot priors
#     if priors_samples is None:
#         pass
#     else:
#         priors_samples_1, priors_samples_2 = priors_samples
#         priors_data_relative = np.zeros((priors_samples_1.shape[0], nfreqs))
#         for i, (sample_1, sample_2) in enumerate(zip(priors_samples_1, priors_samples_2)):
#             data_1 = interp1d(log_freqs, sample_1, kind="cubic", fill_value="extrapolate", bounds_error=False)(
#                 np.log(freqs)
#             )
#             data_2 = interp1d(log_freqs, sample_2, kind="cubic", fill_value="extrapolate", bounds_error=False)(
#                 np.log(freqs)
#             )
#             priors_data_relative[i] = (1 + data_1) / (1 + data_2)
#         priors_data_relative_summary = SamplesSummary(priors_data_relative, average="median", confidence_level=level)
#         plt.plot(freqs, priors_data_relative_summary.lower_absolute_credible_interval, color=color)
#         plt.plot(freqs, priors_data_relative_summary.upper_absolute_credible_interval, color=color)

#     plt.xlim(freq_points.min() - 0.5, freq_points.max() + 50)
#     plt.legend(loc="upper right", prop={"size": 0.75 * font_size})


# def plot_spline_pos_relative_phase(
#     log_freqs: np.ndarray,
#     samples_1: np.ndarray,
#     samples_2: np.ndarray,
#     minimum_frequency: float,
#     maximum_frequency: float,
#     nfreqs: int = 100,
#     level: float = 0.9,
#     injected_values: tuple | None = None,
#     priors_samples: np.ndarray | None = None,
#     errorbar: bool = False,
#     color: str = "k",
#     label: str | None = None,
#     xform: Callable | None = None,
# ):
#     """
#     Plot calibration posterior estimates relative phase for a spline model in log space.
#     Adapted from the function plot_spline_pos in bilby.gw.utils

#     Args:
#         log_freqs (array-like): The (log) location of spline control points.
#         samples_1 (array-like): List of phase posterior draws of function
#             at control points ``log_freqs`` for detector 1
#         samples_2 (array-like): List of phase posterior draws of function
#             at control points ``log_freqs`` for detector 2
#         minimum_frequency (float): Minimum frequency for plotting.
#         maximum_frequency (float): Maximum frequency for plotting.
#         nfreqs (int): Number of points to evaluate spline at for plotting.
#         level (float): Credible level to fill in.
#         injected_values (array-like): List of injected values
#             at control points ``log_freqs``
#         priors_samples (array-like): List of prior draws of function
#             at control points ``log_freqs``
#         errorbar (bool): If True, plot the posterior draws errorbars of function
#             at control points ``log_freqs``
#         color (str): Color to plot with.
#         label (str): Label for plot.
#         xform (Callable):Function to transform the spline into plotted values.
#     """

#     import matplotlib.pyplot as plt
#     from bilby.gw.utils import spline_angle_xform

#     if xform is not spline_angle_xform:
#         raise ValueError(f"Input xform={xform} is not bilby.gw.utils.spline_angle_xform.")

#     font_size = 32

#     freq_points = np.exp(log_freqs)
#     freqs = np.logspace(start=minimum_frequency, end=maximum_frequency, num=nfreqs, base=np.exp(1))

#     # Retrieve posterior samples
#     if xform is not None:
#         errorbar_samples_1 = xform(samples_1)
#         errorbar_samples_2 = xform(samples_2)
#     else:
#         errorbar_samples_1 = samples_1
#         errorbar_samples_2 = samples_2
#     errorbar_samples_relative = errorbar_samples_1 - errorbar_samples_2

#     errorbar_samples_relative_summary = SamplesSummary(
#         errorbar_samples_relative, average="median", confidence_level=level
#     )

#     if not isinstance(errorbar_samples_relative_summary.average, np.ndarray):
#         raise ValueError("errorbar_samples_relative_summary.average is not a numpy array.")

#     # Plot errorbar
#     if errorbar:
#         plt.errorbar(
#             freq_points,
#             errorbar_samples_relative_summary.average,
#             yerr=[
#                 -errorbar_samples_relative_summary.lower_relative_credible_interval,
#                 errorbar_samples_relative_summary.upper_relative_credible_interval,
#             ],
#             fmt=".",
#             color=color,
#             lw=4,
#             alpha=0.5,
#             capsize=0,
#         )

#     # Plot posterior samples
#     data_relative = np.zeros((samples_1.shape[0], nfreqs))
#     for i, (sample_1, sample_2) in enumerate(zip(samples_1, samples_2)):
#         temp_1 = interp1d(log_freqs, sample_1, kind="cubic", fill_value=0, bounds_error=False)(np.log(freqs))
#         temp_2 = interp1d(log_freqs, sample_2, kind="cubic", fill_value=0, bounds_error=False)(np.log(freqs))
#         if xform is not None:
#             data_1 = xform(temp_1)
#             data_2 = xform(temp_2)
#         else:
#             data_1 = temp_1
#             data_2 = temp_2
#         data_relative[i] = data_1 - data_2
#     data_relative_summary = SamplesSummary(data_relative, average="median", confidence_level=level)
#     plt.fill_between(
#         freqs,
#         data_relative_summary.lower_absolute_credible_interval,
#         data_relative_summary.upper_absolute_credible_interval,
#         color=color,
#         alpha=0.2,
#         linewidth=0.1,
#         label=label,
#     )

#     # Plot injected values
#     injected_values_1, injected_values_2 = injected_values
#     if (injected_values_1 is None) or (injected_values_2 is None):
#         pass
#     else:
#         injected_values_1, injected_values_2 = injected_values
#         injected_values_data_1 = interp1d(
#             log_freqs, injected_values_1, kind="cubic", fill_value="extrapolate", bounds_error=False
#         )(np.log(freqs))
#         injected_values_data_2 = interp1d(
#             log_freqs, injected_values_2, kind="cubic", fill_value="extrapolate", bounds_error=False
#         )(np.log(freqs))
#         injected_values_data_1 = xform(injected_values_data_1)
#         injected_values_data_2 = xform(injected_values_data_2)
#         injected_values_data_relative = injected_values_data_1 - injected_values_data_2
#         plt.plot(freqs, injected_values_data_relative, color="k", ls="-.", lw=1.5)

#     # Plot priors
#     priors_samples_1, priors_samples_2 = priors_samples
#     if (priors_samples_1 is None) or (priors_samples_2 is None):
#         pass
#     else:
#         priors_data_relative = np.zeros((priors_samples_1.shape[0], nfreqs))
#         for i, (sample_1, sample_2) in enumerate(zip(priors_samples_1, priors_samples_2)):
#             temp_1 = interp1d(log_freqs, sample_1, kind="cubic", fill_value=0, bounds_error=False)(np.log(freqs))
#             temp_2 = interp1d(log_freqs, sample_2, kind="cubic", fill_value=0, bounds_error=False)(np.log(freqs))
#             data_1 = xform(temp_1)
#             data_2 = xform(temp_2)
#             priors_data_relative[i] = data_1 - data_2
#         priors_data_relative_summary = SamplesSummary(priors_data_relative, average="median", confidence_level=level)
#         plt.plot(freqs, priors_data_relative_summary.lower_absolute_credible_interval, color=color)
#         plt.plot(freqs, priors_data_relative_summary.upper_absolute_credible_interval, color=color)

#     plt.xlim(freq_points.min() - 0.5, freq_points.max() + 50)
#     plt.legend(loc="upper right", prop={"size": 0.75 * font_size})
