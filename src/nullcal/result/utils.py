import numpy as np
from scipy.interpolate import interp1d

from bilby.core.utils import SamplesSummary


def plot_spline_pos(log_freqs, samples, minimum_frequency, maximum_frequency, nfreqs=100, level=0.9, injected_values=None, priors=None, errorbar=False, color='k', label=None, xform=None):
    """
    Plot calibration posterior estimates for a spline model in log space.
    Adapted from the same function in bilby.gw.utils

    Parameters
    ==========
    log_freqs: array-like
        The (log) location of spline control points.
    samples: array-like
        List of posterior draws of function at control points ``log_freqs``
    minimum_frequency: float
        Minimum frequency for plotting.
    maximum_frequency: float
        Maximum frequency for plotting.
    nfreqs: int
        Number of points to evaluate spline at for plotting.
    level: float
        Credible level to fill in.
    injected_values: array-like
        List of injected values at control points ``log_freqs``
    priors: array-like
        List of priors at control points ``log_freqs``
    errorbar: bool
        If True, plot the posterior draws errorbars of function at control points ``log_freqs``
    color: str
        Color to plot with.
    label: str
        Label for plot.
    xform: callable
        Function to transform the spline into plotted values.
    """

    import matplotlib.pyplot as plt

    font_size = 32

    freq_points = np.exp(log_freqs)
    if minimum_frequency is None:
        minimum_frequency = min(log_freqs)
    if maximum_frequency is None:
        maximum_frequency = max(log_freqs)
    freqs = np.logspace(minimum_frequency, maximum_frequency, nfreqs, base=np.exp(1))

    # Retrieve posterior samples
    if xform is None:
        scaled_samples = samples
    else:
        scaled_samples = xform(samples)
    scaled_samples_summary = SamplesSummary(scaled_samples, average='mean', confidence_level=level)

    # Plot errorbar
    if errorbar:
        plt.errorbar(freq_points, scaled_samples_summary.average,
                     yerr=[-scaled_samples_summary.lower_relative_credible_interval,
                           scaled_samples_summary.upper_relative_credible_interval],
                     fmt='.', color=color, lw=4, alpha=0.5, capsize=0)

    # Plot posterior samples
    data = np.zeros((samples.shape[0], nfreqs))
    for i, sample in enumerate(samples):
        temp = interp1d(log_freqs, sample, kind="cubic", fill_value=0,
                        bounds_error=False)(np.log(freqs))
        if xform is None:
            data[i] = temp
        else:
            data[i] = xform(temp)
    data_summary = SamplesSummary(data, average='mean', confidence_level=level)
    plt.fill_between(freqs, data_summary.lower_absolute_credible_interval,
                     data_summary.upper_absolute_credible_interval, color=color, alpha=.2, linewidth=0.1, label=label)

    # Plot injected values
    if injected_values in [False, None]:
        pass
    else:
        injected_values_data = np.zeros((len(injected_values), nfreqs))
        temp = interp1d(log_freqs, injected_values, kind="cubic",
                        fill_value=0, bounds_error=False)(np.log(freqs))
        if xform is None:
            injected_values_data = 100 * temp
        else:
            injected_values_data = xform(temp)
        plt.plot(freqs, injected_values_data, color='k', ls='-.', lw=1.5)

    # Plot priors
    if priors in [False, None]:
        pass
    else:
        # Sample the priors
        n_samples = 1000
        priors_samples = np.transpose([prior.sample(n_samples) for prior in priors])

        # Plot the priors
        priors_data = np.zeros((priors_samples.shape[0], nfreqs))
        for i, sample in enumerate(priors_samples):
            temp = interp1d(log_freqs, sample, kind="cubic", fill_value=0,
                            bounds_error=False)(np.log(freqs))
            if xform is None:
                priors_data[i] = 100 * temp
            else:
                priors_data[i] = xform(temp)
        priors_data_summary = SamplesSummary(priors_data, average='mean', confidence_level=level)
        plt.plot(freqs, priors_data_summary.lower_absolute_credible_interval, color=color)
        plt.plot(freqs, priors_data_summary.upper_absolute_credible_interval, color=color)

    plt.xlim(freq_points.min() - .5, freq_points.max() + 50)
    plt.legend(loc='upper right', prop={'size': .75 * font_size})
