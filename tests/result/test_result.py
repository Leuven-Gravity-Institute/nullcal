from unittest.mock import MagicMock, patch

import numpy as np

from nullcal.result import plot_calibration_posterior
from nullcal.utils import plot_spline_pos

# Mock SamplesSummary class


class MockSamplesSummary:
    def __init__(self, data, average="mean", confidence_level=0.9):
        self.average = np.mean(data, axis=0) if average == "mean" else np.median(data, axis=0)
        self.lower_relative_credible_interval = self.average - np.percentile(
            data, (1 - confidence_level) / 2 * 100, axis=0
        )
        self.upper_relative_credible_interval = (
            np.percentile(data, (1 + confidence_level) / 2 * 100, axis=0) - self.average
        )
        self.lower_absolute_credible_interval = np.percentile(data, (1 - confidence_level) / 2 * 100, axis=0)
        self.upper_absolute_credible_interval = np.percentile(data, (1 + confidence_level) / 2 * 100, axis=0)


# Mock spline_angle_xform function


def mock_spline_angle_xform(x):
    return np.rad2deg(x)


# Test plot_spline_pos


@patch("matplotlib.pyplot.errorbar")
@patch("matplotlib.pyplot.fill_between")
@patch("matplotlib.pyplot.plot")
@patch("matplotlib.pyplot.xlim")
@patch("matplotlib.pyplot.legend")
@patch("scipy.interpolate.interp1d")
def test_plot_spline_pos_basic(mock_interp1d, mock_legend, mock_xlim, mock_plot, mock_fill_between, mock_errorbar):
    # Setup test data
    log_freqs = np.log([10, 100, 1000])
    samples = np.random.randn(100, 3)  # 100 samples, 3 control points
    minimum_frequency = 1
    maximum_frequency = 4
    nfreqs = 50
    level = 0.9
    color = "k"
    label = "Test"

    # Mock interp1d
    mock_interpolator = MagicMock()
    mock_interpolator.return_value = np.random.randn(nfreqs)
    mock_interp1d.return_value = mock_interpolator

    # Mock SamplesSummary
    with patch("__main__.SamplesSummary", MockSamplesSummary):
        # Call the function
        plot_spline_pos(
            log_freqs,
            samples,
            minimum_frequency,
            maximum_frequency,
            nfreqs,
            level,
            injected_values=None,
            priors=None,
            errorbar=False,
            color=color,
            label=label,
            xform=None,
        )

    # Verify calls
    assert mock_interp1d.called
    mock_fill_between.assert_called_once()
    mock_xlim.assert_called_once()
    mock_legend.assert_called_once()
    mock_errorbar.assert_not_called()
    mock_plot.assert_not_called()


@patch("matplotlib.pyplot.errorbar")
@patch("matplotlib.pyplot.fill_between")
@patch("matplotlib.pyplot.plot")
@patch("scipy.interpolate.interp1d")
def test_plot_spline_pos_with_errorbar_and_xform(mock_interp1d, mock_plot, mock_fill_between, mock_errorbar):
    # Setup test data
    log_freqs = np.log([10, 100, 1000])
    samples = np.random.randn(100, 3)
    minimum_frequency = 1
    maximum_frequency = 4
    nfreqs = 50
    level = 0.9
    color = "b"
    xform = mock_spline_angle_xform

    # Mock interp1d
    mock_interpolator = MagicMock()
    mock_interpolator.return_value = np.random.randn(nfreqs)
    mock_interp1d.return_value = mock_interpolator

    # Mock SamplesSummary
    with patch("__main__.SamplesSummary", MockSamplesSummary):
        plot_spline_pos(
            log_freqs,
            samples,
            minimum_frequency,
            maximum_frequency,
            nfreqs,
            level,
            injected_values=None,
            priors=None,
            errorbar=True,
            color=color,
            label=None,
            xform=xform,
        )

    # Verify calls
    mock_errorbar.assert_called_once()
    mock_fill_between.assert_called_once()
    mock_plot.assert_not_called()


@patch("matplotlib.pyplot.plot")
@patch("scipy.interpolate.interp1d")
def test_plot_spline_pos_with_injected_values(mock_interp1d, mock_plot):
    # Setup test data
    log_freqs = np.log([10, 100, 1000])
    samples = np.random.randn(100, 3)
    injected_values = np.random.randn(3)
    minimum_frequency = 1
    maximum_frequency = 4
    nfreqs = 50
    level = 0.9

    # Mock interp1d
    mock_interpolator = MagicMock()
    mock_interpolator.return_value = np.random.randn(nfreqs)
    mock_interp1d.return_value = mock_interpolator

    # Mock SamplesSummary
    with patch("__main__.SamplesSummary", MockSamplesSummary):
        plot_spline_pos(
            log_freqs,
            samples,
            minimum_frequency,
            maximum_frequency,
            nfreqs,
            level,
            injected_values=injected_values,
            priors=None,
            errorbar=False,
        )

    # Verify injected values plotting
    assert mock_plot.call_count == 1


@patch("matplotlib.pyplot.plot")
@patch("scipy.interpolate.interp1d")
def test_plot_spline_pos_with_priors(mock_interp1d, mock_plot):
    # Setup test data
    log_freqs = np.log([10, 100, 1000])
    samples = np.random.randn(100, 3)
    priors = [MagicMock() for _ in range(3)]  # Mock prior objects
    for prior in priors:
        prior.sample.return_value = np.random.randn(1000)
    minimum_frequency = 1
    maximum_frequency = 4
    nfreqs = 50
    level = 0.9

    # Mock interp1d
    mock_interpolator = MagicMock()
    mock_interpolator.return_value = np.random.randn(nfreqs)
    mock_interp1d.return_value = mock_interpolator

    # Mock SamplesSummary
    with patch("__main__.SamplesSummary", MockSamplesSummary):
        plot_spline_pos(
            log_freqs,
            samples,
            minimum_frequency,
            maximum_frequency,
            nfreqs,
            level,
            injected_values=None,
            priors=priors,
            errorbar=False,
        )

    # Verify priors plotting
    assert mock_plot.call_count == 2  # Two calls for upper and lower credible intervals


# Test plot_calibration_posterior


@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.sca")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.show")
@patch("plot_functions.plot_spline_pos")
def test_plot_calibration_posterior_no_ifos(
    mock_plot_spline_pos, mock_show, mock_tight_layout, mock_sca, mock_subplots
):
    # Mock posterior with no calibration parameters
    mock_posterior = MagicMock()
    mock_posterior.keys.return_value = ["param1", "param2"]
    result = MagicMock()
    result.posterior = mock_posterior

    plot_calibration_posterior(result)

    # Verify no plotting occurs
    mock_subplots.assert_not_called()
    mock_plot_spline_pos.assert_not_called()


@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.sca")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.show")
@patch("plot_functions.plot_spline_pos")
def test_plot_calibration_posterior_with_ifos(
    mock_plot_spline_pos, mock_show, mock_tight_layout, mock_sca, mock_subplots
):
    # Mock posterior with calibration parameters
    mock_posterior = MagicMock()
    mock_posterior.keys.return_value = ["recalib_ET1_frequency_0", "recalib_ET1_amplitude_0", "recalib_ET1_phase_0"]
    mock_posterior.__getitem__.return_value.iloc = [np.log(10)]  # Mock frequency
    result = MagicMock()
    result.posterior = mock_posterior
    result.injection_parameters = None
    result.priors = None

    # Mock subplots
    mock_ax1, mock_ax2 = MagicMock(), MagicMock()
    mock_subplots.return_value = (MagicMock(), [mock_ax1, mock_ax2])

    plot_calibration_posterior(result, minimum_frequency=1, maximum_frequency=4, quantile=0.95)

    # Verify plotting calls
    assert mock_subplots.called
    assert mock_plot_spline_pos.call_count == 2  # Once for amplitude, once for phase
    mock_ax1.set_xscale.assert_called_with("log")
    mock_ax2.set_xscale.assert_called_with("log")
    mock_ax1.set_ylabel.assert_called_with(r"Amplitude [$\%$]", fontsize=10)
    mock_ax2.set_ylabel.assert_called_with("Phase [deg]", fontsize=10)
    mock_ax2.set_xlabel.assert_called_with("Frequency [Hz]", fontsize=10)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.sca")
@patch("plot_functions.safe_save_figure")
@patch("plot_functions.plot_spline_pos")
def test_plot_calibration_posterior_save_file(mock_plot_spline_pos, mock_save_figure, mock_sca, mock_subplots):
    # Mock posterior with calibration parameters
    mock_posterior = MagicMock()
    mock_posterior.keys.return_value = ["recalib_ET1_frequency_0", "recalib_ET1_amplitude_0", "recalib_ET1_phase_0"]
    mock_posterior.__getitem__.return_value.iloc = [np.log(10)]
    result = MagicMock()
    result.posterior = mock_posterior
    result.injection_parameters = {"recalib_ET1_amplitude_0": 0.1, "recalib_ET1_phase_0": 0.05}
    result.priors = {"recalib_ET1_amplitude_0": MagicMock(), "recalib_ET1_phase_0": MagicMock()}

    # Mock subplots
    mock_ax1, mock_ax2 = MagicMock(), MagicMock()
    mock_subplots.return_value = (MagicMock(), [mock_ax1, mock_ax2])

    plot_calibration_posterior(result, filename="test.png", show_injected_value=True, show_priors=True)

    # Verify file saving
    assert mock_subplots.called
    assert mock_plot_spline_pos.call_count == 2
    mock_save_figure.assert_called_once()
