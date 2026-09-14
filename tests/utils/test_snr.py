"""Tests for the uncorrelated null-stream optimal SNR.

``optimal_uncorrelated_null_stream_snr_squared`` forms the direct sum of the detector signals,
normalised by :math:`\\sqrt{N_{\\rm det}}`, against the mean of the detector noise power spectra,
and takes the noise-weighted inner product of that with itself.

The anchors are analytic. For :math:`N` detectors carrying the *same* signal and the *same* noise
curve the construction is a coherent sum, so the result is :math:`N` times one detector's optimal
:math:`\\rho^2`; and for signals that cancel, the null stream is empty and :math:`\\rho^2` is zero.
Neither number comes from running this code.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.utils.snr import optimal_uncorrelated_null_stream_snr_squared

DURATION = 4.0
N_FREQUENCIES = 256


def analytic_inner_product(signal, power_spectral_density, duration):
    """One-sided discrete noise-weighted norm, ``4 / duration * sum(|h|^2 / S)``.

    This is the textbook definition evaluated directly, rather than a call to bilby's helper (the
    same helper used by the production function under test).
    """
    return 4.0 / duration * np.sum(np.abs(signal) ** 2 / power_spectral_density)


@pytest.fixture
def signal():
    """A smooth, non-trivial frequency-domain signal at gravitational-wave strain amplitude."""
    frequency = np.linspace(20.0, 512.0, N_FREQUENCIES)
    return 1e-23 * (frequency / 100.0) ** (-7 / 6) * np.exp(1j * frequency / 30.0)


@pytest.fixture
def power_spectral_density():
    frequency = np.linspace(20.0, 512.0, N_FREQUENCIES)
    return 1e-46 * (1.0 + (frequency / 100.0) ** 2)


@pytest.mark.unit
def test_single_detector_reduces_to_the_optimal_snr(signal, power_spectral_density):
    """With one detector the null stream is that detector, so the result is its own :math:`\\rho^2`.

    Anchored on the discrete one-sided noise-weighted inner product
    :math:`4/T \\sum_k |h_k|^2/S_k`, evaluated directly here.  Calling bilby's helper would not be
    independent: production delegates to that same helper.
    """
    result = optimal_uncorrelated_null_stream_snr_squared(
        signals=np.array([signal]),
        power_spectral_densities=np.array([power_spectral_density]),
        duration=DURATION,
    )

    expected = analytic_inner_product(signal, power_spectral_density, DURATION)
    assert np.real(result) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("n_detectors", [2, 3, 5])
def test_identical_detectors_add_coherently(signal, power_spectral_density, n_detectors):
    """:math:`N` identical detectors give exactly :math:`N \\times` one detector's :math:`\\rho^2`.

    This is the defining property of a coherent sum and it fixes the :math:`1/\\sqrt{N}`
    normalisation: an incoherent combination would give :math:`\\rho^2` independent of :math:`N`,
    and an unnormalised sum would give :math:`N^2`. Both wrong answers are excluded by testing more
    than one value of :math:`N`.
    """
    single = optimal_uncorrelated_null_stream_snr_squared(
        signals=np.array([signal]),
        power_spectral_densities=np.array([power_spectral_density]),
        duration=DURATION,
    )

    combined = optimal_uncorrelated_null_stream_snr_squared(
        signals=np.array([signal] * n_detectors),
        power_spectral_densities=np.array([power_spectral_density] * n_detectors),
        duration=DURATION,
    )

    assert np.real(combined) == pytest.approx(n_detectors * np.real(single), rel=1e-12)


@pytest.mark.unit
def test_cancelling_signals_give_a_vanishing_null_stream(signal, power_spectral_density):
    """Signals that sum to zero leave no energy in the null stream.

    This is the property the whole package is built on: a correctly modelled network has a null
    combination in which the signal cancels. Compared against zero with ``atol=0.0`` and a scale set
    by the non-cancelling case, since at strain amplitude an absolute tolerance drawn from
    ``numpy``'s defaults would accept any answer at all.
    """
    non_cancelling = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal, signal]),
            power_spectral_densities=np.array([power_spectral_density] * 2),
            duration=DURATION,
        )
    )

    result = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal, -signal]),
            power_spectral_densities=np.array([power_spectral_density] * 2),
            duration=DURATION,
        )
    )

    assert non_cancelling > 0.0
    assert abs(result) / non_cancelling < 1e-25


@pytest.mark.unit
def test_snr_squared_scales_quadratically_with_signal_amplitude(signal, power_spectral_density):
    """:math:`\\rho^2` is quadratic in the strain: doubling the signal quadruples it."""
    base = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal, signal]),
            power_spectral_densities=np.array([power_spectral_density] * 2),
            duration=DURATION,
        )
    )

    doubled = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([2 * signal, 2 * signal]),
            power_spectral_densities=np.array([power_spectral_density] * 2),
            duration=DURATION,
        )
    )

    assert doubled == pytest.approx(4.0 * base, rel=1e-12)


@pytest.mark.unit
def test_noise_spectra_are_averaged_not_summed(signal, power_spectral_density):
    """The effective spectrum is the *mean* of the detectors', so identical detectors keep theirs.

    Summing instead would divide :math:`\\rho^2` by the detector count and cancel the coherent gain
    exactly, leaving a result that looks plausible and is independent of the network size. Pinned by
    giving two detectors different spectra and comparing against the mean computed here.
    """
    other_spectrum = 3.0 * power_spectral_density

    result = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal, signal]),
            power_spectral_densities=np.array([power_spectral_density, other_spectrum]),
            duration=DURATION,
        )
    )

    mean_spectrum = (power_spectral_density + other_spectrum) / 2
    null_stream = 2 * signal / np.sqrt(2)
    expected = analytic_inner_product(null_stream, mean_spectrum, DURATION)
    assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_snr_squared_is_inversely_proportional_to_the_noise_power(signal, power_spectral_density):
    """Quadrupling the noise power quarters :math:`\\rho^2` -- the weighting is :math:`1/S(f)`."""
    base = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal]),
            power_spectral_densities=np.array([power_spectral_density]),
            duration=DURATION,
        )
    )

    noisier = np.real(
        optimal_uncorrelated_null_stream_snr_squared(
            signals=np.array([signal]),
            power_spectral_densities=np.array([4.0 * power_spectral_density]),
            duration=DURATION,
        )
    )

    assert noisier == pytest.approx(base / 4.0, rel=1e-12)
