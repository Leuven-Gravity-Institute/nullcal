"""Zero-noise calibration-direction and posterior-summary checks."""

import numpy as np
import pytest

from nullcal.calibration import calibration_factor, posterior_median_calibration_factor
from nullcal.data import InterferometerData
from nullcal.likelihood import RecalibrationLikelihood
from nullcal.likelihood.recalibration_likelihood import ET_BEAM_PATTERN
from nullcal.null_stream.null_stream import NullStream
from nullcal.time_frequency_transform.wavelet_transforms import WaveletTransform


@pytest.mark.parametrize("path", ["likelihood", "null_stream"])
@pytest.mark.parametrize(
    "correction",
    [
        "truth",
        pytest.param("inverse", marks=pytest.mark.xfail(strict=True, raises=AssertionError)),
        pytest.param("conjugate", marks=pytest.mark.xfail(strict=True, raises=AssertionError)),
    ],
)
def test_exact_calibration_truth_removes_zero_noise_null_residual(path, correction, monkeypatch):
    frequencies = np.fft.rfftfreq(64, 1.0 / 64.0)
    mask = (frequencies >= 4.0) & (frequencies <= 32.0)
    knots = np.geomspace(4.0, 32.0, 4)
    amplitude = np.array([[0.08, 0.04, -0.03, 0.02], [-0.06, 0.03, 0.05, -0.02], [0.02, -0.04, 0.07, 0.01]])
    phase = np.array([[0.06, -0.03, 0.02, 0.04], [-0.05, 0.02, 0.04, -0.03], [0.03, 0.05, -0.02, 0.01]])
    factors = np.asarray(
        [calibration_factor(frequencies[mask], knots, a, p) for a, p in zip(amplitude, phase, strict=True)]
    )
    waveforms = np.vstack(
        (
            np.exp(0.04j * frequencies[mask]),
            0.7 * np.exp(-0.07j * frequencies[mask]),
        )
    )
    clean_signal = ET_BEAM_PATTERN @ waveforms
    strain = np.zeros((3, frequencies.size), dtype=np.complex128)
    strain[:, mask] = factors * clean_signal
    data = InterferometerData(
        psd=np.ones((3, frequencies.size)),
        strain=strain,
        mask=np.broadcast_to(mask, strain.shape),
        frequency_array=frequencies,
        duration=1.0,
        sampling_frequency=64.0,
        start_time=0.0,
        name=("ET1", "ET2", "ET3"),
    )
    likelihood = RecalibrationLikelihood(
        data,
        knots,
        time_frequency_filter=np.ones((8, 8), dtype=bool),
        wavelet_transform_frequency_resolution=4.0,
    )

    transform = {"truth": lambda factor: factor, "inverse": lambda factor: 1.0 / factor, "conjugate": np.conj}[
        correction
    ]
    if path == "likelihood":
        if correction != "truth":
            original_factor = likelihood._calibration_factor
            monkeypatch.setattr(likelihood, "_calibration_factor", lambda a, p: transform(original_factor(a, p)))
        residual = np.asarray(likelihood._frequency_domain_null_stream(amplitude, phase))[:, mask]
    else:
        null_stream = NullStream(
            data,
            WaveletTransform(duration=1.0, sampling_frequency=64.0, frequency_resolution=4.0, nx=4.0),
            np.ones((8, 8), dtype=bool),
        )
        full_factors = np.zeros_like(strain)
        full_factors[:, mask] = transform(factors)
        residual = np.asarray(null_stream.compute_calibrated_frequency_domain_null_stream(full_factors))[:, mask]

    relative_residual = np.max(np.abs(residual)) / np.max(np.abs(strain))
    assert relative_residual < 2e-15, f"relative residual {relative_residual:.17g} exceeds 2e-15"


def test_posterior_median_is_taken_after_evaluating_each_factor():
    frequencies = np.geomspace(4.0, 32.0, 17)
    knots = np.geomspace(4.0, 32.0, 4)
    amplitudes = np.zeros((3, 1, 4))
    amplitudes[0, 0, 0] = 0.12
    amplitudes[1, 0, 1] = 0.12
    amplitudes[2, 0, 2] = 0.12
    phases = np.zeros_like(amplitudes)
    phases[0, 0, 1] = 0.04
    phases[1, 0, 2] = -0.03
    phases[2, 0, 3] = 0.05

    sample_factors = np.asarray(
        [calibration_factor(frequencies, knots, a[0], p[0]) for a, p in zip(amplitudes, phases, strict=True)]
    )
    expected = np.median(sample_factors.real, axis=0) + 1j * np.median(sample_factors.imag, axis=0)
    actual = np.asarray(posterior_median_calibration_factor(frequencies, knots, amplitudes, phases))
    median_knot_curve = np.asarray(
        calibration_factor(
            frequencies,
            knots,
            np.median(amplitudes[:, 0], axis=0),
            np.median(phases[:, 0], axis=0),
        )
    )

    np.testing.assert_allclose(actual[0], expected, rtol=0.0, atol=1e-15)
    assert np.max(np.abs(actual[0] - median_knot_curve)) > 1e-3
