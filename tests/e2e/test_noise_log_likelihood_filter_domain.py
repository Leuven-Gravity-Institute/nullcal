"""Regression test for the noise-log-likelihood filter-domain defect.

``NullStream.compute_uncalibrated_time_frequency_domain_null_stream`` applies
``[:, ~time_frequency_filter] = 0.0`` to ``uncalibrated_frequency_domain_null_stream`` — the
*frequency-domain* array of shape ``(detector, frequency)`` — using the *2-D* time-frequency
filter of shape ``(n_t, n_f)``. Indexing a 2-D array with a slice plus a 2-D boolean mask
addresses three dimensions, so the line raises ``IndexError``, and the time-frequency array it
returns is never filtered at all.

Consequences, in order of severity:

1. ``RecalibrationLikelihood.noise_log_likelihood()`` cannot return. Anything that reaches it
   fails, including bilby's ``log_likelihood_ratio``.
2. Had it not raised, the uncalibrated branch would have summed residual energy over *every*
   time-frequency pixel while the calibrated branch sums over the filter only, so their
   difference would have been normalised against different domains.

This test asserts the fixed behaviour. **Verified to fail on the unfixed code** with
``IndexError: too many indices for array`` — a regression test that passes before the fix does
not discriminate.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import pipeline

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.xfail(
        strict=True,
        reason=(
            "Known defect: NullStream.compute_uncalibrated_time_frequency_domain_null_stream "
            "indexes a 2-D frequency-domain array with the 2-D time-frequency filter, so "
            "noise_log_likelihood() raises IndexError. Verified to fail on this revision. "
            "strict=True means these turn into an ERROR the moment the defect is fixed, which "
            "is the signal to delete this marker in the fixing commit."
        ),
    ),
]


@pytest.fixture(scope="module")
def likelihood(tmp_path_factory):
    return pipeline.build_likelihood(tmp_path_factory.mktemp("noise_log_likelihood"))


def test_uncalibrated_time_frequency_null_stream_is_confined_to_the_filter(likelihood):
    """The uncalibrated branch must zero pixels outside the filter, as the calibrated one does."""
    null_stream = likelihood.null_stream_calculator
    uncalibrated = null_stream.compute_uncalibrated_time_frequency_domain_null_stream()
    tf_filter = null_stream.time_frequency_filter

    assert uncalibrated.shape[1:] == tf_filter.shape, (
        f"time-frequency array {uncalibrated.shape} is not shaped by the filter {tf_filter.shape}"
    )
    outside = uncalibrated[:, ~tf_filter]
    assert np.count_nonzero(outside) == 0, (
        f"{np.count_nonzero(outside)} non-zero pixels outside the time-frequency filter; "
        "the uncalibrated branch is summing over a different domain than the calibrated one"
    )


def test_noise_log_likelihood_returns_a_finite_value(likelihood):
    """The public bilby Likelihood method must return, not raise."""
    value = likelihood.noise_log_likelihood()
    assert np.isfinite(value), f"noise_log_likelihood() returned {value!r}"
    assert value <= 0.0, f"noise_log_likelihood() = {value!r}; it is -0.5 * residual energy and cannot be positive"


def test_noise_and_signal_log_likelihood_share_a_summation_domain(likelihood):
    """Both branches must sum residual energy over the same pixels.

    Checked structurally rather than by comparing the two scalars: equal counts of non-zero
    pixels is the invariant that was violated, and it is not implied by the two numbers being
    of similar size.
    """
    null_stream = likelihood.null_stream_calculator
    from . import config

    parameters = config.calibration_parameters()

    uncalibrated = null_stream.compute_uncalibrated_time_frequency_domain_null_stream()
    calibrated = null_stream.compute_calibrated_time_frequency_domain_null_stream_from_parameters(parameters=parameters)
    assert np.count_nonzero(uncalibrated) == np.count_nonzero(calibrated), (
        "uncalibrated and calibrated null streams occupy different numbers of pixels, so "
        "noise_log_likelihood and log_likelihood are normalised against different domains"
    )
