"""Regression test for the noise-log-likelihood filter-domain defect, now fixed.

``NullStream.compute_uncalibrated_time_frequency_domain_null_stream`` used to apply
``[:, ~time_frequency_filter] = 0.0`` to ``uncalibrated_frequency_domain_null_stream`` — the
*frequency-domain* array of shape ``(detector, frequency)`` — using the *2-D* time-frequency
filter of shape ``(n_t, n_f)``. Indexing a 2-D array with a slice plus a 2-D boolean mask
addresses three dimensions, so the line raised ``IndexError``, and the time-frequency array it
returned was never filtered at all.

Consequences it had, in order of severity:

1. ``RecalibrationLikelihood.noise_log_likelihood()`` could not return. Anything that reached it
   failed, including bilby's ``log_likelihood_ratio``.
2. Had it not raised, the uncalibrated branch would have summed residual energy over *every*
   time-frequency pixel while the calibrated branch sums over the filter only, so their
   difference would have been normalised against different domains.

The fix applies the filter to the time-frequency array that is returned, which is what the
calibrated path does in its ``_from_parameters`` wrapper.

These tests assert the fixed behaviour. They were committed alongside the characterisation
harness carrying ``xfail(strict=True, raises=IndexError)``, verified failing on the unfixed code —
a regression test that passes before the fix does not discriminate. The fix removed the markers,
which ``strict=True`` forced: an unexpected pass is an error, so the markers could not be left
behind.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import pipeline

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
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


def test_both_direct_methods_are_confined_to_the_filter(likelihood):
    """Both public time-frequency methods must return filter-confined arrays, not just one.

    The uncalibrated method was the one that was broken, but fixing only it would have left the
    two similarly-named public methods with different contracts — and a caller using both directly
    would then sum energies over different pixel domains, recreating this very defect with no
    exception to announce it.
    """
    from . import config

    null_stream = likelihood.null_stream_calculator
    tf_filter = null_stream.time_frequency_filter
    calibration_factor = null_stream.construct_calibration_factor_from_parameters(config.calibration_parameters())

    uncalibrated = null_stream.compute_uncalibrated_time_frequency_domain_null_stream()
    calibrated = null_stream.compute_calibrated_time_frequency_domain_null_stream(
        calibration_factor=calibration_factor
    )

    for name, array in (("uncalibrated", uncalibrated), ("calibrated", calibrated)):
        outside = int(np.count_nonzero(array[:, ~tf_filter]))
        assert outside == 0, f"{name} direct method leaves {outside} non-zero pixels outside the filter"


def test_noise_log_likelihood_returns_a_finite_value(likelihood):
    """The public bilby Likelihood method must return, not raise."""
    value = likelihood.noise_log_likelihood()
    assert np.isfinite(value), f"noise_log_likelihood() returned {value!r}"
    assert value <= 0.0, f"noise_log_likelihood() = {value!r}; it is -0.5 * residual energy and cannot be positive"


def test_noise_and_signal_log_likelihood_share_a_summation_domain(likelihood):
    """Both branches must sum residual energy over the same pixels.

    Checked structurally rather than by comparing the two scalars: which pixels are summed is
    the invariant that was violated, and it is not implied by the two numbers being of similar
    size.

    Compares the *support* rather than the count of non-zero pixels. Equal counts are a weaker
    statement that two disjoint pixel sets of the same size would also satisfy, so counting
    alone would let a genuinely mismatched domain through.
    """
    null_stream = likelihood.null_stream_calculator
    from . import config

    parameters = config.calibration_parameters()

    uncalibrated = null_stream.compute_uncalibrated_time_frequency_domain_null_stream()
    calibrated = null_stream.compute_calibrated_time_frequency_domain_null_stream_from_parameters(parameters=parameters)
    tf_filter = null_stream.time_frequency_filter

    assert uncalibrated.shape == calibrated.shape, (
        f"uncalibrated {uncalibrated.shape} and calibrated {calibrated.shape} arrays are not comparable"
    )

    # Both must be confined to the filter, and to the *same* pixels within it.
    for name, array in (("uncalibrated", uncalibrated), ("calibrated", calibrated)):
        outside = np.count_nonzero(array[:, ~tf_filter])
        assert outside == 0, f"{name} null stream has {outside} non-zero pixels outside the time-frequency filter"

    mismatched = int(np.count_nonzero((uncalibrated != 0.0) != (calibrated != 0.0)))
    assert mismatched == 0, (
        f"{mismatched} pixels are non-zero in one null stream but not the other, so "
        "noise_log_likelihood and log_likelihood are normalised against different domains"
    )

    # Same domain, different values. Without this, the cheapest wrong fix passes: routing the
    # uncalibrated branch through the calibrated computation makes the two arrays *identical*,
    # which satisfies every structural check above — confined to the filter, finite, equal
    # support — while destroying the quantity's meaning, since noise_log_likelihood would then
    # be the signal likelihood and their difference identically zero.
    #
    # The threshold is 1e-6 against a *measured* separation of ~0.8 peak-relative, with round-off
    # at ~1e-15: six orders of margin below the real value and nine above the noise floor. Do not
    # "tighten" this toward the ~2% scale of the injected calibration error — that intuition is
    # wrong and re-creates the hole. The calibration rotates the projector's null subspace, and
    # projecting O(1) noise onto two subspaces differing by ~2% gives an O(1) difference, not a 2%
    # one.
    assert not np.array_equal(uncalibrated, calibrated), (
        "uncalibrated and calibrated null streams are bit-identical; the uncalibrated branch is "
        "applying the calibration factor, so noise_log_likelihood is not a noise likelihood"
    )
    separation = float(np.max(np.abs(uncalibrated - calibrated))) / float(np.max(np.abs(calibrated)))
    assert separation > 1e-6, (
        f"uncalibrated and calibrated null streams differ by only {separation:.3e} peak-relative; "
        "the injected calibration error is ~2%, so this is too small for the calibration to have "
        "been applied to one and not the other"
    )
