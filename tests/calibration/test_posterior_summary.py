"""Input validation for the per-frequency calibration posterior summary."""

import numpy as np
import pytest

from nullcal.calibration import posterior_median_calibration_factor


def test_posterior_median_rejects_empty_samples():
    knots = np.geomspace(4.0, 32.0, 4)
    empty = np.empty((0, 1, knots.size))

    with pytest.raises(ValueError, match="matching nonempty"):
        posterior_median_calibration_factor(knots, knots, empty, empty)


def test_posterior_median_rejects_detector_by_knot_mismatch():
    knots = np.geomspace(4.0, 32.0, 4)
    samples = np.zeros((2, 2, knots.size))
    one_detector_knots = knots[np.newaxis, :]

    with pytest.raises(ValueError, match="detector-by-knot shape"):
        posterior_median_calibration_factor(knots, one_detector_knots, samples, samples)
