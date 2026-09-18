"""Small result-processing helpers."""

from __future__ import annotations

import numpy as np


def spline_percentage_xform(delta_a: float | np.ndarray) -> float | np.ndarray:
    """Convert fractional amplitude error to percent."""
    return delta_a * 100


__all__ = ["spline_percentage_xform"]
