"""Source-cluster quantile uncertainty utilities."""

import warnings
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import DegenerateDataWarning, bootstrap

_MIN_RESAMPLES = 2
_MIN_SOURCE_COUNT = 3
_SOURCE_MATRIX_NDIM = 2


@dataclass(frozen=True)
class SourceBlockQuantileBound:
    """A nominal one-sided BCa result and its resampling distribution."""

    estimate: float
    upper_bound: float
    standard_error: float
    bootstrap_distribution: tuple[float, ...]


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite probability strictly between zero and one")
    return float(value)


def source_block_quantile_upper(
    values: ArrayLike,
    *,
    quantile: float = 0.95,
    confidence_level: float = 0.95,
    resamples: int = 9999,
    seed: int = 0,
) -> SourceBlockQuantileBound:
    """Return a pointwise BCa upper bound for a pooled repeated-source quantile.

    ``values`` has shape (independent sources, repeats per source). A resample
    selects complete source rows, retaining all repeats, and the acceleration
    jackknife deletes one complete source. The quantile uses NumPy's ``linear``
    convention over the pooled values. At least three sources and one repeat
    are required; unequal-sized clusters or non-finite/missing values are not
    accepted. Sources, rather than individual repeats, must be independent.

    This delegates BCa to ``scipy.stats.bootstrap`` with ``alternative='less'``
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
    It does not clamp the upper bound above the observed estimate. Degenerate
    or non-finite intervals raise ValueError, with no percentile fallback.
    The returned nominal level is not evidence of empirical coverage for a
    non-smooth quantile, nor a simultaneous confidence band across many calls.
    Callers must assess resampling precision and sampling-model validity.
    """
    quantile = _probability(quantile, "quantile")
    confidence_level = _probability(confidence_level, "confidence_level")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples < _MIN_RESAMPLES:
        raise ValueError("resamples must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if np.ma.is_masked(values):
        raise ValueError("values must not contain masked entries")
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("values must be real")
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be a finite real source-by-repeat matrix") from exc
    if (
        matrix.ndim != _SOURCE_MATRIX_NDIM
        or matrix.shape[0] < _MIN_SOURCE_COUNT
        or matrix.shape[1] < 1
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("values must be finite, with at least three source rows and one repeat column")

    def statistic(indices: np.ndarray) -> float:
        return float(np.quantile(matrix[indices], quantile, method="linear"))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateDataWarning)
            warnings.simplefilter("error", RuntimeWarning)
            result = bootstrap(
                (np.arange(matrix.shape[0]),),
                statistic,
                vectorized=False,
                method="BCa",
                alternative="less",
                confidence_level=confidence_level,
                n_resamples=int(resamples),
                batch=32,
                rng=np.random.default_rng(int(seed)),
            )
    except (DegenerateDataWarning, RuntimeWarning) as exc:
        raise ValueError("BCa bound unresolved: degenerate or numerically invalid resampling distribution") from exc
    upper_bound = float(result.confidence_interval.high)
    standard_error = float(result.standard_error)
    if (
        not np.isfinite(upper_bound)
        or not np.isfinite(standard_error)
        or not np.all(np.isfinite(result.bootstrap_distribution))
    ):
        raise ValueError("BCa bound unresolved: non-finite interval or resampling distribution")
    return SourceBlockQuantileBound(
        estimate=float(np.quantile(matrix, quantile, method="linear")),
        upper_bound=upper_bound,
        standard_error=standard_error,
        bootstrap_distribution=tuple(float(value) for value in result.bootstrap_distribution),
    )


__all__ = ["SourceBlockQuantileBound", "source_block_quantile_upper"]
