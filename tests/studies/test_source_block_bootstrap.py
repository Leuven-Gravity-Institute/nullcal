"""References for whole-source resampling and one-sided BCa bounds."""

import numpy as np
import pytest
from scipy.stats import bootstrap

from nullcal.studies.bootstrap import SourceBlockQuantileBound, source_block_quantile_upper


@pytest.mark.parametrize("confidence", [0.1, 0.9, 0.95])
def test_repeated_rows_reduce_to_independent_scalar_median(confidence):
    # Repeating each scalar an odd number of times preserves the median for
    # every resample and every delete-one-source jackknife sample.
    sources = np.array([-4.0, -2.0, -1.0, 0.0, 0.5, 2.0, 8.0])
    values = np.repeat(sources[:, None], 3, axis=1)
    reference = bootstrap(
        (sources,),
        np.median,
        method="BCa",
        alternative="less",
        confidence_level=confidence,
        n_resamples=256,
        batch=32,
        rng=np.random.default_rng(42),
    )
    result = source_block_quantile_upper(
        values,
        quantile=0.5,
        confidence_level=confidence,
        resamples=256,
        seed=42,
    )
    assert isinstance(result, SourceBlockQuantileBound)
    assert result.estimate == np.median(sources)
    assert result.upper_bound == reference.confidence_interval.high
    assert result.standard_error == reference.standard_error
    np.testing.assert_array_equal(result.bootstrap_distribution, reference.bootstrap_distribution)
    if confidence == 0.1:
        assert result.upper_bound < result.estimate


def test_bootstrap_distribution_preserves_complete_rows():
    values = np.arange(33.0).reshape(11, 3) ** 2
    indices = np.random.default_rng(123).integers(0, 11, size=(128, 11))
    expected = [np.quantile(values[row].ravel(), 0.8, method="linear") for row in indices]
    result = source_block_quantile_upper(values, quantile=0.8, resamples=128, seed=123)
    np.testing.assert_array_equal(result.bootstrap_distribution, expected)
    assert len(result.bootstrap_distribution) == 128


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[]],
        np.zeros((2, 4)),
        np.zeros((3, 0)),
        np.zeros(4),
        np.zeros((3, 2, 1)),
        np.full((3, 2), np.nan),
        np.full((3, 2), np.inf),
        np.ones((3, 2), dtype=complex),
    ],
)
def test_rejects_invalid_source_matrices(values):
    with pytest.raises(ValueError, match="values"):
        source_block_quantile_upper(values, resamples=32)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantile": 0},
        {"quantile": 1},
        {"quantile": np.nan},
        {"quantile": True},
        {"confidence_level": 0},
        {"confidence_level": 1},
        {"confidence_level": np.inf},
        {"resamples": 0},
        {"resamples": 1},
        {"resamples": 2.5},
        {"resamples": True},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        source_block_quantile_upper(np.arange(12.0).reshape(4, 3), **kwargs)


def test_degenerate_bca_is_unresolved():
    with pytest.raises(ValueError, match=r"BCa.*unresolved"):
        source_block_quantile_upper(np.ones((8, 3)), resamples=128)


def test_bca_is_distinct_from_the_percentile_bound():
    sources = np.arange(50.0) ** 3

    def statistic(sample):
        return np.quantile(sample, 0.8, method="linear")

    reference = bootstrap(
        (sources,),
        statistic,
        vectorized=False,
        method="BCa",
        alternative="less",
        confidence_level=0.95,
        n_resamples=1024,
        batch=32,
        rng=np.random.default_rng(741),
    )
    percentile = np.quantile(reference.bootstrap_distribution, 0.95, method="linear")
    assert reference.confidence_interval.high != percentile
    result = source_block_quantile_upper(
        sources[:, None],
        quantile=0.8,
        confidence_level=0.95,
        resamples=1024,
        seed=741,
    )
    assert result.upper_bound == reference.confidence_interval.high
    assert result.upper_bound != percentile
    assert result.support_saturated is False


def test_rejects_masked_values_instead_of_using_hidden_entries():
    values = np.ma.masked_equal(np.arange(12.0).reshape(4, 3), 3.0)
    with pytest.raises(ValueError, match="values"):
        source_block_quantile_upper(values, resamples=128)


@pytest.mark.parametrize("container", [list, tuple])
def test_rejects_masked_rows_inside_a_sequence(container):
    matrix = np.ma.masked_equal(np.arange(12.0).reshape(4, 3), 3.0)
    values = container(matrix[row] for row in range(4))
    with pytest.raises(ValueError, match="values"):
        source_block_quantile_upper(values, resamples=64)


@pytest.mark.parametrize("dtype", [str, bool, object])
def test_rejects_non_numeric_or_boolean_matrix_types(dtype):
    values = np.arange(12.0).reshape(4, 3).astype(dtype)
    with pytest.raises(ValueError, match="values"):
        source_block_quantile_upper(values, resamples=64)


def test_reports_bootstrap_support_boundary_saturation():
    values = np.array([0.0, 0.01, 0.02, 0.03, 100.0])[:, None]
    result = source_block_quantile_upper(values, resamples=256, seed=5)
    assert result.upper_bound == max(result.bootstrap_distribution)
    assert getattr(result, "support_saturated", None) is True
