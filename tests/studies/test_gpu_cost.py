from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nullcal.studies import gpu_cost


def _device(platform: str, index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(platform=platform, id=index)


def test_devices_with_platform_filters_by_platform():
    devices = (_device("cpu", 0), _device("gpu", 1), _device("gpu", 2))

    assert [device.id for device in gpu_cost.devices_with_platform("gpu", devices)] == [1, 2]
    assert [device.id for device in gpu_cost.devices_with_platform("cpu", devices)] == [0]


def test_require_gpu_devices_rejects_a_cpu_only_node():
    with pytest.raises(RuntimeError, match="no GPU device present"):
        gpu_cost.require_gpu_devices((_device("cpu", 0),))


def test_require_gpu_devices_returns_the_gpu_devices():
    devices = (_device("cpu", 0), _device("gpu", 1))

    assert gpu_cost.require_gpu_devices(devices) == (devices[1],)


def test_summarise_seconds_reports_minimum_median_and_mean():
    summary = gpu_cost.summarise_seconds([3.0, 1.0, 2.0])

    assert summary == {"count": 3, "minimum_seconds": 1.0, "median_seconds": 2.0, "mean_seconds": 2.0}


@pytest.mark.parametrize("sample", [[], [-1.0], [float("nan")]])
def test_summarise_seconds_rejects_unusable_samples(sample):
    with pytest.raises(ValueError, match=r"at least one timing sample|finite and non-negative"):
        gpu_cost.summarise_seconds(sample)


def test_speedup_rejects_zero_and_is_cpu_over_accelerator():
    assert gpu_cost.speedup(10.0, 2.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="both timings must be positive"):
        gpu_cost.speedup(10.0, 0.0)


def test_decompose_per_posterior_splits_likelihood_and_overhead():
    decomposition = gpu_cost.decompose_per_posterior(
        seconds_per_posterior=10.0, gradient_seconds=0.5, gradients_per_posterior=12.0
    )

    assert decomposition == {"likelihood_seconds": 6.0, "sampler_overhead_seconds": 4.0}


def test_decompose_per_posterior_does_not_report_negative_overhead():
    decomposition = gpu_cost.decompose_per_posterior(1.0, 0.5, 12.0)

    assert decomposition["sampler_overhead_seconds"] == 0.0


def test_campaign_gpu_hours_multiplies_the_posterior_count():
    pricing = gpu_cost.campaign_gpu_hours(
        seconds_per_posterior=1.0, event_count=2, noise_realisations=3, calibration_draws=4
    )

    assert pricing["posterior_count"] == 24
    assert pricing["gpu_hours"] == pytest.approx(24.0 / 3600.0)


def test_campaign_gpu_hours_rejects_a_zero_factor():
    with pytest.raises(ValueError, match="event_count"):
        gpu_cost.campaign_gpu_hours(1.0, 0, 1, 1)


def test_synthetic_realisations_are_shape_and_seed_stable():
    first = gpu_cost.synthetic_realisations((2, 5), 3, seed=7)
    second = gpu_cost.synthetic_realisations((2, 5), 3, seed=7)

    assert first.shape == (3, 2, 5)
    assert first.dtype == np.complex128
    np.testing.assert_array_equal(first, second)


def test_fresh_complex_array_has_the_requested_shape_and_dtype():
    import jax

    array = gpu_cost.fresh_complex_array(jax.random.key(0), (2, 3))

    assert array.shape == (2, 3)
    assert array.dtype == jnp.complex128


def test_time_call_regenerates_inputs_for_every_iteration():
    seen: list[int] = []

    def function(value):
        return jnp.sum(value)

    def make_inputs(index):
        seen.append(index)
        return (jnp.asarray(float(index)),)

    times = gpu_cost.time_call(function, make_inputs, repeats=3, warmup=1)

    assert len(times) == 3
    assert seen == [0, 1, 2, 3, 4]


def test_build_synthetic_likelihood_supports_the_per_realisation_path():
    knots = np.geomspace(8.0, 2048.0, 4)
    likelihood = gpu_cost.build_synthetic_likelihood(
        duration=1.0,
        sampling_frequency=256,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        knot_frequencies=knots,
    )
    params = {"amplitude": jnp.zeros((3, 4)), "phase": jnp.zeros((3, 4))}
    realisations = np.zeros((2, 3, likelihood._frequency_count), dtype=np.complex128)

    batched = likelihood.log_likelihood_for_strain(params, jnp.asarray(realisations))

    assert batched.shape == (2,)
    assert float(batched[0]) == pytest.approx(float(batched[1]), abs=1e-12)
