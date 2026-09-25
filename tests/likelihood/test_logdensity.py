from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nullcal.data import InterferometerData
from nullcal.likelihood import RecalibrationLikelihood


@pytest.fixture
def likelihood() -> RecalibrationLikelihood:
    duration = 1.0
    sampling_frequency = 64.0
    frequencies = np.fft.rfftfreq(int(duration * sampling_frequency), 1.0 / sampling_frequency)
    mask = np.broadcast_to((frequencies >= 4.0) & (frequencies <= 32.0), (3, frequencies.size)).copy()
    rng = np.random.default_rng(393)
    strain = rng.normal(size=(3, frequencies.size)) + 1j * rng.normal(size=(3, frequencies.size))
    data = InterferometerData(
        psd=np.ones_like(strain.real),
        strain=strain,
        mask=mask,
        frequency_array=frequencies,
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=0.0,
        name=("ET1", "ET2", "ET3"),
    )
    return RecalibrationLikelihood(
        interferometers=data,
        knot_frequencies=np.geomspace(4.0, 32.0, 4),
        time_frequency_filter=np.ones((8, 8), dtype=bool),
        wavelet_transform_frequency_resolution=4.0,
        amplitude_prior_sigma=0.05,
        phase_prior_sigma=0.05,
    )


def test_logdensity_is_pure_jittable_and_differentiable(likelihood):
    params = {
        "amplitude": jnp.zeros((3, 4), dtype=jnp.float64),
        "phase": jnp.zeros((3, 4), dtype=jnp.float64),
    }

    eager = likelihood.logdensity_fn(params)
    compiled = jax.jit(likelihood.logdensity_fn)(params)
    gradient = jax.grad(likelihood.logdensity_fn)(params)

    assert eager.shape == ()
    assert eager.dtype == jnp.float64
    assert float(compiled) == pytest.approx(float(eager), rel=1e-13)
    assert all(np.all(np.isfinite(value)) for value in gradient.values())
    assert float(likelihood.logdensity_fn(params)) == float(eager)


def test_logdensity_includes_the_normalized_gaussian_prior(likelihood):
    zeros = jnp.zeros((3, 4), dtype=jnp.float64)
    shifted = zeros.at[0, 0].set(0.05)

    zero_params = {"amplitude": zeros, "phase": zeros}
    shifted_params = {"amplitude": shifted, "phase": zeros}
    likelihood_change = likelihood.log_likelihood_fn(shifted_params) - likelihood.log_likelihood_fn(zero_params)
    density_change = likelihood.logdensity_fn(shifted_params) - likelihood.logdensity_fn(zero_params)

    assert float(density_change - likelihood_change) == pytest.approx(-0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"amplitude": jnp.zeros((3, 4))}, "exactly amplitude and phase"),
        (
            {"amplitude": jnp.zeros((3, 5)), "phase": jnp.zeros((3, 5))},
            "shape",
        ),
    ],
)
def test_logdensity_rejects_an_invalid_parameter_pytree(likelihood, params, message):
    with pytest.raises(ValueError, match=message):
        likelihood.logdensity_fn(params)


def test_container_represents_one_realisation_without_a_batch_axis(likelihood):
    assert likelihood.interferometers.strain.ndim == 2
    assert likelihood.interferometers.strain.shape[0] == 3


def _full_frequency_strain(likelihood):
    stored = np.asarray(likelihood._whitened_frequency_domain_strain)
    full = np.zeros((stored.shape[0], likelihood._frequency_count), dtype=np.complex128)
    full[:, np.asarray(likelihood._frequency_indices)] = stored
    return full


def test_supplied_strain_reproduces_the_stored_realisation(likelihood):
    params = {
        "amplitude": jnp.zeros((3, 4), dtype=jnp.float64),
        "phase": jnp.zeros((3, 4), dtype=jnp.float64),
    }
    supplied = _full_frequency_strain(likelihood)

    default = likelihood.log_likelihood_fn(params)
    on_supplied = likelihood.log_likelihood_for_strain(params, jnp.asarray(supplied))
    density = likelihood.logdensity_for_strain(params, jnp.asarray(supplied))

    assert float(on_supplied) == pytest.approx(float(default), rel=1e-12, abs=0.0)
    assert float(density - on_supplied) == pytest.approx(
        float(likelihood.logdensity_fn(params) - default), rel=1e-12, abs=1e-12
    )


def test_strain_path_keeps_one_value_per_realisation(likelihood):
    params = {
        "amplitude": jnp.zeros((3, 4), dtype=jnp.float64),
        "phase": jnp.zeros((3, 4), dtype=jnp.float64),
    }
    first = _full_frequency_strain(likelihood)
    second = 2.0 * first + 0.1
    batch = jnp.asarray(np.stack([first, second]))

    batched = likelihood.log_likelihood_for_strain(params, batch)

    assert batched.shape == (2,)
    assert float(batched[0]) == pytest.approx(float(likelihood.log_likelihood_for_strain(params, batch[0])), rel=1e-12)
    assert float(batched[1]) == pytest.approx(float(likelihood.log_likelihood_for_strain(params, batch[1])), rel=1e-12)
    assert float(batched[0]) != pytest.approx(float(batched[1]))


def test_strain_path_rejects_a_wrong_shape(likelihood):
    params = {
        "amplitude": jnp.zeros((3, 4), dtype=jnp.float64),
        "phase": jnp.zeros((3, 4), dtype=jnp.float64),
    }
    with pytest.raises(ValueError, match="trailing shape"):
        likelihood.log_likelihood_for_strain(params, jnp.zeros((3, 5), dtype=jnp.complex128))
