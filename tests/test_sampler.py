from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nullcal.sampler import sample_nuts


def test_sample_nuts_returns_samples_and_required_diagnostics():
    def standard_normal(position):
        return -0.5 * jnp.sum(position["x"] ** 2)

    result = sample_nuts(
        standard_normal,
        {"x": jnp.zeros(2)},
        seed=393,
        num_chains=2,
        num_warmup=100,
        num_samples=100,
        target_acceptance_rate=0.8,
        initial_position_jitter=0.1,
    )

    assert result.samples["x"].shape == (200, 2)
    assert result.logdensity.shape == (200,)
    assert result.metadata["num_chains"] == 2
    assert result.metadata["num_samples_per_chain"] == 100
    assert result.metadata["num_warmup"] == 100
    assert np.isfinite(result.metadata["max_rhat"])
    assert np.isfinite(result.metadata["min_ess_bulk"])
    assert np.isfinite(result.metadata["min_ess_tail"])
    assert result.metadata["divergences"] >= 0


def test_sample_nuts_rejects_diagnostics_from_one_chain():
    def standard_normal(position):
        return -0.5 * jnp.sum(position["x"] ** 2)

    with pytest.raises(ValueError, match="at least two chains"):
        sample_nuts(standard_normal, {"x": jnp.zeros(1)}, num_chains=1)
