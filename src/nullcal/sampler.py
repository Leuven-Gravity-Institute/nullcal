"""BlackJAX NUTS sampling with convergence diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from .result import Result

MINIMUM_DIAGNOSTIC_CHAINS = 2


def _jitter_position(position, key, scale: float):
    leaves, structure = jax.tree.flatten(position)
    keys = jax.random.split(key, len(leaves))
    jittered = [
        jnp.asarray(leaf) + scale * jax.random.normal(leaf_key, shape=jnp.asarray(leaf).shape)
        for leaf, leaf_key in zip(leaves, keys, strict=True)
    ]
    return jax.tree.unflatten(structure, jittered)


def sample_nuts(
    logdensity_fn: Callable[[Any], jax.Array],
    initial_position: Mapping[str, Any],
    *,
    seed: int = 0,
    num_chains: int = 4,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    target_acceptance_rate: float = 0.8,
    initial_position_jitter: float = 0.01,
) -> Result:
    """Adapt and run independent NUTS chains, returning mandatory diagnostics.

    Chains are executed independently rather than treating detector data as a
    realization batch. Returned parameter arrays flatten chain and draw into a
    single leading sample dimension; chain-aware R-hat and ESS are retained in
    ``Result.metadata``.
    """
    if num_chains < MINIMUM_DIAGNOSTIC_CHAINS:
        raise ValueError("at least two chains are required to compute R-hat")
    if num_warmup <= 0 or num_samples <= 0:
        raise ValueError("num_warmup and num_samples must be positive")
    if initial_position_jitter < 0.0:
        raise ValueError("initial_position_jitter must be non-negative")

    root_key = jax.random.key(seed)
    chain_keys = jax.random.split(root_key, num_chains)
    chain_states = []
    chain_infos = []

    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
    )

    for chain_key in chain_keys:
        jitter_key, warmup_key, sample_key = jax.random.split(chain_key, 3)
        chain_initial_position = _jitter_position(initial_position, jitter_key, initial_position_jitter)
        adapted, _ = adaptation.run(warmup_key, chain_initial_position, num_steps=num_warmup)
        kernel = blackjax.nuts(logdensity_fn, **adapted.parameters)
        sample_keys = jax.random.split(sample_key, num_samples)

        def step(state, key, kernel=kernel):
            next_state, info = kernel.step(key, state)
            return next_state, (next_state, info)

        _, (states, infos) = jax.lax.scan(step, adapted.state, sample_keys)
        chain_states.append(states)
        chain_infos.append(infos)

    positions_by_chain = jax.tree.map(lambda *values: jnp.stack(values), *(state.position for state in chain_states))
    logdensity_by_chain = jnp.stack([state.logdensity for state in chain_states])
    divergence_by_chain = jnp.stack([info.is_divergent for info in chain_infos])

    diagnostic_leaves = [value.reshape((num_chains, num_samples, -1)) for value in jax.tree.leaves(positions_by_chain)]
    diagnostic_array = jnp.concatenate(diagnostic_leaves, axis=-1)
    rhat = blackjax.diagnostics.rhat(diagnostic_array)
    ess_bulk = blackjax.diagnostics.ess_bulk(diagnostic_array)
    ess_tail = blackjax.diagnostics.ess_tail(diagnostic_array)

    flattened_positions = jax.tree.map(
        lambda value: value.reshape((num_chains * num_samples, *value.shape[2:])), positions_by_chain
    )
    flattened_logdensity = logdensity_by_chain.reshape(num_chains * num_samples)
    flattened_divergence = divergence_by_chain.reshape(num_chains * num_samples)
    metadata = {
        "seed": seed,
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples_per_chain": num_samples,
        "target_acceptance_rate": target_acceptance_rate,
        "max_rhat": float(np.asarray(jnp.max(rhat))),
        "min_ess_bulk": float(np.asarray(jnp.min(ess_bulk))),
        "min_ess_tail": float(np.asarray(jnp.min(ess_tail))),
        "divergences": int(np.asarray(jnp.sum(divergence_by_chain))),
    }
    return Result(
        samples=flattened_positions,
        logdensity=flattened_logdensity,
        info={"is_divergent": flattened_divergence},
        metadata=metadata,
    )


__all__ = ["sample_nuts"]
