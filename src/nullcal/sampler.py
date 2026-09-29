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


def _run_chain(
    logdensity_fn: Callable[[Any], jax.Array],
    initial_position: Mapping[str, Any],
    chain_key: jax.Array,
    *,
    num_warmup: int,
    num_samples: int,
    target_acceptance_rate: float,
    initial_position_jitter: float,
):
    """Adapt and sample one NUTS chain from a single jittered start."""
    jitter_key, warmup_key, sample_key = jax.random.split(chain_key, 3)
    chain_initial_position = _jitter_position(initial_position, jitter_key, initial_position_jitter)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
    )
    adapted, _ = adaptation.run(warmup_key, chain_initial_position, num_steps=num_warmup)
    kernel = blackjax.nuts(logdensity_fn, **adapted.parameters)
    sample_keys = jax.random.split(sample_key, num_samples)

    def step(state, key):
        next_state, info = kernel.step(key, state)
        return next_state, (next_state.position, next_state.logdensity, info.is_divergent, info.num_integration_steps)

    _, (positions, logdensity, divergences, integration_steps) = jax.lax.scan(step, adapted.state, sample_keys)
    return positions, logdensity, divergences, jnp.sum(integration_steps)


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
    chain_positions = []
    chain_logdensity = []
    chain_divergences = []
    chain_integration_steps = []

    for chain_key in chain_keys:
        positions, logdensity, divergences, integration_steps = _run_chain(
            logdensity_fn,
            initial_position,
            chain_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
            initial_position_jitter=initial_position_jitter,
        )
        chain_positions.append(positions)
        chain_logdensity.append(logdensity)
        chain_divergences.append(divergences)
        chain_integration_steps.append(integration_steps)

    positions_by_chain = jax.tree.map(lambda *values: jnp.stack(values), *chain_positions)
    logdensity_by_chain = jnp.stack(chain_logdensity)
    divergence_by_chain = jnp.stack(chain_divergences)
    integration_steps_by_chain = jnp.stack(chain_integration_steps)

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
        "integration_steps": int(np.asarray(jnp.sum(integration_steps_by_chain))),
    }
    return Result(
        samples=flattened_positions,
        logdensity=flattened_logdensity,
        info={"is_divergent": flattened_divergence},
        metadata=metadata,
    )


def sample_nuts_batched(
    logdensity_fn: Callable[[Any, Any], jax.Array],
    initial_position: Mapping[str, Any],
    realisation_data: Any,
    *,
    seed: int = 0,
    num_chains_per_realisation: int = 1,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    target_acceptance_rate: float = 0.8,
    initial_position_jitter: float = 0.01,
) -> Result:
    """Run one NUTS chain per (realisation, chain) pair under a single ``vmap``.

    ``logdensity_fn(position, realisation)`` evaluates one posterior target:
    the leading axis of every leaf of ``realisation_data`` is the realisation
    index. All chains advance inside one ``vmap``, so a campaign pays one
    dispatch and one accelerator launch per batch instead of one per
    realisation. The returned sample arrays flatten ``(realisation, chain,
    draw)`` into the leading axis; the split sizes are in ``Result.metadata``.

    The single-realisation :func:`sample_nuts` is the diagnostic path and
    computes R-hat and ESS; this batched path trades those diagnostics for
    throughput and is the cost-measurement target.
    """
    if num_chains_per_realisation < 1:
        raise ValueError("num_chains_per_realisation must be at least one")
    if num_warmup <= 0 or num_samples <= 0:
        raise ValueError("num_warmup and num_samples must be positive")
    if initial_position_jitter < 0.0:
        raise ValueError("initial_position_jitter must be non-negative")

    leaves = jax.tree.leaves(realisation_data)
    if not leaves:
        raise ValueError("realisation_data must contain at least one array")
    realisation_count = int(jnp.shape(leaves[0])[0]) if jnp.ndim(leaves[0]) > 0 else 0
    if realisation_count < 1:
        raise ValueError("realisation_data must contain at least one realisation on a leading axis")

    total_chains = realisation_count * num_chains_per_realisation
    realisation_index = jnp.repeat(jnp.arange(realisation_count), num_chains_per_realisation)
    batched_realisations = jax.tree.map(lambda value: jnp.asarray(value)[realisation_index], realisation_data)
    broadcast_positions = jax.tree.map(
        lambda value: jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), (total_chains, *jnp.shape(value))),
        initial_position,
    )
    chain_keys = jax.random.split(jax.random.key(seed), total_chains)

    def chain(position, chain_key, realisation):
        return _run_chain(
            lambda sample_position: logdensity_fn(sample_position, realisation),
            position,
            chain_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
            initial_position_jitter=initial_position_jitter,
        )

    positions_by_chain, logdensity_by_chain, divergence_by_chain, integration_steps_by_chain = jax.vmap(chain)(
        broadcast_positions, chain_keys, batched_realisations
    )

    flattened_positions = jax.tree.map(
        lambda value: value.reshape((total_chains * num_samples, *value.shape[2:])), positions_by_chain
    )
    flattened_logdensity = logdensity_by_chain.reshape(total_chains * num_samples)
    flattened_divergence = divergence_by_chain.reshape(total_chains * num_samples)
    metadata = {
        "seed": seed,
        "num_realisations": realisation_count,
        "num_chains_per_realisation": num_chains_per_realisation,
        "num_warmup": num_warmup,
        "num_samples_per_chain": num_samples,
        "target_acceptance_rate": target_acceptance_rate,
        "initial_position_jitter": initial_position_jitter,
        "divergences": int(np.asarray(jnp.sum(divergence_by_chain))),
        "integration_steps": int(np.asarray(jnp.sum(integration_steps_by_chain))),
    }
    return Result(
        samples=flattened_positions,
        logdensity=flattened_logdensity,
        info={"is_divergent": flattened_divergence},
        metadata=metadata,
    )


__all__ = ["sample_nuts", "sample_nuts_batched"]
