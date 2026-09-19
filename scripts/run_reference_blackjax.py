"""Run the fixed BlackJAX acceptance posterior against immutable R1 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullcal.sampler import sample_nuts  # noqa: E402
from tests.e2e import pipeline  # noqa: E402
from tests.e2e.test_blackjax_posterior import (  # noqa: E402
    MAX_DIVERGENCES,
    MAX_EIGENMODE_WIDTH_RATIO,
    MAX_KS_STATISTIC,
    MAX_LOGDENSITY_ABS_DIFF,
    MAX_MARGINAL_WIDTH_RATIO,
    MAX_RHAT,
    MIN_EIGENMODE_WIDTH_RATIO,
    MIN_ESS_BULK,
    MIN_ESS_TAIL,
    MIN_MARGINAL_WIDTH_RATIO,
    MIN_PERMUTATION_SEPARATION,
)

DEFAULT_OUTPUT_DIR = ROOT / "tests" / "e2e" / "reference"
SEED = 20260919
NUM_CHAINS = 4
NUM_WARMUP = 1_000
NUM_SAMPLES = 1_500
TARGET_ACCEPTANCE_RATE = 0.9
INITIAL_POSITION_JITTER = 0.005


def _parameter_names() -> list[str]:
    return [
        f"recalib_{detector}_{quantity}_{knot}"
        for detector in pipeline.config.DETECTOR_NAMES
        for quantity in ("amplitude", "phase")
        for knot in range(pipeline.config.N_POINTS)
    ]


def _git(command: list[str]) -> str:
    return subprocess.check_output(["git", *command], text=True).strip()  # noqa: S603, S607


def _source_is_dirty() -> bool:
    return bool(
        _git(
            [
                "status",
                "--porcelain",
                "--",
                "src/nullcal",
                "tests/e2e",
                "scripts/run_reference_blackjax.py",
                "pyproject.toml",
                "uv.lock",
            ]
        )
    )


def _ordered_posterior(samples: dict[str, np.ndarray], parameters: list[str]) -> np.ndarray:
    detector_index = {name: index for index, name in enumerate(pipeline.config.DETECTOR_NAMES)}
    columns = []
    for name in parameters:
        _, detector, quantity, knot = name.split("_")
        columns.append(samples[quantity][:, detector_index[detector], int(knot)])
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    if _source_is_dirty():
        raise RuntimeError("refusing to produce an acceptance artifact from dirty source or harness code")

    likelihood = pipeline.build_likelihood_from_reference_inputs()
    initial_position = {
        "amplitude": jnp.zeros(likelihood.parameter_shape),
        "phase": jnp.zeros(likelihood.parameter_shape),
    }
    started = time.perf_counter()
    result = sample_nuts(
        likelihood.logdensity_fn,
        initial_position,
        seed=SEED,
        num_chains=NUM_CHAINS,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        target_acceptance_rate=TARGET_ACCEPTANCE_RATE,
        initial_position_jitter=INITIAL_POSITION_JITTER,
    )
    wall_seconds = time.perf_counter() - started

    parameters = _parameter_names()
    posterior = _ordered_posterior({key: np.asarray(value) for key, value in result.samples.items()}, parameters)
    posterior_by_chain = posterior.reshape(NUM_CHAINS, NUM_SAMPLES, -1)
    is_divergent = np.asarray(result.info["is_divergent"]).reshape(NUM_CHAINS, NUM_SAMPLES)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / "blackjax_posterior_samples.npz"
    manifest_path = arguments.output_dir / "blackjax_posterior_manifest.json"
    np.savez_compressed(
        output_path,
        posterior_by_chain=posterior_by_chain,
        parameters=np.asarray(parameters),
        is_divergent=is_divergent,
    )
    manifest = {
        "source_git_revision": _git(["rev-parse", "HEAD"]),
        "source_git_dirty": False,
        "posterior_sha256": hashlib.sha256(np.ascontiguousarray(posterior_by_chain).tobytes()).hexdigest(),
        "posterior_shape": list(posterior_by_chain.shape),
        "parameters": parameters,
        "sampler": {
            "name": "blackjax.nuts",
            "seed": SEED,
            "num_chains": NUM_CHAINS,
            "num_warmup": NUM_WARMUP,
            "num_samples_per_chain": NUM_SAMPLES,
            "target_acceptance_rate": TARGET_ACCEPTANCE_RATE,
            "initial_position_jitter": INITIAL_POSITION_JITTER,
        },
        "acceptance": {
            "historical_density": {
                "max_abs_difference": MAX_LOGDENSITY_ABS_DIFF,
                "min_permutation_separation": MIN_PERMUTATION_SEPARATION,
            },
            "chain_diagnostics": {
                "max_rhat": MAX_RHAT,
                "min_ess_bulk": MIN_ESS_BULK,
                "min_ess_tail": MIN_ESS_TAIL,
                "max_divergences": MAX_DIVERGENCES,
            },
            "curvature": {
                "marginal_width_ratio": [MIN_MARGINAL_WIDTH_RATIO, MAX_MARGINAL_WIDTH_RATIO],
                "eigenmode_width_ratio": [MIN_EIGENMODE_WIDTH_RATIO, MAX_EIGENMODE_WIDTH_RATIO],
                "require_positive_definite_hessian": True,
            },
        },
        "results": {
            **result.metadata,
            "wall_seconds": wall_seconds,
        },
        "anchors": {
            "fixed_log_likelihood": -203.88870383371767,
            "likelihood_agreement": "tests/e2e/diagnostics/likelihood_agreement.json",
        },
        "legacy_diagnostic": {
            "ks_threshold": MAX_KS_STATISTIC,
            "is_acceptance_reference": False,
            "reason": "the historical dynesty protocol undercovers an exact correlated-Gaussian target",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
