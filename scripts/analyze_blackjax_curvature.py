"""Compare fixed BlackJAX samples with local posterior curvature."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

DETECTORS = ("ET1", "ET2", "ET3")
KINDS = ("amplitude", "phase")
KNOTS = 10
DIMENSIONS = len(DETECTORS) * len(KINDS) * KNOTS
MARGINAL_WIDTH_RATIO_RANGE = (0.90, 1.10)
EIGENMODE_WIDTH_RATIO_RANGE = (0.85, 1.15)


def canonical_names() -> list[str]:
    return [f"recalib_{detector}_{kind}_{knot}" for detector in DETECTORS for kind in KINDS for knot in range(KNOTS)]


def vector_to_arrays(vector):
    detector_kind_knot = vector.reshape((len(DETECTORS), len(KINDS), KNOTS))
    return {kind: detector_kind_knot[:, kind_index, :] for kind_index, kind in enumerate(KINDS)}


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tests.e2e import pipeline  # noqa: PLC0415

    with np.load(arguments.samples) as stored:
        samples = stored["posterior_by_chain"].reshape(-1, DIMENSIONS)
        names = stored["parameters"].tolist()
    if names != canonical_names():
        raise RuntimeError("BlackJAX artifact parameter order is not the declared canonical order")
    sampler_manifest = json.loads(arguments.manifest.read_text())

    likelihood = pipeline.build_likelihood_from_reference_inputs()

    def objective(vector):
        arrays = {key: jnp.asarray(value) for key, value in vector_to_arrays(vector).items()}
        return -likelihood.logdensity_fn(arrays)

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=float)

    optimization = minimize(
        scipy_objective,
        samples.mean(axis=0),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1.0e-13, "gtol": 1.0e-8, "maxiter": 1_000, "maxls": 50},
    )
    if not optimization.success:
        raise RuntimeError(f"MAP optimization failed: {optimization.message}")

    hessian = np.asarray(jax.hessian(objective)(jnp.asarray(optimization.x)))
    laplace_covariance = np.linalg.inv(hessian)
    sample_covariance = np.cov(samples, rowvar=False, ddof=1)
    marginal_ratios = np.sqrt(np.diag(sample_covariance) / np.diag(laplace_covariance))

    laplace_eigenvalues, laplace_eigenvectors = np.linalg.eigh(laplace_covariance)
    sample_eigenvalues = np.linalg.eigvalsh(sample_covariance)
    projected_variance = np.diag(laplace_eigenvectors.T @ sample_covariance @ laplace_eigenvectors)
    projected_ratios = np.sqrt(projected_variance / laplace_eigenvalues)
    sorted_ratios = np.sqrt(sample_eigenvalues / laplace_eigenvalues)

    payload = {
        "provenance": {
            "diagnostic_commit": git_revision(),
            "r5_source_commit": sampler_manifest["source_git_revision"],
            "posterior_sha256": sampler_manifest["posterior_sha256"],
        },
        "criteria": {
            "marginal_width_ratio_range": list(MARGINAL_WIDTH_RATIO_RANGE),
            "eigenmode_width_ratio_range": list(EIGENMODE_WIDTH_RATIO_RANGE),
        },
        "optimization": {
            "iterations": int(optimization.nit),
            "gradient_infinity_norm": float(np.max(np.abs(optimization.jac))),
            "hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(hessian).min()),
        },
        "marginal_width_ratios": marginal_ratios.tolist(),
        "projected_eigenmode_width_ratios": projected_ratios.tolist(),
        "sorted_eigenvalue_width_ratios": sorted_ratios.tolist(),
        "summary": {
            "marginal_minimum": float(marginal_ratios.min()),
            "marginal_median": float(np.median(marginal_ratios)),
            "marginal_maximum": float(marginal_ratios.max()),
            "projected_eigenmode_minimum": float(projected_ratios.min()),
            "projected_eigenmode_median": float(np.median(projected_ratios)),
            "projected_eigenmode_maximum": float(projected_ratios.max()),
            "sorted_eigenvalue_minimum": float(sorted_ratios.min()),
            "sorted_eigenvalue_median": float(np.median(sorted_ratios)),
            "sorted_eigenvalue_maximum": float(sorted_ratios.max()),
        },
        "sampler_diagnostics": {
            key: sampler_manifest["results"][key] for key in ("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences")
        },
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": payload["summary"], "sampler_diagnostics": payload["sampler_diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
