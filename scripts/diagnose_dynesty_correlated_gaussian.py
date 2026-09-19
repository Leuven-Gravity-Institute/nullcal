"""Validate the frozen bilby/dynesty protocol on an analytic Gaussian.

This is a 60-dimensional correlated Gaussian likelihood under the same
independent normalized Gaussian priors used by the calibration problem.  Its
posterior mean and covariance are analytic.  The protocol and pass bands below
are fixed in source before sampling; a failed run must not be tuned or repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version as package_version
from pathlib import Path

import bilby
import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Gaussian, PriorDict
from scipy.optimize import linear_sum_assignment

DIMENSIONS = 60
PRIOR_SIGMA = 0.05
PROBLEM_SEED = 20260922
SAMPLER_SEED = 20260918
SELECTION_SEED = 20260920
RETAINED_DRAWS = 5_000
NPOOL = 8

SAMPLER_KWARGS = {
    "sampler": "dynesty",
    "nlive": 1_000,
    "sample": "rslice",
    "slices": 10,
    "dlogz": 0.1,
    "check_point": True,
    "check_point_delta_t": 1_800,
    "check_point_plot": False,
    "resume": True,
    "print_method": "interval-60",
}

# Fixed before sampling.  Five thousand independent Gaussian draws would have
# about 1% relative standard-deviation noise.  These wider bands admit nested-
# sampling dependence and finite-dimensional covariance noise while rejecting
# the approximately 20% width deficit seen in the calibration references.
MARGINAL_WIDTH_RATIO_RANGE = (0.90, 1.10)
EIGENMODE_WIDTH_RATIO_RANGE = (0.85, 1.15)


def parameter_names() -> list[str]:
    return [f"x_{index:02d}" for index in range(DIMENSIONS)]


def problem_definition() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return likelihood mean/covariance and analytic posterior mean/covariance."""
    rng = np.random.default_rng(PROBLEM_SEED)
    orthogonal, triangular = np.linalg.qr(rng.normal(size=(DIMENSIONS, DIMENSIONS)))
    orthogonal *= np.sign(np.diag(triangular))[None, :]
    likelihood_widths = np.geomspace(0.02, 0.20, DIMENSIONS)
    likelihood_covariance = (orthogonal * likelihood_widths**2) @ orthogonal.T
    likelihood_mean = np.linspace(-0.02, 0.02, DIMENSIONS)

    likelihood_precision = np.linalg.inv(likelihood_covariance)
    prior_precision = np.eye(DIMENSIONS) / PRIOR_SIGMA**2
    posterior_covariance = np.linalg.inv(likelihood_precision + prior_precision)
    posterior_mean = posterior_covariance @ likelihood_precision @ likelihood_mean
    return likelihood_mean, likelihood_covariance, posterior_mean, posterior_covariance


class CorrelatedGaussianLikelihood(Likelihood):
    """Normalized correlated Gaussian with a deterministic dense covariance."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        super().__init__(parameters=dict.fromkeys(parameter_names(), 0.0))
        self.mean = mean
        self.precision = np.linalg.inv(covariance)
        _, log_determinant = np.linalg.slogdet(covariance)
        self.normalization = -0.5 * (DIMENSIONS * np.log(2.0 * np.pi) + log_determinant)

    def log_likelihood(self) -> float:
        vector = np.asarray([self.parameters[name] for name in parameter_names()])
        displacement = vector - self.mean
        return float(self.normalization - 0.5 * displacement @ self.precision @ displacement)

    def noise_log_likelihood(self) -> float:
        return 0.0


def sha256_of_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S607


def within(values: np.ndarray, limits: tuple[float, float]) -> bool:
    return bool(np.all((values >= limits[0]) & (values <= limits[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--npool", type=int, default=NPOOL)
    arguments = parser.parse_args()
    if arguments.npool != NPOOL:
        parser.error("the frozen diagnostic protocol requires --npool 8")
    if arguments.outdir.exists() and any(arguments.outdir.iterdir()):
        raise RuntimeError("output directory must be absent or empty; this diagnostic cannot resume or overwrite a run")
    arguments.outdir.mkdir(parents=True, exist_ok=True)

    if package_version("bilby") != "2.8.2" or package_version("dynesty") != "3.1.0":
        raise RuntimeError("the diagnostic requires bilby 2.8.2 and dynesty 3.1.0")

    likelihood_mean, likelihood_covariance, analytic_mean, analytic_covariance = problem_definition()
    likelihood = CorrelatedGaussianLikelihood(likelihood_mean, likelihood_covariance)
    priors = PriorDict(
        {name: Gaussian(mu=0.0, sigma=PRIOR_SIGMA, name=name, boundary="reflective") for name in parameter_names()}
    )

    bilby.core.utils.random.seed(SAMPLER_SEED)
    started = time.perf_counter()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=str(arguments.outdir),
        label="correlated_gaussian_protocol",
        npool=arguments.npool,
        save="hdf5",
        plot=False,
        use_ratio=False,
        **SAMPLER_KWARGS,
    )
    wall_seconds = time.perf_counter() - started

    raw_posterior = result.posterior[parameter_names()].to_numpy()
    if raw_posterior.shape[0] < RETAINED_DRAWS:
        raise RuntimeError(
            f"dynesty returned {raw_posterior.shape[0]} draws, fewer than the predeclared {RETAINED_DRAWS}"
        )
    selection_rng = np.random.default_rng(SELECTION_SEED)
    indices = selection_rng.choice(raw_posterior.shape[0], size=RETAINED_DRAWS, replace=False)
    posterior = raw_posterior[indices]

    sample_mean = posterior.mean(axis=0)
    sample_covariance = np.cov(posterior, rowvar=False, ddof=1)
    analytic_marginal_width = np.sqrt(np.diag(analytic_covariance))
    sample_marginal_width = np.sqrt(np.diag(sample_covariance))
    marginal_ratios = sample_marginal_width / analytic_marginal_width

    analytic_eigenvalues, analytic_eigenvectors = np.linalg.eigh(analytic_covariance)
    sample_eigenvalues, sample_eigenvectors = np.linalg.eigh(sample_covariance)
    sorted_eigenmode_ratios = np.sqrt(sample_eigenvalues / analytic_eigenvalues)
    projected_sample_variance = np.diag(analytic_eigenvectors.T @ sample_covariance @ analytic_eigenvectors)
    projected_eigenmode_ratios = np.sqrt(projected_sample_variance / analytic_eigenvalues)
    overlap = np.abs(analytic_eigenvectors.T @ sample_eigenvectors)
    analytic_indices, sample_indices = linear_sum_assignment(-overlap)
    matched_overlaps = overlap[analytic_indices, sample_indices]

    marginal_pass = within(marginal_ratios, MARGINAL_WIDTH_RATIO_RANGE)
    projected_mode_pass = within(projected_eigenmode_ratios, EIGENMODE_WIDTH_RATIO_RANGE)
    sorted_mode_pass = within(sorted_eigenmode_ratios, EIGENMODE_WIDTH_RATIO_RANGE)
    passed = marginal_pass and projected_mode_pass and sorted_mode_pass

    np.savez_compressed(
        arguments.outdir / "correlated_gaussian_protocol_samples.npz",
        posterior=posterior,
        parameters=np.asarray(parameter_names()),
    )
    payload = {
        "status": "pass" if passed else "fail",
        "provenance": {
            "git_revision": git_revision(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {name: package_version(name) for name in ("bilby", "dynesty", "numpy", "scipy")},
        },
        "problem": {
            "dimensions": DIMENSIONS,
            "problem_seed": PROBLEM_SEED,
            "prior_sigma": PRIOR_SIGMA,
            "likelihood_width_range": [0.02, 0.20],
            "analytic_marginal_widths": analytic_marginal_width.tolist(),
            "analytic_eigenmode_widths": np.sqrt(analytic_eigenvalues).tolist(),
        },
        "protocol": SAMPLER_KWARGS
        | {
            "npool": arguments.npool,
            "sampler_seed": SAMPLER_SEED,
            "selection_seed": SELECTION_SEED,
            "retained_draws": RETAINED_DRAWS,
            "retention": "uniform without replacement from bilby's equal-weight posterior",
            "warmup": "not applicable to static nested sampling",
            "thinning": "not applicable to static nested sampling",
        },
        "criteria": {
            "marginal_width_ratio_range": list(MARGINAL_WIDTH_RATIO_RANGE),
            "eigenmode_width_ratio_range": list(EIGENMODE_WIDTH_RATIO_RANGE),
        },
        "results": {
            "raw_draws": int(raw_posterior.shape[0]),
            "retained_draws": int(posterior.shape[0]),
            "wall_seconds": wall_seconds,
            "sampling_time_seconds": float(result.sampling_time),
            "log_evidence": float(result.log_evidence),
            "log_evidence_error": float(result.log_evidence_err),
            "posterior_sha256": sha256_of_array(posterior),
            "sample_mean": sample_mean.tolist(),
            "analytic_mean": analytic_mean.tolist(),
            "marginal_width_ratios": marginal_ratios.tolist(),
            "marginal_width_ratio_minimum": float(marginal_ratios.min()),
            "marginal_width_ratio_median": float(np.median(marginal_ratios)),
            "marginal_width_ratio_maximum": float(marginal_ratios.max()),
            "projected_eigenmode_width_ratios": projected_eigenmode_ratios.tolist(),
            "projected_eigenmode_width_ratio_minimum": float(projected_eigenmode_ratios.min()),
            "projected_eigenmode_width_ratio_median": float(np.median(projected_eigenmode_ratios)),
            "projected_eigenmode_width_ratio_maximum": float(projected_eigenmode_ratios.max()),
            "sorted_eigenvalue_width_ratios": sorted_eigenmode_ratios.tolist(),
            "sorted_eigenvalue_width_ratio_minimum": float(sorted_eigenmode_ratios.min()),
            "sorted_eigenvalue_width_ratio_median": float(np.median(sorted_eigenmode_ratios)),
            "sorted_eigenvalue_width_ratio_maximum": float(sorted_eigenmode_ratios.max()),
            "matched_eigenvector_overlap_minimum": float(matched_overlaps.min()),
            "matched_eigenvector_overlap_median": float(np.median(matched_overlaps)),
        },
        "checks": {
            "marginal_widths": marginal_pass,
            "projected_analytic_eigenmodes": projected_mode_pass,
            "sorted_covariance_eigenvalues": sorted_mode_pass,
        },
    }
    manifest_path = arguments.outdir / "correlated_gaussian_protocol_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"status": payload["status"], "checks": payload["checks"], "results": payload["results"]}, indent=2)
    )


if __name__ == "__main__":
    main()
