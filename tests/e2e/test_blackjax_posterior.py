"""Acceptance of BlackJAX against fixed values, diagnostics, and curvature."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from . import pipeline

REFERENCE_DIR = Path(__file__).parent / "reference"
DIAGNOSTIC_DIR = Path(__file__).parent / "diagnostics"
BLACKJAX_POSTERIOR_PATH = REFERENCE_DIR / "blackjax_posterior_samples.npz"
BLACKJAX_MANIFEST_PATH = REFERENCE_DIR / "blackjax_posterior_manifest.json"
POSTERIOR_MANIFEST_PATH = REFERENCE_DIR / "posterior_manifest.json"
LIKELIHOOD_AGREEMENT_PATH = DIAGNOSTIC_DIR / "likelihood_agreement.json"
LIKELIHOOD_AGREEMENT_SHA256 = "d32940f353b15518debaf6d8e65fdc90b6f733a7c8da675f29251085d55904f4"

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# The historical value is retained unchanged for provenance, but the diagnosed
# bilby/dynesty posterior is no longer an acceptance reference.
MAX_KS_STATISTIC = 0.10
MAX_LOGDENSITY_ABS_DIFF = 1.0e-9
MIN_PERMUTATION_SEPARATION = 1.0
MAX_RHAT = 1.01
MIN_ESS_BULK = 400.0
MIN_ESS_TAIL = 400.0
MAX_DIVERGENCES = 0
MIN_MARGINAL_WIDTH_RATIO = 0.90
MAX_MARGINAL_WIDTH_RATIO = 1.10
MIN_EIGENMODE_WIDTH_RATIO = 0.85
MAX_EIGENMODE_WIDTH_RATIO = 1.15


def _vector_to_parameters(vector):
    detector_kind_knot = vector.reshape((len(pipeline.config.DETECTOR_NAMES), 2, pipeline.config.N_POINTS))
    return {"amplitude": detector_kind_knot[:, 0, :], "phase": detector_kind_knot[:, 1, :]}


def _ordered_posterior(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return arrays["posterior_by_chain"].reshape(-1, arrays["posterior_by_chain"].shape[-1])


@pytest.fixture(scope="module")
def blackjax_artifact():
    if not BLACKJAX_POSTERIOR_PATH.exists() or not BLACKJAX_MANIFEST_PATH.exists():
        pytest.fail("BlackJAX posterior acceptance artifact is absent; run scripts/run_reference_blackjax.py")
    with np.load(BLACKJAX_POSTERIOR_PATH) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return arrays, json.loads(BLACKJAX_MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def likelihood_agreement_artifact():
    if not LIKELIHOOD_AGREEMENT_PATH.exists():
        pytest.fail("the adopted historical likelihood-agreement artifact is absent")
    encoded = LIKELIHOOD_AGREEMENT_PATH.read_bytes()
    assert hashlib.sha256(encoded).hexdigest() == LIKELIHOOD_AGREEMENT_SHA256
    return json.loads(encoded)


def test_blackjax_artifact_integrity_and_provenance(blackjax_artifact):
    arrays, manifest = blackjax_artifact
    historical_manifest = json.loads(POSTERIOR_MANIFEST_PATH.read_text())
    digest = hashlib.sha256(np.ascontiguousarray(arrays["posterior_by_chain"]).tobytes()).hexdigest()

    assert digest == manifest["posterior_sha256"]
    assert list(arrays["posterior_by_chain"].shape) == manifest["posterior_shape"]
    assert arrays["parameters"].tolist() == manifest["parameters"]
    assert manifest["source_git_dirty"] is False
    assert len(manifest["source_git_revision"]) == 40
    assert manifest["legacy_diagnostic"]["ks_threshold"] == MAX_KS_STATISTIC
    assert manifest["legacy_diagnostic"]["is_acceptance_reference"] is False
    assert manifest["legacy_diagnostic"]["bilby_posterior_sha256"] == historical_manifest["posterior_sha256"]
    assert (
        manifest["anchors"]["fixed_log_likelihood"] == historical_manifest["results"]["frozen_reference_log_likelihood"]
    )
    assert manifest["acceptance"]["historical_density"] == {
        "max_abs_difference": MAX_LOGDENSITY_ABS_DIFF,
        "min_permutation_separation": MIN_PERMUTATION_SEPARATION,
    }


def test_current_density_matches_historical_fixed_points(likelihood_agreement_artifact):
    likelihood = pipeline.build_likelihood_from_reference_inputs()
    maximum_differences = {"log_likelihood": 0.0, "log_prior": 0.0, "log_posterior": 0.0}

    assert likelihood_agreement_artifact["status"] == "pass"
    assert likelihood_agreement_artifact["criterion"] == {
        "maximum_absolute_difference": MAX_LOGDENSITY_ABS_DIFF,
        "permutation_minimum_absolute_separation": MIN_PERMUTATION_SEPARATION,
    }
    assert len(likelihood_agreement_artifact["points"]) == 8
    assert len(likelihood_agreement_artifact["provenance"]["producer_commit"]) == 40
    assert "diagnostic_commit" not in likelihood_agreement_artifact["provenance"]
    assert "current_nullcal" not in likelihood_agreement_artifact["versions"]

    for point in likelihood_agreement_artifact["points"].values():
        vector = jnp.asarray(point["canonical_vector"])
        parameters = _vector_to_parameters(vector)
        log_likelihood = float(likelihood.log_likelihood_fn(parameters))
        log_posterior = float(likelihood.logdensity_fn(parameters))
        current = {
            "log_likelihood": log_likelihood,
            "log_prior": log_posterior - log_likelihood,
            "log_posterior": log_posterior,
        }
        for quantity, previous_maximum in maximum_differences.items():
            difference = abs(current[quantity] - point["premigration"][quantity])
            maximum_differences[quantity] = max(previous_maximum, difference)

    assert all(difference <= MAX_LOGDENSITY_ABS_DIFF for difference in maximum_differences.values())
    probe = jnp.asarray(likelihood_agreement_artifact["points"]["ordering_probe"]["canonical_vector"])
    correct = float(likelihood.logdensity_fn(_vector_to_parameters(probe)))
    permuted = float(likelihood.logdensity_fn(_vector_to_parameters(jnp.roll(probe, 1))))
    assert abs(correct - permuted) > MIN_PERMUTATION_SEPARATION
    assert likelihood_agreement_artifact["permutation_check"]["absolute_separation"] > MIN_PERMUTATION_SEPARATION


def test_blackjax_chain_diagnostics_pass_fixed_thresholds(blackjax_artifact):
    arrays, manifest = blackjax_artifact
    posterior = arrays["posterior_by_chain"]
    max_rhat = float(np.max(np.asarray(blackjax.diagnostics.rhat(posterior))))
    min_ess_bulk = float(np.min(np.asarray(blackjax.diagnostics.ess_bulk(posterior))))
    min_ess_tail = float(np.min(np.asarray(blackjax.diagnostics.ess_tail(posterior))))
    divergences = int(np.count_nonzero(arrays["is_divergent"]))

    assert max_rhat <= MAX_RHAT
    assert min_ess_bulk >= MIN_ESS_BULK
    assert min_ess_tail >= MIN_ESS_TAIL
    assert divergences <= MAX_DIVERGENCES
    assert manifest["acceptance"]["chain_diagnostics"] == {
        "max_divergences": MAX_DIVERGENCES,
        "max_rhat": MAX_RHAT,
        "min_ess_bulk": MIN_ESS_BULK,
        "min_ess_tail": MIN_ESS_TAIL,
    }


def test_blackjax_covariance_matches_local_curvature(blackjax_artifact):
    arrays, manifest = blackjax_artifact
    posterior = _ordered_posterior(arrays)
    likelihood = pipeline.build_likelihood_from_reference_inputs()

    def objective(vector):
        return -likelihood.logdensity_fn(_vector_to_parameters(vector))

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=float)

    optimization = minimize(
        scipy_objective,
        posterior.mean(axis=0),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1.0e-13, "gtol": 1.0e-8, "maxiter": 1_000, "maxls": 50},
    )
    assert optimization.success
    hessian = np.asarray(jax.hessian(objective)(jnp.asarray(optimization.x)))
    hessian_eigenvalues = np.linalg.eigvalsh(hessian)
    assert float(hessian_eigenvalues.min()) > 0.0

    laplace_covariance = np.linalg.inv(hessian)
    sample_covariance = np.cov(posterior, rowvar=False, ddof=1)
    marginal_ratios = np.sqrt(np.diag(sample_covariance) / np.diag(laplace_covariance))
    laplace_eigenvalues, laplace_eigenvectors = np.linalg.eigh(laplace_covariance)
    sample_eigenvalues = np.linalg.eigvalsh(sample_covariance)
    projected_variance = np.diag(laplace_eigenvectors.T @ sample_covariance @ laplace_eigenvectors)
    projected_ratios = np.sqrt(projected_variance / laplace_eigenvalues)
    sorted_ratios = np.sqrt(sample_eigenvalues / laplace_eigenvalues)

    assert np.all((marginal_ratios >= MIN_MARGINAL_WIDTH_RATIO) & (marginal_ratios <= MAX_MARGINAL_WIDTH_RATIO))
    assert np.all((projected_ratios >= MIN_EIGENMODE_WIDTH_RATIO) & (projected_ratios <= MAX_EIGENMODE_WIDTH_RATIO))
    assert np.all((sorted_ratios >= MIN_EIGENMODE_WIDTH_RATIO) & (sorted_ratios <= MAX_EIGENMODE_WIDTH_RATIO))
    assert manifest["acceptance"]["curvature"] == {
        "marginal_width_ratio": [MIN_MARGINAL_WIDTH_RATIO, MAX_MARGINAL_WIDTH_RATIO],
        "eigenmode_width_ratio": [MIN_EIGENMODE_WIDTH_RATIO, MAX_EIGENMODE_WIDTH_RATIO],
        "require_positive_definite_hessian": True,
    }
