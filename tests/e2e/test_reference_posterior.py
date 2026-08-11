"""Checks on the reference posterior, which is **provisional and not the anchor**.

Owner's decision, 2026-08-11: this posterior does *not* serve as the distributional anchor for the
BlackJAX rebuild, and it does not block the migration. It is kept as a worked example and as a
regression guard on ``generate_reference_posterior.py``. The samples are regenerable from the
pre-migration revision, so what matters is that this revision and the generator survive — not these
particular draws.

It is not the anchor because the number is not yet anchored. Marginal widths (median
``sigma_post/sigma_prior`` 0.720) are explained by measured degeneracy, but the conditional widths
recovered from the samples, 0.292 median, still disagree with the 0.41 that a sampler-independent
per-parameter scan predicts, and one covariance direction comes out broader than the prior beyond
finite-sample noise. Closing those is what "anchored" would mean.

The tests below are therefore integrity and sanity checks on a provisional artifact, not acceptance
criteria for a reference. They still earn their place: they are what would catch a regenerated
posterior that had silently reverted to returning the prior.

The tests here deliberately do **not** re-run the sampler — that is a 12-minute 32-core job. They
check the shipped artifact's integrity, and they check the property that makes it worth having at
all.

**Why an informativeness test exists.** The first production run of this posterior converged on
``dlogz``, wrote a complete manifest, and returned the *prior*: median ``sigma_post/sigma_prior``
0.999 across all 60 parameters. dynesty's ``rwalk`` had never mixed — efficiency 0.0%, calls per
iteration pinned at its ceiling, and an autocorrelation warning buried in stderr — and nothing in
the output said the result was worthless. A posterior equal to its prior is reproduced equally well
by a correct implementation and a broken one, so as an anchor it would pass anything. The artifact
now records its own shrinkage and this test enforces it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REFERENCE_DIR = Path(__file__).parent / "reference"
POSTERIOR_PATH = REFERENCE_DIR / "posterior_samples.npz"
POSTERIOR_MANIFEST_PATH = REFERENCE_DIR / "posterior_manifest.json"

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

#: A posterior whose marginals are this close to the prior is not a usable anchor. Set well away
#: from both the failure (0.999, observed) and the accepted artifact (0.720 median, 0.533 min), so
#: it catches a prior-returning run without tracking the exact value of a healthy one.
MAX_ACCEPTABLE_SHRINKAGE = 0.90


@pytest.fixture(scope="module")
def posterior() -> np.ndarray:
    if not POSTERIOR_PATH.exists():
        pytest.fail(f"reference posterior absent: {POSTERIOR_PATH}")
    with np.load(POSTERIOR_PATH, allow_pickle=True) as data:
        return np.asarray(data["posterior"])


@pytest.fixture(scope="module")
def posterior_manifest() -> dict:
    if not POSTERIOR_MANIFEST_PATH.exists():
        pytest.fail(f"reference posterior manifest absent: {POSTERIOR_MANIFEST_PATH}")
    return json.loads(POSTERIOR_MANIFEST_PATH.read_text())


def test_posterior_matches_its_manifest(posterior, posterior_manifest):
    """Digest, shape and sample count must describe the shipped samples."""
    digest = hashlib.sha256(np.ascontiguousarray(posterior).tobytes()).hexdigest()
    assert digest == posterior_manifest["posterior_sha256"], "posterior digest does not match its manifest"
    assert list(posterior.shape) == list(posterior_manifest["posterior_shape"])
    assert posterior.shape[0] == posterior_manifest["results"]["n_posterior_samples"]
    assert posterior.shape[1] == posterior_manifest["prior"]["n_sampled_dimensions"]


def test_posterior_is_not_the_prior(posterior, posterior_manifest):
    """The samples must be narrower than the prior they were drawn under.

    Recomputed from the shipped samples rather than read from the manifest: a manifest can be
    edited, and this is the assertion the artifact exists to satisfy.
    """
    amplitude_sigma = posterior_manifest["prior"]["amplitude_sigma"]
    phase_sigma = posterior_manifest["prior"]["phase_sigma"]
    names = posterior_manifest["parameters"]
    prior_sigma = np.array([phase_sigma if "_phase_" in name else amplitude_sigma for name in names])

    shrinkage = posterior.std(axis=0) / prior_sigma
    unconstrained = int(np.count_nonzero(shrinkage > MAX_ACCEPTABLE_SHRINKAGE))
    assert unconstrained == 0, (
        f"{unconstrained} of {len(names)} parameters have sigma_post/sigma_prior > "
        f"{MAX_ACCEPTABLE_SHRINKAGE}; median {float(np.median(shrinkage)):.3f}. A posterior at the "
        "prior width is not a discriminating anchor — check whether the sampler mixed."
    )


def test_manifest_records_its_own_informativeness(posterior_manifest):
    """The recorded diagnostic must agree with the shipped samples.

    The manifest is what a reader consults; if it claimed a healthy shrinkage while the samples
    said otherwise, the artifact would be self-certifying and worthless.
    """
    recorded = posterior_manifest["informativeness"]
    assert recorded["n_parameters_above_0.9"] == 0
    assert recorded["sigma_post_over_sigma_prior_median"] < MAX_ACCEPTABLE_SHRINKAGE
    assert recorded["n_parameters"] == posterior_manifest["prior"]["n_sampled_dimensions"]


def test_manifest_records_the_sampler_that_produced_it(posterior_manifest):
    """The sampler settings are part of the artifact's meaning, not incidental metadata.

    ``rwalk`` is pinned as *not* the method here: it is what produced the prior-returning run, and
    a future regeneration that silently reverted to it would reintroduce that failure.
    """
    sampler = posterior_manifest["sampler"]
    assert sampler["sampler"] == "dynesty"
    assert sampler["sample"] == "rslice", (
        "the frozen posterior was produced with slice sampling; rwalk failed to mix on this "
        "60-dimensional problem and returned the prior"
    )
    assert sampler["nlive"] == 1000
    assert sampler["dlogz"] == 0.1
    assert posterior_manifest["smoke"] is False, "a smoke-configuration run must never be the reference"
