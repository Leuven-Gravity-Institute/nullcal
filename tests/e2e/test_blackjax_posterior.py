"""Distributional acceptance of BlackJAX against the frozen bilby posterior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import blackjax
import numpy as np
import pytest
from scipy.stats import ks_2samp

REFERENCE_DIR = Path(__file__).parent / "reference"
BLACKJAX_POSTERIOR_PATH = REFERENCE_DIR / "blackjax_posterior_samples.npz"
BLACKJAX_MANIFEST_PATH = REFERENCE_DIR / "blackjax_posterior_manifest.json"
BILBY_POSTERIOR_PATH = REFERENCE_DIR / "posterior_samples.npz"
BILBY_MANIFEST_PATH = REFERENCE_DIR / "posterior_manifest.json"

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# Fixed before the BlackJAX production run. These are acceptance criteria, not
# summaries chosen after inspecting the samples.
MAX_KS_STATISTIC = 0.10
MAX_RHAT = 1.01
MIN_ESS_BULK = 400.0
MIN_ESS_TAIL = 400.0
MAX_DIVERGENCES = 0


@pytest.fixture(scope="module")
def blackjax_artifact():
    if not BLACKJAX_POSTERIOR_PATH.exists() or not BLACKJAX_MANIFEST_PATH.exists():
        pytest.fail("BlackJAX posterior acceptance artifact is absent; run scripts/run_reference_blackjax.py")
    with np.load(BLACKJAX_POSTERIOR_PATH) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return arrays, json.loads(BLACKJAX_MANIFEST_PATH.read_text())


def test_blackjax_artifact_integrity_and_provenance(blackjax_artifact):
    arrays, manifest = blackjax_artifact
    digest = hashlib.sha256(np.ascontiguousarray(arrays["posterior_by_chain"]).tobytes()).hexdigest()

    assert digest == manifest["posterior_sha256"]
    assert list(arrays["posterior_by_chain"].shape) == manifest["posterior_shape"]
    assert arrays["parameters"].tolist() == manifest["parameters"]
    assert manifest["source_git_dirty"] is False
    assert len(manifest["source_git_revision"]) == 40


def test_blackjax_marginals_match_the_frozen_bilby_posterior(blackjax_artifact):
    arrays, manifest = blackjax_artifact
    with np.load(BILBY_POSTERIOR_PATH) as stored:
        bilby_posterior = stored["posterior"]
        bilby_parameters = stored["parameters"]
    bilby_manifest = json.loads(BILBY_MANIFEST_PATH.read_text())
    blackjax_posterior = arrays["posterior_by_chain"].reshape(-1, arrays["posterior_by_chain"].shape[-1])

    assert bilby_posterior.shape == (5937, 60)
    assert bilby_parameters.tolist() == arrays["parameters"].tolist()
    statistics = np.array(
        [ks_2samp(blackjax_posterior[:, index], bilby_posterior[:, index]).statistic for index in range(60)]
    )
    assert float(np.max(statistics)) <= MAX_KS_STATISTIC
    assert manifest["acceptance"]["max_ks_statistic"] == MAX_KS_STATISTIC
    assert manifest["anchors"]["bilby_posterior_sha256"] == bilby_manifest["posterior_sha256"]


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
    assert manifest["acceptance"] == {
        "max_divergences": MAX_DIVERGENCES,
        "max_ks_statistic": MAX_KS_STATISTIC,
        "max_rhat": MAX_RHAT,
        "min_ess_bulk": MIN_ESS_BULK,
        "min_ess_tail": MIN_ESS_TAIL,
    }
