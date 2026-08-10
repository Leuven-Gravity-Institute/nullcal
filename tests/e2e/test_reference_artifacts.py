"""End-to-end characterisation tests against frozen reference artifacts.

These pin the *current* numerical behaviour of the numpy/numba pipeline so that the BlackJAX
rewrite has something external to be checked against. Agreement between the JAX port and the
numba implementation would bound neither; agreement with an artifact produced before the port
started, at a stated tolerance, is what makes a regression visible.

Regenerate with ``uv run python -m tests.e2e.generate_reference`` — deliberately not automatic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from . import pipeline

REFERENCE_DIR = Path(__file__).parent / "reference"
ARTIFACT_PATH = REFERENCE_DIR / "artifacts.npz"
MANIFEST_PATH = REFERENCE_DIR / "manifest.json"

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

#: Same-implementation reruns must agree to round-off, not to a loose engineering tolerance:
#: the configuration is seeded, so the only admissible difference is floating-point
#: non-determinism in threaded reductions and in the FFT. ``atol=0.0`` is mandatory — the
#: numpy default of 1e-8 exceeds the magnitude of whitened strain quantities, which would make
#: the comparison vacuously true.
RTOL = 1e-12
ATOL = 0.0

#: Names whose reference is stored but which are compared exactly rather than approximately,
#: because they are integer- or boolean-valued and any change is structural.
EXACT_KEYS = frozenset({"frequency_mask", "time_frequency_filter"})


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    # fail, not skip: these artifacts are committed inputs, so their absence is a broken
    # checkout rather than an unavailable optional dependency. Skipping would let an explicit
    # ``pytest -m e2e`` run report success having compared nothing at all, which is the one
    # outcome this suite exists to prevent.
    if not ARTIFACT_PATH.exists():
        pytest.fail(f"reference artifacts absent: {ARTIFACT_PATH} (run tests.e2e.generate_reference)")
    with np.load(ARTIFACT_PATH) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.fail(f"reference manifest absent: {MANIFEST_PATH} (run tests.e2e.generate_reference)")
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def computed(tmp_path_factory) -> dict[str, np.ndarray]:
    likelihood = pipeline.build_likelihood(tmp_path_factory.mktemp("reference"))
    return {key: np.asarray(value) for key, value in pipeline.compute_artifacts(likelihood).items()}


def test_reference_covers_every_computed_artifact(reference, computed):
    """A quantity that stops being frozen must fail loudly, not vanish from the comparison."""
    assert set(computed) == set(reference), (
        f"computed-only: {sorted(set(computed) - set(reference))}; "
        f"reference-only: {sorted(set(reference) - set(computed))}"
    )


@pytest.mark.parametrize(
    "key",
    [
        "frequency_mask",
        "time_frequency_filter",
        "whitened_antenna_response",
        "projector",
        "calibration_factor",
        "uncalibrated_frequency_domain_null_stream",
        "calibrated_frequency_domain_null_stream",
        "calibrated_time_frequency_domain_null_stream",
        "wavelet_probe_output",
        "log_likelihood",
    ],
)
def test_artifact_matches_reference(reference, computed, key):
    expected = reference[key]
    actual = computed[key]
    assert actual.shape == expected.shape, f"{key}: shape {actual.shape} != reference {expected.shape}"
    assert actual.dtype == expected.dtype, f"{key}: dtype {actual.dtype} != reference {expected.dtype}"

    if key in EXACT_KEYS:
        assert np.array_equal(actual, expected), f"{key}: exact comparison failed"
        return

    # Peak-relative, never per-sample relative: these arrays cross zero, where a per-sample
    # relative error is meaningless.
    peak = float(np.max(np.abs(expected))) if expected.size else 0.0
    difference = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    relative = difference / peak if peak > 0.0 else difference
    assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"{key}: max|diff| = {difference:.6e}, peak = {peak:.6e}, peak-relative = {relative:.3e} (rtol={RTOL})"
    )


def test_filter_is_not_degenerate(computed):
    """Guard the guard: an all-False or all-True filter would make every comparison vacuous."""
    tf_filter = computed["time_frequency_filter"]
    selected = int(tf_filter.sum())
    assert 0 < selected < tf_filter.size, f"degenerate time-frequency filter: {selected} of {tf_filter.size}"


def test_calibrated_null_stream_is_confined_to_the_filter(computed):
    """The calibrated path must zero every pixel outside the time-frequency filter."""
    calibrated = computed["calibrated_time_frequency_domain_null_stream"]
    tf_filter = computed["time_frequency_filter"]
    outside = calibrated[:, ~tf_filter]
    assert np.count_nonzero(outside) == 0, f"{np.count_nonzero(outside)} non-zero pixels outside the filter"
    assert np.count_nonzero(calibrated) == int(tf_filter.sum()) * calibrated.shape[0]


def test_manifest_matches_artifacts(reference, manifest):
    """The manifest's digests must describe the artifacts actually shipped beside it."""
    import hashlib

    # Iterating the manifest alone cannot notice a deleted entry: drop a key from the manifest
    # and the loop simply checks one fewer artifact, silently. Compare the two key sets first so
    # the manifest is required to describe the npz exactly, in both directions.
    assert set(manifest["artifacts"]) == set(reference), (
        f"manifest describes {sorted(set(manifest['artifacts']) - set(reference))} "
        f"which are absent from the npz, and omits {sorted(set(reference) - set(manifest['artifacts']))}"
    )

    for key, meta in manifest["artifacts"].items():
        raw = np.asarray(reference[key])
        # Digest the contiguous bytes, but report shape from `raw`: np.ascontiguousarray
        # promotes a 0-d array to shape (1,), so the scalar log_likelihood would appear to
        # contradict the manifest's correct []. The promotion does not change the bytes.
        stored = np.ascontiguousarray(raw)
        assert hashlib.sha256(stored.tobytes()).hexdigest() == meta["sha256"], f"{key}: manifest digest mismatch"
        # The digest already pins the bytes, but shape and dtype are what a reader trusts when
        # deciding whether an artifact is the thing they think it is; an unchecked field drifts.
        assert list(raw.shape) == list(meta["shape"]), f"{key}: shape {raw.shape} != manifest {meta['shape']}"
        assert raw.dtype.name == meta["dtype"], f"{key}: dtype {raw.dtype.name} != manifest {meta['dtype']}"
