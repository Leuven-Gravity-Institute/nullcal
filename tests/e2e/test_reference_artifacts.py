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

#: Tolerance on ``max|actual - expected|`` scaled by the reference array's peak. Applied as a
#: peak-scaled absolute budget rather than a per-element relative one, because these arrays
#: cross zero and are mostly zero outside the filter: the calibrated time-frequency array is
#: ~97.7% exact zeros, and a per-element relative budget against an exact-zero reference is
#: exactly 0.0, which silently demands bit-exactness. The smallest non-zero elements are ~1e-4,
#: whose per-element budget would be ~1e-16 — at or below the round-off of the wavelet and FFT
#: sums, so a per-element rule would not in fact admit the reduction-order non-determinism it
#: was meant to allow.
#:
#: Note on ``atol``: the numpy default of 1e-8 is wrong here, but not because it would be
#: vacuous. The quantities compared are *whitened* — the null streams peak at ~1.7 and
#: ``log_likelihood`` is ~-204 — so a 1e-8 absolute floor would not swamp them; it would loosen
#: the effective tolerance to ~1e-8 relative, a ~1e4 loss of sensitivity. (The ~1e-24 quantity
#: is the raw unwhitened strain, which is not among the compared artifacts.) A default atol
#: *would* be vacuous on unwhitened strain; that argument simply does not apply to these arrays.
RTOL = 1e-12

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


#: The artifacts compared value-by-value. Spelled out rather than derived from the npz at
#: collection time on purpose: parametrising over the file's keys would mean a missing or
#: truncated artifact file silently collects fewer tests — the same "compares nothing quietly"
#: failure the fixtures were changed to prevent, arriving from the other direction.
#: ``test_compared_keys_cover_every_artifact`` is what stops this list drifting from reality.
COMPARED_KEYS = (
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
)


def test_compared_keys_cover_every_artifact(reference, computed):
    """Every frozen and every computed artifact must be in the value-comparison list.

    Without this, adding a quantity to ``compute_artifacts`` and regenerating the reference
    leaves it frozen, digested and set-equal — but never value-compared, so a later change to
    that quantity alone passes the whole suite.
    """
    compared = set(COMPARED_KEYS)
    assert compared == set(reference), (
        f"frozen but not compared: {sorted(set(reference) - compared)}; "
        f"compared but not frozen: {sorted(compared - set(reference))}"
    )
    assert compared == set(computed), (
        f"computed but not compared: {sorted(set(computed) - compared)}; "
        f"compared but not computed: {sorted(compared - set(computed))}"
    )


@pytest.mark.parametrize("key", COMPARED_KEYS)
def test_artifact_matches_reference(reference, computed, key):
    expected = reference[key]
    actual = computed[key]
    assert actual.shape == expected.shape, f"{key}: shape {actual.shape} != reference {expected.shape}"
    assert actual.dtype == expected.dtype, f"{key}: dtype {actual.dtype} != reference {expected.dtype}"

    if key in EXACT_KEYS:
        assert np.array_equal(actual, expected), f"{key}: exact comparison failed"
        return

    # Peak-relative, never per-element relative: these arrays cross zero and are mostly zero
    # outside the filter, where a per-element relative budget is 0.0 and therefore demands
    # bit-exactness. Assert exactly the quantity the failure message reports — asserting one
    # tolerance while reporting another lets a failure print a number that reads like a pass.
    assert expected.size > 0, f"{key}: reference array is empty, so the comparison is vacuous"
    peak = float(np.max(np.abs(expected)))
    difference = float(np.max(np.abs(actual - expected)))
    relative = difference / peak if peak > 0.0 else difference
    assert relative <= RTOL, (
        f"{key}: max|diff| = {difference:.6e}, peak = {peak:.6e}, peak-relative = {relative:.3e} exceeds rtol={RTOL}"
    )


def test_manifest_configuration_matches_the_live_config(manifest):
    """The manifest's recorded configuration must equal what ``config.py`` says today.

    These are the same quantities encoded in two places, so they are a latent inconsistency
    until something compares them. The arrays are protected by their digests; the *provenance*
    is not, and a manifest whose recorded seed or source parameters have drifted from the module
    misdescribes how the artifacts were made while every numerical test still passes. That turns
    the manifest from a record into a decoration.
    """
    from . import config

    recorded = manifest["configuration"]
    expected = {
        "duration": config.DURATION,
        "sampling_frequency": config.SAMPLING_FREQUENCY,
        "minimum_frequency": config.MINIMUM_FREQUENCY,
        "maximum_frequency": config.MAXIMUM_FREQUENCY,
        "n_points": config.N_POINTS,
        "frequency_resolution": config.FREQUENCY_RESOLUTION,
        "nx": config.NX,
        "clustering_threshold": config.CLUSTERING_THRESHOLD,
        "seed": config.SEED,
        "source_parameters": dict(config.SOURCE_PARAMETERS),
        "waveform_arguments": dict(config.WAVEFORM_ARGUMENTS),
    }
    assert set(recorded) == set(expected), (
        f"manifest configuration keys differ from config.py: "
        f"manifest-only {sorted(set(recorded) - set(expected))}, "
        f"config-only {sorted(set(expected) - set(recorded))}"
    )
    for key, value in expected.items():
        assert recorded[key] == value, f"manifest configuration[{key!r}] = {recorded[key]!r}, config.py says {value!r}"


def test_manifest_records_the_provenance_fields_it_promises(manifest):
    """The traceability block must be present and non-empty.

    Not compared against the running environment: ``git_revision``, ``platform``, ``python`` and
    ``packages`` describe where the artifacts *were generated*, which is deliberately not where
    the test now runs. Asserting they match HEAD would force a regeneration on every commit,
    which the regeneration policy exists to prevent. What can be checked is that they are
    recorded at all.
    """
    for field in ("git_revision", "platform", "python", "packages"):
        assert manifest.get(field), f"manifest is missing the {field!r} provenance field"
    for package in ("bilby", "numpy", "scipy", "numba", "lalsuite"):
        assert package in manifest["packages"], f"manifest packages omits {package!r}"


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
    # Inside the filter, require that the array is substantially populated rather than that
    # *every* pixel is non-zero: an exactly-zero wavelet coefficient at a filtered pixel is
    # numerically possible and would fail an equality on the count without anything being wrong.
    # The point of this assertion is to catch an all-zero array, not to pin the realisation.
    inside = np.count_nonzero(calibrated[:, tf_filter])
    total_inside = int(tf_filter.sum()) * calibrated.shape[0]
    assert inside > 0.9 * total_inside, f"only {inside} of {total_inside} pixels inside the filter are non-zero"


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
