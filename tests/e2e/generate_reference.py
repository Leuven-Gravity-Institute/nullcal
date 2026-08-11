"""Regenerate the frozen reference artifacts.

Regeneration is an explicit, reviewed action, never a side effect of running the test suite:
the artifacts exist to detect change, so a run that silently rewrites them detects nothing.

    uv run python -m tests.e2e.generate_reference

The manifest records the code version, dependency versions and per-array digests, so a
reference can be traced to the commit that produced it. Changing the reference and changing
the code that produces it must happen in the same commit, with the reason in the message.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from . import config, pipeline

REFERENCE_DIR = Path(__file__).parent / "reference"
ARTIFACT_PATH = REFERENCE_DIR / "artifacts.npz"
MANIFEST_PATH = REFERENCE_DIR / "manifest.json"

TRACKED_PACKAGES = ("nullcal", "bilby", "numpy", "scipy", "numba", "lalsuite", "rocket-fft")


def _digest(array: np.ndarray) -> str:
    """Content digest of an array, independent of how numpy chooses to lay it out."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _git_is_dirty() -> bool:
    """Whether tracked sources differ from HEAD, ignoring the reference artifacts being written.

    Recorded because ``git_revision`` alone can lie. The R17 artifacts were first generated from a
    working tree carrying an uncommitted fix, so the manifest named a revision that raises
    ``IndexError`` and cannot produce them — provenance pointing at the opposite implementation
    from the one that ran. A reviewer caught it; this makes it self-reporting.
    """
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "tests"],  # noqa: S607
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return any(line.strip() and "tests/e2e/reference/" not in line for line in output.splitlines())


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def main() -> int:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    likelihood = pipeline.build_likelihood()
    artifacts = pipeline.compute_artifacts(likelihood)

    arrays = {key: np.asarray(value) for key, value in artifacts.items()}
    np.savez_compressed(ARTIFACT_PATH, **arrays)

    manifest = {
        "git_revision": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "configuration": {
            "minimum_frequency": config.MINIMUM_FREQUENCY,
            "maximum_frequency": config.MAXIMUM_FREQUENCY,
            "sampling_frequency": config.SAMPLING_FREQUENCY,
            "duration": config.DURATION,
            "n_points": config.N_POINTS,
            "frequency_resolution": config.FREQUENCY_RESOLUTION,
            "nx": config.NX,
            "clustering_threshold": config.CLUSTERING_THRESHOLD,
            "seed": config.SEED,
            "source_parameters": config.SOURCE_PARAMETERS,
            "waveform_arguments": config.WAVEFORM_ARGUMENTS,
        },
        "artifacts": {
            key: {
                "shape": list(np.shape(value)),
                "dtype": str(np.asarray(value).dtype),
                "sha256": _digest(np.asarray(value)),
            }
            for key, value in artifacts.items()
        },
        "fixed_defects": {
            "noise_log_likelihood": (
                "Fixed in R17. Before that, RecalibrationLikelihood.noise_log_likelihood() raised "
                "IndexError, because compute_uncalibrated_time_frequency_domain_null_stream applied "
                "the 2-D time-frequency filter to the 2-D frequency-domain array it had already "
                "consumed and returned the time-frequency array unfiltered. The fix filters the "
                "array it returns, matching the calibrated path. noise_log_likelihood and "
                "uncalibrated_time_frequency_domain_null_stream are frozen artifacts from this "
                "revision onward; earlier manifests record them as absent under known_defects."
            )
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    size_kib = ARTIFACT_PATH.stat().st_size / 1024
    print(f"wrote {ARTIFACT_PATH} ({size_kib:.0f} KiB)")
    print(f"wrote {MANIFEST_PATH}")
    for key, meta in manifest["artifacts"].items():
        print(f"  {key:48s} {meta['shape']!s:18s} {meta['sha256'][:16]}")
    print(f"\nlog_likelihood = {float(artifacts['log_likelihood'])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
