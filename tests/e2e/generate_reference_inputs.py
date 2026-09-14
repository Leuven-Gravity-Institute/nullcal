"""Freeze the architecture-sensitive inputs to the reference pipeline.

This is an explicit maintenance command, never a test side effect:

    uv run python -m tests.e2e.generate_reference_inputs

The existing output artifacts are deliberately not regenerated. Freezing the whitened strain
and power spectral density separately removes waveform generation from later comparisons while
preserving the original outputs as the independent numerical anchor.
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
INPUT_ARTIFACT_PATH = REFERENCE_DIR / "inputs.npz"
INPUT_MANIFEST_PATH = REFERENCE_DIR / "inputs_manifest.json"

TRACKED_PACKAGES = ("nullcal", "bilby", "numpy", "scipy", "numba", "lalsuite", "rocket-fft")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _source_is_dirty() -> bool:
    """Whether source or the input-generating harness differs from the recorded revision."""
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "src", "tests/e2e"],  # noqa: S607
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    return bool(output.strip())


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def main() -> int:
    source_git_dirty = _source_is_dirty()
    if source_git_dirty:
        raise RuntimeError("source or the e2e harness is dirty; inputs require an unmodified revision")

    likelihood = pipeline.build_likelihood()
    inputs = {
        "whitened_frequency_domain_strain": np.array(
            likelihood.null_stream_calculator._whitened_frequency_domain_strain_array,
            copy=True,
        ),
        "power_spectral_density": np.array(
            [interferometer.power_spectral_density_array for interferometer in likelihood.interferometers],
            copy=True,
        ),
    }

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUT_ARTIFACT_PATH, **inputs)
    manifest = {
        "source_git_revision": _git_revision(),
        "source_git_dirty": source_git_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "configuration": {
            "minimum_frequency": config.MINIMUM_FREQUENCY,
            "maximum_frequency": config.MAXIMUM_FREQUENCY,
            "sampling_frequency": config.SAMPLING_FREQUENCY,
            "duration": config.DURATION,
            "detector_names": list(config.DETECTOR_NAMES),
            "start_time": config.start_time(),
            "n_points": config.N_POINTS,
            "frequency_resolution": config.FREQUENCY_RESOLUTION,
            "nx": config.NX,
            "clustering_threshold": config.CLUSTERING_THRESHOLD,
            "seed": config.SEED,
            "wavelet_probe_seed": config.SEED + 1,
            "source_parameters": config.SOURCE_PARAMETERS,
            "waveform_arguments": config.WAVEFORM_ARGUMENTS,
        },
        "artifacts": {
            key: {
                "shape": list(value.shape),
                "dtype": value.dtype.name,
                "sha256": _digest(value),
            }
            for key, value in inputs.items()
        },
    }
    INPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {INPUT_ARTIFACT_PATH}")
    print(f"wrote {INPUT_MANIFEST_PATH}")
    for key, meta in manifest["artifacts"].items():
        print(f"  {key:38s} {meta['shape']!s:18s} {meta['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
