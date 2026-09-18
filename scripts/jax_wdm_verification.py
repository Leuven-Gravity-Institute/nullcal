"""Print the reproducible verification table for the JAX WDM implementation."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from shutil import which

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullcal.null_stream.calibration import (  # noqa: E402
    compute_calibrated_whitened_antenna_response,
)
from nullcal.time_frequency_transform import transform_wavelet_freq  # noqa: E402
from nullcal.time_frequency_transform.wavelet_transforms import inverse_wavelet_freq  # noqa: E402
from tests.e2e import config, pipeline  # noqa: E402

REFERENCE_TOLERANCE = 1e-12
GRADIENT_TOLERANCE = 1e-9
REFERENCE_PATH = ROOT / "tests/e2e/reference/artifacts.npz"
TONE_LAYER = 10
TONE_CONTAINMENT_MINIMUM = 0.999


def report_identity(revision: str) -> str:
    """Identify the code and host behind platform-sensitive last-bit values."""
    return f"Revision: `{revision}`\nPlatform: `{platform.platform()}`"


def peak_comparison(actual, reference):
    """Return maximum absolute and peak-relative errors with no absolute floor."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if reference.dtype == np.bool_:
        difference = float(np.any(actual != reference))
        return difference, 1.0, difference
    difference = float(np.max(np.abs(actual - reference))) if reference.size else 0.0
    peak = float(np.max(np.abs(reference))) if reference.size else 0.0
    relative = difference / peak if peak else difference
    return difference, peak, relative


def reference_table():
    """Compare independently reproduced outputs with their frozen artifacts."""
    with np.load(REFERENCE_PATH) as archive:
        reference = {name: archive[name] for name in archive.files}
    likelihood = pipeline.build_likelihood_from_reference_inputs()
    actual = pipeline.compute_artifacts(likelihood)
    rows = []
    for name in sorted(reference.keys() - pipeline.FED_BACK_INPUT_KEYS):
        difference, peak, relative = peak_comparison(actual[name], reference[name])
        rows.append((name, difference, peak, relative))
    return rows


def gradient_checks():
    """Compare JAX gradients with independent central differences."""
    response = jnp.arange(24, dtype=jnp.float64).reshape(4, 3, 2) / 10.0
    mask = jnp.array([True, False, True, True])

    def calibration_objective(scale):
        factors = jnp.ones((3, 4), dtype=jnp.complex128).at[1, 2].set(1.0 + 1j * scale)
        folded = compute_calibrated_whitened_antenna_response(response, factors, mask)
        return jnp.sum(jnp.abs(folded) ** 2)

    n_t, n_f = 32, 8
    base = jnp.asarray(np.fft.rfft(np.random.default_rng(391).normal(size=n_t * n_f)))
    direction = jnp.linspace(0.2, 1.0, base.size) * (1.0 + 0.25j)

    def transform_objective(scale):
        wave = transform_wavelet_freq(base + scale * direction, n_f=n_f, n_t=n_t)
        return jnp.sum(wave**2) / wave.size

    inverse_base = transform_wavelet_freq(
        np.fft.rfft(np.random.default_rng(393).normal(size=n_t * n_f)), n_f=n_f, n_t=n_t
    )
    inverse_direction = jnp.reshape(jnp.linspace(-0.4, 0.7, n_t * n_f), (n_t, n_f))

    def inverse_objective(scale):
        recovered = inverse_wavelet_freq(inverse_base + scale * inverse_direction, n_f=n_f, n_t=n_t)
        return jnp.sum(jnp.abs(recovered) ** 2) / recovered.size

    cases = (
        ("calibration fold", calibration_objective, 0.17),
        ("forward WDM", transform_objective, 0.13),
        ("inverse WDM", inverse_objective, -0.21),
    )
    step = 3e-3
    rows = []
    for name, objective, point in cases:
        autodiff = float(jax.grad(objective)(point))
        finite = float((objective(point + step) - objective(point - step)) / (2.0 * step))
        relative = abs(autodiff - finite) / abs(finite)
        rows.append((name, autodiff, finite, relative))
    return rows


def precision_comparison():
    """Measure frozen-probe error in explicit float64 and float32 modes."""
    with np.load(REFERENCE_PATH) as archive:
        reference = archive["wavelet_probe_output"]
    n_t, n_f = reference.shape
    probe = config.wavelet_probe_input(n_frequencies=n_t * n_f // 2 + 1)
    output64 = transform_wavelet_freq(probe, n_f=n_f, n_t=n_t)
    with jax.enable_x64(False):
        output32 = transform_wavelet_freq(jnp.asarray(probe, dtype=jnp.complex64), n_f=n_f, n_t=n_t)
    return (
        ("float64", str(output64.dtype), peak_comparison(output64, reference)[2]),
        ("float32", str(output32.dtype), peak_comparison(output32, reference)[2]),
    )


def external_anchors():
    """Measure Parseval, round-trip, and analytic tone-bin anchors."""
    n_t, n_f = 64, 32
    n_samples = n_t * n_f
    sampling_frequency = 512.0
    frequency_resolution = sampling_frequency / (2 * n_f)
    rng = np.random.default_rng(20260914)
    noise = rng.normal(size=n_samples)
    frequency_data = np.fft.rfft(noise)
    wave = transform_wavelet_freq(frequency_data, n_f=n_f, n_t=n_t)
    recovered = inverse_wavelet_freq(wave, n_f=n_f, n_t=n_t)
    energy_ratio = float(jnp.sum(wave**2) / np.sum(noise**2))
    round_trip = peak_comparison(recovered, frequency_data)[2]

    layer = TONE_LAYER
    time = np.arange(n_samples) / sampling_frequency
    tone = np.sin(2.0 * np.pi * layer * frequency_resolution * time)
    tone_wave = transform_wavelet_freq(np.fft.rfft(tone), n_f=n_f, n_t=n_t)
    power = np.sum(np.asarray(tone_wave) ** 2, axis=0)
    return n_samples, energy_ratio, round_trip, int(np.argmax(power)), float(power[layer] / np.sum(power))


def main():
    """Run all checks and print Markdown tables."""
    git = which("git")
    if git is None:
        raise RuntimeError("git is required to record the verification revision")
    revision = subprocess.run(  # noqa: S603 - executable is resolved from the local PATH above
        [git, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(  # noqa: S603 - executable is resolved from the local PATH above
        [git, "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status:
        raise RuntimeError("verification must run from a clean tree so the revision identifies the measured code")
    print(report_identity(revision))
    print(f"\nFrozen-reference tolerance: peak-relative <= {REFERENCE_TOLERANCE:.0e}, atol = 0.0\n")
    print("| artifact | max abs diff | reference peak | peak-relative | pass |")
    print("| --- | ---: | ---: | ---: | :---: |")
    reference_rows = reference_table()
    for name, difference, peak, relative in reference_rows:
        print(
            f"| {name} | {difference:.6e} | {peak:.6e} | {relative:.3e} | {'yes' if relative <= REFERENCE_TOLERANCE else 'no'} |"
        )
    if not all(row[3] <= REFERENCE_TOLERANCE for row in reference_rows):
        raise RuntimeError("one or more frozen-reference comparisons failed")

    print(f"\nGradient tolerance: relative <= {GRADIENT_TOLERANCE:.0e}\n")
    print("| path | autodiff | central difference | relative error | pass |")
    print("| --- | ---: | ---: | ---: | :---: |")
    gradient_rows = gradient_checks()
    for name, autodiff, finite, relative in gradient_rows:
        print(
            f"| {name} | {autodiff:.12e} | {finite:.12e} | {relative:.3e} | {'yes' if relative <= GRADIENT_TOLERANCE else 'no'} |"
        )
    if not all(row[3] <= GRADIENT_TOLERANCE for row in gradient_rows):
        raise RuntimeError("one or more gradient comparisons failed")

    print("\n| requested mode | output dtype | frozen peak-relative error |")
    print("| --- | --- | ---: |")
    precision_rows = precision_comparison()
    for mode, dtype, relative in precision_rows:
        print(f"| {mode} | {dtype} | {relative:.3e} |")
    if precision_rows[0][2] > REFERENCE_TOLERANCE or precision_rows[1][2] <= REFERENCE_TOLERANCE:
        raise RuntimeError("the precision comparison did not establish the x64 requirement")
    print("\nJAX x64 is required: float32 does not satisfy the frozen-reference tolerance.")

    n_samples, energy, round_trip, tone_layer, containment = external_anchors()
    print("\n| external anchor | measured | expected |")
    print("| --- | ---: | ---: |")
    print(f"| Parseval energy ratio | {energy:.12e} | {n_samples:.12e} |")
    print(f"| round-trip peak-relative error | {round_trip:.3e} | <= {REFERENCE_TOLERANCE:.0e} |")
    print(f"| tone peak layer | {tone_layer} | {TONE_LAYER} |")
    print(f"| tone power containment | {containment:.12f} | > {TONE_CONTAINMENT_MINIMUM} |")
    anchors_pass = (
        abs(energy - n_samples) / n_samples <= REFERENCE_TOLERANCE
        and round_trip <= REFERENCE_TOLERANCE
        and tone_layer == TONE_LAYER
        and containment > TONE_CONTAINMENT_MINIMUM
    )
    if not anchors_pass:
        raise RuntimeError("one or more external anchor checks failed")


if __name__ == "__main__":
    main()
