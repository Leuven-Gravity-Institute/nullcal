"""Measure finite-arm ET null leakage and its induced LWA calibration bias."""

import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from nullcal.studies.lwa_leakage import (
    ARM_LENGTH_METRES,
    SPEED_OF_LIGHT_METRES_PER_SECOND,
    evaluate_frequency,
    generate_population,
)

FREQUENCIES_HZ = np.array(
    [20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0]
)
POPULATION_SIZE = 256
NOISE_REALIZATIONS = 16
NETWORK_SNR = 120.0
PRIOR_SIGMA = 0.05
SIGNIFICANCE_ALPHA = 0.002699796063260207
POPULATION_SEED = 397
NOISE_SEED = 9381
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 2718
AMPLITUDE_BIAS_LIMIT_PERCENT = 4.0
PHASE_BIAS_LIMIT_DEGREES = 4.0


def validity_band_maximum(results: list[dict[str, float | int]]) -> float | None:
    """Return the end of the contiguous p95-bias-valid frequency grid."""
    valid_maximum = None
    for row in results:
        if (
            row["amplitude_bias_percent_p95_upper95"] >= AMPLITUDE_BIAS_LIMIT_PERCENT
            or row["phase_bias_degrees_p95_upper95"] >= PHASE_BIAS_LIMIT_DEGREES
        ):
            break
        valid_maximum = float(row["frequency_hz"])
    return valid_maximum


def write_csv(path: Path, results: list[dict[str, float | int]]) -> None:
    """Write the complete bias-versus-frequency and sky-factor table."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)


def write_figure(path: Path, results: list[dict[str, float | int]]) -> None:
    """Plot sky factor, leakage significance, and induced MAP bias."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - plotting is an optional driver-only dependency

    frequency = np.array([row["frequency_hz"] for row in results])
    figure, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    axes[0].fill_between(
        frequency,
        [row["sky_factor_p05"] for row in results],
        [row["sky_factor_p95"] for row in results],
        alpha=0.25,
        label="sky/inclination p05-p95",
    )
    axes[0].plot(frequency, [row["sky_factor_median"] for row in results], marker="o", label="median")
    axes[0].set_ylabel(r"$|h_{null}| / [\rho_{net}(2\pi fL/c)]$")
    axes[0].legend(frameon=False)

    axes[1].fill_between(
        frequency,
        [row["leakage_snr_p05"] for row in results],
        [row["leakage_snr_p95"] for row in results],
        alpha=0.25,
        label="leaked null SNR p05-p95",
    )
    axes[1].plot(frequency, [row["leakage_snr_median"] for row in results], marker="o")
    axes[1].axhline(np.sqrt(-np.log(SIGNIFICANCE_ALPHA)), color="black", linestyle="--", label="3-sigma null threshold")
    axes[1].set_ylabel("leaked null SNR")
    axes[1].legend(frameon=False)

    axes[2].plot(
        frequency,
        [row["amplitude_bias_percent_p95"] for row in results],
        marker="o",
        label="amplitude-bias p95 (%)",
    )
    axes[2].plot(
        frequency,
        [row["amplitude_bias_percent_p95_upper95"] for row in results],
        linestyle=":",
        label="amplitude p95 upper 95% bound",
    )
    axes[2].plot(
        frequency,
        [row["phase_bias_degrees_p95"] for row in results],
        marker="s",
        label="phase-bias p95 (deg)",
    )
    axes[2].plot(
        frequency,
        [row["phase_bias_degrees_p95_upper95"] for row in results],
        linestyle=":",
        label="phase p95 upper 95% bound",
    )
    axes[2].axhline(4.0, color="black", linestyle="--", label="4% / 4 deg criterion")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("paired maximum-detector bias")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n")


def _table_rows(results: list[dict[str, float | int]]) -> str:
    return "\n".join(
        "| {frequency_hz:.0f} | {two_pi_f_l_over_c:.4f} | {sky_factor_median:.3f} "
        "[{sky_factor_p05:.3f}, {sky_factor_p95:.3f}] | {leakage_snr_median:.2f} "
        "[{leakage_snr_p05:.2f}, {leakage_snr_p95:.2f}] | {present_detection_fraction:.3f} | "
        "{amplitude_bias_percent_p95:.3f} ({amplitude_bias_percent_p95_upper95:.3f}) | "
        "{phase_bias_degrees_p95:.3f} ({phase_bias_degrees_p95_upper95:.3f}) |".format(**row)
        for row in results
    )


def write_readme(path: Path, results: list[dict[str, float | int]], commit: str, elapsed_seconds: float) -> None:
    """Write the result, validity-band recommendation, anchors, and limitations."""
    validity_maximum = validity_band_maximum(results)
    if validity_maximum is None:
        recommendation = "No tested frequency satisfies the predeclared p95 bias criterion."
    else:
        first_invalid = next((row for row in results if row["frequency_hz"] > validity_maximum), None)
        bracket = (
            f"; the first failing grid point is {first_invalid['frequency_hz']:.0f} Hz"
            if first_invalid is not None
            else "; no failure occurs on the tested grid"
        )
        recommendation = (
            f"Use **20-{validity_maximum:.0f} Hz** as the conservative M3 LWA validity band on this grid{bracket}. "
            "This is a pre-check recommendation, not a measured continuous-frequency boundary."
        )
    maximum_failure_count = max(int(row["map_failure_count"]) for row in results)
    text = f"""# ET finite-arm leakage and LWA calibration-bias pre-check

This study compares the full frequency-dependent response of the directed 10 km Einstein Telescope
triangle with the long-wavelength-approximation (LWA) response. It uses 256 isotropic
sky/polarization/inclination draws, 16 complex Gaussian ET-D noise realizations per draw, and 15
frequencies from 20 to 2000 Hz. Every narrowband signal is normalized to coherent LWA network SNR
120. Signal-present and signal-absent calculations reuse the **identical noise draw**.

## Result

| f (Hz) | 2 pi fL/c | sky factor median [p05, p95] | leaked null SNR median [p05, p95] | 3-sigma detection fraction | amplitude-bias p95 (upper 95% bound), % | phase-bias p95 (upper 95% bound), deg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_table_rows(results)}

The complete numerical table is `bias_vs_frequency.csv`; it also reports p99 and maxima, the paired
signal-minus-null statistic shift, the noise-only false-alarm fraction, ET-D PSD, and MAP convergence.
No frequency has more than {maximum_failure_count} failed MAP fits.

## Validity-band recommendation for M3

{recommendation}

The criterion is fixed before the clean run: at every included grid point, the one-sided 95% source-block
bootstrap upper bound on the population 95th percentile of the maximum absolute detector amplitude bias
must be below 4%, and the corresponding phase bound must be below 4 degrees. A later point cannot reopen
the contiguous band. The bootstrap resamples all 16 noise draws with each of 256 source rows, preserving
their clustered design. The maximum over detectors is used for every realization; no RMS calibration
error is reported.

## Stated null and significance

The null hypothesis is independent, correctly modelled Gaussian ET-D noise with no signal. After
whitening, the fixed ET sum divided by sqrt(3) is a unit circular complex Gaussian variate, so
`2 |z_null|^2` follows chi-square with two degrees of freedom and the single-bin p-value is
`exp(-|z_null|^2)`. The predeclared threshold is the two-sided 3-sigma tail probability
`alpha = {SIGNIFICANCE_ALPHA:.16g}`. The detection fraction in the table is the signal-present
fraction below that p-value; the CSV's false-alarm fraction applies the same test to the paired
signal-absent sample.

## Calibration-bias measurement

For each one-frequency datum, the recovery minimizes the same LWA null-projector Gaussian likelihood
and independent 5% Gaussian amplitude/latent-phase priors used by `nullcal`. The ET LWA signal plane
has left-null vector `(1, 1, 1)`; after detector calibration factors `c_i`, the exact normalized null
vector is proportional to `1/conj(c_i)`. The reported induced bias is the paired MAP difference between
exact-response signal data and an LWA-response signal control at identical noise, converted to physical
calibration amplitude and phase, then maximized over the three detectors. This control removes the
ordinary change in calibration information caused by adding any loud, perfectly LWA signal. The separate
signal-present versus signal-absent pair is retained for the leakage significance test above.

## Anchors and quantities that remain unanchored

- The finite-arm transfer is Eq. (6) of [Virtuoso and Milotti (2024)](https://arxiv.org/abs/2412.01693).
  Its zero-frequency closure and a transverse-arm closed form are tested independently. The leading
  amplitude scaling `2 pi fL/c` is externally stated by
  [Goncharov, Nitz, and Harms (2022)](https://arxiv.org/abs/2204.08533).
- ET-D PSD values are pinned from Bilby's tabulation at
  [commit 75f834b](https://github.com/bilby-dev/bilby/blob/75f834b14c0f2d8314d61b7276c39ed0ad1c00c9/bilby/gw/detector/noise_curves/ET_D_psd.txt).
  The run uses the tabulated values at every frequency grid point, not an analytic fit.
- Network SNR 120 and the 4% / 4 degree comparison are anchored to
  [Sinha, Sun, and Ma (2025)](https://arxiv.org/abs/2506.15979), but that paper studied a broadband
  Cosmic Explorer ringdown and a narrow calibration bump. Transferring its threshold to ET and to
  this one-bin diagnostic remains **unanchored**.
- The sky-factor distribution, leakage SNRs, MAP biases, and recommended band are new computational
  results produced by this run. No external numerical replication was found; they remain **unanchored
  to an independent numerical reference**. Agreement between code paths is not presented as accuracy.
- This pre-check is not the M4 D2 result: it uses monochromatic bins, fixed SNR, independent equal ET-D
  noise, a MAP rather than a posterior, and calibration parameters free at each frequency. M4 must use
  a broadband source, the 19-knot calibration curve, posterior coverage, more noise realizations, and
  frequency-dependent null mitigation. The ET-D curve affects physical strain/noise scaling, while the
  fixed-SNR whitened statistics deliberately isolate the geometric frequency dependence.

## Provenance

Run `2026-09-21-lwa-leakage-study` used `scripts/lwa_leakage_study.py` at nullcal commit `{commit}`.
The bounded run comprised 256 population draws x 15 frequencies x 16 noise realizations = 61,440
paired signal/null cases (122,880 MAP fits) and completed in {elapsed_seconds:.1f} seconds. Exact seeds,
versions, configuration, all table rows, and the reproduction command are in `run.json`.
"""
    path.write_text(text, encoding="utf-8")


def git_metadata() -> tuple[str, bool]:
    """Return the exact nullcal revision and whether its worktree is dirty."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to stamp study provenance")
    commit = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
    )
    return commit, dirty


def main() -> None:
    """Run the bounded study and write its table, figure, and provenance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_metadata()
    population = generate_population(POPULATION_SIZE, seed=POPULATION_SEED)
    started = time.perf_counter()
    results = []
    for frequency in FREQUENCIES_HZ:
        print(f"evaluating {frequency:g} Hz", flush=True)
        results.append(
            evaluate_frequency(
                float(frequency),
                population,
                noise_realizations=NOISE_REALIZATIONS,
                noise_seed=NOISE_SEED,
                network_snr=NETWORK_SNR,
                prior_sigma=PRIOR_SIGMA,
                significance_alpha=SIGNIFICANCE_ALPHA,
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
        )
    elapsed_seconds = time.perf_counter() - started
    if any(row["map_failure_count"] for row in results):
        raise RuntimeError("one or more calibration MAP fits failed; refusing to publish partial summaries")

    write_csv(arguments.output_directory / "bias_vs_frequency.csv", results)
    write_figure(arguments.output_directory / "lwa_leakage_and_bias.svg", results)
    write_readme(arguments.output_directory / "README.md", results, commit, elapsed_seconds)
    run = {
        "run_id": "2026-09-21-lwa-leakage-study",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "code": {
            "repository": "https://github.com/Leuven-Gravity-Institute/nullcal",
            "commit": commit,
            "dirty": dirty,
            "script": "scripts/lwa_leakage_study.py",
        },
        "command": (
            "uv run --with matplotlib python scripts/lwa_leakage_study.py "
            "--output-directory <paper-repo>/exploratory-tests/lwa-leakage-band"
        ),
        "execution_environment": "local arm64 reference platform",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "configuration": {
            "arm_length_metres": ARM_LENGTH_METRES,
            "speed_of_light_metres_per_second": SPEED_OF_LIGHT_METRES_PER_SECOND,
            "frequencies_hz": FREQUENCIES_HZ.tolist(),
            "population_size": POPULATION_SIZE,
            "population_seed": POPULATION_SEED,
            "noise_realizations": NOISE_REALIZATIONS,
            "noise_seed_reused_at_every_frequency": NOISE_SEED,
            "network_snr": NETWORK_SNR,
            "calibration_prior_sigma": PRIOR_SIGMA,
            "significance_alpha": SIGNIFICANCE_ALPHA,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "amplitude_bias_limit_percent": AMPLITUDE_BIAS_LIMIT_PERCENT,
            "phase_bias_limit_degrees": PHASE_BIAS_LIMIT_DEGREES,
            "noise": "independent equal ET-D PSD; signal-present and signal-absent pairs share exact draws",
            "response": "Virtuoso and Milotti arXiv:2412.01693 Eq. 6",
        },
        "results": results,
        "recommendation": {
            "m3_lwa_validity_band_minimum_hz": float(FREQUENCIES_HZ[0]),
            "m3_lwa_validity_band_maximum_hz": validity_band_maximum(results),
            "criterion": (
                "contiguous grid with one-sided 95% source-block bootstrap upper bound on p95 paired "
                "max-detector amplitude <4% and phase <4 deg"
            ),
        },
        "claims": {
            "finite_arm_leakage_detected_against_stated_noise_null": {
                "supported_by": "bias_vs_frequency.csv present_detection_fraction"
            },
            "finite_arm_leakage_induces_lwa_calibration_bias": {
                "supported_by": "bias_vs_frequency.csv paired MAP bias columns"
            },
            "m3_validity_band_recommendation": {"supported_by": "recommendation and p95 bias columns"},
        },
    }
    (arguments.output_directory / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
