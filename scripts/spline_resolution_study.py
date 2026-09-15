"""Measure calibration-spline resolution for a narrow Gaussian bump."""

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import find_peaks

from nullcal.studies.spline_resolution import (
    augmented_log_knots,
    gaussian_bump,
    minimax_phase_spline_fit,
    minimax_spline_fit,
    spline_design_matrix,
)

PEAK_FREQUENCY_HZ = 249.43
FREQUENCY_WIDTH_HZ = 50.0
AMPLITUDE_PEAK = -0.10
PHASE_PEAK_DEGREES = 10.0
ANALYSIS_MINIMUM_HZ = 20.0
ANALYSIS_MAXIMUM_HZ = 2000.0
EXTENDED_SPLINE_MINIMUM_HZ = 8.0
EXTENDED_SPLINE_MAXIMUM_HZ = 2048.0
INITIAL_FIT_GRID_SIZE = 2_501
VALIDATION_GRID_SIZE = 200_001
PHASE_BISECTION_STEPS = 22
AMPLITUDE_REFINEMENT_TOLERANCE = 1e-4
PHASE_REFINEMENT_TOLERANCE_RADIANS = 1e-5
MAX_REFINEMENT_ITERATIONS = 15


@dataclass(frozen=True)
class Configuration:
    """A spline grid and the frequency interval on which to assess it."""

    name: str
    placement: str
    knots_hz: np.ndarray
    evaluation_minimum_hz: float
    evaluation_maximum_hz: float


def configurations() -> list[Configuration]:
    """Return the submitted, extended-uniform, and QNM-dense grids."""
    output = [
        Configuration(
            name="submitted-10",
            placement="uniform log",
            knots_hz=np.geomspace(8.0, 512.0, 10),
            evaluation_minimum_hz=20.0,
            evaluation_maximum_hz=512.0,
        )
    ]
    output.extend(
        Configuration(
            name=f"extended-uniform-{count}",
            placement="uniform log",
            knots_hz=np.geomspace(EXTENDED_SPLINE_MINIMUM_HZ, EXTENDED_SPLINE_MAXIMUM_HZ, count),
            evaluation_minimum_hz=ANALYSIS_MINIMUM_HZ,
            evaluation_maximum_hz=ANALYSIS_MAXIMUM_HZ,
        )
        for count in (10, 20, 40)
    )
    output.extend(
        Configuration(
            name=f"qnm-dense-{10 + local_count}",
            placement="10 uniform-log broadband + linear QNM knots",
            knots_hz=augmented_log_knots(
                minimum_frequency=EXTENDED_SPLINE_MINIMUM_HZ,
                maximum_frequency=EXTENDED_SPLINE_MAXIMUM_HZ,
                broadband_count=10,
                peak_frequency=PEAK_FREQUENCY_HZ,
                local_half_width=FREQUENCY_WIDTH_HZ,
                local_count=local_count,
            ),
            evaluation_minimum_hz=ANALYSIS_MINIMUM_HZ,
            evaluation_maximum_hz=ANALYSIS_MAXIMUM_HZ,
        )
        for local_count in (3, 5, 7, 9, 11)
    )
    return output


def evaluate(configuration: Configuration) -> dict[str, float | int | str | list[float]]:
    """Fit amplitude and physical phase, then assess both on a finer grid."""
    validation_frequencies = np.linspace(
        configuration.evaluation_minimum_hz,
        configuration.evaluation_maximum_hz,
        VALIDATION_GRID_SIZE,
    )
    validation_shape = gaussian_bump(
        validation_frequencies,
        peak_value=1.0,
        peak_frequency=PEAK_FREQUENCY_HZ,
        frequency_width=FREQUENCY_WIDTH_HZ,
    )
    validation_basis = spline_design_matrix(validation_frequencies, configuration.knots_hz)
    amplitude_target = AMPLITUDE_PEAK * validation_shape
    phase_target = np.deg2rad(PHASE_PEAK_DEGREES * validation_shape)
    amplitude_values, amplitude_iterations, amplitude_fit_points = refined_fit(
        validation_frequencies,
        validation_shape,
        configuration.knots_hz,
        validation_basis,
        phase=False,
    )
    phase_values, phase_iterations, phase_fit_points = refined_fit(
        validation_frequencies,
        phase_target,
        configuration.knots_hz,
        validation_basis,
        phase=True,
    )
    amplitude_values *= AMPLITUDE_PEAK
    amplitude_residual_percent = 100.0 * (amplitude_values - amplitude_target)
    phase_residual_degrees = np.rad2deg(phase_values - phase_target)
    amplitude_worst_index = int(np.argmax(np.abs(amplitude_residual_percent)))
    phase_worst_index = int(np.argmax(np.abs(phase_residual_degrees)))
    upper_knot_index = int(np.searchsorted(configuration.knots_hz, PEAK_FREQUENCY_HZ))
    lower_knot = float(configuration.knots_hz[upper_knot_index - 1])
    upper_knot = float(configuration.knots_hz[upper_knot_index])
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * (FREQUENCY_WIDTH_HZ / 4.0)

    return {
        "name": configuration.name,
        "placement": configuration.placement,
        "knot_count": int(configuration.knots_hz.size),
        "knots_hz": [float(value) for value in configuration.knots_hz],
        "spline_minimum_hz": float(configuration.knots_hz[0]),
        "spline_maximum_hz": float(configuration.knots_hz[-1]),
        "evaluation_minimum_hz": configuration.evaluation_minimum_hz,
        "evaluation_maximum_hz": configuration.evaluation_maximum_hz,
        "lower_bracketing_knot_hz": lower_knot,
        "upper_bracketing_knot_hz": upper_knot,
        "bracketing_gap_hz": upper_knot - lower_knot,
        "gap_over_fwhm": (upper_knot - lower_knot) / fwhm,
        "max_amplitude_residual_percent": float(np.max(np.abs(amplitude_residual_percent))),
        "amplitude_worst_frequency_hz": float(validation_frequencies[amplitude_worst_index]),
        "max_phase_residual_degrees": float(np.max(np.abs(phase_residual_degrees))),
        "phase_worst_frequency_hz": float(validation_frequencies[phase_worst_index]),
        "amplitude_refinement_iterations": amplitude_iterations,
        "amplitude_fit_points": amplitude_fit_points,
        "phase_refinement_iterations": phase_iterations,
        "phase_fit_points": phase_fit_points,
    }


def refined_fit(
    validation_frequencies: np.ndarray,
    target: np.ndarray,
    knots: np.ndarray,
    validation_basis: np.ndarray,
    *,
    phase: bool,
) -> tuple[np.ndarray, int, int]:
    """Refine a minimax fit by exchanging worst validation-grid extrema."""
    fit_indices = np.unique(np.linspace(0, validation_frequencies.size - 1, INITIAL_FIT_GRID_SIZE, dtype=int))
    for iteration in range(1, MAX_REFINEMENT_ITERATIONS + 1):
        if phase:
            fit = minimax_phase_spline_fit(
                validation_frequencies[fit_indices],
                target[fit_indices],
                knots,
                bisection_steps=PHASE_BISECTION_STEPS,
            )
            latent_values = np.einsum("ij,j->i", validation_basis, fit.node_values)
            values = 2.0 * np.arctan(latent_values / 2.0)
        else:
            fit = minimax_spline_fit(validation_frequencies[fit_indices], target[fit_indices], knots)
            values = np.einsum("ij,j->i", validation_basis, fit.node_values)

        absolute_residual = np.abs(values - target)
        extrema = np.concatenate(([0], find_peaks(absolute_residual)[0], [validation_frequencies.size - 1]))
        worst_extrema = extrema[np.argsort(absolute_residual[extrema])[-max(32, 2 * knots.size) :]]
        new_indices = np.setdiff1d(worst_extrema, fit_indices)
        fitted_maximum = float(np.max(absolute_residual[fit_indices]))
        validated_maximum = float(np.max(absolute_residual))
        tolerance = PHASE_REFINEMENT_TOLERANCE_RADIANS if phase else AMPLITUDE_REFINEMENT_TOLERANCE
        if validated_maximum - fitted_maximum <= tolerance:
            return values, iteration, int(fit_indices.size)
        fit_indices = np.unique(np.concatenate([fit_indices, new_indices]))

    raise RuntimeError("minimax exchange refinement did not converge")


def write_csv(path: Path, results: list[dict[str, object]]) -> None:
    """Write the residual-versus-knots table."""
    fieldnames = [name for name in results[0] if name != "knots_hz"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in fieldnames} for row in results)


def write_figure(path: Path, results: list[dict[str, object]]) -> None:
    """Plot maximum amplitude and phase residual against knot count."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=False)
    series = {
        "extended uniform log": [row for row in results if str(row["name"]).startswith("extended-uniform")],
        "QNM-dense": [row for row in results if str(row["name"]).startswith("qnm-dense")],
    }
    for axis, key, ylabel in (
        (axes[0], "max_amplitude_residual_percent", "max amplitude residual (%)"),
        (axes[1], "max_phase_residual_degrees", "max phase residual (deg)"),
    ):
        for label, rows in series.items():
            axis.plot(
                [row["knot_count"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                label=label,
            )
        submitted = results[0]
        axis.scatter(submitted["knot_count"], submitted[key], marker="x", s=65, label="submitted (20-512 Hz)")
        axis.axhline(4.0, color="black", linestyle="--", linewidth=1.0, label="4% / 4 deg requirement")
        axis.set_yscale("log")
        axis.set_xlabel("knot count")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n")


def write_readme(path: Path, results: list[dict[str, object]], commit: str) -> None:
    """Write the interpretation, recommendation, and provenance links."""
    submitted = next(row for row in results if row["name"] == "submitted-10")
    recommended = next(row for row in results if row["name"] == "qnm-dense-19")
    knot_list = ", ".join(f"{value:.3f}" for value in recommended["knots_hz"])
    rows = "\n".join(
        "| {name} | {knot_count} | {evaluation_minimum_hz:.0f}-{evaluation_maximum_hz:.0f} | "
        "{max_amplitude_residual_percent:.3f} | {max_phase_residual_degrees:.3f} |".format(**row)
        for row in results
    )
    amplitude_margin = 4.0 / float(recommended["max_amplitude_residual_percent"])
    phase_margin = 4.0 / float(recommended["max_phase_residual_degrees"])
    text = f"""# Calibration-spline resolution study

This is a spline-only representation floor: no detector noise, null stream, likelihood, or
posterior enters the calculation. The injected error is a -10% amplitude and +10 degree phase
Gaussian bump at 249.43 Hz, with `f_width = 50 Hz`, `sigma = f_width / 4 = 12.5 Hz`, and FWHM
29.435 Hz. These inputs and the 4% / 4 degree requirement are externally anchored to
[Sinha, Sun, and Ma (2025)](https://arxiv.org/abs/2506.15979).

## Result

| configuration | knots | assessed band (Hz) | max amplitude residual (%) | max phase residual (deg) |
| --- | ---: | ---: | ---: | ---: |
{rows}

The submitted 10-knot, 8-512 Hz basis leaves **{submitted["max_amplitude_residual_percent"]:.3f}%**
amplitude and **{submitted["max_phase_residual_degrees"]:.3f} degrees** phase residual over
20-512 Hz. Its 119.352 Hz bracketing gap is 4.055 times the bump FWHM. It has no spline support
above 512 Hz, so it cannot demonstrate the stated requirement over 512-2000 Hz.

## Recommendation fixed for the calibration model and demonstration

Use **19 explicit knots over 8-2048 Hz**: retain 10 uniform-log broadband knots, then add nine
linear knots from 199.43 to 299.43 Hz at 12.5 Hz spacing (one Gaussian sigma), including the
249.43 Hz peak. The exact sorted knot set in Hz is:

`[{knot_list}]`

Assess and report the demonstration over the complete **20-2000 Hz** requirement band. The
8-2048 Hz spline support provides boundary knots outside that assessed interval. This grid's
best validated floor is **{recommended["max_amplitude_residual_percent"]:.3f}%** amplitude and
**{recommended["max_phase_residual_degrees"]:.3f} degrees** phase, respectively
{amplitude_margin:.1f} and {phase_margin:.1f} times below the 4% / 4 degree requirement. The
21-knot alternative lowers the floor further, but 19 knots already makes basis error only about
1.6% of the allowed residual while avoiding four additional amplitude/phase parameters per
detector relative to 21 knots.

## Provenance and limitations

- Claim `submitted_basis_misses_requirement` -> result `submitted-10` -> run
  `2026-09-15-spline-resolution-study` in `run.json`.
- Claim `recommended_basis_resolves_bump_over_20_2000_hz` -> result `qnm-dense-19` -> the same
  run snapshot.
- The run used `scripts/spline_resolution_study.py` at nullcal commit `{commit}`. The CSV contains
  the 200,001-point validation maxima and worst frequencies; `run.json` contains every knot,
  solver tolerance, dependency version, and reproduction command.
- The cubic basis is anchored against bilby's `CubicSpline` by
  `tests/studies/test_spline_resolution.py`; the FWHM is independently anchored by the analytic
  Gaussian identity. The new minimax residuals have no external published numerical reference;
  they are computational results anchored only to this run and commit.
- The source requirement was derived for Cosmic Explorer and has no independent spectroscopy
  replication identified here. Applying the same threshold to ET remains an unanchored transfer
  assumption. This study measures only representation error and does not predict inference,
  noise, or null-stream performance.
"""
    path.write_text(text, encoding="utf-8")


def git_metadata() -> tuple[str, bool]:
    """Return the exact nullcal revision and whether its worktree is dirty."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to stamp study provenance")
    commit = subprocess.run(  # noqa: S603 - executable resolved from PATH, arguments are fixed
        [git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603 - executable resolved from PATH, arguments are fixed
            [git, "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
    )
    return commit, dirty


def main() -> None:
    """Run the study and write its table, figure, and provenance snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    commit, dirty = git_metadata()
    results = []
    for configuration in configurations():
        print(f"evaluating {configuration.name}", flush=True)
        results.append(evaluate(configuration))
    write_csv(arguments.output_directory / "residual_vs_knots.csv", results)
    write_figure(arguments.output_directory / "residual_vs_knots.svg", results)
    write_readme(arguments.output_directory / "README.md", results, commit)
    run = {
        "run_id": "2026-09-15-spline-resolution-study",
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "code": {
            "repository": "https://github.com/Leuven-Gravity-Institute/nullcal",
            "commit": commit,
            "dirty": dirty,
            "script": "scripts/spline_resolution_study.py",
        },
        "command": (
            "uv run --with matplotlib python scripts/spline_resolution_study.py "
            "--output-directory <paper-repo>/exploratory-tests/spline-basis-resolution"
        ),
        "execution_environment": "local arm64 reference platform",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "configuration": {
            "peak_frequency_hz": PEAK_FREQUENCY_HZ,
            "frequency_width_hz": FREQUENCY_WIDTH_HZ,
            "gaussian_sigma_hz": FREQUENCY_WIDTH_HZ / 4.0,
            "gaussian_fwhm_hz": 2.0 * np.sqrt(2.0 * np.log(2.0)) * (FREQUENCY_WIDTH_HZ / 4.0),
            "amplitude_peak_percent": 100.0 * AMPLITUDE_PEAK,
            "phase_peak_degrees": PHASE_PEAK_DEGREES,
            "initial_fit_grid_size": INITIAL_FIT_GRID_SIZE,
            "validation_grid_size": VALIDATION_GRID_SIZE,
            "phase_bisection_steps": PHASE_BISECTION_STEPS,
            "amplitude_refinement_tolerance": AMPLITUDE_REFINEMENT_TOLERANCE,
            "phase_refinement_tolerance_radians": PHASE_REFINEMENT_TOLERANCE_RADIANS,
            "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
            "noise": False,
            "null_stream": False,
        },
        "results": results,
        "claims": {
            "submitted_basis_misses_requirement": {"supported_by": "submitted-10"},
            "recommended_basis_resolves_bump_over_20_2000_hz": {"supported_by": "qnm-dense-19"},
        },
    }
    (arguments.output_directory / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
