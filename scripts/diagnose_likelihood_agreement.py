"""Compare the pre-migration and pure-JAX posterior densities at fixed points.

The canonical vector order is detector-major (ET1, ET2, ET3), with amplitude
knots 0..9 followed by phase knots 0..9 for each detector.  The script runs the
historical evaluator in its own pinned environment, so the two implementations
never share imported package code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

R5_BASE = "3e5bf7a0cd8919fe160a4503ab3d89917e54770b"
PREMIGRATION_BASE = "a02bd921f6c7834aa6e9d61227643cc32bcc4a98"
DETECTORS = ("ET1", "ET2", "ET3")
KINDS = ("amplitude", "phase")
KNOTS = 10
DIMENSIONS = len(DETECTORS) * len(KINDS) * KNOTS
PRIOR_SIGMA = 0.05
AGREEMENT_ATOL = 1.0e-9
POINT_SEED = 20260921


def canonical_names() -> list[str]:
    """Return the shared detector-major parameter order."""
    return [f"recalib_{detector}_{kind}_{knot}" for detector in DETECTORS for kind in KINDS for knot in range(KNOTS)]


def vector_to_arrays(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Map the canonical vector to R5's quantity-keyed detector-by-knot arrays."""
    values = dict(zip(canonical_names(), np.asarray(vector), strict=True))
    return {
        kind: np.asarray(
            [[values[f"recalib_{detector}_{kind}_{knot}"] for knot in range(KNOTS)] for detector in DETECTORS]
        )
        for kind in KINDS
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()  # noqa: S603, S607


def evaluate_r5(points: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Evaluate R5 and return point values plus the permutation-sensitive probe."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from scipy.optimize import minimize  # noqa: PLC0415

    from tests.e2e import pipeline  # noqa: PLC0415

    likelihood = pipeline.build_likelihood_from_reference_inputs()

    def objective(vector):
        arrays = vector_to_arrays(vector)
        return -likelihood.logdensity_fn({key: jnp.asarray(value) for key, value in arrays.items()})

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(vector):
        value, gradient = value_and_grad(jnp.asarray(vector))
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        scipy_objective,
        np.zeros(DIMENSIONS),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1.0e-13, "gtol": 1.0e-8, "maxiter": 1_000, "maxls": 50},
    )
    if not result.success:
        raise RuntimeError(f"MAP optimization failed: {result.message}")
    points["map"] = np.asarray(result.x)

    evaluated = {}
    for name, vector in points.items():
        arrays = {key: jnp.asarray(value) for key, value in vector_to_arrays(vector).items()}
        log_likelihood = float(likelihood.log_likelihood_fn(arrays))
        log_density = float(likelihood.logdensity_fn(arrays))
        evaluated[name] = {
            "log_likelihood": log_likelihood,
            "log_prior": log_density - log_likelihood,
            "log_posterior": log_density,
        }

    ordering_probe = np.linspace(-0.12, 0.12, DIMENSIONS)
    correct = float(
        likelihood.logdensity_fn({key: jnp.asarray(value) for key, value in vector_to_arrays(ordering_probe).items()})
    )
    permuted = float(
        likelihood.logdensity_fn(
            {key: jnp.asarray(value) for key, value in vector_to_arrays(np.roll(ordering_probe, 1)).items()}
        )
    )
    mapping_check = {
        "correct_log_posterior": correct,
        "cyclically_permuted_log_posterior": permuted,
        "absolute_separation": abs(correct - permuted),
    }
    map_metadata = {
        "iterations": int(result.nit),
        "gradient_infinity_norm": float(np.max(np.abs(result.jac))),
        "negative_log_posterior": float(result.fun),
    }
    return evaluated, mapping_check | {"map": map_metadata}


def build_fixed_points() -> dict[str, np.ndarray]:
    """Create prior and tail points before either implementation is evaluated."""
    rng = np.random.default_rng(POINT_SEED)
    points = {f"prior_{index}": rng.normal(0.0, PRIOR_SIGMA, DIMENSIONS) for index in range(3)}
    points.update(
        {
            "tail_single_positive_5sigma": np.eye(1, DIMENSIONS, 0).ravel() * (5.0 * PRIOR_SIGMA),
            "tail_alternating_3sigma": np.where(np.arange(DIMENSIONS) % 2 == 0, 3.0, -3.0) * PRIOR_SIGMA,
            "tail_ramp_3sigma": np.linspace(-3.0, 3.0, DIMENSIONS) * PRIOR_SIGMA,
            "ordering_probe": np.linspace(-0.12, 0.12, DIMENSIONS),
        }
    )
    return points


def premigration_worker(root: Path, points_path: Path, output_path: Path) -> None:
    """Evaluate the historical bilby likelihood in the pinned historical tree."""
    root = root.resolve()
    os_paths = [str(root), str(root / "src")]
    for path in reversed(os_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    from bilby.gw.prior import CalibrationPriorDict  # noqa: PLC0415
    from scipy.special import ndtr  # noqa: PLC0415

    from tests.e2e import config, pipeline  # noqa: PLC0415

    if _git(root, "rev-parse", "HEAD") != PREMIGRATION_BASE:
        raise RuntimeError("pre-migration worktree is not pinned to the declared commit")

    likelihood = pipeline.build_likelihood_from_reference_inputs()
    prior = CalibrationPriorDict()
    for detector in DETECTORS:
        prior.update(
            CalibrationPriorDict.constant_uncertainty_spline(
                amplitude_sigma=PRIOR_SIGMA,
                phase_sigma=PRIOR_SIGMA,
                minimum_frequency=config.MINIMUM_FREQUENCY,
                maximum_frequency=config.MAXIMUM_FREQUENCY,
                n_nodes=KNOTS,
                label=detector,
            )
        )

    with np.load(points_path) as stored:
        points = {name: stored[name] for name in stored.files}
    names = canonical_names()
    base_parameters = dict(config.calibration_parameters())
    evaluated = {}
    for point_name, vector in points.items():
        parameters = base_parameters | dict(zip(names, vector, strict=True))
        likelihood.parameters = parameters
        log_likelihood = float(likelihood.log_likelihood())
        log_prior = float(sum(prior[name].ln_prob(parameters[name]) for name in names))
        evaluated[point_name] = {
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "log_posterior": log_likelihood + log_prior,
        }

    transform_probabilities = np.asarray([1.0e-6, 0.01, 0.5, 0.99, 1.0 - 1.0e-6])
    representative = prior[names[0]]
    transformed = np.asarray([representative.rescale(value) for value in transform_probabilities])
    round_trip = ndtr(transformed / PRIOR_SIGMA)
    payload = {
        "evaluated": evaluated,
        "prior": {
            "family": type(representative).__name__,
            "minimum": str(representative.minimum),
            "maximum": str(representative.maximum),
            "boundary": representative.boundary,
            "mean": float(representative.mu),
            "sigma": float(representative.sigma),
            "transform_round_trip_max_abs": float(np.max(np.abs(round_trip - transform_probabilities))),
        },
        "versions": {
            "bilby": package_version("bilby"),
            "numpy": package_version("numpy"),
            "nullcal": package_version("nullcal"),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_diagnosis(premigration_root: Path, premigration_python: Path, output_path: Path) -> None:
    """Run both isolated evaluators and write their direct comparison."""
    current_root = Path(__file__).resolve().parents[1]
    ancestor_check = subprocess.run(  # noqa: S603
        ["git", "-C", str(current_root), "merge-base", "--is-ancestor", R5_BASE, "HEAD"],  # noqa: S607
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError("diagnostic branch does not descend from the pinned R5 head")

    points = build_fixed_points()
    r5_values, mapping_check = evaluate_r5(points)
    with tempfile.TemporaryDirectory(prefix="nullcal-likelihood-diagnosis-") as directory:
        directory_path = Path(directory)
        points_path = directory_path / "points.npz"
        worker_output = directory_path / "premigration.json"
        np.savez(points_path, **points)
        subprocess.run(  # noqa: S603
            [
                str(premigration_python),
                str(Path(__file__).resolve()),
                "--worker",
                "--premigration-root",
                str(premigration_root),
                "--points",
                str(points_path),
                "--output",
                str(worker_output),
            ],
            check=True,
            cwd=premigration_root,
        )
        historical = json.loads(worker_output.read_text())

    comparisons = {}
    maxima = {"log_likelihood": 0.0, "log_prior": 0.0, "log_posterior": 0.0}
    for point_name, r5_point in r5_values.items():
        historical_point = historical["evaluated"][point_name]
        differences = {key: r5_point[key] - historical_point[key] for key in maxima}
        comparisons[point_name] = {
            "premigration": historical_point,
            "r5": r5_point,
            "r5_minus_premigration": differences,
        }
        for key, difference in differences.items():
            maxima[key] = max(maxima[key], abs(difference))

    mapping_check["passes_permutation_sensitivity"] = mapping_check["absolute_separation"] > 1.0
    passes = (
        all(value <= AGREEMENT_ATOL for value in maxima.values()) and mapping_check["passes_permutation_sensitivity"]
    )
    payload = {
        "status": "pass" if passes else "fail",
        "criterion": {
            "maximum_absolute_difference": AGREEMENT_ATOL,
            "permutation_minimum_absolute_separation": 1.0,
        },
        "provenance": {
            "diagnostic_commit": _git(current_root, "rev-parse", "HEAD"),
            "r5_base": R5_BASE,
            "premigration_base": PREMIGRATION_BASE,
        },
        "parameter_mapping": {
            "canonical_order": "ET1/ET2/ET3; within each detector amplitude 0..9 then phase 0..9",
            "r5": "canonical names are gathered into amplitude[detector,knot] and phase[detector,knot]",
            "premigration": "canonical names are assigned directly to bilby's recalib_<detector>_<kind>_<knot> keys",
        },
        "points": comparisons,
        "maximum_absolute_differences": maxima,
        "permutation_check": mapping_check,
        "premigration_prior": historical["prior"],
        "versions": historical["versions"] | {"jax": package_version("jax"), "r5_nullcal": package_version("nullcal")},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("status", "maximum_absolute_differences", "permutation_check")}, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--premigration-root", type=Path, required=True)
    parser.add_argument("--premigration-python", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.points is None:
            parser.error("--points is required in worker mode")
        premigration_worker(arguments.premigration_root, arguments.points, arguments.output)
        return
    if arguments.premigration_python is None:
        parser.error("--premigration-python is required")
    run_diagnosis(arguments.premigration_root, arguments.premigration_python, arguments.output)


if __name__ == "__main__":
    main()
