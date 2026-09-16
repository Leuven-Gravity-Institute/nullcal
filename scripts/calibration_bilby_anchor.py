"""Print the R13 calibration-factor and gradient anchor table."""

from __future__ import annotations

import argparse

import bilby
import jax
import jax.numpy as jnp
import numpy as np
from bilby.gw.detector.calibration import CubicSpline

from nullcal.calibration import calibration_factor

ANCHOR_PEAK_RELATIVE_TOLERANCE = 1e-11
FINITE_DIFFERENCE_STEP = 1e-6
GRADIENT_RTOL = 5e-7
GRADIENT_ATOL = 5e-9

CONFIGURATIONS = (
    ("compact", 8.0, 512.0, 4),
    ("requirement", 20.0, 2000.0, 7),
    ("broadband", 8.0, 2048.0, 10),
    ("dense", 8.0, 2048.0, 19),
)
NODE_SCALES = (0.0, 0.01, 0.2)


def bilby_factor(frequencies, knots, amplitude, phase):
    """Evaluate the installed bilby model for one uniform-log configuration."""
    model = CubicSpline(
        prefix="recalib_",
        minimum_frequency=knots[0],
        maximum_frequency=knots[-1],
        n_points=knots.size,
    )
    parameters = {f"recalib_amplitude_{index}": value for index, value in enumerate(amplitude)}
    parameters.update({f"recalib_phase_{index}": value for index, value in enumerate(phase)})
    return model.get_calibration_factor(frequencies, **parameters)


def factor_rows():
    """Yield bilby/JAX comparison rows over placements, counts, and values."""
    for placement, minimum, maximum, count in CONFIGURATIONS:
        knots = np.geomspace(minimum, maximum, count)
        frequencies = np.geomspace(minimum, maximum, 257)
        coordinate = np.linspace(-1.0, 1.0, count)
        for node_scale in NODE_SCALES:
            amplitude = node_scale * (coordinate**3 - 0.2 * coordinate)
            phase = node_scale * (coordinate**2 - 0.4)
            expected = bilby_factor(frequencies, knots, amplitude, phase)
            actual = np.asarray(calibration_factor(frequencies, knots, amplitude, phase))
            deviation = float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))
            if deviation >= ANCHOR_PEAK_RELATIVE_TOLERANCE:
                raise AssertionError(f"bilby anchor failed for {placement}/{count}/{node_scale}: {deviation}")
            yield placement, minimum, maximum, count, node_scale, deviation


def gradient_rows():
    """Yield automatic-versus-centred-finite-difference gradient metrics."""
    frequencies = jnp.geomspace(8.0, 2048.0, 73)
    knots = jnp.geomspace(8.0, 2048.0, 7)
    amplitude = jnp.linspace(-0.04, 0.03, knots.size)
    phase = jnp.linspace(0.05, -0.02, knots.size)
    weights = jnp.linspace(0.5, 1.5, frequencies.size) + 1j * jnp.linspace(-0.7, 0.4, frequencies.size)

    def objective(node_amplitude, node_phase):
        return jnp.real(jnp.vdot(weights, calibration_factor(frequencies, knots, node_amplitude, node_phase)))

    automatic = jax.grad(objective, argnums=(0, 1))(amplitude, phase)
    for argument_index, (name, values) in enumerate((("amplitude", amplitude), ("phase", phase))):
        finite_difference = np.empty(values.size)
        values_np = np.asarray(values)
        for index in range(values.size):
            step = np.zeros(values.size)
            step[index] = FINITE_DIFFERENCE_STEP
            above = [amplitude, phase]
            below = [amplitude, phase]
            above[argument_index] = values_np + step
            below[argument_index] = values_np - step
            finite_difference[index] = (float(objective(*above)) - float(objective(*below))) / (
                2.0 * FINITE_DIFFERENCE_STEP
            )
        automatic_values = np.asarray(automatic[argument_index])
        absolute = float(np.max(np.abs(automatic_values - finite_difference)))
        peak_relative = absolute / float(np.max(np.abs(finite_difference)))
        if not np.allclose(
            automatic_values,
            finite_difference,
            rtol=GRADIENT_RTOL,
            atol=GRADIENT_ATOL,
        ):
            raise AssertionError(f"finite-difference gradient anchor failed for {name}")
        yield name, absolute, peak_relative


def main():
    """Run the anchors and print a self-contained Markdown report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="commit that produced the measurements")
    commit = parser.parse_args().commit
    rows = list(factor_rows())
    gradients = list(gradient_rows())
    print("# Bilby calibration-model anchor")
    print()
    print(f"Producing commit: `{commit}`")
    print(f"Versions: bilby {bilby.__version__}, JAX {jax.__version__}, NumPy {np.__version__}.")
    print()
    print(
        "The factor tolerance was fixed before comparison at peak-relative "
        f"`{ANCHOR_PEAK_RELATIVE_TOLERANCE:.0e}`. Both paths solve the same at-most-19-dimensional "
        "float64 system; the bound is over 100 times `n^2 * eps` at `n=19`, allowing different "
        "LAPACK/XLA reductions while remaining negligible on the physical scale."
    )
    print()
    print("| Placement | Band (Hz) | Knots | Node scale | Peak-relative deviation |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for placement, minimum, maximum, count, node_scale, deviation in rows:
        print(f"| {placement} | {minimum:g}-{maximum:g} | {count} | {node_scale:g} | {deviation:.17g} |")
    print()
    print(
        f"Worst factor deviation: `{max(row[-1] for row in rows):.17g}` "
        f"(tolerance `{ANCHOR_PEAK_RELATIVE_TOLERANCE:.0e}`)."
    )
    print()
    print(
        "The gradient comparison used centred differences with step "
        f"`{FINITE_DIFFERENCE_STEP:.0e}` and fixed tolerances `rtol={GRADIENT_RTOL:.0e}`, "
        f"`atol={GRADIENT_ATOL:.0e}`. The basis is the O(h^2) truncation and O(eps/h) round-off "
        "of a centred float64 difference."
    )
    print()
    print("| Gradient | Max absolute deviation | Peak-relative deviation |")
    print("| --- | ---: | ---: |")
    for name, absolute, peak_relative in gradients:
        print(f"| {name} | {absolute:.17g} | {peak_relative:.17g} |")


if __name__ == "__main__":
    main()
