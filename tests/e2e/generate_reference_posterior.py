"""Generate the frozen bilby reference posterior for the R1 characterisation harness.

This is the last R1 artifact and the only one with a deadline: bilby generates it, and R5
removes bilby, after which it cannot be produced again at any price. It exists so that the
BlackJAX rebuild has a *distributional* anchor to be checked against, not only the
fixed-parameter evaluations already frozen in ``reference/artifacts.npz``.

Design decisions, all fixed here rather than discovered during the run:

*Sampled parameters.* Only the calibration spline parameters. ``RecalibrationLikelihood.
log_likelihood`` reads nothing else -- the source parameters enter once, at construction, through
the injection-clustering time-frequency filter -- so sampling them would add cost and no anchor.
That is 60 free dimensions (3 detectors x 10 amplitude + 10 phase) plus 30 DeltaFunction node
frequencies.

*Prior.* Gaussian, sigma = 0.05 in amplitude and 0.05 rad in phase, against an injection drawn
from sigma = 0.02. Wider than the injection on purpose: a prior tighter than the data's
constraining power would hand back the prior as the posterior, and a prior-dominated posterior is
reproduced equally well by a correct implementation and a broken one -- it would be an anchor that
cannot fail.

An earlier version of this docstring justified that with "the injection sits 154.6 nats above the
median prior draw, so the likelihood, not the prior, sets the posterior". **That test is close to
meaningless here and the claim was wrong.** In 60 dimensions a random prior draw is far from the
peak by concentration of measure alone, whatever the marginals do -- and the first production run
duly returned the prior. The check that actually bears on it is per-parameter: move one parameter by
one prior sigma with the rest fixed and measure delta log L. Median 2.51 nats on this configuration,
which is informative. It costs half a second; run it before spending a cluster job, not after.

*Sampler.* ``rslice``, not ``rwalk``. This is not a preference. With ``rwalk`` this script produced
a posterior identical to its prior (median sigma_post/sigma_prior 0.999 over all 60 parameters)
while reporting convergence at dlogz=0.1 and writing a complete manifest: the proposal never mixed,
at efficiency 0.0% with calls per iteration pinned at its ceiling and dynesty's "Hit maximum number"
autocorrelation warning buried in stderr. Slice sampling is the standard remedy for random-walk
proposals failing in tens of dimensions, and here it reached the expected widths in 12 minutes using
8.4e5 likelihood calls against rwalk's 5.7e7. The manifest records an ``informativeness`` block so a
future run that regresses says so in its own artifact.

*bilby's noise-evidence call.* ``bilby.run_sampler`` calls ``likelihood.noise_log_likelihood()``
unconditionally once sampling completes, on both branches of its ``use_ratio`` test. On this
revision that raises ``IndexError`` (the R1 defect, fixed later in R17), which would destroy a
finished run at the point where all the cost has already been paid. The subclass below returns NaN
instead, so ``log_noise_evidence`` and ``log_bayes_factor`` come out NaN and are recorded as
unavailable-at-this-revision. ``log_likelihood`` is inherited untouched, so the posterior samples
are produced by exactly the code under characterisation.

Usage::

    python generate_reference_posterior.py --outdir <dir> [--smoke] [--npool N]

``--smoke`` runs a deliberately cheap configuration to validate the script and measure the cost
scaling; it writes its manifest with ``"smoke": true`` so a smoke artifact can never be mistaken
for the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np

sys.path.insert(0, "tests")
sys.path.insert(0, ".")

import bilby

from nullcal.likelihood import RecalibrationLikelihood
from nullcal.prior import CalibrationPriorDict
from tests.e2e import config, pipeline

#: Prior widths. Fixed here; changing either invalidates the artifact.
AMPLITUDE_SIGMA = 0.05
PHASE_SIGMA = 0.05

#: Sampler settings for the reference run. Stated in advance, recorded in the manifest.
REFERENCE_SAMPLER_KWARGS = {
    "sampler": "dynesty",
    "nlive": 1000,
    # rslice, NOT rwalk. rwalk returned the prior on this 60-dimensional problem while reporting
    # convergence; see the module docstring. Changing this back would silently reproduce that.
    "sample": "rslice",
    "slices": 10,
    "dlogz": 0.1,
    # Checkpoint so a walltime kill costs the last half hour rather than the whole run.
    "check_point": True,
    "check_point_delta_t": 1800,
    "check_point_plot": False,
    "resume": True,
    # The default per-iteration progress line produces hundreds of megabytes of log over a long
    # run; this keeps it readable without hiding progress.
    "print_method": "interval-60",
}

#: Cheap settings for --smoke. Not a reference; used to measure scaling and shake out the script.
SMOKE_SAMPLER_KWARGS = {
    "sampler": "dynesty",
    "nlive": 50,
    "sample": "rslice",
    "slices": 5,
    "dlogz": 5.0,
    "print_method": "interval-60",
}

#: Seeds the sampler. Distinct from config.SEED, which seeds the data, so the two cannot be
#: confused when a difference has to be attributed to one or the other.
SAMPLER_SEED = 20260810


class ReferenceLikelihood(RecalibrationLikelihood):
    """``RecalibrationLikelihood`` with bilby's post-run noise-evidence call neutralised.

    ``log_likelihood`` is inherited unchanged -- this subclass exists only so that the R1 defect
    in ``noise_log_likelihood`` cannot destroy a completed sampler run. See the module docstring.
    """

    def noise_log_likelihood(self) -> float:
        """Return NaN rather than raising ``IndexError``.

        The real method is broken on this revision and is pinned by its own regression tests. NaN
        propagates into ``log_noise_evidence`` and ``log_bayes_factor``, which the manifest marks
        as unavailable; it does not touch the samples.
        """
        return float("nan")


def build_prior() -> CalibrationPriorDict:
    """Calibration spline prior over the three ET detectors.

    bilby's ``constant_uncertainty_spline`` prepends ``recalib_`` to the label, so the label is
    the bare detector name. Getting this wrong is not loud: a wrongly prefixed key still passes
    bilby's substring test in ``set_calibration_parameters``, is sliced to a junk key, and leaves
    the model's previous values in place -- the likelihood then silently stops responding to the
    parameters. The assertion below is what makes that failure visible.
    """
    prior = CalibrationPriorDict()
    for name in config.DETECTOR_NAMES:
        prior.update(
            CalibrationPriorDict.constant_uncertainty_spline(
                amplitude_sigma=AMPLITUDE_SIGMA,
                phase_sigma=PHASE_SIGMA,
                minimum_frequency=config.MINIMUM_FREQUENCY,
                maximum_frequency=config.MAXIMUM_FREQUENCY,
                n_nodes=config.N_POINTS,
                label=name,
            )
        )
    expected = {
        f"recalib_{name}_{kind}_{index}"
        for name in config.DETECTOR_NAMES
        for kind in ("amplitude", "phase")
        for index in range(config.N_POINTS)
    }
    missing = expected - set(prior)
    if missing:
        raise AssertionError(f"prior keys do not match the likelihood's prefix; missing {sorted(missing)}")
    return prior


def build_likelihood() -> ReferenceLikelihood:
    """The reference likelihood, built through the frozen e2e construction path."""
    interferometers = pipeline.build_interferometers()
    waveform_generator = pipeline.build_waveform_generator()

    directory = Path("scratchpad") / "clustering"
    directory.mkdir(parents=True, exist_ok=True)
    parameter_file = directory / "clustering_parameters.csv"
    import pandas as pd

    pd.DataFrame([config.SOURCE_PARAMETERS]).to_csv(parameter_file, index=False)

    return ReferenceLikelihood(
        interferometers=interferometers,
        waveform_generator=waveform_generator,
        wavelet_transform_frequency_resolution=config.FREQUENCY_RESOLUTION,
        wavelet_transform_nx=config.NX,
        clustering_parameter_file=str(parameter_file),
        clustering_threshold=config.CLUSTERING_THRESHOLD,
    )


def check_responds_to_parameters(likelihood: ReferenceLikelihood, prior: CalibrationPriorDict) -> None:
    """Refuse to launch if the likelihood does not move with the parameters.

    A frozen likelihood produces a posterior that is exactly the prior, at full cost, and looks
    entirely healthy in the output files. Cheaper to check here than to discover afterwards.
    """
    first = {k: float(np.asarray(v).item()) for k, v in prior.sample().items()}
    second = {k: float(np.asarray(v).item()) for k, v in prior.sample().items()}
    likelihood.parameters = first
    first_value = float(likelihood.log_likelihood())
    likelihood.parameters = second
    second_value = float(likelihood.log_likelihood())
    if first_value == second_value:
        raise AssertionError(
            f"log_likelihood identical at two distinct prior draws ({first_value!r}); "
            "the calibration parameters are not reaching the calibration model."
        )


def git_revision() -> str:
    """Current revision of the checkout that produced the artifact."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def sha256_of_array(array: np.ndarray) -> str:
    """Digest of an array's bytes, in a canonical layout."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="Output directory for the run.")
    parser.add_argument("--smoke", action="store_true", help="Cheap configuration; not a reference.")
    parser.add_argument("--npool", type=int, default=1, help="Number of likelihood-evaluation processes.")
    parser.add_argument("--sample", help="Override dynesty's sampling method, e.g. rslice.")
    parser.add_argument("--nlive", type=int, help="Override nlive.")
    parser.add_argument("--dlogz", type=float, help="Override the dlogz stopping criterion.")
    parser.add_argument("--slices", type=int, help="Slices per iteration, for slice-based samplers.")
    parser.add_argument("--maxmcmc", type=int, help="Cap on MCMC steps per proposal, for rwalk.")
    parser.add_argument("--label", help="Override the output label, so a diagnostic cannot overwrite a reference.")
    arguments = parser.parse_args()

    outdir = Path(arguments.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sampler_kwargs = dict(SMOKE_SAMPLER_KWARGS if arguments.smoke else REFERENCE_SAMPLER_KWARGS)
    label = "smoke_posterior" if arguments.smoke else "reference_posterior"

    # Overrides exist because the first production run (job 96907/96939) converged on dlogz while
    # returning the *prior*: dynesty's rwalk hit its MCMC-step ceiling without meeting the
    # autocorrelation requirement, so live points barely moved. A sampler-independent scan showed
    # the likelihood is informative (median delta log L = 2.5 nats at 1 prior sigma, implying
    # sigma_post/sigma_prior ~ 0.45 against the 0.999 observed). Any override is recorded in the
    # manifest, and --label keeps a diagnostic from overwriting a reference.
    overrides = {
        key: getattr(arguments, key)
        for key in ("sample", "nlive", "dlogz", "slices", "maxmcmc")
        if getattr(arguments, key) is not None
    }
    sampler_kwargs.update(overrides)
    if arguments.label:
        label = arguments.label
    if overrides:
        print(f"sampler overrides: {overrides}")

    likelihood = build_likelihood()
    prior = build_prior()
    check_responds_to_parameters(likelihood, prior)

    # Confirm the fixed-parameter anchor still holds in this environment before sampling.
    likelihood.parameters = dict(config.calibration_parameters())
    log_likelihood_at_injection = float(likelihood.log_likelihood())

    injection_parameters = {key: value for key, value in config.calibration_parameters().items() if key in prior}

    # Seed the sampler LAST. bilby ignores run_sampler's `seed` kwarg when it builds `rstate`
    # (it warns "ignoring 'seed'") and derives the state from this global generator instead --
    # and `pipeline.build_interferometers()` reseeds that same generator with the *data* seed
    # while constructing the likelihood above. Seeding before construction therefore leaves the
    # sampler running off config.SEED, silently, with SAMPLER_SEED having no effect at all.
    # Everything that consumes randomness before this line (the prior draws in the guard) must
    # stay before it for the sampler stream to be reproducible.
    bilby.core.utils.random.seed(SAMPLER_SEED)

    start = time.perf_counter()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=prior,
        outdir=str(outdir),
        label=label,
        injection_parameters=injection_parameters,
        npool=arguments.npool,
        save="hdf5",
        plot=False,
        use_ratio=False,
        **sampler_kwargs,
    )
    wall_seconds = time.perf_counter() - start

    free_parameters = list(result.search_parameter_keys)
    posterior = result.posterior[free_parameters].to_numpy()

    # Did the sampler actually learn anything? Recorded in the manifest so the artifact carries its
    # own verdict. Job 96907/96939 converged on dlogz and returned marginals indistinguishable from
    # the prior (median sigma_post/sigma_prior = 0.999) because rwalk never mixed; nothing in the
    # output said so, and it took a separate analysis to notice. A posterior equal to its prior is
    # reproduced equally well by a correct implementation and a broken one, so as an anchor for the
    # BlackJAX port it is worse than useless — it would pass anything.
    prior_sigma = np.array([PHASE_SIGMA if "_phase_" in name else AMPLITUDE_SIGMA for name in free_parameters])
    posterior_sigma = posterior.std(axis=0)
    shrinkage = posterior_sigma / prior_sigma
    truth = np.array([injection_parameters[name] for name in free_parameters])
    lower, upper = np.percentile(posterior, [5, 95], axis=0)
    covered = int(np.count_nonzero((truth >= lower) & (truth <= upper)))
    informativeness = {
        "sigma_post_over_sigma_prior_median": float(np.median(shrinkage)),
        "sigma_post_over_sigma_prior_min": float(shrinkage.min()),
        "n_parameters_above_0.9": int(np.count_nonzero(shrinkage > 0.9)),
        "n_parameters_below_0.5": int(np.count_nonzero(shrinkage < 0.5)),
        "injected_value_in_90pc_credible_interval": covered,
        "n_parameters": len(free_parameters),
        "interpretation": (
            "Median shrinkage near 1.0 means the posterior is the prior and the artifact is NOT a "
            "discriminating anchor, regardless of whether the sampler reported convergence. A "
            "sampler-independent scan of this configuration predicts ~0.45."
        ),
    }

    manifest = {
        "smoke": bool(arguments.smoke),
        "artifact": "bilby reference posterior over the calibration spline parameters",
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("bilby", "dynesty", "numpy", "scipy", "numba", "lalsuite", "rocket-fft", "nullcal")
        },
        "configuration": {
            "duration": config.DURATION,
            "sampling_frequency": config.SAMPLING_FREQUENCY,
            "minimum_frequency": config.MINIMUM_FREQUENCY,
            "maximum_frequency": config.MAXIMUM_FREQUENCY,
            "n_points": config.N_POINTS,
            "frequency_resolution": config.FREQUENCY_RESOLUTION,
            "nx": config.NX,
            "clustering_threshold": config.CLUSTERING_THRESHOLD,
            "data_seed": config.SEED,
            "source_parameters": config.SOURCE_PARAMETERS,
        },
        "prior": {
            "family": "Gaussian (bilby CalibrationPriorDict.constant_uncertainty_spline)",
            "amplitude_sigma": AMPLITUDE_SIGMA,
            "phase_sigma": PHASE_SIGMA,
            "n_sampled_dimensions": len(free_parameters),
            "injected_sigma_for_comparison": 0.02,
        },
        "sampler": {
            **sampler_kwargs,
            "sampler_seed": SAMPLER_SEED,
            "seeding_mechanism": (
                "bilby.core.utils.random.seed() called immediately before run_sampler; "
                "run_sampler's own `seed` kwarg is ignored by bilby in favour of rstate"
            ),
            "npool": arguments.npool,
            "use_ratio": False,
        },
        "results": {
            "log_evidence": float(result.log_evidence),
            "log_evidence_err": float(result.log_evidence_err),
            "n_posterior_samples": int(posterior.shape[0]),
            "sampling_time_seconds": float(result.sampling_time),
            "wall_seconds": round(wall_seconds, 1),
            "log_likelihood_at_injection": log_likelihood_at_injection,
            "frozen_reference_log_likelihood": -203.88870383371767,
        },
        "informativeness": informativeness,
        "unavailable_at_this_revision": {
            "log_noise_evidence": (
                "NaN by construction: noise_log_likelihood() raises IndexError on this revision "
                "(the R1 defect, fixed in R17), and bilby calls it unconditionally after sampling. "
                "log_bayes_factor is NaN for the same reason. Neither is part of the anchor."
            )
        },
        "parameters": free_parameters,
        "posterior_sha256": sha256_of_array(posterior),
        "posterior_shape": list(posterior.shape),
    }

    np.savez_compressed(outdir / f"{label}_samples.npz", posterior=posterior, parameters=np.array(free_parameters))
    (outdir / f"{label}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest["results"] | {"smoke": manifest["smoke"]}, indent=2))


if __name__ == "__main__":
    main()
