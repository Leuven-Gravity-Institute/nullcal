# End-to-end characterisation harness

Frozen reference artifacts pinning the numerical behaviour of the numpy/numba
pipeline, plus the tests that check the current code against them.

## Why this exists

These artifacts are the external anchor for a planned rewrite of the likelihood
onto JAX and BlackJAX. Agreement between a new implementation and the old one
bounds the accuracy of neither — both could be wrong in the same way. Agreement
with an artifact produced _before_ the rewrite began, at a tolerance stated in
advance, is what makes a regression visible.

## Layout

| Path                                         | Role                                                                                                                           |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `config.py`                                  | The frozen configuration. Single source of truth, imported by both the generator and the tests so they cannot drift.           |
| `pipeline.py`                                | Builds the interferometers, waveform generator and likelihood, and computes every frozen quantity. One construction path only. |
| `generate_reference.py`                      | Writes `reference/artifacts.npz` and `reference/manifest.json`.                                                                |
| `test_reference_artifacts.py`                | Compares current output against the frozen artifacts.                                                                          |
| `test_noise_log_likelihood_filter_domain.py` | Regression test for the defect below, now fixed.                                                                               |
| `reference/posterior_samples.npz`            | A bilby posterior over the calibration parameters — **provisional, not the anchor** (below).                                   |
| `test_reference_posterior.py`                | Integrity and informativeness checks on that posterior.                                                                        |

## Running

The `e2e` marker is deselected by default, so a normal `pytest` run does not pay
for these:

```bash
uv run pytest                       # unit + integration-excluded suite, no e2e
uv run pytest tests/e2e -m e2e      # the characterisation tests
```

## Regenerating the reference

Regeneration is an explicit action, never a side effect of running the tests —
artifacts exist to detect change, so a run that silently rewrites them detects
nothing.

```bash
uv run python -m tests.e2e.generate_reference
```

**Regenerate only with a reason, and commit the new artifacts in the same commit
as the code change that justified them, with that reason in the message.** A
reference regenerated to make a failing test pass is not a reference.

`manifest.json` records the git revision, Python and platform, the versions of
`nullcal`, `bilby`, `numpy`, `scipy`, `numba`, `lalsuite` and `rocket-fft`, the
scalar configuration and source parameters, and a SHA-256 per array.
`test_manifest_matches_artifacts` checks the digests against the shipped `.npz`,
so a hand-edited artifact fails rather than passing quietly, and
`test_manifest_configuration_matches_the_live_config` checks the recorded
configuration against `config.py` field by field, so a stale manifest fails too.

It is **not** a complete description of the inputs: `DETECTOR_NAMES`, the
derived segment `start_time()` and the wavelet probe's own seed (`SEED + 1`)
live only in `config.py`. Reproducing from the manifest alone is therefore not
possible — reproduce from `config.py` at the recorded revision. Recording those
fields is folded into the follow-up that freezes the strain and PSD as inputs,
because that change regenerates the manifest anyway; doing it here would rewrite
the recorded provenance for a documentation-only gain.

The provenance fields are asserted _present_, never compared against the running
environment. They say where the artifacts were generated, which is deliberately
not where the tests later run; requiring them to match `HEAD` would force a
regeneration on every commit, which is exactly what the policy above forbids.

## Tolerances, and why they are what they are

`max|diff| / max|reference| <= 1e-12` — a **peak-scaled** budget, asserted on
exactly the quantity the failure message reports.

The configuration is seeded end to end, so a rerun of the _same_ implementation
may differ only by floating-point non-determinism in threaded reductions and in
the FFT. A loose engineering tolerance would hide a real change.

**Why peak-scaled and not per-element.** A per-element relative budget
(`np.allclose(rtol=1e-12, atol=0.0)`) is 0.0 wherever the reference element is
exactly zero, and the calibrated time-frequency array is ~97.7% exact zeros — so
that rule silently demands bit-exactness outside the filter. The smallest
non-zero elements are ~1e-4, giving per-element budgets of ~1e-16, at or below
the round-off of the wavelet and FFT sums; a per-element rule would therefore
_not_ admit the reduction-order non-determinism stated above, contradicting its
own justification. Peak-scaling admits round-off while still catching a 1e-12
relative change in the array as a whole.

**On `atol`.** An earlier version of this file justified `atol=0.0` by claiming
the compared quantities are of order 1e-24, so that numpy's default `atol=1e-8`
would make any comparison vacuously true. That was wrong and is recorded here
because the conclusion survived the reasoning. The frozen arrays are not of
order 1e-24: the null streams peak at ~1.7, `log_likelihood` is ~-204, and
`whitened_antenna_response` peaks at ~1.3e24 — they span some 24 orders between
them, so no single statement about `atol` covers all of them. For the O(1) bulk
a 1e-8 floor would not swamp anything; it would loosen the effective tolerance
to ~1e-8 relative, costing about four orders of sensitivity. At the smallest
wavelet-tail pixels (~1e-4) it would swamp them outright. Either way it is the
wrong floor, but not for the originally stated reason. The ~1e-24 quantity is
the raw unwhitened strain, which is not among the frozen artifacts; a default
`atol` _would_ be vacuous there, and that is the case this argument was borrowed
from.

The tolerance for the future JAX implementation is a separate decision and will
be larger. It must be stated with its basis _before_ the port is compared, not
chosen after seeing a diff.

## The defect this harness found, and its fix (R17)

`RecalibrationLikelihood.noise_log_likelihood()` **raised `IndexError`** on the
revision the original reference was taken from. R17 fixed it, and the repaired
quantities are now frozen artifacts: `noise_log_likelihood` and
`uncalibrated_time_frequency_domain_null_stream`.

`NullStream.compute_uncalibrated_time_frequency_domain_null_stream` applied
`[:, ~time_frequency_filter] = 0.0` to
`uncalibrated_frequency_domain_null_stream` — the _frequency-domain_ array,
shape `(detector, frequency)` — using the _2-D_ time-frequency filter, shape
`(n_t, n_f)`. A slice plus a 2-D boolean mask addresses three dimensions of a
2-D array:

```text
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

Two consequences, in order:

1. The method could not return, so `noise_log_likelihood()` and anything
   reaching it failed — including bilby's `log_likelihood_ratio`.
2. Had it not raised, the returned time-frequency array would still be
   unfiltered, so the uncalibrated branch would sum residual energy over every
   pixel while the calibrated branch sums over the filter only, normalising the
   two against different domains.

The calibrated path is unaffected and `log_likelihood()` is correct, so
published results that never called `noise_log_likelihood()` are not implicated.

**The fix** applies the filter to the time-frequency array the method returns,
matching what the calibrated path does in its `_from_parameters` wrapper. With
it, `noise_log_likelihood` = -235.791023711555 against `log_likelihood` =
-203.88870383371767, so the log Bayes factor is +31.90 — the correct sign, since
the uncalibrated stream still carries the injected ~2% calibration error and
must therefore hold _more_ residual energy. The energy ratio
uncalibrated/calibrated is 1.156, which matches the 1.16 an independent reviewer
measured by simulating the repaired branch before the fix existed.

Structural checks alone would not have pinned that: they constrain the shape of
the result (confined to the filter, finite, sharing the calibrated branch's
support) but not its values, so the two repaired quantities are frozen as
artifacts too.

Historic note: `test_noise_log_likelihood_filter_domain.py` asserts the fixed
behaviour and **failed on the unfixed code** — verified, not assumed. Its three
tests carried a module-level `xfail(strict=True, raises=IndexError)` until R17.
They did not flip to passing by themselves: `strict=True` makes an unexpected
pass an _error_, which is deliberate — it forced the fixing commit to delete the
markers rather than leave stale ones behind. Re-verified when the fix landed:
reverting the one-line change fails these three tests with the original
`IndexError`.

`raises=IndexError` matters as much as `strict`. Without it a strict xfail
accepts any failure as the expected one, so an assertion failing for an
unrelated reason would still report `XFAIL` and hide real breakage.

## The reference posterior — PROVISIONAL, not the anchor

> **Status (owner's decision, 2026-08-11): this posterior is provisional and is
> _not_ the frozen distributional anchor for the rebuild.** It is kept as a
> worked example and as a regression guard on the generator, not as something
> the JAX port will be checked against. It does not block the migration: the
> posterior is regenerable from the pre-migration commit, so what has to be
> preserved is the _ability_ to regenerate — this revision plus
> `generate_reference_posterior.py` — rather than these particular samples.
>
> Why it is not yet the anchor: the marginal widths (median
> `sigma_post/sigma_prior` 0.720) are explained by measured degeneracy, but the
> _conditional_ widths recovered from the samples, 0.292 median, still disagree
> with the 0.41 predicted by a sampler-independent per-parameter scan — a ~40%
> gap on exactly the quantity that scan predicts. One covariance direction also
> comes out broader than the prior (2.08x the prior variance), which
> finite-sample noise does not explain. Anchoring means closing those two, not
> re-running the sampler. Until then, do not cite these numbers as the reference
> distribution.

`reference/posterior_samples.npz` (5937 samples x 60 calibration parameters) and
`reference/posterior_manifest.json` hold a bilby posterior over the calibration
parameters. The intent was that it become the _distributional_ anchor, with R5
testing the ported sampler for statistical consistency against it — that is
**not** its status; see the note above. R5's acceptance test is therefore not
yet defined against these samples, and defining it waits on the number being
anchored.

An earlier version of this file said the artifact **cannot be regenerated
later**, because bilby produces it and the rewrite removes bilby. That is wrong
as stated: it can be regenerated from any pre-migration revision, so what has to
be preserved is that revision together with `generate_reference_posterior.py`,
not these particular draws. The urgency it implied — generate before bilby goes,
or lose the chance — does not hold.

Only the calibration spline parameters are sampled — 60 free, plus 30
`DeltaFunction` node frequencies. `RecalibrationLikelihood.log_likelihood` reads
nothing else; the source parameters enter once, at construction, through the
injection-clustering filter.

Produced with dynesty `sample="rslice", slices=10, nlive=1000, dlogz=0.1` on
Linux x86-64 in 12 minutes on 32 cores.

**`rwalk` must not be used here, and the reason is recorded because it cost a
day.** The first production run used dynesty's default random-walk proposal. It
reported convergence at `dlogz=0.1`, wrote a complete manifest, and returned the
**prior**: median `sigma_post/sigma_prior` of 0.999 across all 60 parameters,
100% of injected values inside their 90% intervals, medians at the prior mean.
`rwalk` had never mixed — efficiency fell to 0.0%, calls per iteration pinned at
the 5001 ceiling, and dynesty warned `Hit maximum number` (its
autocorrelation-not-met signal) into stderr. Nothing in the output said the
result was worthless. A posterior equal to its prior is reproduced equally well
by a correct implementation and a broken one, so as an anchor it would pass
anything at all.

A sampler-independent check settled that the fault was the sampler and not the
data: varying one parameter at a time by one prior sigma gives a median
`delta log L` of 2.51 nats. The likelihood is informative. Slice sampling then
reached the expected widths in 12 minutes, using 8.4e5 likelihood calls against
`rwalk`'s 5.7e7.

The frozen `posterior_manifest.json` also lists `walks` and `nact`, which are
inert leftovers: the run was launched from the then-`rwalk` defaults with
`--sample rslice` on the command line, and the recorded settings are the merged
result. The generator now defaults to `rslice` and no longer sets them, so a
regeneration produces a manifest without those two keys. The settings that
matter — sampler, sample, slices, nlive, dlogz — agree.

`test_reference_posterior.py` therefore asserts the shrinkage directly, and pins
`sample="rslice"`, so a regeneration that silently reverted would fail.

**What the coverage number does and does not show.** 58 of 60 injected values
lie inside their 90% credible intervals. That is _not_ a calibration test: the
injection was drawn from N(0, 0.02) while the prior is N(0, 0.05), so truths
come from a narrower distribution than the prior and land inside intervals more
often than nominal. Real calibration would need many injections drawn _from the
prior_ and a P-P plot.

**Marginals are wider than conditionals, by design of the problem.** Conditional
`sigma/sigma_prior` is 0.292 median against a marginal 0.720, and 26 of 60
covariance directions are constrained below 0.5 prior sigma. The likelihood
constrains _combinations_ of spline nodes rather than individual nodes, so
per-parameter marginals understate how much the data says. One covariance
direction comes out broader than the prior (2.08x the prior variance), which
finite-sample noise does not explain and which remains unaccounted for.
