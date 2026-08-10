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
| `test_noise_log_likelihood_filter_domain.py` | Regression test for a known defect (below).                                                                                    |

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

It is **not** a complete description of the inputs: `DETECTOR_NAMES`, the derived
segment `start_time()` and the wavelet probe's own seed (`SEED + 1`) live only in
`config.py`. Reproducing from the manifest alone is therefore not possible —
reproduce from `config.py` at the recorded revision. Recording those fields is
folded into the follow-up that freezes the strain and PSD as inputs, because that
change regenerates the manifest anyway; doing it here would rewrite the recorded
provenance for a documentation-only gain.

The provenance fields are asserted *present*, never compared against the running
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
because the conclusion survived the reasoning: the arrays compared are
_whitened_ — the null streams peak at ~1.7 and `log_likelihood` is ~-204 — so a
1e-8 absolute floor would not swamp them, it would loosen the effective
tolerance to ~1e-8 relative, costing about four orders of sensitivity. The
~1e-24 quantity is the raw unwhitened strain, which is not among the frozen
artifacts. A default `atol` _would_ be vacuous on unwhitened strain; that
argument just does not apply here.

The tolerance for the future JAX implementation is a separate decision and will
be larger. It must be stated with its basis _before_ the port is compared, not
chosen after seeing a diff.

## Known defect pinned here

`RecalibrationLikelihood.noise_log_likelihood()` **raises `IndexError` on the
revision this reference was taken from**, so no reference value for it exists.

`NullStream.compute_uncalibrated_time_frequency_domain_null_stream` applies
`[:, ~time_frequency_filter] = 0.0` to
`uncalibrated_frequency_domain_null_stream` — the _frequency-domain_ array,
shape `(detector, frequency)` — using the _2-D_ time-frequency filter, shape
`(n_t, n_f)`. A slice plus a 2-D boolean mask addresses three dimensions of a
2-D array:

```text
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

Two consequences, in order:

1. The method cannot return, so `noise_log_likelihood()` and anything reaching
   it fails — including bilby's `log_likelihood_ratio`.
2. Had it not raised, the returned time-frequency array would still be
   unfiltered, so the uncalibrated branch would sum residual energy over every
   pixel while the calibrated branch sums over the filter only, normalising the
   two against different domains.

The calibrated path is unaffected and `log_likelihood()` is correct, so
published results that never called `noise_log_likelihood()` are not implicated.

`test_noise_log_likelihood_filter_domain.py` asserts the fixed behaviour and
**fails on the unfixed code** — verified, not assumed. Its three tests carry a
module-level `xfail(strict=True, raises=IndexError)`, so on this revision they
report `XFAIL`. They do **not** flip to passing by themselves: `strict=True`
turns an unexpected pass into an _error_, which is deliberate — it forces the
fixing commit to delete the marker rather than leave a stale one behind. So the
repair, a separate change with its own review, consists of fixing the defect
**and** removing that marker in the same commit; the tests then pass normally.

`raises=IndexError` matters as much as `strict`. Without it a strict xfail
accepts any failure as the expected one, so an assertion failing for an
unrelated reason would still report `XFAIL` and hide real breakage.
