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
full configuration, and a SHA-256 per array. `test_manifest_matches_artifacts`
checks the digests against the shipped `.npz`, so a hand-edited artifact fails
rather than passing quietly.

## Tolerances, and why they are what they are

`rtol=1e-12`, **`atol=0.0`**.

The configuration is seeded end to end, so a rerun of the _same_ implementation
may differ only by floating-point non-determinism in threaded reductions and in
the FFT. A loose engineering tolerance would hide a real change.

`atol=0.0` is not optional. Whitened strain quantities here are of order 1e-24;
numpy's default `atol=1e-8` exceeds them by sixteen orders of magnitude, which
would make `allclose` return `True` for any two such arrays and the assertion
incapable of failing.

Differences are reported **peak-relative** (`max|diff| / max|reference|`), never
per-sample relative: these arrays cross zero, where a per-sample relative error
is meaningless.

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
turns an unexpected pass into an *error*, which is deliberate — it forces the
fixing commit to delete the marker rather than leave a stale one behind. So the
repair, a separate change with its own review, consists of fixing the defect
**and** removing that marker in the same commit; the tests then pass normally.

`raises=IndexError` matters as much as `strict`. Without it a strict xfail
accepts any failure as the expected one, so an assertion failing for an
unrelated reason would still report `XFAIL` and hide real breakage.
