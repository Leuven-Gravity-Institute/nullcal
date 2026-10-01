# D1 prerequisite audit

No D1 posterior has been generated or examined. This record reports an input
availability check, not the requested scientific measurement. Both inference
runners, the spectroscopy comparison and the inspiral-only experiment remain
unimplemented and unexecuted in this change.

## Producing run

The checker ran from clean commit `6bbd3a4fe43610ae66b336a2b83fb13f7f9b8445`:

```sh
uv run python scripts/d1_control_preflight.py \
  --inputs docs/dev/d1-control/inputs \
  --output /tmp/401-preflight.json
```

It returned exit **2**, `status: blocked`, and `scientific_results: null`.
`prerequisite-audit.json` preserves the output. The input directory is the
expected bundle location, not evidence that any upstream producer has already
exported that bundle. The checker cannot distinguish an absent artifact from one
available elsewhere; it does not establish that the data cannot be acquired.

The registration was originally committed at
`5242cdd4740c82eb0d879f3b361c18d983440928`; formatting-only changes are in the
producing commit. The registration file hash used by the audit is
`1291a8a49cf8f430580c1de2a2c81481ce86a6f8db59a23ec71a2c85ecc39f20`. No result
inspection occurred between those commits.

Environment: Python 3.13.12, Darwin arm64; NumPy 2.5.3, SciPy 1.18.1, JAX
0.11.2, BlackJAX 1.6.2, pytest 9.1.1 and Ruff 0.16.9. This command is a small
filesystem audit, not an experiment or accelerator run.

## Missing inputs

The audit lists `waveform.h5`, `waveform-manifest.json`, `ringdown-anchor.json`,
`gr-imr-validation.json`, `injection-prescription.json` and
`resolution-precheck.json` as missing or empty. The benchmark comparison, exact
deformation construction, source-prior prescription and feasibility pre-check
must be produced or located before the registered scientific cell can run. No
substitute posterior or synthetic measurement is supplied.

## What this shows and does not show

The committed protocol establishes prospective absorption criteria and the audit
establishes that its prerequisite bundle was absent at the checked location. It
supplies no absorption fraction or interval, no measured knot posterior shift,
no nullcal posterior and no inspiral-sufficiency result; every one of those
quantities is unmeasured and unanchored. The geometric argument `P(cF) = P(F)`
for a nonzero common factor c is an analytic identifiability constraint on the
proposed comparison, not an experimental validation of nullcal's calibration
posterior. This preparation therefore supports neither success nor failure of
conventional calibration and cannot decide a journal venue.

## Software verification

With numerical-library thread caps set to two, `uv run pytest -q` returned
`274 passed, 6 skipped, 40 deselected, 6 xfailed` with coverage 71.49%. The
seven prerequisite-checker tests use explicitly synthetic file fixtures; they do
not run either scientific fit. `uv run ruff check .` passed;
`uv run ruff format --check .` passed. Applicable targeted pre-commit checks
passed after the initial formatter normalized the registration files.
