# End-to-end characterisation harness

This directory holds immutable inputs and outputs produced before the JAX and
BlackJAX migration, plus the committed evidence for the migration acceptance
gate. Agreement between two live implementations alone would not establish
that either is correct.

## Frozen inputs

- `reference/inputs.npz` contains the whitened strain and PSD.
- `reference/artifacts.npz` contains fixed-parameter intermediate values and
  log likelihoods from the pre-migration implementation.
- `reference/posterior_samples.npz` contains the final bilby posterior: 5,937
  draws over 60 calibration parameters.

The bilby posterior is read-only. Its generator was removed with bilby, and no
test or script in the current tree can overwrite it. Its manifest records the
producing revision and SHA-256 digest. It is retained as a historical
diagnostic, not as an acceptance reference.

`pipeline.py` builds the current likelihood directly from the frozen arrays;
it does not generate a waveform. The archived time-frequency filter is fed
back as input and is therefore not claimed as a newly reproduced output.

## Acceptance

`test_reference_artifacts.py` checks the fixed-parameter path against the
pre-migration artifacts at a peak-scaled relative tolerance of `1e-12`. This is the primary
numerical anchor.

`test_blackjax_posterior.py` applies three independently checkable gates:

- at committed MAP, prior and tail probes, current and historical log
  likelihood, log prior and log posterior agree to absolute error `1e-9`; an
  intentionally permuted probe must differ by more than `1` nat;
- maximum split R-hat is `1.01`, minimum bulk and tail ESS are `400`, and
  divergences are `0`;
- every BlackJAX/Laplace marginal width ratio is in `[0.90, 1.10]`, every
  projected and sorted covariance-eigenvalue width ratio is in `[0.85, 1.15]`,
  and the MAP Hessian is positive definite.

The historical dynesty posterior was removed from acceptance after an exact
correlated-Gaussian diagnosis found marginal width ratios with minimum
`0.8165`, median `0.8873`, and maximum `0.9538`; 37 of 60 marginals fell below
the acceptance band. The producing diagnostic commit is
`3a9d1c31bc45fe460c9ed20e0e6a529e77951545`. The legacy KS threshold remains
`0.10` in the manifest solely as provenance; it is not changed or enforced.

The curvature gate is a local Gaussian consistency check. It does not anchor
nonlinear posterior tails or rule out separated modes; those quantities remain
unanchored by an external reference.

Run the harness explicitly:

```console
uv run pytest tests/e2e -m e2e
```

The `e2e` and `slow` markers are deselected by default.

## Producing the BlackJAX acceptance artifact

`scripts/run_reference_blackjax.py` consumes the immutable likelihood inputs,
runs four independent BlackJAX NUTS chains, and writes only the distinct
`blackjax_posterior_*` files. It reads the frozen posterior after sampling only
to reproduce the explicitly non-acceptance legacy KS diagnostics and their
provenance. It refuses to run from dirty source or a dirty harness so every
reported number names the exact producing commit. The parameter order is
derived directly from the fixed detector, quantity and knot layout and checked
against the historical order.

`scripts/diagnose_likelihood_agreement.py` regenerates the fixed-point density
comparison from a historical tree pinned to the commit declared in that script
and an independent historical Python environment. Its committed output is
`diagnostics/likelihood_agreement.json`. The output's `producer_commit` names
the clean source revision used to generate it; reproduce the artifact by
checking out that revision and running the script with the same pinned
historical environment.

The original reference generators remain historical utilities for the
fixed-value arrays, but must not be run as part of this migration. In
particular, there is no generator for the frozen bilby posterior.
