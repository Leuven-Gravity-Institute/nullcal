# End-to-end characterisation harness

This directory holds immutable inputs and outputs produced before the JAX and
BlackJAX migration. They are external anchors: agreement between two live
implementations would not establish that either is correct.

## Frozen inputs

- `reference/inputs.npz` contains the whitened strain and PSD.
- `reference/artifacts.npz` contains fixed-parameter intermediate values and
  log likelihoods from the pre-migration implementation.
- `reference/posterior_samples.npz` contains the final bilby posterior: 5,937
  draws over 60 calibration parameters.

The bilby posterior is read-only. Its generator was removed with bilby, and no
test or script in the current tree can overwrite it. Its manifest records the
producing revision and SHA-256 digest.

`pipeline.py` builds the current likelihood directly from the frozen arrays;
it does not generate a waveform. The archived time-frequency filter is fed
back as input and is therefore not claimed as a newly reproduced output.

## Acceptance

`test_reference_artifacts.py` checks the fixed-parameter path against the
pre-migration artifacts at a peak-scaled relative tolerance of `1e-12`. This is the primary
numerical anchor.

`test_blackjax_posterior.py` checks the separately generated BlackJAX chains
against the frozen bilby posterior using thresholds fixed before the run:

- maximum per-parameter two-sample KS statistic: `0.10`;
- maximum split R-hat: `1.01`;
- minimum bulk and tail ESS: `400`;
- divergences: `0`.

Run the harness explicitly:

```console
uv run pytest tests/e2e -m e2e
```

The `e2e` and `slow` markers are deselected by default.

## Producing the BlackJAX acceptance artifact

`scripts/run_reference_blackjax.py` consumes the immutable inputs and bilby
posterior, runs four independent BlackJAX NUTS chains, and writes only the
distinct `blackjax_posterior_*` files. It refuses to run from dirty source or a
dirty harness so every reported number names the exact producing commit.

The original reference generators remain historical utilities for the
fixed-value arrays, but must not be run as part of this migration. In
particular, there is no generator for the frozen bilby posterior.
