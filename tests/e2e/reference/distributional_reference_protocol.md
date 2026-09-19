# Additional pre-migration distributional reference protocol

This protocol was committed before the reference run.  It defines an additional
artifact and does not replace or modify `posterior_samples.npz`, the original
5,937-by-60 frozen posterior.

## Conditioning and parameter order

The likelihood is constructed by
`tests.e2e.pipeline.build_likelihood_from_reference_inputs`, conditioned on
`inputs.npz` (file SHA-256
`f55e7a56d0ce6abab5ec15dff4322eb4c2f15be62dab086aa724a6a499a79982`)
and the archived time-frequency filter in `artifacts.npz` (file SHA-256
`b5cdd469e7746ef9097de1ba73f0cd518e9a00d71627bdfd77839b2ed6a20977`).
This is the same frozen conditioning consumed by the BlackJAX path.

The 60 sampled parameters use bilby's `search_parameter_keys` order: ET1,
ET2, then ET3; within each detector, amplitude knots 0 through 9 followed by
phase knots 0 through 9.  All independent Gaussian priors have mean zero and
standard deviation 0.05.

## Sampler and retained draws

- Source base: pre-migration `a02bd921f6c7834aa6e9d61227643cc32bcc4a98`.
- Locked sampler versions: bilby 2.8.2 and dynesty 3.1.0.
- Sampler: dynesty static nested sampling through bilby, `nlive=1000`,
  `sample="rslice"`, `slices=10`, and `dlogz=0.1`.
- Sampler seed: 20260918, applied with `bilby.core.utils.random.seed`
  immediately before `run_sampler`.
- Parallelism: `npool=8`.
- Retained sample count: exactly 5,000.  If bilby returns fewer than 5,000
  equal-weight posterior draws, generation fails.  Otherwise 5,000 are selected
  uniformly without replacement using NumPy seed 20260920.
- Warmup and Markov-chain thinning: not applicable to static nested sampling;
  no chain thinning is performed.

## Fixed acceptance decision

The already-produced four-chain BlackJAX posterior is compared to this new
reference with a two-sample KS statistic for each of the 60 parameters.  The
distributional gate passes only if every statistic is at most 0.10.  The
pre-existing BlackJAX requirements remain split R-hat at most 1.01, bulk and
tail ESS at least 400, and zero divergences.

The new reference must also be supported by the likelihood: its median
marginal standard deviation divided by the Laplace/Hessian marginal standard
deviation must be close to one and clearly separated from the original frozen
posterior's recorded ratio of 0.8048.  If this sanity check fails, generation
stops rather than tuning or repeating the run.
