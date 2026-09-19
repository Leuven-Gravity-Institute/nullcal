# Likelihood and reference-protocol diagnosis

This bounded investigation follows a staged decision tree.  It stops at the
first failed branch: the historical and current posterior densities agree, but
the frozen bilby/dynesty protocol under-covers a correlated Gaussian with a
known analytic posterior.  No harder calibration run was attempted.

## Step 1: posterior-density agreement — pass

`scripts/diagnose_likelihood_agreement.py`, run from commit
`ef9a8541b2a8bbc7a9a525a97d46c5aa4bc8f12b`, compared the historical bilby
implementation at `a02bd921f6c7834aa6e9d61227643cc32bcc4a98` with the pure-JAX
implementation based on `3e5bf7a0cd8919fe160a4503ab3d89917e54770b`.

The canonical vector is detector-major: ET1, ET2, ET3; within each detector,
amplitude knots 0..9 followed by phase knots 0..9.  The historical evaluator
assigns those values directly to bilby's
`recalib_<detector>_<kind>_<knot>` keys.  The JAX evaluator reshapes the same
vector to `(detector, kind, knot)` and selects the amplitude and phase planes.
An intentional one-coordinate cyclic permutation changes the log posterior by
6.641330974624225 nats, so the comparison would detect an ordering error.

Both paths use normalized, independent, unbounded Gaussian priors with mean 0
and standard deviation 0.05.  Bilby labels the infinite-boundary prior as
reflective and maps unit-cube coordinates through the Gaussian quantile; the
transform round trip differs by at most 2.6757770803763348e-17.  R5 consumes
the resulting physical coordinates directly.  There is no constant prior or
likelihood offset: differences fluctuate around floating-point zero rather
than sharing a nonzero constant.

The table reports R5 minus historical bilby at identical points:

| Point | log likelihood | log prior | log posterior |
| --- | ---: | ---: | ---: |
| MAP | 2.842170943040401e-14 | 0 | 2.842170943040401e-14 |
| prior draw 0 | 0 | -1.4210854715202004e-14 | -2.842170943040401e-14 |
| prior draw 1 | 0 | -2.842170943040401e-14 | -5.684341886080802e-14 |
| prior draw 2 | -5.684341886080802e-14 | -2.842170943040401e-14 | -8.526512829121202e-14 |
| single coordinate at +5 sigma | 5.684341886080802e-14 | 2.842170943040401e-14 | 8.526512829121202e-14 |
| alternating coordinates at +/-3 sigma | 5.684341886080802e-14 | -1.1368683772161603e-13 | -5.684341886080802e-14 |
| ramp from -3 to +3 sigma | -2.2737367544323206e-13 | -5.684341886080802e-14 | -2.2737367544323206e-13 |
| ordering probe | 0 | -1.4210854715202004e-14 | 0 |

The maximum absolute differences are 2.2737367544323206e-13 for the
likelihood, 1.1368683772161603e-13 for the prior, and
2.2737367544323206e-13 for the posterior, all below the predeclared 1e-9
tolerance.  The MAP optimization took 35 iterations and ended with gradient
infinity norm 4.3538719819102845e-05.  The complete values are in
`likelihood_agreement.json`.

## Step 2: known correlated Gaussian — fail

`scripts/diagnose_dynesty_correlated_gaussian.py` fixed the problem, protocol,
and validation bands in signed commit
`3a9d1c31bc45fe460c9ed20e0e6a529e77951545` before sampling.  The test is a
60-dimensional dense Gaussian likelihood with a deterministic orthogonal
basis and likelihood eigenwidths from 0.02 to 0.20, under independent
`N(0, 0.05^2)` priors.  Gaussian precision addition gives its posterior mean,
covariance, marginal widths, and eigenmode widths analytically; those analytic
quantities are independent of the sampler under test.

The sampler was bilby 2.8.2 with dynesty 3.1.0, static `rslice`, `nlive=1000`,
`slices=10`, `dlogz=0.1`, eight workers, and seed 20260918.  Dynesty returned
6,170 equal-weight draws; the predeclared seed 20260920 selected exactly 5,000
without replacement.  Marginal ratios had to lie in [0.90, 1.10], while
analytic projected eigenmode and sorted covariance-eigenvalue width ratios had
to lie in [0.85, 1.15].

All three checks fail:

| Width ratio | Minimum | Median | Maximum | Below band | In band | Above band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| per-coordinate marginal | 0.816507723261325 | 0.8872855208248145 | 0.9537822619641384 | 37 | 23 | 0 |
| variance projected on analytic eigenmodes | 0.7889179642709937 | 0.879859338615419 | 0.9666884367651004 | 7 | 53 | 0 |
| sorted covariance-eigenvalue widths | 0.650887438970961 | 0.7665222259182963 | 1.1517092493179217 | 40 | 19 | 1 |

The dominant effect is systematic under-coverage, not a symmetric scatter
around the analytic answer.  The retained posterior's canonical array digest
is `9feba6f296891e0c10d841bef4f5de42c863dbe82727d29a5a1221afee9432fb`.
The compressed sample file digest is
`38f9609537cbc3c4f63edf913b26b0f126b6ef6b95c3ce8712e9069b2f9a886f`;
the manifest digest is
`2235e9a33b1be875c30a84fddc140176b04ad4579725d937eeda03287347ca33`.

This is the first failed decision-tree branch.  The harder-converged bilby run
is therefore deliberately absent.

## BlackJAX curvature post-check

The post-failure gate analysis in `scripts/analyze_blackjax_curvature.py`, run
from `d872ec2d6ac3b7896c8488d21fefb91603812a74`, uses the fixed BlackJAX
posterior produced from R5 commit
`3e5bf7a0cd8919fe160a4503ab3d89917e54770b`.  It compares the sample covariance
with the inverse Hessian of the same agreed posterior density:

| Width ratio | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| per-parameter marginal | 0.9689002159388045 | 0.9990824755447066 | 1.0266553538576064 |
| variance projected on Hessian eigenmodes | 0.9715329544283706 | 0.9988591323998669 | 1.0401944716967855 |
| sorted covariance-eigenvalue widths | 0.9196451965930268 | 1.0001056329469722 | 1.0811321754620509 |

The chain diagnostics are max split R-hat 1.0039834916500208, minimum bulk ESS
6909.782172601377, minimum tail ESS 3931.484899476682, and zero divergences.

## Proposed replacement gate

The failed Gaussian calibration shows that a KS gate against a posterior made
by this bilby/dynesty protocol is not a valid correctness oracle.  Replace it,
subject to maintainer approval, with the conjunction below:

1. Keep the immutable fixed-value numerical anchors.  At the committed MAP,
   seeded prior, and tail probes, require historical and current normalized
   log likelihood, log prior, and log posterior to agree within absolute 1e-9,
   with the permutation-sensitive probe required to separate by more than 1
   nat.  Continue the existing array-level fixed-reference checks at their
   tighter committed tolerances.
2. Keep the predeclared BlackJAX chain gates unchanged: split R-hat at most
   1.01, bulk and tail ESS at least 400, and zero divergences.
3. Require every per-parameter BlackJAX-to-Laplace marginal-width ratio to lie
   in [0.90, 1.10].  Require every Hessian-eigenmode projected-width ratio and
   every sorted covariance-eigenvalue width ratio to lie in [0.85, 1.15].
   Require the MAP Hessian to be positive definite.

The current BlackJAX artifact passes all proposed numerical bands above.  This
proposal is not applied by this branch.  Curvature is a local Gaussian anchor;
it does not independently validate nonlinear posterior tails or multimodality,
and those quantities remain unanchored here.  The analytic correlated-Gaussian
experiment anchors the protocol diagnosis, but no independent exact posterior
is available for the nonlinear calibration problem itself.
