# D1 control: preregistered criteria

This protocol fixes the primary decision rule before any D1 posterior has been
examined. It is a design, not an experimental result. The machine-readable
settings in `preregistration.json` are authoritative. A producing run must name
the registration commit and its file hash. Any changed setting requires a new
version, committed before new results are examined; keep the original version.

The scientific question is whether a GR IMR joint source-and-calibration fit can
hide a deliberately non-GR ringdown in otherwise correctly calibrated ET
triangle data, whereas null-stream calibration preserves it. Failure to observe
absorption is a possible result. A missing input, failed validation, or
unconverged chain is an inconclusive experiment, not evidence of either outcome.

## Inputs and feasibility gate

Use SXS:BBH:0305, a reference 220 frequency of 249.43 Hz, and an injected
fractional 220 frequency deviation of +0.05 with no damping-time deviation.
These are design choices. The reference configuration and the calibration
benchmark come from [Sinha, Sun and Ma](https://arxiv.org/abs/2506.15979). The
benchmark is a Cosmic Explorer rational-filter result: it is not an ET or
`ringdown` posterior-width requirement.

Before sampling, archive the original waveform and extraction/version metadata;
its SHA-256; the mass, remnant spin, units, PSD, sample rate, duration, detector
geometry, sky positions, orientation and ringdown start convention. Archive the
exact mode-deformation and taper prescription, including how it leaves the
inspiral intact. Verify the injected frequency against an independent QNM
reference, not against the injection implementation itself. Do not substitute an
analytic damped sinusoid for the selected numerical-relativity IMR signal. The
injection prescription is a required input: the decision rule alone does not
make an otherwise unspecified signal construction reproducible.

Archive a comparison of `ringdown` and the rational filter on the published
SXS/CE configuration with modes {220, 221}, including both configurations,
posterior artifacts, producing commits and the measured discrepancy. Quantify
that discrepancy before invoking the published 4%/4° threshold. A later
inspiral-sufficiency claim must use the directly measured spectroscopy outcome;
posterior width alone cannot establish sufficiency.

Validate the GR IMR source model against the undeformed SXS signal at the chosen
SNR, with zero calibration and the same priors used in the main experiment. An
undeformed-signal calibration bias is a waveform-model confound, not non-GR
absorption. Independently check frequency/time resolution and the chosen spline
basis against the required deformation response. Archive convergence under
doubled waveform resolution and denser knots. Require discretization changes in
recovered frequency to be below 0.1 posterior standard deviations, and the
independent injection-frequency check to agree within 0.005 fractionally. These
are prospective tolerances, not measured accuracies. Do not sample the
registered cell until these feasibility checks pass.

## Population and arms

Use 16 independent sky/orientation draws (geometry seed 40100) in a co-located,
equilateral, long-wavelength ET triangle; isotropic sky, uniform polarization
and isotropic inclination. Scale distance to network ringdown SNR 120, recording
both ringdown and whole-signal SNR per event. The population shares one true
calibration factor, exactly unity. Repeat the entire population over eight
independent noise draws with seeds 40101 through 40108. Use a GR injection with
identical noise as the negative control. This is a controlled fiducial-waveform
population, not an astrophysical mass/redshift population.

Freeze detector strain, PSD, frequency masks, response and noise arrays once.
Each arm must read the same immutable data bundle and record its SHA-256. Use
the established 19-knot spectroscopy basis spanning 20–2000 Hz; archive the
exact frequencies in the input prescription. Each detector's knot prior has
independent zero-mean Gaussians with amplitude sigma 0.1 and latent phase sigma
10 degrees expressed in radians, with the reference cubic spline in
log-frequency and phase factor `(2 + i*p)/(2 - i*p)`. Report physical phase as
`2*atan(p/2)`. The shared calibration posterior sums event likelihoods once and
applies its prior once. Do not treat events as independent calibration posterior
samples.

1. **Identity calibration:** run `ringdown` on the uncorrected data to establish
   that the injected deviation is detectable. Infer the expected GR frequency
   from the independent inspiral source/remnant posterior; retain its
   uncertainty.
2. **Conventional joint fit:** use a GR IMR template and jointly infer source
   and detector spline calibration parameters. Archive the precise IMR
   approximant, engine versions, full source priors and sampled parameters. Do
   not fix the source to injection truth. Spline conventions must agree with the
   reference implementation. This external control must not reintroduce bilby as
   a nullcal runtime dependency or add a compatibility backend.
3. **Null-stream calibration:** sum the nullcal event log likelihoods and sample
   the shared calibration. Apply the inverse per-frequency posterior-median
   calibration factor to the frozen data, then run the same `ringdown` model.
4. **Inspiral-only calibration:** repeat the conventional fit with a recorded
   pre-merger time gate excluding the injected deformation. Propagate that
   calibration to the original ringdown data and repeat spectroscopy. A low
   frequency cut alone is not a time-domain inspiral selection.

Apply the same spectroscopy start time, modes, priors and covariance treatment
in all arms. For every correction, report both the posterior-median-factor
analysis and propagation over calibration draws with consistent PSD/covariance
transformation. Record the complete conventional source model and prior
prescription before running; missing source priors are a feasibility failure.

## Intervals and primary decision

Report 90% equal-tailed credible intervals (5th, 50th and 95th percentiles), not
standard deviations relabelled as intervals. For each noise replicate define
`A = 1 - delta_omega_after_joint_correction / delta_omega_injected`. Evaluate
this transformation draw by draw on the calibration-marginalized spectroscopy
posterior. Do not form a ratio of medians. The denominator is the fixed injected
0.05, so its uncertainty is not inferred from another posterior. Report the
posterior-median-factor result separately from the marginalized result.

An individual replicate supports absorption only if **all** of the following
hold:

- Identity-calibration spectroscopy excludes zero deviation and includes the
  injected deviation in its 90% interval.
- The lower 90% bound of A exceeds 0.5 and the joint-corrected
  frequency-deviation interval includes zero.
- At the predefined ET1 knot nearest 249.43 Hz, at least one of amplitude or
  latent phase has a posterior-median shift greater than one prior sigma and its
  90% interval excludes zero. Report both components and every other knot; a
  selected maximum over knots cannot replace the predefined test.
- The corresponding undeformed GR control does not meet those calibration-shift
  and apparent-deviation criteria.
- Nullcal-corrected spectroscopy excludes zero deviation and includes the
  injected deviation, with the identifiability qualifications below.

Support the controlled-population absorption claim only if at least six of the
eight replicates meet every criterion. Report the count, its binomial interval,
and the full replicate table even when the claim fails. This fraction is a
prospective design threshold, not a measured success rate. Do not generalize it
to an astrophysical population or use it as an automatic venue decision.

Require R-hat <=1.01, bulk and tail ESS >=400 for all reported parameters and
zero divergences in HMC runs. Record the corresponding diagnostics for a non-HMC
conventional sampler. Failed diagnostics make that replicate inconclusive; never
drop it from the denominator or call it a null result.

## What nullcal can identify

Geometric closure cancels arbitrary incident polarization in the stated LWA
geometry. It does **not** identify a calibration factor common to all three
detectors: multiplying the entire response by the same nonzero complex factor
preserves its signal subspace and hence its null projector. This common factor
must remain prior-controlled. Do not impose an exact reference-detector
calibration using injection truth and then call it waveform-independent
recovery.

Report relative calibration through `C_ET1/C_ET3` and `C_ET2/C_ET3`, evaluated
from each posterior draw, separately from the common factor. Report their 90%
intervals and zero-error coverage across noise replicates. Unbiased relative
calibration is compatible with broad absolute calibration uncertainty. A null
residual alone proves cancellation, not an unbiased posterior or preserved
spectroscopy. If common-factor uncertainty prevents resolving the deviation,
report that limitation and do not claim success of the nullcal arm.

## Inspiral counter-hypothesis and scope

The inspiral-only arm is sufficient for this controlled population only if at
least six of eight replicates still exclude zero and include the injected
deviation in their calibration-marginalized spectroscopy intervals. Report
calibration uncertainty at ringdown frequencies, maximum residual error across
the stated band, common/relative factors and source-remnant uncertainty. If this
counter succeeds, report it: D1 has not established that waveform-based
calibration is inapplicable generally. Do not invoke 4%/4° as an exact
substitute for the direct `ringdown` test.

The result paragraph must state the actual absorption interval, the knot shift,
the nullcal result, and the inspiral-only result. It must also state that the
experiment assumes stationary independent noise, co-located LWA geometry, shared
calibration over a specified epoch and a selected waveform/deformation. It
cannot establish finite-arm validity, arbitrary common-calibration recovery, or
all-source failure of conventional inference. Missing benchmark, waveform,
source-model validation or posterior artifacts leave these quantities
unmeasured.
