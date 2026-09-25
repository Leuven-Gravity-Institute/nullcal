# GPU cost ledger — vmap over realisations

Per-posterior cost is the campaign budget. The accelerator arm runs one NUTS
chain per realisation inside a single `vmap`; the CPU arm is the same-day
baseline sampling the same realisation ensemble, so the two arms target the same
posterior family.

## How these numbers were produced

`scripts/gpu_cost_ledger.py measure` writes one
`<configuration>-<platform>.json` per arm; `merge` combines them and computes
the speedups and the per-posterior split. The accelerator arm asserts
`any(device.platform == "gpu" for device in jax.devices())` and exits non-zero
otherwise, so a CPU-only node cannot emit GPU-labelled numbers. Every timing
call regenerates its input array, so a device-cached value cannot be mistaken
for a fresh transfer. Run the CPU arm with `JAX_PLATFORMS=cpu`.

| field                         | value                                      |
| ----------------------------- | ------------------------------------------ |
| ledger created                | 2026-09-25T18:27:22.191460+00:00           |
| reference accelerator         | NVIDIA A30 (gpu)                           |
| reference CPU                 | cpu                                        |
| reference committed revision  | `608d3d46c83906ab3de79b300039ad16c000a607` |
| production accelerator        | NVIDIA A30 (gpu)                           |
| production CPU                | cpu                                        |
| production committed revision | `608d3d46c83906ab3de79b300039ad16c000a607` |

## reference

Model: 3 detectors, 8033 selected bins over 20-1024 Hz, 10 knots, transform
[256, 64].

| stage               |     CPU (s) | accelerator (s) | speedup |
| ------------------- | ----------: | --------------: | ------: |
| likelihood_gradient |   0.0402903 |      0.00161606 |  24.93x |
| projector           |  0.00906722 |      0.00052536 |  17.26x |
| transform           | 0.000673123 |      0.00012037 |   5.59x |

Posterior: accelerator 6.17371 s/posterior (vmap-over-realisations); CPU 157.088
s/posterior; speedup 25.44x.

accelerator split: likelihood 4.12095 s, sampler overhead 2.05276 s.

CPU split: likelihood 67.3654 s, sampler overhead 89.7229 s.

## production

Model: 3 detectors, 15841 selected bins over 20-2000 Hz, 19 knots, transform
[256, 128].

| stage               |    CPU (s) | accelerator (s) | speedup |
| ------------------- | ---------: | --------------: | ------: |
| likelihood_gradient |  0.0779725 |       0.0024289 |  32.10x |
| projector           |  0.0182114 |      0.00071227 |  25.57x |
| transform           | 0.00166399 |      0.00012998 |  12.80x |

Posterior: accelerator 325.18 s/posterior (vmap-over-realisations); CPU 475.707
s/posterior; speedup 1.46x.

accelerator split: likelihood 130.937 s, sampler overhead 194.242 s.

CPU split: likelihood 238.908 s, sampler overhead 236.8 s.

## Campaign pricing

| configuration |        1 event |   100x10x10 |  1000x10x10 |
| ------------- | -------------: | ----------: | ----------: |
| reference     | 0.001715 GPU-h | 17.15 GPU-h | 171.5 GPU-h |
| production    |  0.09033 GPU-h | 903.3 GPU-h |  9033 GPU-h |

## Anchors

- `reference_configuration` = frozen e2e inputs, 8 s / 2048 Hz / 20-1024 Hz / 10
  knots
- `reference_log_likelihood` = -203.88870383371773

## Unanchored

- the timed realisations are synthetic same-shape unit-variance complex arrays,
  not the M3 noise-and-calibration realisations; a surrogate does not reproduce
  the sampler's trajectory length, so the per-posterior cost is a shape-anchored
  estimate rather than an M3 prediction
- the production configuration selects every time-frequency pixel (all-ones
  filter), an upper bound on the cost of a clustered M3 analysis, which keeps
  only a sparse pixel set
- the production accelerator posterior averaged 269 NUTS integration steps per
  sample (near maximum tree depth) on the synthetic target, so its per-posterior
  cost is a pathological surrogate upper bound; the single-evaluation stage
  speedups are the better model-size scaling
- the CPU baseline is one realisation draw from the same ensemble, while the
  accelerator arm averages over its batch; the CPU arm's own integration-step
  count is recorded so the trajectory lengths can be compared
- warmup gradient count is a lower bound: one gradient per warmup iteration
- the per-posterior split uses the single-evaluation likelihood-gradient cost,
  not the batched one, so the accelerator likelihood share is an upper bound and
  its sampler overhead a lower bound
- the campaign size is not fixed by M2 beyond 'a population'; the pricing table
  is per-posterior cost times an illustrative grid
