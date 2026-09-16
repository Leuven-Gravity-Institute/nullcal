# Bilby calibration-model anchor

Producing commit: `21d6b8b91809bafbed35b2805ba52353fff8e3a4` Versions: bilby
2.8.2, JAX 0.11.1, NumPy 2.3.5.

The factor tolerance was fixed before comparison at peak-relative `1e-11`. Both
paths solve the same at-most-19-dimensional float64 system; the bound is over
100 times `n^2 * eps` at `n=19`, allowing different LAPACK/XLA reductions while
remaining negligible on the physical scale.

Bilby fixes knot placement to uniform spacing in log frequency. The table
therefore sweeps only band, knot count, and node scale; it does not sweep knot
placement.

| Band (Hz) | Knots | Node scale | Peak-relative deviation |
| --------: | ----: | ---------: | ----------------------: |
|     8-512 |     4 |          0 |                       0 |
|     8-512 |     4 |       0.01 |  2.2088821710794792e-16 |
|     8-512 |     4 |        0.2 |  9.0291578161108082e-16 |
|   20-2000 |     7 |          0 |                       0 |
|   20-2000 |     7 |       0.01 |  2.2061150193781349e-16 |
|   20-2000 |     7 |        0.2 |  4.6458258310676654e-16 |
|    8-2048 |    10 |          0 |                       0 |
|    8-2048 |    10 |       0.01 |  2.2221665135454867e-16 |
|    8-2048 |    10 |        0.2 |  1.0333124624958732e-15 |
|    8-2048 |    19 |          0 |                       0 |
|    8-2048 |    19 |       0.01 |  2.2082619598722291e-16 |
|    8-2048 |    19 |        0.2 |  5.0869968260434048e-16 |

Worst factor deviation: `1.0333124624958732e-15` (tolerance `1e-11`).

## Extrapolation

On the 7-knot 20-2000 Hz uniform-log model, 200 frequencies from 10-19.9 Hz and
200 from 2000.1-4000 Hz give peak-relative bilby/JAX deviation
`1.1759054902345757e-15` (tolerance `1e-11`).

## Nonuniform knot placement

The 19-knot nonuniform spectroscopy grid has no bilby anchor by construction
because bilby hard-codes uniform-log knots. SciPy's not-a-knot `CubicSpline`
covers that generalisation: SciPy agrees with bilby on the corresponding 19-knot
uniform-log grid to `2.222122336258544e-16`, and JAX agrees with SciPy on the
nonuniform grid at 1001 off-knot frequencies over 20-2000 Hz to
`8.8969464739234496e-16` (tolerance `1e-11`).

The gradient comparison used centred differences with step `1e-06` and fixed
tolerances `rtol=5e-07`, `atol=5e-09`. The basis is the O(h^2) truncation and
O(eps/h) round-off of a centred float64 difference.

| Gradient  | Max absolute deviation | Peak-relative deviation |
| --------- | ---------------------: | ----------------------: |
| amplitude | 6.8701990763031517e-09 |  3.2800087922912774e-10 |
| phase     | 6.3656138138412643e-09 |  7.4340774908128764e-10 |

This runner intentionally requires bilby. The checked-in table is the durable
record for use after bilby is removed from the runtime dependency set.
