# Isolated science environment

The environment was installed with Python 3.12.3 in a separate virtual
environment. The complete resolved package set is pinned in
`inputs/science-requirements.txt`. Its SHA-256 is
`7e86cddc317ab0a6f4b8ffdbb2d0ca033b4a9180a954051e99bd05484fc0abc6`.

A scheduler-controlled CPU job ran the import-verification script from clean
producing commit `52179bb9136a928d33851362cc67f2372f128206`:

```sh
python -u d1_science_environment.py \
  --output science-environment.json \
  --requirements science-requirements.txt \
  --producing-commit 52179bb9136a928d33851362cc67f2372f128206
```

The job completed with exit 0 and printed:

```text
import ringdown: OK (1.1.0)
import sxs: OK (2026.0.7)
import ripplegw: OK (0.0.9)
```

`inputs/science-environment.json` preserves the producing commit, Python and
platform information, verified versions, requirement hash and script hash. This
confirms imports only; it does not validate a GPU backend, numerical accuracy,
extraction, inference or the prerequisite bundle.

The anchor paper cites
[CE-T2000017-v5](https://dcc.cosmicexplorer.org/CE-T2000017-v5/public). Its
README distinguishes two strain sensitivity curves:

| Filename                          | Detector       | SHA-256                                                            |
| --------------------------------- | -------------- | ------------------------------------------------------------------ |
| `cosmic_explorer_strain.txt`      | baseline 40 km | `ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c` |
| `cosmic_explorer_20km_strain.txt` | baseline 20 km | `ced9a60e6a304a5809ee81ac24752d8fecf867c0a64f3056c915105a22869af2` |

These hashes were computed directly from the downloaded published archive. They
are source-file identifiers, not science measurements. Both files contain strain
amplitude spectral density, so their squares supply a PSD. Selecting a curve
affects the covariance and weighting, even if strain is later scaled to the same
SNR. The selected benchmark curve is the baseline 40 km
`cosmic_explorer_strain.txt` from CE-T2000017-v5, selected on 2026-10-02.
Its SHA-256 is
`ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c`.
The exact published bytes are preserved in `inputs/cosmic_explorer_strain.txt`;
`inputs/ce-curve-manifest.json` records the source archive and extraction hashes.
The 20 km file is excluded from this benchmark. No scientific comparison has
yet run.

The SXS waveform, spectroscopy anchor comparison, GR IMR validation, injection
prescription and resolution pre-check remain unproduced. Environment
verification must not be counted as completion of any of those artifacts.
