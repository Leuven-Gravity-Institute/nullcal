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
`cosmic_explorer_strain.txt` from CE-T2000017-v5, selected on 2026-10-02. Its
SHA-256 is `ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c`.
The exact published bytes are preserved in `inputs/cosmic_explorer_strain.txt`;
`inputs/ce-curve-manifest.json` records the source archive and extraction
hashes. The 20 km file is excluded from this benchmark. No scientific comparison
has yet run.

## Fiducial catalog inspection

The catalog probe ran from clean producing commit
`28c1ff4b11bd032f7cf5fcca81b49f78bacf72d8` through a scheduler-controlled CPU
job (129463, completed with exit 0):

```sh
python -u d1_waveform_probe.py --output waveform-catalog-probe.json \
  --producing-commit 28c1ff4b11bd032f7cf5fcca81b49f78bacf72d8
```

The script SHA-256 is
`385a6500e62393784f78b5d044248fdc4fd1912527a9a8450b95632919a72278`.
`inputs/waveform-catalog-probe.json` preserves the catalog entry, including
published download URLs and checksums. Its SHA-256 is
`d6097306a0e9b462e4b3974830ea2d0528c1fde3c13e75d7b7a748a5d893fe0f`. The observed
catalog tag is `v3.0.0`; the unversioned simulation resolves to
`SXS:BBH:0305v3.0/Lev6`. Older versions, including `v2.0`, remain listed. This
was metadata inspection: no strain extraction or inference ran.

[SXS documentation](https://sxs.readthedocs.io/en/main/tutorials/02-Simulation/)
states that versions identify modifications to the data files. Its
[waveform tutorial](https://sxs.readthedocs.io/en/stable/tutorials/04-Waveforms/)
notes that the third catalog includes waveform memory. The
[anchor paper](https://arxiv.org/html/2506.15979v2) names the simulation but
does not pin its release. The release used to produce its figures could not be
established from that paper. The benchmark selection made on 2026-10-04 is
catalog tag `v3.0.0`, simulation **`SXS:BBH:0305v3.0/Lev6`** (data release
`v3.0`, resolution `Lev6`). Future acquisition must use this explicit
identifier, rather than an unversioned loader default. The authors' data
release, resolution and SXS package version remain **unknown**; this selection
does not establish their choice. The prior probe used SXS package `2026.0.7`,
which is separate from the data-release pin. `inputs/waveform-release-pin.json`
records the selection and links the prior catalog evidence to the exact
published Lev6 metadata bytes in `inputs/sxs-bbh-0305-v3-lev6-metadata.json`.
The downloaded metadata's MD5 matches the catalog's published
`0217071352f64284b14400ae4338f5c7`; the pin record also preserves its SHA-256.

The SXS waveform, spectroscopy anchor comparison, GR IMR validation, injection
prescription and resolution pre-check remain unproduced. The prerequisite
checker still returns exit 2, `status=blocked`, all six required files missing,
and `scientific_results=null`. Environment and catalog verification must not be
counted as completion of any of those artifacts.

## Table 1 metadata anchor recipe

The external target is
[Table 1 of the anchor paper, v2](https://arxiv.org/html/2506.15979v2#S3.T1):
SXS:BBH:0305 has remnant mass **67.21 solar masses** and dimensionless spin
**0.69**, with the total binary mass fixed to **70.6 solar masses** in the
caption. Fix the rounding tolerances before comparison: absolute difference at
most **0.005 solar masses** for mass and **0.005** for spin (half a unit in the
last printed decimal place); relative tolerance is zero. These are rounding
checks, not measurement uncertainties or bounds on waveform accuracy. Only a
beyond-rounding disagreement triggers the same check with `v2.0`.

Run the following standard-library calculation in a scheduled CPU job from a
clean revision containing this recipe and its metadata inputs. Set
`D1_PRODUCING_COMMIT` to that full revision. Execute from the input directory;
the output is `remnant-metadata-anchor.json`, which is separate from the
spectroscopy comparison `ringdown-anchor.json` required by the preflight.

```python
import hashlib
import json
import math
import os
import platform
from pathlib import Path

pin = json.loads(Path("waveform-release-pin.json").read_text())
metadata_path = Path("sxs-bbh-0305-v3-lev6-metadata.json")
raw = metadata_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == pin["artifacts"][metadata_path.name]
assert "md5:" + hashlib.md5(raw).hexdigest() == pin["metadata_published_checksum"]
metadata = json.loads(raw)
catalog = json.loads(Path("waveform-catalog-probe.json").read_text())
assert hashlib.sha256(Path("waveform-catalog-probe.json").read_bytes()).hexdigest() == pin["catalog_probe_sha256"]
assert catalog["catalog_tag"] == pin["catalog_tag"] == "v3.0.0"
assert pin["simulation"] == "SXS:BBH:0305v3.0/Lev6"
assert metadata["simulation_name"].endswith("/Lev6")
assert "SXS:BBH:0305" in metadata["alternative_names"]
for key in ("remnant_mass", "remnant_dimensionless_spin", "reference_mass1", "reference_mass2"):
    assert metadata[key] == catalog["catalog_entry"][key]

mass_fraction = metadata["remnant_mass"]
binary_mass = 70.6
spin_vector = metadata["remnant_dimensionless_spin"]
spin = math.hypot(*spin_vector)
mass = mass_fraction * binary_mass
# Check sensitivity to normalizing by the component masses at reference time.
reference_total = metadata["reference_mass1"] + metadata["reference_mass2"]
reference_normalized_mass = mass_fraction / reference_total * binary_mass
mass_pass = abs(mass - 67.21) <= 0.005
spin_pass = abs(spin - 0.69) <= 0.005
record = {
    "scope": "Table 1 remnant metadata anchor only; no spectroscopy or inference",
    "producing_commit": os.environ["D1_PRODUCING_COMMIT"],
    "dirty": False,
    "command": "python - < anchor recipe in environment-production.md",
    "scheduler_job_id": os.environ["SLURM_JOB_ID"],
    "versions": {"python": platform.python_version()},
    "catalog_tag": pin["catalog_tag"],
    "simulation": pin["simulation"],
    "authors_data_release": "unknown",
    "anchor": {
        "url": pin["anchor_paper"],
        "table": 1,
        "binary_mass_solar_masses": binary_mass,
        "remnant_mass_solar_masses": 67.21,
        "remnant_spin": 0.69,
    },
    "metadata": {
        "remnant_mass": mass_fraction,
        "remnant_dimensionless_spin": spin_vector,
        "reference_mass1": metadata["reference_mass1"],
        "reference_mass2": metadata["reference_mass2"],
    },
    "calculation": {
        "mass_formula": "remnant_mass * 70.6",
        "spin_formula": "Euclidean norm of remnant_dimensionless_spin",
        "remnant_mass_solar_masses": mass,
        "remnant_spin": spin,
        "mass_difference_solar_masses": mass - 67.21,
        "spin_difference": spin - 0.69,
        "reference_total_mass": reference_total,
        "reference_normalized_mass_solar_masses": reference_normalized_mass,
        "reference_normalized_mass_pass": abs(reference_normalized_mass - 67.21) <= 0.005,
    },
    "rounding_tolerance": {
        "mass_solar_masses": 0.005,
        "spin": 0.005,
        "relative": 0.0,
        "comparison": "absolute difference <= tolerance",
    },
    "mass_pass": mass_pass,
    "spin_pass": spin_pass,
    "v2_fallback_required": not (mass_pass and spin_pass),
    "artifacts": {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in (metadata_path.name, "waveform-release-pin.json", "waveform-catalog-probe.json")
    },
    "limitations": [
        "Authors' data release, resolution and package version are unknown.",
        "The table does not specify the mass-normalization epoch; both conventions are checked.",
        "Rounded metadata agreement does not identify the authors' release or reproduce their posteriors.",
    ],
}
Path("remnant-metadata-anchor.json").write_text(json.dumps(record, indent=4) + "\n")
print(json.dumps(record, indent=4))
raise SystemExit(0 if mass_pass and spin_pass else 2)
```
