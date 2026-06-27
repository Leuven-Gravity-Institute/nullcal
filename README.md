# nullcal

[![Python CI](https://github.com/Leuven-Gravity-Institute/nullcal/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/nullcal/actions/workflows/ci.yml)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/nullcal/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/nullcal)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/nullcal/branch/main/graph/badge.svg)](https://codecov.io/gh/Leuven-Gravity-Institute/nullcal)
[![PyPI Version](https://img.shields.io/pypi/v/nullcal)](https://pypi.org/project/nullcal/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nullcal)](https://pypi.org/project/nullcal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/ID.svg)](https://doi.org/DOI)

A Python package for constraining calibration errors of a closed-geometry
network of gravitational-wave detectors. It provides the null stream formalism
for noise-only data combinations and Bayesian recalibration likelihoods to
quantify detector miscalibration.

## Features

- **Null stream construction** for closed-geometry detector networks
- **Calibration error constraint** via Bayesian recalibration likelihood
- **Time-frequency transforms** (Short-Time Fourier Transform, wavelet
  transforms)
- **Clustering algorithms** for time-frequency map analysis
- **Modular architecture** with clean separation between null stream,
  likelihood, prior, result, and utility layers

## Installation

We recommend using `uv` to manage virtual environments for installing nullcal.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

**Note:** The package requires Python 3.10 or later and is built and tested
against Python 3.10–3.12. When creating a virtual environment with `uv`, specify
the Python version to ensure compatibility: `uv venv --python 3.12` (replace
`3.12` with your preferred supported version: 3.10, 3.11, or 3.12). This avoids
potential issues with unsupported Python versions.

### From PyPI

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install nullcal
```

### From Source

```bash
git clone git@github.com:Leuven-Gravity-Institute/nullcal.git
cd nullcal
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## Quick Start

```python
import numpy as np
from bilby.gw.detector import InterferometerList
from nullcal.likelihood import RecalibrationLikelihood
from nullcal.null_stream.null_stream import NullStream
from nullcal.time_frequency_transform.wavelet_transforms import WaveletTransform

# Set up the Einstein Telescope interferometer triplet
interferometers = InterferometerList(["ET"])
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=4096, duration=4, start_time=0
)

# Compute the null stream
wavelet_transform = WaveletTransform(nx=4, frequency_resolution=4)
null_stream = NullStream(
    interferometers=interferometers,
    time_frequency_transform=wavelet_transform,
    time_frequency_filter=None,
)
null_data = null_stream.compute_calibrated_frequency_domain_null_stream(
    calibration_factor=np.ones_like(interferometers[0].frequency_array)
)

# Set up a recalibration likelihood for calibration error constraints
likelihood = RecalibrationLikelihood(
    interferometers=interferometers,
    waveform_generator=...,
    wavelet_transform_frequency_resolution=4,
    wavelet_transform_nx=4,
)
```

## Documentation

Full documentation is available at
[https://leuven-gravity-institute.github.io/nullcal/](https://leuven-gravity-institute.github.io/nullcal/).

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/nullcal/releases).

## Testing

Run the test suite:

```bash
uv run pytest
```

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE)
file for the full license text.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/nullcal/issues/new) or
contact the maintainers.
