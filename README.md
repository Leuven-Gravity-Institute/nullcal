# nullcal

A Python package for constraining calibration errors of a closed-geometry
network of gravitational-wave detectors.

[![PyPI version](https://badge.fury.io/py/nullcal.svg)](https://pypi.org/project/nullcal/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/Leuven-Gravity-Institute/nullcal/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/nullcal/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/nullcal/branch/main/graph/badge.svg)](https://codecov.io/gh/Leuven-Gravity-Institute/nullcal)
[![Python Version](https://img.shields.io/pypi/pyversions/nullcal)](https://pypi.org/project/nullcal/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/documentation-online-brightgreen)](https://leuven-gravity-institute.github.io/nullcal/)
[![DOI](https://zenodo.org/badge/ID.svg)](https://doi.org/DOI)

## Features

- **Null stream construction** for closed-geometry detector networks
- **Calibration error constraint** via Bayesian recalibration likelihood
- **Time-frequency transforms** (Short-Time Fourier Transform, wavelet
  transforms)
- **Clustering algorithms** for time-frequency map analysis
- **Modular architecture** with clean separation between null stream,
  likelihood, prior, result, and utility layers

## Installation

### Requirements

- Python 3.10 or later
- Linux, macOS, or Windows

### From PyPI

```console
pip install nullcal
```

### From Source

```console
git clone https://github.com/Leuven-Gravity-Institute/nullcal.git
cd nullcal
uv sync
```

### Development

Install development dependencies and pre-commit hooks:

```console
uv sync --group dev
uv run prek install
```

## Quick Start

```python
from nullcal.null_stream import NullStreamComputer
from nullcal.likelihood import RecalibrationLikelihood

# Construct a null stream for a detector network
null_stream = NullStreamComputer(
    ifos=["ET1", "ET2", "ET3"],
    sampling_frequency=4096,
    waveform_duration=4,
)

# Apply calibration parameters
# likelihood = RecalibrationLikelihood(...)
```

## Documentation

Full documentation is available at
[https://leuven-gravity-institute.github.io/nullcal/](https://leuven-gravity-institute.github.io/nullcal/).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.
