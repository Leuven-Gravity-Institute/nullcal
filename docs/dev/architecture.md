# Architecture

## Design Principles

- **Modular separation**: The package is organized into distinct layers — null
  stream construction, calibration, likelihood, priors, and result analysis
- **Scientific correctness**: All transforms preserve the physical meaning of
  gravitational-wave data
- **Testability**: Each module has a corresponding test module under `tests/`

## Project Structure

````text
src/nullcal/
├── clustering/          # Time-frequency clustering algorithms
│   ├── base.py          # Base clustering interface
│   ├── injection.py     # Signal injection and clustering
│   ├── precompute.py    # Precomputed clustering
│   ├── single.py        # Single-detector clustering
│   └── time_frequency_map.py  # Time-frequency map representation
├── likelihood/          # Likelihood computations
│   └── recalibration_likelihood.py  # Bayesian recalibration likelihood
├── metadata/            # Metadata handling
│   └── yaml.py          # YAML metadata serialization
├── null_stream/         # Core null stream logic
│   ├── calibration.py   # Calibration parameter handling
│   ├── null_stream.py   # Null stream computer
│   ├── projector.py     # Null stream projector
│   └── whiten.py        # Data whitening
├── prior/               # Prior distributions
│   └── prior.py         # Prior definitions
├── result/              # Result analysis
│   ├── result.py        # Result container
│   └── utils.py         # Result utilities
├── time_frequency_transform/  # Wavelet and STFT transforms
│   ├── inverse_wavelet_freq_funcs.py
│   ├── inverse_wavelet_time_funcs.py
│   ├── stft.py
│   ├── transform_freq_funcs.py
│   ├── transform_time_funcs.py
│   ├── utils.py
│   └── wavelet_transforms.py
├── utils/               # Shared utilities
│   ├── log.py           # Logging utilities
│   └── snr.py           # Signal-to-noise ratio
└── version.py           # Package version
```text

## Data Flow

1. **Input**: Strain data from a closed-geometry detector network
2. **Null Stream**: The `NullStreamComputer` constructs a data combination that
   cancels the gravitational-wave signal, leaving only noise
3. **Calibration**: The `RecalibrationLikelihood` injects calibration parameters
   to model deviations from perfect calibration
4. **Output**: Calibration error constraints with associated likelihood values

## Extension Points

- New clustering algorithms can be added under `clustering/`
- Additional likelihood models can extend the `likelihood/` module
- Custom time-frequency transforms can be added to `time_frequency_transform/`
````
