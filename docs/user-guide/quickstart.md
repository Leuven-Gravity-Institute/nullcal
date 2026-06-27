# Quick Start

This guide walks you through the basic usage of nullcal.

## 1. Import the Package

```python
import numpy as np
from bilby.gw.detector import InterferometerList
from nullcal.null_stream.null_stream import NullStream
from nullcal.time_frequency_transform.wavelet_transforms import WaveletTransform
```

## 2. Set Up a Detector Network

nullcal is designed for closed-geometry networks. The typical use case is the
Einstein Telescope triangular configuration with three interferometers (ET1,
ET2, ET3):

```python
interferometers = InterferometerList(["ET"])
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=4096, duration=4, start_time=0
)
```

## 3. Compute the Null Stream

```python
wavelet_transform = WaveletTransform(nx=4, frequency_resolution=4)
null_stream = NullStream(
    interferometers=interferometers,
    time_frequency_transform=wavelet_transform,
    time_frequency_filter=None,
)
null_data = null_stream.compute_calibrated_frequency_domain_null_stream(
    calibration_factor=np.ones_like(interferometers[0].frequency_array)
)
```

The null stream is a data combination that cancels the gravitational-wave signal
while preserving noise. In a perfectly calibrated network, the null stream
contains only noise.

## 4. Constrain Calibration Errors

```python
from nullcal.likelihood import RecalibrationLikelihood

likelihood = RecalibrationLikelihood(
    interferometers=interferometers,
    waveform_generator=...,
    wavelet_transform_frequency_resolution=4,
    wavelet_transform_nx=4,
)
```

## Next Steps

- See [Installation](installation.md) for environment setup
- See the [API Reference](../reference/index.md) for module documentation
