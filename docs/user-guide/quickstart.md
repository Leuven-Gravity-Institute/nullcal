# Quick Start

This guide walks you through the basic usage of nullcal.

## 1. Import the Package

```python
import numpy as np
from nullcal.data import InterferometerData
from nullcal.null_stream.null_stream import NullStream
from nullcal.time_frequency_transform.wavelet_transforms import WaveletTransform
```

## 2. Set Up a Detector Network

nullcal is designed for closed-geometry networks. Load detector data from your
strain source into the immutable array container. The convenience loader
`InterferometerData.from_interferometers(...)` accepts detector objects with the
corresponding attributes.

```python
duration = 4.0
sampling_frequency = 4096.0
frequency_array = np.fft.rfftfreq(int(duration * sampling_frequency), 1 / sampling_frequency)
data = InterferometerData(
    psd=np.ones((3, frequency_array.size)),
    strain=np.zeros((3, frequency_array.size), dtype=complex),
    mask=np.broadcast_to(frequency_array >= 4.0, (3, frequency_array.size)).copy(),
    frequency_array=frequency_array,
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=0.0,
    name=("ET1", "ET2", "ET3"),
)
```

## 3. Compute the Null Stream

```python
wavelet_transform = WaveletTransform(
    duration=duration, sampling_frequency=sampling_frequency, nx=4, frequency_resolution=4
)
time_frequency_filter = np.ones(wavelet_transform.shape, dtype=bool)
null_stream = NullStream(
    interferometers=data,
    time_frequency_transform=wavelet_transform,
    time_frequency_filter=time_frequency_filter,
)
null_data = null_stream.compute_calibrated_frequency_domain_null_stream(calibration_factor=np.ones_like(data.strain))
```

The null stream is a data combination that cancels the gravitational-wave signal
while preserving noise. In a perfectly calibrated network, the null stream
contains only noise.

## 4. Constrain Calibration Errors

```python
from nullcal.likelihood import RecalibrationLikelihood

likelihood = RecalibrationLikelihood(
    interferometers=data,
    knot_frequencies=np.geomspace(4.0, 2048.0, 10),
    time_frequency_filter=time_frequency_filter,
    wavelet_transform_frequency_resolution=4,
    wavelet_transform_nx=4,
)

params = {
    "amplitude": np.zeros((3, 10)),
    "phase": np.zeros((3, 10)),
}
log_posterior = likelihood.logdensity_fn(params)
```

`logdensity_fn` is a pure JAX function and can be passed directly to BlackJAX.
Use `nullcal.sampler.sample_nuts` when the standard window-adapted NUTS run and
R-hat, ESS, and divergence diagnostics are desired together.

## 5. Summarize and Apply a Calibration Posterior

```python
from nullcal.calibration import posterior_median_calibration_factor

# result is the output of nullcal.sampler.sample_nuts(...).
frequencies = data.frequency_array[np.all(data.mask, axis=0)]
factor = posterior_median_calibration_factor(
    frequencies,
    likelihood.knot_frequencies,
    result.samples["amplitude"],
    result.samples["phase"],
)
corrected_strain = np.asarray(data.strain).copy()
corrected_strain[:, np.all(data.mask, axis=0)] /= np.asarray(factor)
```

The likelihood multiplies the whitened antenna response by the calibration
factor. Correcting observed strain uses its inverse, as above. The point
estimate evaluates the complex factor for every posterior sample at every
frequency, then takes separate medians of its real and imaginary parts.
Evaluating a spline from median knot values is a different estimator.

## Next Steps

- See [Installation](installation.md) for environment setup
- See the [API Reference](../reference/index.md) for module documentation
