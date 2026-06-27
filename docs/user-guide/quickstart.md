# Quick Start

This guide walks you through the basic usage of nullcal.

## 1. Import the Package

```python
import nullcal
from nullcal.null_stream import NullStreamComputer
```

## 2. Set Up a Detector Network

nullcal is designed for closed-geometry networks. The typical use case is the
Einstein Telescope triangular configuration with three interferometers:

```python
sampling_frequency = 4096  # Hz
waveform_duration = 4  # seconds

null_stream = NullStreamComputer(
    ifos=["ET1", "ET2", "ET3"],
    sampling_frequency=sampling_frequency,
    waveform_duration=waveform_duration,
)
```

## 3. Compute the Null Stream

```python
null_data = null_stream.compute_null_stream(strain_data)
```

The null stream is a data combination that cancels the gravitational-wave signal
while preserving noise. In a perfectly calibrated network, the null stream
contains only noise.

## 4. Constrain Calibration Errors

```python
from nullcal.likelihood import RecalibrationLikelihood

likelihood = RecalibrationLikelihood(
    null_stream=null_stream,
    calibration_parameters={"amplitude": 1.0, "phase": 0.0},
)
result = likelihood.evaluate(null_data)
```

## Next Steps

- See [Installation](installation.md) for environment setup
- See the [API Reference](../reference/index.md) for module documentation
