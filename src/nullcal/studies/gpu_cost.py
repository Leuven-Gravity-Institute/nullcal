"""Measure and price the vmap-over-realisations campaign path.

R6's number is the cost per posterior when the correction loop runs over many
noise and calibration realisations, not the latency of a single chain. This
module holds the pieces that can be checked without an accelerator: the device
assertion that stops a CPU-only node from producing GPU-labelled numbers, the
per-stage timing summary, the additive decomposition of a posterior into
likelihood and sampler overhead, and the campaign arithmetic. The driver in
``scripts/gpu_cost_ledger.py`` runs the measurements on the requested backend.

Every timing call regenerates its input arrays for that iteration, so a value
cached on the device by a previous call can never be mistaken for a transfer or
a fresh computation.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..data import InterferometerData
from ..likelihood import RecalibrationLikelihood
from ..sampler import sample_nuts, sample_nuts_batched
from ..time_frequency_transform.wavelet_transforms import WaveletTransform
from .lwa_leakage import et_d_psd

GPU_PLATFORM = "gpu"
CPU_PLATFORM = "cpu"
SECONDS_PER_HOUR = 3_600.0
DETECTOR_NAMES = ("ET1", "ET2", "ET3")


def devices_with_platform(platform: str, devices: Sequence | None = None) -> tuple:
    """Return the JAX devices whose platform matches ``platform``."""
    available = jax.devices() if devices is None else devices
    return tuple(device for device in available if getattr(device, "platform", None) == platform)


def require_gpu_devices(devices: Sequence | None = None) -> tuple:
    """Return the GPU devices, or raise if the process cannot see one.

    A job that lands on a CPU-only node must fail loudly rather than emit
    numbers labelled GPU, so callers turn this into a non-zero exit.
    """
    gpus = devices_with_platform(GPU_PLATFORM, devices)
    if not gpus:
        available = jax.devices() if devices is None else devices
        platforms = sorted({getattr(device, "platform", "unknown") for device in available})
        raise RuntimeError(f"no GPU device present (JAX platforms: {platforms}); refusing to report CPU numbers as GPU")
    return gpus


def summarise_seconds(seconds: Sequence[float]) -> dict[str, float | int]:
    """Return min/median/mean of a non-negative timing sample."""
    values = np.asarray(seconds, dtype=float)
    if values.size == 0:
        raise ValueError("at least one timing sample is required")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("timing samples must be finite and non-negative")
    return {
        "count": int(values.size),
        "minimum_seconds": float(values.min()),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(values.mean()),
    }


def speedup(cpu_seconds: float, accelerator_seconds: float) -> float:
    """Return the CPU/accelerator ratio, rejecting unusable inputs."""
    if cpu_seconds <= 0.0 or accelerator_seconds <= 0.0:
        raise ValueError("both timings must be positive")
    return float(cpu_seconds) / float(accelerator_seconds)


def decompose_per_posterior(
    seconds_per_posterior: float, gradient_seconds: float, gradients_per_posterior: float
) -> dict[str, float]:
    """Split a posterior wall time into likelihood and sampler overhead.

    ``gradient_seconds`` is the measured cost of one log-posterior gradient
    evaluation and ``gradients_per_posterior`` the number of such evaluations a
    posterior performs. The remainder after the likelihood term is the
    integrator and adaptation overhead.
    """
    if seconds_per_posterior < 0.0 or gradient_seconds < 0.0 or gradients_per_posterior < 0.0:
        raise ValueError("times and gradient counts must be non-negative")
    likelihood_seconds = gradient_seconds * gradients_per_posterior
    return {
        "likelihood_seconds": float(likelihood_seconds),
        "sampler_overhead_seconds": float(max(seconds_per_posterior - likelihood_seconds, 0.0)),
    }


def campaign_gpu_hours(
    seconds_per_posterior: float, event_count: int, noise_realisations: int, calibration_draws: int
) -> dict[str, float | int]:
    """Price a population campaign from the measured per-posterior cost."""
    factors = {
        "event_count": event_count,
        "noise_realisations": noise_realisations,
        "calibration_draws": calibration_draws,
    }
    for name, value in factors.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be at least one")
    posterior_count = int(event_count) * int(noise_realisations) * int(calibration_draws)
    if seconds_per_posterior < 0.0:
        raise ValueError("seconds_per_posterior must be non-negative")
    return {
        **{name: int(value) for name, value in factors.items()},
        "posterior_count": posterior_count,
        "gpu_hours": float(seconds_per_posterior) * posterior_count / SECONDS_PER_HOUR,
    }


def fresh_complex_array(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Return a new complex128 array on the default device from a fresh key."""
    real_key, imaginary_key = jax.random.split(key)
    real = jax.random.normal(real_key, shape, dtype=jnp.float64)
    imaginary = jax.random.normal(imaginary_key, shape, dtype=jnp.float64)
    return jnp.asarray(1.0 / np.sqrt(2.0), dtype=jnp.complex128) * (
        real + jnp.asarray(1j, dtype=jnp.complex128) * imaginary
    )


def synthetic_realisations(full_shape: tuple[int, ...], count: int, *, seed: int) -> np.ndarray:
    """Return ``count`` unit-variance complex realisations shaped like the data.

    These are same-shape surrogates for the throughput measurement, not the M3
    science realisations: the cost of the path depends on the array shapes, and
    the ledger says so.
    """
    if count < 1 or any(size < 1 for size in full_shape):
        raise ValueError("shape entries and count must be positive")
    rng = np.random.default_rng(seed)
    shape = (count, *full_shape)
    noise = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
    return noise.astype(np.complex128)


def synthetic_parameters(full_shape: tuple[int, ...], count: int, *, seed: int, scale: float = 0.01) -> dict:
    """Return fresh calibration parameters, batched on a leading axis."""
    if count < 1 or scale <= 0.0:
        raise ValueError("count must be positive and scale positive")
    rng = np.random.default_rng(seed)
    shape = (count, *full_shape)
    return {
        "amplitude": jnp.asarray(rng.normal(0.0, scale, shape), dtype=jnp.float64),
        "phase": jnp.asarray(rng.normal(0.0, scale, shape), dtype=jnp.float64),
    }


def build_synthetic_likelihood(
    *,
    duration: float,
    sampling_frequency: int,
    minimum_frequency: float,
    maximum_frequency: float,
    knot_frequencies: np.ndarray,
    frequency_resolution: float = 16.0,
    nx: float = 4.0,
) -> RecalibrationLikelihood:
    """Build a likelihood on synthetic same-shape data for cost measurement.

    The ET-D PSD is the only physical input; the whitened strain is zero. The
    time-frequency filter is all-ones, so the transform's cost is measured on
    its full output rather than on a sparse pixel selection.
    """
    frequency_array = np.fft.rfftfreq(int(duration * sampling_frequency), 1.0 / sampling_frequency)
    shared_mask = (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)
    if not np.any(shared_mask):
        raise ValueError("the requested band contains no frequency bin")
    clipped = np.clip(frequency_array, minimum_frequency, maximum_frequency)
    psd = np.broadcast_to(et_d_psd(clipped), (len(DETECTOR_NAMES), frequency_array.size)).copy()
    mask = np.broadcast_to(shared_mask, psd.shape).copy()
    strain = np.zeros_like(psd, dtype=np.complex128)
    transform_shape = WaveletTransform(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_resolution=frequency_resolution,
        nx=nx,
    ).shape
    time_frequency_filter = np.ones(transform_shape, dtype=bool)
    data = InterferometerData(
        psd=psd,
        strain=strain,
        mask=mask,
        frequency_array=frequency_array,
        duration=float(duration),
        sampling_frequency=float(sampling_frequency),
        start_time=0.0,
        name=DETECTOR_NAMES,
    )
    return RecalibrationLikelihood(
        data,
        knot_frequencies,
        time_frequency_filter=time_frequency_filter,
        wavelet_transform_frequency_resolution=frequency_resolution,
        wavelet_transform_nx=nx,
        whitened_frequency_domain_strain=strain,
    )


def time_call(function: Callable, make_inputs: Callable[[int], tuple], *, repeats: int, warmup: int = 1) -> list[float]:
    """Time compiled calls, regenerating the inputs for every iteration.

    ``make_inputs(iteration)`` must return a fresh argument tuple; nothing is
    reused between iterations, so a device-cached value cannot be timed as if
    it had just been produced.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative")
    for index in range(warmup + 1):
        jax.block_until_ready(function(*make_inputs(index)))
    seconds = []
    for index in range(repeats):
        inputs = make_inputs(warmup + 1 + index)
        started = time.perf_counter()
        jax.block_until_ready(function(*inputs))
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True)
class StageFunctions:
    """Compiled entry points for the decomposed stage measurements."""

    transform: Callable
    projector: Callable
    likelihood_gradient: Callable


def compile_stages(likelihood: RecalibrationLikelihood) -> StageFunctions:
    """Return the jitted stage functions the timing driver calls."""
    return StageFunctions(
        transform=jax.jit(likelihood._transform_to_time_frequency),
        projector=jax.jit(lambda params: likelihood._calibrated_projector(params["amplitude"], params["phase"])),
        likelihood_gradient=jax.jit(jax.value_and_grad(likelihood.logdensity_fn)),
    )


def measure_stages(
    likelihood: RecalibrationLikelihood,
    *,
    repeats: int,
    seed: int,
    stage_functions: StageFunctions | None = None,
) -> dict[str, dict[str, float | int]]:
    """Time the transform, projector, and one likelihood gradient evaluation."""
    stages = compile_stages(likelihood) if stage_functions is None else stage_functions
    detector_count, frequency_count = likelihood.parameter_shape[0], likelihood._frequency_count
    parameter_shape = likelihood.parameter_shape

    transform_times = time_call(
        stages.transform,
        lambda index: (
            fresh_complex_array(jax.random.fold_in(jax.random.key(seed), index), (detector_count, frequency_count)),
        ),
        repeats=repeats,
    )
    projector_times = time_call(
        stages.projector,
        lambda index: _parameter_inputs(parameter_shape, seed, index),
        repeats=repeats,
    )
    gradient_times = time_call(
        stages.likelihood_gradient,
        lambda index: _parameter_inputs(parameter_shape, seed + 10_000, index),
        repeats=repeats,
    )
    return {
        "transform": summarise_seconds(transform_times),
        "projector": summarise_seconds(projector_times),
        "likelihood_gradient": summarise_seconds(gradient_times),
    }


def _parameter_inputs(parameter_shape: tuple[int, int], seed: int, index: int) -> tuple[dict]:
    position = synthetic_parameters(parameter_shape, 1, seed=seed + index)  # leading axis stripped below
    return ({"amplitude": position["amplitude"][0], "phase": position["phase"][0]},)


def measure_batched_posterior(
    likelihood: RecalibrationLikelihood,
    initial_position: Mapping,
    realisations: np.ndarray,
    *,
    seed: int,
    num_chains_per_realisation: int = 1,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    target_acceptance_rate: float = 0.8,
    initial_position_jitter: float = 0.01,
) -> dict[str, float | int]:
    """Time one vmapped NUTS batch and return the per-posterior share.

    The first call compiles the vmapped kernel and is reported separately as
    ``compile_wall_seconds``; the campaign budget is the steady-state
    ``wall_seconds`` of the second, identically shaped call.
    """
    realisation_count = int(np.asarray(realisations).shape[0])

    def run():
        return sample_nuts_batched(
            likelihood.logdensity_for_strain,
            initial_position,
            jnp.asarray(realisations),
            seed=seed,
            num_chains_per_realisation=num_chains_per_realisation,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
            initial_position_jitter=initial_position_jitter,
        )

    started = time.perf_counter()
    jax.block_until_ready(run().samples)
    compile_wall_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = run()
    jax.block_until_ready(result.samples)
    wall_seconds = time.perf_counter() - started
    metadata = dict(result.metadata)
    return {
        "wall_seconds": wall_seconds,
        "compile_wall_seconds": compile_wall_seconds,
        "seconds_per_posterior": wall_seconds / realisation_count,
        "num_realisations": realisation_count,
        "num_chains_per_realisation": metadata["num_chains_per_realisation"],
        "chains": metadata["num_chains_per_realisation"],
        "num_warmup": metadata["num_warmup"],
        "num_samples_per_chain": metadata["num_samples_per_chain"],
        "integration_steps": metadata["integration_steps"],
        "divergences": metadata["divergences"],
    }


def measure_posterior(
    likelihood: RecalibrationLikelihood,
    initial_position: Mapping,
    *,
    seed: int,
    num_chains: int = 1,
    num_warmup: int = 1_000,
    num_samples: int = 1_000,
    target_acceptance_rate: float = 0.8,
    initial_position_jitter: float = 0.01,
) -> dict[str, float | int]:
    """Time the single-realisation NUTS path (the same-day CPU baseline)."""

    def run():
        return sample_nuts(
            likelihood.logdensity_fn,
            initial_position,
            seed=seed,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            target_acceptance_rate=target_acceptance_rate,
            initial_position_jitter=initial_position_jitter,
        )

    started = time.perf_counter()
    jax.block_until_ready(run().samples)
    compile_wall_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = run()
    jax.block_until_ready(result.samples)
    wall_seconds = time.perf_counter() - started
    return {
        "wall_seconds": wall_seconds,
        "compile_wall_seconds": compile_wall_seconds,
        "seconds_per_posterior": wall_seconds,
        "num_chains": num_chains,
        "chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples_per_chain": num_samples,
        "integration_steps": int(result.metadata["integration_steps"]),
        "divergences": int(result.metadata["divergences"]),
    }


__all__ = [
    "CPU_PLATFORM",
    "GPU_PLATFORM",
    "StageFunctions",
    "build_synthetic_likelihood",
    "campaign_gpu_hours",
    "compile_stages",
    "decompose_per_posterior",
    "devices_with_platform",
    "fresh_complex_array",
    "measure_batched_posterior",
    "measure_posterior",
    "measure_stages",
    "require_gpu_devices",
    "speedup",
    "summarise_seconds",
    "synthetic_parameters",
    "synthetic_realisations",
    "time_call",
]
