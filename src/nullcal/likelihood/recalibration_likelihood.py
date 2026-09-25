"""Pure JAX log density for null-stream recalibration inference."""

from __future__ import annotations

from collections.abc import Mapping

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ..calibration import MINIMUM_CUBIC_SPLINE_KNOTS, calibration_factor, calibration_log_prior  # noqa: E402
from ..data import InterferometerData  # noqa: E402
from ..null_stream.whiten import (  # noqa: E402
    compute_whitened_antenna_response,
    compute_whitened_frequency_domain_strain,
)
from ..time_frequency_transform.transform_freq_funcs import (  # noqa: E402
    phitilde_vec_norm,
    transform_wavelet_freq_helper,
)
from ..time_frequency_transform.wavelet_transforms import WaveletTransform  # noqa: E402

PARAMETER_NAMES = frozenset({"amplitude", "phase"})
PARAMETER_ARRAY_NDIM = 2
ET_BEAM_PATTERN = np.array(
    [[-1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)], [np.sqrt(6.0) / 3.0, 0.0], [-1.0 / np.sqrt(6.0), 1.0 / np.sqrt(2.0)]]
)


def _broadcast_prior(value, shape: tuple[int, int], name: str) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.float64)
    try:
        return jnp.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or broadcast to detector-by-knot shape {shape}") from error


def _validated_knots(knot_frequencies, detector_count: int) -> np.ndarray:
    knots = np.asarray(knot_frequencies, dtype=np.float64)
    if knots.ndim == 1:
        knots = np.broadcast_to(knots, (detector_count, knots.size)).copy()
    if knots.ndim != PARAMETER_ARRAY_NDIM or knots.shape[0] != detector_count:
        raise ValueError("knot_frequencies must have shape (knot,) or (detector, knot)")
    if knots.shape[1] < MINIMUM_CUBIC_SPLINE_KNOTS:
        raise ValueError("a cubic spline requires at least four knots")
    if np.any(knots <= 0.0) or np.any(np.diff(knots, axis=1) <= 0.0):
        raise ValueError("knot_frequencies must be positive and strictly increasing")
    return knots


def _whitened_inputs(interferometers, shared_mask, supplied_whitened_strain):
    delta_f = 1.0 / float(interferometers.duration)
    psd = np.asarray(interferometers.psd)
    whitened_response = compute_whitened_antenna_response(ET_BEAM_PATTERN, psd, delta_f, shared_mask)
    if supplied_whitened_strain is None:
        whitened_strain = compute_whitened_frequency_domain_strain(
            np.asarray(interferometers.strain), psd, delta_f, shared_mask
        )
    else:
        whitened_strain = np.asarray(supplied_whitened_strain)
        if whitened_strain.shape != psd.shape:
            raise ValueError("whitened_frequency_domain_strain must match the detector data shape")
        if not np.all(whitened_strain[:, ~shared_mask] == 0.0):
            raise ValueError("whitened_frequency_domain_strain must be zero outside the shared mask")
    return psd, whitened_response, whitened_strain


class RecalibrationLikelihood:
    """Immutable-data recalibration posterior with a pure ``logdensity_fn``.

    One instance represents one detector-network realization. Parameters are a
    pytree with ``amplitude`` and ``phase`` arrays of shape ``(detector, knot)``.
    Knot frequencies and Gaussian prior hyperparameters are fixed model data,
    not sampled coordinates.
    """

    def __init__(
        self,
        interferometers: InterferometerData,
        knot_frequencies,
        *,
        time_frequency_filter: np.ndarray,
        wavelet_transform_frequency_resolution: float = 4.0,
        wavelet_transform_nx: float = 4.0,
        amplitude_prior_mean=0.0,
        amplitude_prior_sigma=0.05,
        phase_prior_mean=0.0,
        phase_prior_sigma=0.05,
        whitened_frequency_domain_strain=None,
    ) -> None:
        if not isinstance(interferometers, InterferometerData):
            raise TypeError("interferometers must be an InterferometerData instance")
        if len(interferometers) != ET_BEAM_PATTERN.shape[0]:
            raise ValueError("the recalibration likelihood currently requires the three-detector ET triangle")

        transform = WaveletTransform(
            duration=float(interferometers.duration),
            sampling_frequency=float(interferometers.sampling_frequency),
            frequency_resolution=wavelet_transform_frequency_resolution,
            nx=wavelet_transform_nx,
        )
        time_frequency_filter = np.asarray(time_frequency_filter, dtype=bool)
        if time_frequency_filter.shape != transform.shape:
            raise ValueError(
                f"time_frequency_filter has shape {time_frequency_filter.shape}; expected {transform.shape}"
            )

        shared_mask = np.all(np.asarray(interferometers.mask, dtype=bool), axis=0)
        frequency_indices = np.flatnonzero(shared_mask)
        if frequency_indices.size == 0:
            raise ValueError("the shared detector frequency mask must not be empty")
        masked_frequencies = np.asarray(interferometers.frequency_array)[frequency_indices]
        if np.any(masked_frequencies <= 0.0):
            raise ValueError("calibration frequencies must be positive")

        knots = _validated_knots(knot_frequencies, len(interferometers))
        psd, whitened_response, whitened_strain = _whitened_inputs(
            interferometers, shared_mask, whitened_frequency_domain_strain
        )

        parameter_shape = knots.shape
        self.interferometers = interferometers
        self.knot_frequencies = jnp.asarray(knots)
        self.time_frequency_filter = jnp.asarray(time_frequency_filter)
        self.time_frequency_transform = transform
        self.amplitude_prior_mean = _broadcast_prior(amplitude_prior_mean, parameter_shape, "amplitude_prior_mean")
        self.amplitude_prior_sigma = _broadcast_prior(amplitude_prior_sigma, parameter_shape, "amplitude_prior_sigma")
        self.phase_prior_mean = _broadcast_prior(phase_prior_mean, parameter_shape, "phase_prior_mean")
        self.phase_prior_sigma = _broadcast_prior(phase_prior_sigma, parameter_shape, "phase_prior_sigma")
        if np.any(np.asarray(self.amplitude_prior_sigma) <= 0.0) or np.any(np.asarray(self.phase_prior_sigma) <= 0.0):
            raise ValueError("prior standard deviations must be positive")

        self._frequency_indices = jnp.asarray(frequency_indices)
        self._masked_frequencies = jnp.asarray(masked_frequencies)
        self._whitened_antenna_response = jnp.asarray(whitened_response[frequency_indices])
        self._whitened_frequency_domain_strain = jnp.asarray(whitened_strain[:, frequency_indices])
        self._frequency_count = psd.shape[1]
        n_time, n_frequency = transform.shape
        self._wavelet_n_time = n_time
        self._wavelet_n_frequency = n_frequency
        self._wavelet_window = (2.0 / n_frequency) * phitilde_vec_norm(n_frequency, n_time, wavelet_transform_nx)
        self._wavelet_scale = jnp.sqrt((self._frequency_count - 1) * 2.0)

    @property
    def parameter_shape(self) -> tuple[int, int]:
        """Required shape of each parameter array."""
        return self.knot_frequencies.shape

    def _validated_parameters(self, params: Mapping[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        if not isinstance(params, Mapping) or set(params) != PARAMETER_NAMES:
            raise ValueError("params must contain exactly amplitude and phase")
        amplitude = jnp.asarray(params["amplitude"], dtype=jnp.float64)
        phase = jnp.asarray(params["phase"], dtype=jnp.float64)
        if amplitude.shape != self.parameter_shape or phase.shape != self.parameter_shape:
            raise ValueError(
                f"amplitude and phase must each have shape {self.parameter_shape}; "
                f"received {amplitude.shape} and {phase.shape}"
            )
        return amplitude, phase

    def _calibration_factor(self, amplitude: jax.Array, phase: jax.Array) -> jax.Array:
        return jax.vmap(calibration_factor, in_axes=(None, 0, 0, 0))(
            self._masked_frequencies,
            self.knot_frequencies,
            amplitude,
            phase,
        )

    def _validated_strain(self, whitened_strain) -> jax.Array:
        strain = jnp.asarray(whitened_strain)
        expected_shape = (self.parameter_shape[0], self._frequency_count)
        if tuple(strain.shape[-2:]) != expected_shape:
            raise ValueError(
                f"whitened_frequency_domain_strain must have trailing shape {expected_shape}; "
                f"received {tuple(strain.shape)}"
            )
        return strain

    def _calibrated_projector(self, amplitude: jax.Array, phase: jax.Array) -> jax.Array:
        """Build the per-frequency null projector for the calibrated response."""
        factors = self._calibration_factor(amplitude, phase)
        response = self._whitened_antenna_response * jnp.swapaxes(factors, 0, 1)[:, :, None]
        response_dagger = jnp.swapaxes(jnp.conj(response), 1, 2)
        gram = response_dagger @ response
        projected_response = response @ jnp.linalg.solve(gram, response_dagger)
        return jnp.eye(response.shape[1], dtype=response.dtype)[None, :, :] - projected_response

    def _project(self, projector: jax.Array, selected_strain: jax.Array) -> jax.Array:
        """Apply the null projector to a masked frequency-domain strain."""
        return jnp.einsum("fij,...jf->...if", projector, selected_strain)

    def _frequency_domain_null_stream(
        self, amplitude: jax.Array, phase: jax.Array, whitened_strain: jax.Array | None = None
    ) -> jax.Array:
        if whitened_strain is None:
            selected_strain = self._whitened_frequency_domain_strain
            realisation_shape = selected_strain.shape[:-1]
        else:
            strain = self._validated_strain(whitened_strain)
            selected_strain = strain[..., self._frequency_indices]
            realisation_shape = strain.shape[:-1]
        projector = self._calibrated_projector(amplitude, phase)
        masked_null_stream = self._project(projector, selected_strain)
        full_shape = (*realisation_shape, self._frequency_count)
        return (
            jnp.zeros(full_shape, dtype=masked_null_stream.dtype)
            .at[..., self._frequency_indices]
            .set(masked_null_stream)
        )

    def _transform_to_time_frequency(self, frequency_domain: jax.Array) -> jax.Array:
        """Map a ``(..., detector, frequency)`` array through the WDM transform."""

        def transform(detector_data):
            return (
                transform_wavelet_freq_helper(
                    detector_data,
                    self._wavelet_n_frequency,
                    self._wavelet_n_time,
                    self._wavelet_window,
                )
                * self._wavelet_scale
            )

        leading_shape = frequency_domain.shape[:-1]
        flattened = frequency_domain.reshape((-1, frequency_domain.shape[-1]))
        transformed = jax.vmap(transform)(flattened)
        return transformed.reshape(leading_shape + transformed.shape[-2:])

    def _time_frequency_null_stream(
        self, amplitude: jax.Array, phase: jax.Array, whitened_strain: jax.Array | None = None
    ) -> jax.Array:
        frequency_domain = self._frequency_domain_null_stream(amplitude, phase, whitened_strain)
        return jnp.where(self.time_frequency_filter, self._transform_to_time_frequency(frequency_domain), 0.0)

    def _log_likelihood_from_stream(self, null_stream: jax.Array) -> jax.Array:
        # The detector axis always sits immediately before the time and
        # frequency axes, so summing the trailing three leaves exactly one
        # value per leading realisation (and a scalar for a single realisation).
        return -0.5 * jnp.sum(jnp.abs(null_stream) ** 2, axis=(-1, -2, -3))

    def log_likelihood_fn(self, params: Mapping[str, jax.Array]) -> jax.Array:
        """Return the pure JAX null-stream log likelihood for ``params``."""
        amplitude, phase = self._validated_parameters(params)
        return self._log_likelihood_from_stream(self._time_frequency_null_stream(amplitude, phase))

    def log_likelihood_for_strain(self, params: Mapping[str, jax.Array], whitened_frequency_domain_strain) -> jax.Array:
        """Evaluate the likelihood on a supplied whitened strain realisation.

        ``whitened_frequency_domain_strain`` has trailing shape
        ``(detector, frequency)`` and may carry any number of leading
        realisation axes. The return value drops those leading axes and keeps
        one log likelihood per realisation, so a ``vmap`` over the leading axis
        yields one posterior target per realisation.
        """
        amplitude, phase = self._validated_parameters(params)
        null_stream = self._time_frequency_null_stream(amplitude, phase, whitened_frequency_domain_strain)
        return self._log_likelihood_from_stream(null_stream)

    def logdensity_for_strain(self, params: Mapping[str, jax.Array], whitened_frequency_domain_strain) -> jax.Array:
        """Return log prior plus the per-realisation log likelihood on supplied strain."""
        amplitude, phase = self._validated_parameters(params)
        return self.log_likelihood_for_strain(params, whitened_frequency_domain_strain) + calibration_log_prior(
            amplitude,
            phase,
            self.amplitude_prior_mean,
            self.amplitude_prior_sigma,
            self.phase_prior_mean,
            self.phase_prior_sigma,
        )

    def logdensity_fn(self, params: Mapping[str, jax.Array]) -> jax.Array:
        """Return normalized Gaussian log prior plus null-stream log likelihood."""
        amplitude, phase = self._validated_parameters(params)
        return self.log_likelihood_fn(params) + calibration_log_prior(
            amplitude,
            phase,
            self.amplitude_prior_mean,
            self.amplitude_prior_sigma,
            self.phase_prior_mean,
            self.phase_prior_sigma,
        )

    def noise_log_likelihood(self) -> jax.Array:
        """Return the likelihood for an exactly uncalibrated response."""
        zeros = jnp.zeros(self.parameter_shape, dtype=jnp.float64)
        return self.log_likelihood_fn({"amplitude": zeros, "phase": zeros})


__all__ = ["RecalibrationLikelihood"]
