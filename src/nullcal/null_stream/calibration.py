"""
Functions for including calibration factors into
the antenna response function.
"""

from __future__ import annotations

import jax

# Calibration inference is numerically unstable in JAX's default float32 mode.
# Keep this kernel consistent with nullcal.calibration, which establishes x64 as
# a process-wide requirement for the calibration path.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def compute_calibrated_whitened_antenna_response(
    whitened_antenna_response: np.ndarray, calibration_factor: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Compute the whitened antenna response function
    with the calibration factor included.

    Args:
        whitened_antenna_response (np.ndarray): Whitened antenna response function.
            Dimensions: (frequency, detector, polarization).
        calibration_factor (np.ndarray): Calibration factor.
            Dimensions: (detector, frequency).
        frequency_mask (np.ndarray): Frequency mask.
            Dimensions: (frequency,)

    Returns:
        np.ndarray: Calibrated antenna response function.
    """
    calibration_factor = jnp.asarray(calibration_factor)
    response = jnp.asarray(whitened_antenna_response, dtype=calibration_factor.dtype)
    mask = jnp.asarray(frequency_mask, dtype=bool)[:, None, None]
    calibrated = response * jnp.swapaxes(calibration_factor, 0, 1)[:, :, None]
    return jnp.where(mask, calibrated, jnp.zeros((), dtype=calibrated.dtype))
