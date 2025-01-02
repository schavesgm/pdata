"""Module containing implementations for diffusion noise-schedulers."""

from collections.abc import Callable

import jax.numpy as jnp

from .typing import Scalar

__all__ = [
    "NoiseScheduler",
    "DiffusionTerm",
    "compute_variance_exploding_std",
    "compute_variance_exploding_coefficient",
]


# Type defining a generic noise scheduler depending on some scalar parameter
type NoiseScheduler = Callable[[Scalar], Scalar]

# Type defining a generic diffusion coefficient scheduler depending on some scalar parameter
type DiffusionTerm = Callable[[Scalar], Scalar]


def compute_variance_exploding_std(
    time: Scalar, sigma_max: float = 1.0, sigma_min: float = 0.01
) -> Scalar:
    """Return the variance exploding standard deviation at a given time.

    Args:
        time (Scalar): Time at which the standard deviation is generated.
        sigma_max (float): Maximum value of the standard deviation.
        sigma_min (float): Minimum value of the standard deviation.

    Returns:
        Scalar: Standard deviation at the requested time.
    """
    return sigma_min * (sigma_max / sigma_min) ** time


def compute_variance_exploding_coefficient(
    time: Scalar, sigma_max: float = 1.0, sigma_min: float = 0.01
) -> Scalar:
    """Return the variance exploding diffusion coefficient at a given time.

    Args:
        time (Scalar): Time at which the standard deviation is generated.
        sigma_max (float): Maximum value of the standard deviation.
        sigma_min (float): Minimum value of the standard deviation.

    Returns:
        Scalar: Diffusion coefficient at the requested time.
    """
    std_part = compute_variance_exploding_std(time, sigma_max, sigma_min)
    log_part = jnp.sqrt(2.0 * jnp.log(sigma_max / sigma_min))
    return std_part * log_part
