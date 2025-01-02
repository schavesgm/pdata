"""Module containing functionality to sample time variables using different schedulers."""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr

from pdata.typing import Batched, RandomKey, Scalar

__all__ = ["TimeSampler", "sample_time_uniform", "sample_time_flow_matching"]

# Protocol defining a generic time sampler
type TimeSampler = Callable[[RandomKey, int], Batched[Scalar]]


def sample_time_uniform(
    rng_key: RandomKey, num_samples: int, min_value: float = 0.0
) -> Batched[Scalar]:
    """Return some sampled time variables uniformly sampled in :math:`[0, 1]`.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        num_samples (int): Number of samples to generate.
        min_value (float): Minimum value allowed.

    Returns:
        Batched[Scalar]: Sampled times to generate.
    """
    return jr.uniform(rng_key, (num_samples,), minval=min_value, maxval=1.0)


def sample_time_flow_matching(
    rng_key: RandomKey, num_samples: int, epsilon: float = 1e-5
) -> Batched[Scalar]:
    """Return some sampled time variables uniformly sampled in :math:`[0, 1]`.

    Note:
        This function implements the time-sampler from the unofficial flow-matching implementation
        found `here <https://github.com/gle-bellier/flow-matching>`_.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        num_samples (int): Number of samples to generate.
        epsilon (float): Value defining the maximum available sampled value, ``1.0 - epsilon``.

    Returns:
        Batched[Scalar]: Sampled times to generate.
    """
    starting_time = jr.uniform(rng_key, (1,))
    linspace_time = jnp.linspace(0.0, 1.0, num_samples)
    return (starting_time + linspace_time) % (1.0 - epsilon)
    return jr.uniform(rng_key, (num_samples,))
