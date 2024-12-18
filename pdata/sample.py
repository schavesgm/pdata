"""Module containing functionality to sample data from different 2-dimensional distributions."""

from collections.abc import Callable, Generator

import jax.numpy as jnp
import jax.random as jr

from .typing import Batched, Matrix, RandomKey, Vector

__all__ = ["Sampler", "generate_batches", "sample_from_checkerboard", "sample_from_gaussian"]

# Constants defining the standard normal mean and covariance matrix
STANDARD_NORMAL_MEAN: Vector = jnp.array([0.0, 0.0])
STANDARD_NORMAL_COV: Matrix = jnp.identity(2)

# Constants defining some transformations on points
FLIP_Y_AXIS: Vector = jnp.array([1.0, -1.0])
SCALE_Y_AXIS: Vector = jnp.array([0.0, 1.0])

# Type defining a simple sampler function
type Sampler = Callable[[RandomKey, int], Batched[Vector]]


def generate_batches(
    rng_key: RandomKey, sampler: Sampler, batch_size: int
) -> Generator[Batched[Vector], None, None]:
    """Return a generator over batches sampled from a given distribution.

    Warning:
        ⚠️ This function produces a generator that never exhausts. Do not try to exhaust it.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        sampler (Sampler): Sampler function generating the batch according to a given distribution.
        batch_size (int): Number of samples to generate per batch.

    Returns:
        Generator[Batch[Vector], None, None]: Generator producing batches of data.
    """
    while True:
        rng_key, sub_key = jr.split(rng_key)
        yield sampler(sub_key, batch_size)


def sample_from_gaussian(
    rng_key: RandomKey,
    num_samples: int,
    mean: Vector = STANDARD_NORMAL_MEAN,
    covariance: Matrix = STANDARD_NORMAL_COV,
) -> Batched[Vector]:
    """Return some samples from the 2-dimensional Gaussian distribution.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        num_samples (int): Number of samples to generate.
        mean (Vector): Mean of the 2-dimensional Gaussian distribution.
        covariance (Matrix): Covariance of the 2-dimensional Gaussian distribution.

    Returns:
        Batched[Vector]: Collection of samples from the 2-dimensional Gaussian distribution.
    """
    return jr.multivariate_normal(rng_key, mean, covariance, (num_samples,))


def sample_from_checkerboard(
    rng_key: RandomKey, num_samples: int, num_squares: int = 2
) -> Batched[Vector]:
    """Return some samples from the checkerboard distribution.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        num_samples (int): Number of samples to generate.
        num_squares (int): Number of "black" squares in each direction. A value of ``num_squares=2``
            will create a checkerboard with ``4`` squares in each direction.

    Returns:
        Batched[Vector]: Collection of samples from the checkerboard distribution.
    """
    # Generate the frequency used in the checkerboard pattern
    frequency: float = num_squares * jnp.pi
    min_frequency: float = -frequency
    max_frequency: float = frequency

    # Generate some uniformly distributed points in the interval
    points = jr.uniform(rng_key, (num_samples, 2), minval=min_frequency, maxval=max_frequency)

    # Generate the mask selecting the points in the correct quadrant
    is_checkerboard = jnp.logical_or(
        jnp.logical_and(jnp.sin(points[:, 0]) > 0.0, jnp.sin(points[:, 1]) > 0.0),
        jnp.logical_and(jnp.sin(points[:, 0]) < 0.0, jnp.sin(points[:, 1]) < 0.0),
    )

    # Transform the points to the [0, 1] X [0, 1] area
    points = (points - min_frequency) / (max_frequency - min_frequency)

    # Use the Y-symmetry to transform all points out-of-distribution to the pattern
    return jnp.where(is_checkerboard[:, None], points, FLIP_Y_AXIS * points + SCALE_Y_AXIS)

