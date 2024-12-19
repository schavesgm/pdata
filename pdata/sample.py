"""Module containing functionality to sample data from different 2-dimensional distributions."""

from collections.abc import Callable, Generator

import jax.numpy as jnp
import jax.random as jr

from .typing import Batched, Matrix, RandomKey, Vector

__all__ = [
    "Sampler",
    "generate_batches",
    "generate_dataset",
    "sample_from_checkerboard",
    "sample_from_gaussian",
    "sample_from_swiss_roll",
]

# Constants defining the standard normal mean and covariance matrix
STANDARD_NORMAL_MEAN: Vector = jnp.array([0.0, 0.0])
STANDARD_NORMAL_COV: Matrix = jnp.identity(2)

# Constants defining some transformations on points
FLIP_Y_AXIS: Vector = jnp.array([1.0, -1.0])
SCALE_Y_AXIS: Vector = jnp.array([0.0, 1.0])

# Type defining a simple sampler function
type Sampler = Callable[[RandomKey, int], Batched[Vector]]


def generate_dataset(
    rng_key: RandomKey,
    target_sampler: Sampler,
    num_entries: int,
    batch_size: int,
) -> list[Batched[Vector]]:
    """Return a dataset by sampling some data from a target and source distribution.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        target_sampler (Sampler): Sampler function to generate samples from the target distribution.
        num_entries (int): Number of entries in the dataset.
        batch_size (int): Number of elements per batch.

    Returns:
        list[Batched[Vector]]: List containing the dataset entries.
    """
    num_generated: int = 0
    dataset: list[Batched[Vector]] = []
    for targets in generate_batches(rng_key, target_sampler, batch_size):
        dataset.append(targets)
        num_generated += batch_size
        if num_generated >= num_entries:
            break

    return dataset


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

def sample_from_swiss_roll(
    rng_key: RandomKey, num_samples: int, noise_std: float = 0.05
) -> Batched[Vector]:
    """Return some samples from the swiss-roll dataset.

    Note:
        This function is inspired by the one in `sklearn
        <https://scikit-learn.org/1.5/modules/generated/sklearn.datasets.make_swiss_roll.html>`_.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        num_samples (int): Number of samples to generate.
        noise_std (float): Standard deviation of the noise added to the dataset.

    Returns:
        Batched[Vector]: Collection of points sampled from the swiss-roll distribution.
    """
    rng_key_1, rng_key_2 = jr.split(rng_key)

    # Generate the x-values to be used to generate the function
    x_values = (3.0 * jnp.pi / 3.0) * (1.0 + 2.0 * jr.uniform(rng_key_1, (num_samples, 1)))

    # Generate the y-values of the swiss roll and add some noise to them
    y_values = jnp.concatenate([x_values * jnp.cos(x_values), x_values * jnp.sin(x_values)], axis=1)
    y_values = y_values + noise_std * jr.normal(rng_key_2, (num_samples, 2))

    return y_values


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
