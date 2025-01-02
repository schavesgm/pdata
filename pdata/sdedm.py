"""Module containing functionality to implement SDE score-matching diffusion models."""

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from .encode import DEFAULT_SCALAR_ENCODING_DIM, encode_scalar
from .noise import DiffusionTerm, NoiseScheduler
from .typing import Batched, RandomKey, Scalar, Vector

# Vectorised version of the scalar encoding function
_encode_time = jax.vmap(encode_scalar, in_axes=(0, None))

__all__ = ["ScoreModel", "compute_sde_loss", "generate_trajectory"]


class ScoreModel(nnx.Module):
    r"""Type defining a simple neural network score model.

    Note:
        This module implements: :math:`v: [0, 1] \times \mathbb{R}^2 \to \mathbb{R}^2`.
    """

    def __init__(self, num_hidden_layers: int = 5, *, rngs: nnx.Rngs) -> None:
        """Instantiate a ``ScoreModel`` object.

        Args:
            num_hidden_layers (int): Number of hidden layers in the model.
            rngs (nnx.Rngs): Keys to use in the random engine in the model.
        """
        inner_layers = [
            nnx.Sequential(nnx.Linear(512, 512, rngs=rngs), nnx.leaky_relu)
            for _ in range(num_hidden_layers)
        ]

        self._layers = nnx.Sequential(
            nnx.Linear(2 + DEFAULT_SCALAR_ENCODING_DIM, 512, rngs=rngs),
            nnx.leaky_relu,
            *inner_layers,
            nnx.Linear(512, 2, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(2, 2, rngs=rngs),
        )

    def __call__(self, inputs: Batched[Vector], times: Batched[Scalar]) -> Batched[Vector]:
        """Return the output of the model given some inputs.

        Args:
            inputs (Batched[Vector]): Input vectors to process by the network.
            times (Batched[Scalar]): Time parameter at which each input is processed.

        Returns:
            Batched[Vector]: Transformed inputs.
        """
        return self._layers(
            jnp.concatenate([inputs, _encode_time(times, DEFAULT_SCALAR_ENCODING_DIM)], axis=-1)
        )


def compute_sde_loss(
    score_model: ScoreModel,
    target_samples: Batched[Vector],
    epsilon: Batched[Vector],
    times: Batched[Scalar],
    noise_scheduler: NoiseScheduler,
) -> Scalar:
    """Compute the score-matching SDE denoising loss.

    Args:
        score_model (ScoreModel): Neural network acting as the score.
        target_samples (Batched[Vector]): Samples of the target distribution.
        epsilon (Batched[Vector]): Noise values to use in the diffusion process.
        times (Batched[Scalar]): Time values at which the diffusion process is evaluated.
        noise_scheduler (NoiseScheduler): Scheduler used to control the standard deviation of the
            process.

    Returns:
        Scalar: Diffusion score-matching loss.
    """
    # Compute the standard deviation using the noise scheduler
    transition_std = noise_scheduler(times)[:, None]

    # Perturb the data with some noise -> sampled from p_{0t}
    perturbed_target = target_samples + epsilon * transition_std

    # Compute the predicted score
    pred_score = score_model(perturbed_target, times)

    # Compute the loss function. Note that the score of the Gaussian is ``- (x - \mu) / \sigma^2``,
    # where ``x = x_0 + \epsilon \sigma(t)``. As the perturbation kernel is ``N(x_0, \sigma(t))``,
    # we find that the loss is just ``s_\theta + \epsilon / \sigma(t)``, which can be transformed
    # in ``\sigma(t) s_\theta + \epsilon``.
    return jnp.mean(jnp.sum(jnp.square(pred_score * transition_std + epsilon), axis=1))


def generate_trajectory(
    rng_key: RandomKey,
    model: ScoreModel,
    source_samples: Batched[Vector],
    num_steps: int,
    diffusion_coefficient: DiffusionTerm,
    minimum_time: float = 1e-3,
) -> Batched[Vector]:
    """Generate an SDE trajectory using the Euler-Murayama method.

    Args:
        rng_key (RandomKey): Key to use in the random engine.
        model (ScoreModel): Model used to match the score of the marginal distribution.
        source_samples (Batched[Vector]): Samples generating the initial point of the trajectory.
        num_steps (int): Number of steps in the trajectory.
        diffusion_coefficient (DiffusionTerm): Function scheduling diffusion coefficients.
        minimum_time (float): Minimum time to use in the trajectory.

    Returns:
        Batched[Vector]: Collection of vectors in the SDE trajectory.
    """
    delta_step: float = 1.0 / num_steps
    num_samples: int = source_samples.shape[0]

    @jax.jit
    def _sde_update(rng_key: RandomKey, samples: Batched[Vector], time: Scalar) -> Batched[Vector]:
        """Return the updated SDE values at a given point in the trajectory using Euler-Murayama."""
        times = time * jnp.ones((num_samples, 1))
        diffusion_coeff = diffusion_coefficient(times)
        mean_sde = samples + (diffusion_coeff**2) * model(samples, times) * delta_step
        stochastic = jnp.sqrt(delta_step) * diffusion_coeff * jr.normal(rng_key, (num_samples, 1))
        return mean_sde + stochastic

    trajectory: list[Batched[Vector]] = [source_samples]
    for time in jnp.linspace(1.0, minimum_time, num_steps - 1):
        rng_key, sub_key = jr.split(rng_key)
        source_samples = _sde_update(sub_key, source_samples, time)
        trajectory.append(source_samples)
    return jnp.stack(trajectory, axis=0)
