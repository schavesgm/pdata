"""Module containing functionality to implement flow-matching in 2-dimensions."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from pdata.typing import Batched, Scalar, Vector

__all__ = ["VectorFieldModel", "compute_optimal_transport_loss", "generate_trajectory"]


# Constant defining the default number of encoding dimensions of scalars
SCALAR_ENCODING_DIM: int = 32


class VectorFieldModel(nnx.Module):
    r"""Type defining a simple neural network vector field.

    Note:
        This module implements: :math:`v: [0, 1] \times \mathbb{R}^2 \to \mathbb{R}^2`.
    """

    def __init__(self, num_hidden_layers: int = 5, *, rngs: nnx.Rngs) -> None:
        """Instantiate a ``VectorFieldModel`` object.

        Args:
            num_hidden_layers (int): Number of hidden layers in the model.
            rngs (nnx.Rngs): Keys to use in the random engine in the model.
        """
        inner_layers = [
            nnx.Sequential(nnx.Linear(512, 512, rngs=rngs), nnx.leaky_relu)
            for _ in range(num_hidden_layers)
        ]

        self._layers = nnx.Sequential(
            nnx.Linear(2 + SCALAR_ENCODING_DIM, 512, rngs=rngs),
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
        inputs = jnp.concatenate([inputs, _encode_scalar(times, SCALAR_ENCODING_DIM)], axis=-1)
        return self._layers(inputs)


def compute_optimal_transport_loss(
    model: VectorFieldModel,
    target_samples: Batched[Vector],
    source_samples: Batched[Vector],
    time: Batched[Scalar],
    sigma_1: float = 0.001,
) -> Scalar:
    r"""Return the optimal transport conditional flow matching loss.

    Note:
        This function uses :math:`\sigma_1` to denote :math:`\sigma_{min}` in `[Lipman, 2023]
        <https://arxiv.org/abs/2210.02747>`_.

    Args:
        model (VectorFieldModel): Model used to match the target vector field.
        target_samples (Batched[Vector]): Samples from the target conditional distribution.
        source_samples (Batched[Vector]): Samples from the source conditional distribution.
        time (Batched[Scalar]): Time parameters at which the flow is evaluated.
        sigma_1 (float): Standard deviation of the end-point of the trajectory: :math:`p_1(x_1,
            \sigma_1^2 \mathbb{R})`.

    Returns:
        Scalar: Loss value between the learned vector field and the target vector field.
    """
    # Compute the conditional flow interpolated value
    pushed_samples = _compute_optimal_transport_flow(source_samples, target_samples, time, sigma_1)

    # Compute the prediction of the model on the pushed value
    vf_pred = model(pushed_samples, time)

    # Compute the target vector field to approximate
    vf_target = target_samples - (1.0 - sigma_1) * source_samples

    # NOTE: doing this yields better training than doing ``mean(linalg.norm(a - b, axis=-1))``
    return jnp.mean(jnp.square(vf_pred - vf_target))


def generate_trajectory(
    model: VectorFieldModel, source_samples: Batched[Vector], num_steps: int
) -> Batched[Vector]:
    """Return the flow-matching trajectory from source to data by integrating the FM-ODE.

    Note:
        This function uses the Euler method to integrate the trajectory.

    Args:
        model (VectorFieldModel): Vector field model used to instantiate the trajectory.
        source_samples (Batched[Vector]): Collection of initial points to use in the trajectory.
        num_steps (int): Number of steps to use in the integrator.

    Returns:
        Batched[Vector]: Integrated trajectory with dimensions ``(num_steps, num_samples, 2)``.
    """
    delta_step: float = 1.0 / num_steps
    num_samples: int = source_samples.shape[0]

    @jax.jit
    def _evaluate_ode(samples: Batched[Vector], time: Scalar) -> Batched[Vector]:
        """Return the evaluated ODE at a given time for a given set of samples."""
        return model(samples, time * jnp.ones((num_samples, 1))) * delta_step

    trajectory: list[Batched[Vector]] = [source_samples]
    for time in jnp.linspace(0.0, 1.0, num_steps):
        source_samples = source_samples + _evaluate_ode(source_samples, time)
        trajectory.append(source_samples)
    return jnp.stack(trajectory, axis=0)


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def _compute_optimal_transport_flow(
    sample: Vector, target_sample: Vector, time: Scalar, sigma_1: float
) -> Vector:
    r"""Compute the optimal transport conditional flow :math:`\psi_t(x_0)` between two Gaussians.

    Note:
        This function uses :math:`\sigma_1` to denote :math:`\sigma_{min}` in `[Lipman, 2023]
        <https://arxiv.org/abs/2210.02747>`_.

    Args:
        sample (Vector): Sample to push using the flow.
        target_sample (Vector): Sample in the target conditional distribution.
        time (Scalar): Time parameter at which the flow is evaluated.
        sigma_1 (float): Standard deviation of the end-point of the trajectory: :math:`p_1(x_1,
            \sigma_1^2 \mathbb{R})`.

    Returns:
        Vector: Result of applying the conditional flow at a given time.
    """
    return (1.0 - (1.0 - sigma_1) * time) * sample + time * target_sample


@partial(jax.vmap, in_axes=(0, None))
def _encode_scalar(scalar: Scalar, encoding_dim: int) -> Vector:
    """Return an encoded version of an scalar using frequencies.

    Args:
        scalar (Scalar): Scalar to encode.
        encoding_dim (int): Total number of dimensions of the output vector.

    Returns:
        Vector: Encoded version of the input scalar.
    """
    frequencies = 2.0 * jnp.arange(encoding_dim // 2) * jnp.pi * scalar
    return jnp.concatenate([jnp.sin(frequencies), jnp.cos(frequencies)], axis=-1)
