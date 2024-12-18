"""Module containing functionality to implement flow-matching in 2-dimensions."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from pdata.typing import Batched, Scalar, Vector

__all__ = ["VectorFieldModel"]


# Constant defining the default number of encoding dimensions of scalars
SCALAR_ENCODING_DIM: int = 32


class VectorFieldModel(nnx.Module):
    r"""Type defining a simple neural network vector field.

    Note:
        This module implements: :math:`v: [0, 1] \times \mathbb{R}^2 \to \mathbb{R}^2`.
    """

    def __init__(self, num_hidden_layers: int = 4, *, rngs: nnx.Rngs) -> None:
        """Instantiate a ``VectorFieldModel`` object.

        Args:
            num_hidden_layers (int): Number of hidden layers in the model.
            rngs (nnx.Rngs): Keys to use in the random engine in the model.
        """
        inner_layers = [
            nnx.Sequential(nnx.Linear(256, 256, rngs=rngs), nnx.relu)
            for _ in range(num_hidden_layers)
        ]

        self._layers = nnx.Sequential(
            nnx.Linear(2 + SCALAR_ENCODING_DIM, 256, rngs=rngs),
            nnx.relu,
            *inner_layers,
            nnx.Linear(256, 2, rngs=rngs),
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
