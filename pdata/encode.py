"""Module containing functionality to encode objects into higher-dimensional spaces."""

import jax.numpy as jnp

from .typing import Scalar, Vector

__all__ = ["DEFAULT_SCALAR_ENCODING_DIM", "encode_scalar"]


# Constant defining the default number of encoding dimensions of scalars
DEFAULT_SCALAR_ENCODING_DIM: int = 32


def encode_scalar(scalar: Scalar, encoding_dim: int) -> Vector:
    """Return an encoded version of an scalar using frequencies.

    Args:
        scalar (Scalar): Scalar to encode.
        encoding_dim (int): Total number of dimensions of the output vector.

    Returns:
        Vector: Encoded version of the input scalar.
    """
    frequencies = 2.0 * jnp.arange(encoding_dim // 2) * jnp.pi * scalar
    return jnp.concatenate([jnp.sin(frequencies), jnp.cos(frequencies)], axis=-1)
