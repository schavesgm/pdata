"""Module containing some type-hints used across the package."""

from typing import Annotated

import jax

__all__ = ["Batched", "Matrix", "RandomKey", "Scalar", "Vector"]


# Type annotation for types that are batched over the leading dimension
type Batched[T: jax.Array] = Annotated[T, tuple[int, ...]]

# Core types used for simple arrays
type Scalar = Annotated[jax.Array, tuple[()]]
type Vector = Annotated[jax.Array, tuple[int]]
type Matrix = Annotated[jax.Array, tuple[int, int]]

# Type for JAX random key
type RandomKey = Annotated[jax.Array, "JAX random key"]
