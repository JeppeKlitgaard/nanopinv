"""
This module contains covariance kernels (spatial covariance functions)
and related utilities.

Note that these are not implemented in JAX due to a lack of support for `scipy.spatial` in JAX
and given they are only calculated once during setup, their performance is not critical.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import pdist, squareform

from nanopinv._typing import Array, Float


def get_distance_matrix(*r: Sequence[Float[Array, "ax"]]) -> Float[Array, "a b"]:
    """
    Compute the pairwise distance matrix for a set of points.

    Parameters:
    r (Sequence[Float[Array, "ax"]]): A sequence of arrays representing the coordinates of the points.

    Returns:
    Float[Array, "a b"]: A 2D array containing the pairwise distances between the points.
    """
    R = jnp.meshgrid(*r, indexing="ij")
    points = jnp.column_stack([R_i.ravel() for R_i in R])
    h = squareform(pdist(points))
    return h


@jax.jit
def _add_nugget(cov: Float[Array, "a b"], nugget: float) -> Float[Array, "a b"]:
    return cov + nugget * jnp.eye(cov.shape[0])


@jax.jit
def spherical(
    h: Float[Array, "a b"],
    range_: float,
    partial_sill: float,
    nugget: float,
):
    """
    Spherical covariance kernel.

    Parameters:
    h (Float[Array, "a b"]): A 2D array containing the pairwise distances between the points.
    range_ (float): The range parameter of the spherical model.
    partial_sill (float): The partial sill (variance) of the spherical model.
    nugget (float): The nugget effect (small-scale variance) of the spherical model.

    Returns:
    Float[Array, "a b"]: A 2D array containing the covariance values for each pair of points.
    """
    cov = jnp.where(
        h < range_,
        partial_sill * (1 - 1.5 * (h / range_) + 0.5 * (h / range_) ** 3),
        0.0,
    )

    cov = _add_nugget(cov, nugget)
    return cov
