from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from scipy import linalg as la

from nanopinv._typing import Array, Float, typecheck


class DistributionBase(eqx.Module):
    mean: Float[Array, "..."]

    @abstractmethod
    def __call__(
        self, key: jaxtyping.Key, num_samples: int | None = None
    ) -> Float[Array, "num_samples {self.shape}"]:
        pass


class MultivariateNormalCholesky(DistributionBase):
    shape: tuple[int, ...] = eqx.field(static=True)
    L: Float[Array, "..."]

    # @typecheck
    def __init__(
        self,
        shape: tuple[int, ...],
        mean: Float[Array, "..."] | float,
        cov: Float[Array, "..."],
    ):
        self.shape = shape
        self.L = jax.scipy.linalg.cholesky(cov, lower=True)
        self.mean = jnp.asarray(mean).ravel()

    @eqx.filter_jit
    def __call__(
        self, key: jaxtyping.Key, num_samples: int | None = None
    ) -> Float[Array, "num_samples {r.shape}"]:
        if num_samples is None:
            rvs_shape = (np.prod(self.shape),)
            result_shape = self.shape
        else:
            rvs_shape = (
                num_samples,
                np.prod(self.shape),
            )
            result_shape = (
                num_samples,
                *self.shape,
            )

        # Sample from standard normal
        z = jax.random.normal(
            key,
            shape=rvs_shape,
        )  # (num_samples, r.size)
        # Transform to match the covariance structure
        samples = self.mean + z @ self.L.T  # (num_samples, r.size)
        return jnp.reshape(samples, result_shape)
