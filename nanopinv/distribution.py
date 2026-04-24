from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

from nanopinv._typing import Array, Float


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

    def __init__(
        self,
        L: Float[Array, "..."],
        mean: Float[Array, "..."],
        shape: tuple[int, ...],
    ):
        self.L = L
        self.mean = mean
        self.shape = shape

    @classmethod
    def from_covariance(
        cls,
        shape: tuple[int, ...],
        mean: Float[Array, "..."] | float,
        cov: Float[Array, "..."],
    ) -> Self:
        L = jax.scipy.linalg.cholesky(cov, lower=True)
        mean = jnp.asarray(mean).ravel()

        return cls(L=L, mean=mean, shape=shape)


    @classmethod
    def from_cholesky(
        cls,
        shape: tuple[int, ...],
        mean: Float[Array, "..."] | float,
        L: Float[Array, "..."],
    ) -> Self:
        mean = jnp.asarray(mean).ravel()

        return cls(L=L, mean=mean, shape=shape)

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


def log_likelihood_gaussian(data: Float[Array, "..."], data_obs: Float[Array, "..."], data_std: Float[Array, "..."]) -> Float[Array, ""]:
    """
    Compute the log-likelihood of observed data under a Gaussian noise model.
    """
    normalised_residual = (data_obs - data) / data_std
    log_likelihood = -0.5 * jnp.sum(normalised_residual**2)
    return jnp.sum(log_likelihood)
