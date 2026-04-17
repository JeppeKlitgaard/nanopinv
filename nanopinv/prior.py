import equinox as eqx
import numpy as np
from gstools.covmodel import CovModel
from scipy import linalg as la

from nanopinv._typing import Array, Float, typecheck


class PriorBase:
    def __init__(self, **kwargs):
        pass


class CholeskyPrior(PriorBase):
    @typecheck
    def __init__(
        self,
        r: Float[Array, "*dims"],
        m0: Float[Array, "{r.ndim}"],
        cov: CovModel | Float[Array, "{r.size} {r.size}"],
        rng: None | np.random.Generator = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        try:
            L = la.cholesky(cov, lower=True, check_finite=True)
        except la.LinAlgError as e:
            raise ValueError("Covariance matrix is not positive definite") from e

        self.L = L
        self.m0 = m0

    # def __call__()

    # self.L = L
