from collections.abc import Callable
from copy import deepcopy

import gstools as gs
import numpy as np

from nanopinv._typing import Array, Float, NDGrid, typecheck


class BaseProposalDistribution: ...


class SRFProposalDistribution(BaseProposalDistribution):
    # @typecheck
    def __init__(
        self,
        srf: gs.SRF,
        r: NDGrid,
        step_size: float,
        rng: np.random.Generator,
    ):
        self.srf = srf
        self.mean = srf.mean

        self.r = r
        self.rng = rng
        self.step_size = step_size

    def propose(
        self,
        state_current: Float[
            Array, "{', '.join([f'ax_{i}' for i in range(len(self.r))])}"
        ]
        | None,
        step_size: float | None = None,
        seed: int | None = None,
    ):
        step_size = step_size if step_size is not None else self.step_size
        seed: int = seed if seed is not None else self.rng.integers(0, 2**31 - 1)

        # Implement the logic to propose a new state based on the current state
        new_realisation = self.srf(
            pos=self.r, mesh_type="structured", seed=seed, store=False
        )

        if state_current is None:
            return new_realisation

        # Do Precondition Crank-Nicolson to combine the current state and the new realisation
        state_proposal = (
            self.mean
            + np.sqrt(1.0 - step_size**2) * (state_current - self.mean)
            + step_size * (new_realisation - self.mean)
        )

        return state_proposal


class BaseSampler: ...


class ExtendedMetropolisSampler(BaseSampler):
    def __init__(
        self,
        n_chains: int,
        n_burn_in: int | list[int],
        n_samples: int,
        temperatures: float | list[float] = 1.0,
        # forward_model:
    ):
        """
        Extended Metropolis sampler.

        Args:
            n_chains: Number of parallel chains to run.
            n_burn_in: Number of burn-in iterations (can be a list for each chain).
            n_samples: Number of samples to draw after burn-in.
            temperatures: Temperature(s) for the chains
                If it is a single float of value `1.0`, this is the regular case without parallel tempering.
                If it is a list of floats, each chain will be assigned a temperature from the list.
        """
