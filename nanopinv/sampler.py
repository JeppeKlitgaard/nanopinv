import math
from collections.abc import Callable
from functools import partial
from types import EllipsisType
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce, repeat
from jax_tqdm import PBar, scan_tqdm
from statsmodels.tsa.stattools import acf

from nanopinv._typing import Array, Bool, Float, Int, Key, Shaped
from nanopinv.distribution import DistributionBase
from nanopinv.types import ObservationsT
from nanopinv.utils import make_pytree_spec

_DEFAULT_JAX_TQDM_KWARGS = {
    "print_rate": 50,
    "tqdm_type": "auto",
}


def _resolve_scan_tqdm_fn(
    scan_fn: Callable,
    *,
    progress: bool,
    total_steps: int,
    desc: str,
    jax_tqdm_kwargs: dict[str, Any] | None,
) -> Callable:
    if not progress:
        return scan_fn

    tqdm_kwargs = {} if jax_tqdm_kwargs is None else jax_tqdm_kwargs
    print_rate = tqdm_kwargs.get(
        "print_rate", _DEFAULT_JAX_TQDM_KWARGS.get("print_rate", 50)
    )
    print_rate = min(print_rate, total_steps) if total_steps > 0 else 1

    tqdm_kwargs = (
        _DEFAULT_JAX_TQDM_KWARGS
        | {"desc": desc, "print_rate": print_rate}
        | tqdm_kwargs
    )
    return scan_tqdm(total_steps, **tqdm_kwargs)(scan_fn)


def _scan_with_optional_progress(
    scan_fn: Callable,
    init_carry: Any,
    scan_inputs: Any,
    *,
    progress: bool,
):
    if progress:
        wrapped_init = PBar(id=0, carry=init_carry)
        final_pbar, history = jax.lax.scan(scan_fn, wrapped_init, scan_inputs)
        return final_pbar.carry, history

    return jax.lax.scan(scan_fn, init_carry, scan_inputs)


class ProposalDistribution(eqx.Module):
    dist: DistributionBase
    step_size: Float[Array, ""]
    mean: Float[Array, "..."]

    def __init__(
        self,
        dist: DistributionBase,
        step_size: Float[Array, ""] | float,
    ):
        self.dist = dist
        self.step_size = jnp.asarray(step_size)
        self.mean = dist.mean

    def propose(
        self,
        state_current: Float[Array, "..."],
        key: Key,
        step_size: Float[Array, ""] | None = None,
    ):
        if step_size is None:
            step_size = self.step_size

        new_realisation = self.dist(key)
        state_proposal = (
            self.mean
            + jnp.sqrt(1.0 - step_size**2) * (state_current - self.mean)
            + step_size * (new_realisation - self.mean)
        )
        return state_proposal


class IterationState(eqx.Module):
    state: Float[Array, "*grid"]
    log_likelihood: Float[Array, ""]


def initialize_betas(
    n_chains: int, base: float = 1.2, last_is_zero: bool = True
) -> Float[Array, "n_chains"]:
    exponents = -jnp.arange(n_chains)
    betas = jnp.power(base, exponents)

    if last_is_zero:
        return jnp.append(betas[:-1], 0.0)

    return betas


def take_first(tensor, axis):
    # Sort in reverse to prevent axis shifting as dimensions are dropped
    for ax in sorted(axis, reverse=True):
        tensor = jnp.take(tensor, 0, axis=ax)

    return tensor


class History(eqx.Module):
    # Iteration numbers for n_saved axis
    iterations: Int[Array, "*batch n_saved"]

    # Saved states for n_saved axis
    states: IterationState  # Ensembled ["n_chains", "n_saved"]

    # Whether the saved state was accepted for n_saved axis for each chain
    states_accepted: Bool[Array, "*batch n_saved n_chains"]
    # Number of accepted proposals in the keep_interval for n_saved axis for each chain
    iterand_n_accepted: Int[Array, "*batch n_saved n_chains"]

    # Swaps: None if not using parallel tempering
    # Whether the swap was accepted for n_saved axis for each adjacent pair of chains
    states_swap_accepted: Bool[Array, "*batch n_saved n_chains-1"] | None
    # Number of accepted swaps in the keep_interval for n_saved axis for each adjacent pair of chains
    iterand_n_swap_accepted: Int[Array, "*batch n_saved n_chains-1"] | None

    # Temperatures (betas) and step sizes
    # Either 0-d for single chain, 1-d for multiple chains, 2-d for multiple chains with tuning history
    betas: (
        Float[Array, ""] | Float[Array, "n_chains"] | Float[Array, "n_saved n_chains"] | Float[Array, "*batch n_chains"] | Float[Array, "*batch n_chains n_saved"]
    )
    step_sizes: (
        Float[Array, ""] | Float[Array, "n_chains"] | Float[Array, "n_saved n_chains"] | Float[Array, "*batch n_chains"] | Float[Array, "*batch n_chains n_saved"]
    )

    # Whether these were varied (i.e. tuned)
    varying_step_sizes: bool
    varying_betas: bool

    @property
    def n_chains_flat(self) -> int:
        """Returns the 'flat' number of chains, i.e. the total number of chains across all batches."""
        return self.get_flat_shape()[0]

    @property
    def n_saved(self) -> int:
        return self.get_flat_shape()[1]

    def get_shapes(self) -> tuple[list[int], int, int]:
        """Returns the shapes of the batch dimensions, n_chains, and n_saved."""
        shape = self.states.log_likelihood.shape
        if len(shape) == 1:
            # Non-batched single-chain EMC histories can be saved as (n_saved,)
            return [], 1, shape[0]

        *batch_shape, n_chains, n_saved = shape
        return batch_shape, n_chains, n_saved

    def get_flat_shape(self) -> tuple[int, int]:
        """Returns the shape (n_chains_flat, n_saved)"""
        batch_shape, n_chains, n_saved = self.get_shapes()
        n_chains_flat = math.prod(batch_shape) * n_chains
        return n_chains_flat, n_saved

    def _get_iterations(self) -> Int[Array, "n_saved"]:
        if self.iterations.ndim <= 1:
            return self.iterations

        return reduce(self.iterations, "... n_saved -> n_saved", take_first)

    def get_flat_cold_accepted_states(self) -> Bool[Array, "n_accepted *grid"]:
        cold_mask = self.betas == 1.0
        states_accepted_cold = self.states_accepted[cold_mask]
        return self.states.state[cold_mask][states_accepted_cold]

    @staticmethod
    def _flatten_batched_scalar(
        arr: Float[Array, "*batch n_chains n_saved"],
    ) -> Float[Array, "flat_n_chains n_saved"]:
        if arr.ndim == 1:
            return jnp.expand_dims(arr, axis=0)

        return rearrange(arr, "... n_chains n_saved -> (n_chains ...) n_saved")

    def _get_log_likelihoods(self) -> Float[Array, "flat_n_chains n_saved"]:
        return self._flatten_batched_scalar(self.states.log_likelihood)

    def _get_iterand_n_accepted(self) -> Int[Array, "flat_n_chains n_saved"]:
        return self._flatten_batched_scalar(self.iterand_n_accepted)

    def _get_iterand_n_swap_accepted(self) -> Int[Array, "flat_n_pairs n_saved"]:
        if self.iterand_n_swap_accepted is None:
            raise ValueError(
                "No swap acceptance data available for non-parallel tempering history."
            )
        return self._flatten_batched_scalar(self.iterand_n_swap_accepted)

    def __resolve_get_hyperparameter(
        self,
        varying: bool,
        arr: Float[Array, ""]
        | Float[Array, "n_chains"]
        | Float[Array, "n_saved n_chains"],
    ) -> Float[Array, "flat_n_chains n_saved"]:
        batch_shape, n_chains, n_saved = self.get_shapes()
        n_chains_flat = math.prod(batch_shape) * n_chains

        match varying, arr.shape:
            case (False, ()):
                return jnp.full((n_chains_flat, n_saved), arr)
            case (False, (1,)):
                return repeat(
                    arr,
                    "s -> (s n_chains) n_saved",
                    n_chains=n_chains_flat,
                    n_saved=n_saved,
                )
            case (False, (_n_chains,)) if _n_chains == n_chains:
                return repeat(arr, "n_chains -> n_chains n_saved", n_saved=n_saved)
            case (False, (*_batch_shape, _n_chains)) if (
                _batch_shape == batch_shape and _n_chains == n_chains
            ):
                return repeat(
                    arr, "... n_chains -> (n_chains ...) n_saved", n_saved=n_saved
                )
            # Varying hyperparameters must have a value for each saved iteration
            case (True, (_n_saved,)) if _n_saved == n_saved:
                return repeat(
                    arr, "n_saved -> n_chains n_saved", n_chains=n_chains_flat
                )
            case (True, (_n_chains, _n_saved)) if (
                _n_chains == n_chains and _n_saved == n_saved
            ):
                return arr
            case (True, (*_batch_shape, _n_chains, _n_saved)) if (
                _batch_shape == batch_shape
                and _n_chains == n_chains
                and _n_saved == n_saved
            ):
                return rearrange(arr, "... n_chains n_saved -> (n_chains ...) n_saved")
            case _:
                raise ValueError(f"Invalid shape: {arr.shape} given {self}")

    def _get_betas(self) -> Float[Array, "flat_n_chains n_saved"]:
        return self.__resolve_get_hyperparameter(self.varying_betas, self.betas)

    def _get_step_sizes(self) -> Float[Array, "flat_n_chains n_saved"]:
        return self.__resolve_get_hyperparameter(
            self.varying_step_sizes, self.step_sizes
        )

    def _get_colors_and_labels(self):
        n = self.n_chains_flat
        betas = self._get_betas()
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(n, 2)))[:n]
        stride = max(1, n // 10)

        def c_label(i):
            if i % stride == 0:
                beta = betas[i, 0]
                return f"Chain {i} ($β_0={beta:.3g}$)"
            return "_nolegend_"

        labels = [c_label(i) for i in range(n)]
        return colors, labels

    @staticmethod
    def _plot_chain_series(
        ax,
        x,
        y,
        *,
        colors,
        labels,
        lw: float,
        alpha: float,
    ):
        """Plots one line per chain with consistent fallback styling."""
        n_chains_flat = y.shape[0]
        for i in range(n_chains_flat):
            color = colors[i] if i < len(colors) else "black"
            label = labels[i] if i < len(labels) else f"Chain {i}"
            ax.plot(x, y[i], color=color, lw=lw, alpha=alpha, label=label)

    def plot_step_sizes(self, ax=None, **kwargs):
        """
        Plots the step size trajectories across MCMC iterations.
        If step sizes are fixed, these will appear as horizontal lines.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        steps = self._get_iterations()
        step_sizes = self._get_step_sizes()

        colors, labels = self._get_colors_and_labels()
        self._plot_chain_series(
            ax,
            steps,
            step_sizes,
            colors=colors,
            labels=labels,
            lw=1.5,
            alpha=0.9,
        )

        # Update title based on whether tuning occurred
        title = "Tuning: Step Sizes" if self.varying_step_sizes else "Fixed Step Sizes"
        ax.set_title(title)
        ax.set_xlabel("MCMC Step")
        ax.set_ylabel("Step size")
        ax.legend(fontsize=8, ncol=kwargs.get("legend_ncol", 2), frameon=True)

        return ax

    def plot_betas(self, ax=None, **kwargs):
        """
        Plots beta (inverse temperature) trajectories across MCMC iterations.
        If betas are fixed, these will appear as horizontal lines.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        steps = self._get_iterations()
        betas = self._get_betas()

        colors, labels = self._get_colors_and_labels()
        self._plot_chain_series(
            ax,
            steps,
            betas,
            colors=colors,
            labels=labels,
            lw=1.5,
            alpha=0.9,
        )

        title = "Tuning: Betas" if self.varying_betas else "Fixed Betas"
        ax.set_title(title)
        ax.set_xlabel("MCMC Step")
        ax.set_ylabel("Beta")
        ax.legend(fontsize=8, ncol=kwargs.get("legend_ncol", 2), frameon=True)

        return ax

    def plot_log_likelihoods(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        steps = self._get_iterations()
        log_likelihoods = self._get_log_likelihoods()

        n_chains_flat = self.n_chains_flat
        mean_log_likelihoods = reduce(
            log_likelihoods, "n_chains_flat n_saved -> n_saved", "mean"
        )

        colors, labels = self._get_colors_and_labels()
        self._plot_chain_series(
            ax,
            steps,
            log_likelihoods,
            colors=colors,
            labels=labels,
            lw=1.3,
            alpha=0.85,
        )

        if n_chains_flat > 1:
            ax.plot(steps, mean_log_likelihoods, color="black", lw=2.6, label="Average")

        ax.set_title("Log-Likelihood Trace")
        ax.set_xlabel("MCMC Step")
        ax.set_ylabel("Log-likelihood")
        ax.legend(fontsize=8, frameon=True)
        return ax

    def plot_autocorrelation(self, ax=None, max_lag: int = 250, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        log_likelihoods = self._get_log_likelihoods()
        n_chains_flat, n_saved = self.get_flat_shape()

        steps = self._get_iterations()
        keep_interval = int(steps[1] - steps[0]) if len(steps) > 1 else 1

        max_lag = min(max_lag, n_saved - 1)
        lags_idx = np.arange(max_lag + 1)
        lags_steps = lags_idx * keep_interval

        # Compute ACF via statsmodels
        acf_logL = np.array(
            [
                acf(log_likelihoods[i], nlags=max_lag, fft=True)
                for i in range(n_chains_flat)
            ]
        )

        colors, labels = self._get_colors_and_labels()
        self._plot_chain_series(
            ax,
            lags_steps,
            acf_logL,
            colors=colors,
            labels=labels,
            lw=1.2,
            alpha=0.85,
        )

        if n_chains_flat > 1:
            ax.plot(
                lags_steps, acf_logL.mean(axis=0), color="black", lw=2.6, label="Average ACF"
            )

        ax.axhline(0.0, color="gray", lw=1.0)
        ax.set_title("Log-Likelihood Autocorrelation")
        ax.set_xlabel("Lag (MCMC Steps)")
        ax.set_ylabel("ACF")
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=8, frameon=True)
        return ax

    def plot_local_acceptance(
        self, ax=None, window: int = 75, target: float = 0.25, **kwargs
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        steps = self._get_iterations()
        n_acc = self._get_iterand_n_accepted()
        n_chains_flat = n_acc.shape[0]

        # Translate iteration step window into empirical array indices
        keep_interval = int(steps[1] - steps[0]) if len(steps) > 1 else 1
        window_idx = max(1, window // keep_interval)

        # Prevent NumPy convolve shape mismatch when history is shorter than the window
        window_idx = max(1, min(window_idx, len(steps)))

        block_sizes = np.diff(np.concatenate(([0], steps)))

        kernel = np.ones(window_idx)
        roll_attempts = np.convolve(block_sizes, kernel, mode="valid")
        roll_steps = steps[window_idx - 1 :]

        colors, labels = self._get_colors_and_labels()

        roll_rates = []
        for i in range(n_chains_flat):
            roll_acc = np.convolve(n_acc[i], kernel, mode="valid")
            rate = roll_acc / roll_attempts
            roll_rates.append(rate)

        roll_rates = np.asarray(roll_rates)
        self._plot_chain_series(
            ax,
            roll_steps,
            roll_rates,
            colors=colors,
            labels=labels,
            lw=1.2,
            alpha=0.9,
        )

        if n_chains_flat > 1:
            ax.plot(
                roll_steps,
                np.mean(roll_rates, axis=0),
                color="black",
                lw=2.6,
                label="Average",
            )

        ax.axhline(
            target,
            color="crimson",
            ls="--",
            lw=1.4,
            alpha=0.9,
            label=f"Target {target:.2f}",
        )
        ax.set_title(f"Within-Chain Acceptance (Window={window} steps)")
        ax.set_xlabel("MCMC Step")
        ax.set_ylabel("Acceptance rate")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, frameon=True)
        return ax

    def plot_swap_acceptance(
        self, ax=None, window: int = 75, target: float | None = None, **kwargs
    ):
        if self.iterand_n_swap_accepted is None:
            raise ValueError(
                "Swap acceptance can only be plotted for parallel tempering histories"
            )
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 5)))

        steps = self._get_iterations()
        n_acc = self._get_iterand_n_swap_accepted()
        n_chains_flat = n_acc.shape[0]

        # Translate iteration step window into empirical array indices
        keep_interval = int(steps[1] - steps[0]) if len(steps) > 1 else 1
        window_idx = max(1, window // keep_interval)

        # Prevent NumPy convolve shape mismatch when history is shorter than the window
        window_idx = max(1, min(window_idx, len(steps)))

        # Insert 0 at the beginning to accurately pinpoint alternating swap availabilities
        steps_padded = np.concatenate(([0], steps))
        start = steps_padded[:-1]
        end = steps_padded[1:]
        length = end - start

        kernel = np.ones(window_idx)
        roll_steps = steps[window_idx - 1 :]

        colors, labels = self._get_colors_and_labels()

        roll_rates = []
        for i in range(n_chains_flat):
            # Safely capture exact local eligibility offsets irrespective of iteration length bounds
            attempts = length // 2 + np.where(
                (length % 2 == 1) & (start % 2 == i % 2), 1, 0
            )
            roll_acc = np.convolve(n_acc[i], kernel, mode="valid")
            roll_att = np.convolve(attempts, kernel, mode="valid")

            rate = np.divide(
                roll_acc,
                roll_att,
                out=np.zeros_like(roll_acc, dtype=float),
                where=roll_att != 0,
            )
            roll_rates.append(rate)

        roll_rates = np.asarray(roll_rates)
        self._plot_chain_series(
            ax,
            roll_steps,
            roll_rates,
            colors=colors,
            labels=labels,
            lw=1.2,
            alpha=0.9,
        )

        if n_chains_flat > 1:
            ax.plot(
                roll_steps,
                np.mean(roll_rates, axis=0),
                color="black",
                lw=2.6,
                label="Average",
            )

        if target is not None:
            ax.axhline(
                target,
                color="crimson",
                ls="--",
                lw=1.4,
                alpha=0.9,
                label=f"Target {target:.2f}",
            )

        ax.set_title(f"Inter-Chain Acceptance (Window={window} steps)")
        ax.set_xlabel("MCMC Step")
        ax.set_ylabel("Swap acceptance rate")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, frameon=True)
        return ax

    def plot_diagnostics(
        self,
        window_chain: int = 75,
        window_swap: int = 75,
        max_lag: int = 250,
        target_chain: float = 0.25,
        target_swap: float | None = None,
        title: str = "MCMC Diagnostics",
    ):
        plt.style.use("seaborn-v0_8-whitegrid")
        plotters = [
            lambda ax: self.plot_log_likelihoods(ax=ax),
            lambda ax: self.plot_autocorrelation(ax=ax, max_lag=max_lag),
            lambda ax: self.plot_local_acceptance(
                ax=ax, window=window_chain, target=target_chain
            ),
        ]

        if self.iterand_n_swap_accepted is not None:
            plotters.append(
                lambda ax: self.plot_swap_acceptance(
                    ax=ax, window=window_swap, target=target_swap
                )
            )

        if self.varying_step_sizes:
            plotters.append(lambda ax: self.plot_step_sizes(ax=ax))

        if self.varying_betas:
            plotters.append(lambda ax: self.plot_betas(ax=ax))

        n_plots = len(plotters)
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        fig_height = max(5 * n_rows, 5)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(16, fig_height),
            constrained_layout=True,
        )

        axes_flat = np.atleast_1d(axes).reshape(-1)

        for ax, plot_fn in zip(axes_flat, plotters, strict=False):
            plot_fn(ax)

        for ax in axes_flat[n_plots:]:
            ax.set_visible(False)

        fig.suptitle(title, fontsize=16, y=1.02)
        plt.show()
        return fig, axes



class ExtendedMetropolisChain(eqx.Module):
    beta: Float[Array, ""]

    proposal_dist: ProposalDistribution
    forward_model: Callable[[Float[Array, "*grid"]], Float[Array, "N_data"]] = (
        eqx.field(static=True)
    )
    log_likelihood_fn: Callable = eqx.field(static=True)

    def __init__(
        self,
        *,
        proposal_dist: ProposalDistribution,
        forward_model: Callable[[Float[Array, "*grid"]], Float[Array, "N_data"]],
        log_likelihood_fn: Callable,
        beta: float | Float[Array, ""] = 1.0,
    ):
        self.proposal_dist = proposal_dist
        self.forward_model = forward_model
        self.log_likelihood_fn = log_likelihood_fn
        self.beta = jnp.asarray(beta)

    @eqx.filter_jit
    def log_likelihood(
        self, data: Float[Array, "N_data"], obs: ObservationsT
    ) -> Float[Array, ""]:
        return self.log_likelihood_fn(data, obs.data_obs, obs.data_std)

    @eqx.filter_jit
    def get_iteration_state(
        self, state: Float[Array, "*grid"], obs: ObservationsT
    ) -> IterationState:
        data = self.forward_model(state)
        log_likelihood = self.log_likelihood(data, obs)
        return IterationState(state=state, log_likelihood=log_likelihood)

    @eqx.filter_jit
    def __call__(
        self,
        key: Key,
        iter_state: IterationState,
        observations: ObservationsT,
        step_size: Float[Array, ""] | None = None,
    ) -> tuple[IterationState, Bool[Array, ""]]:
        proposal_key, accept_key = jax.random.split(key)

        state_proposal = self.proposal_dist.propose(
            iter_state.state, proposal_key, step_size=step_size
        )
        iter_state_proposal = self.get_iteration_state(state_proposal, observations)

        log_ratio = (
            iter_state_proposal.log_likelihood - iter_state.log_likelihood
        ) * self.beta
        log_P_accept = jnp.minimum(0.0, log_ratio)
        P_accept = jnp.exp(log_P_accept)

        accept = jax.random.uniform(accept_key) < P_accept
        iter_state_next = jax.lax.cond(
            accept, lambda: iter_state_proposal, lambda: iter_state
        )

        return iter_state_next, accept

    @eqx.filter_jit
    def step_n(
        self,
        n: int,
        key: Key,
        iter_state: IterationState,
        observations: ObservationsT,
        step_size: Float[Array, ""] | None = None,
        keep_interval: int = 1,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        if keep_interval <= 0:
            raise ValueError(
                f"keep_interval ({keep_interval}) must be a positive integer"
            )
        if n % keep_interval != 0:
            raise ValueError(
                f"n ({n}) must be a multiple of keep_interval ({keep_interval})"
            )

        n_saved = n // keep_interval

        def inner_scan_fn(inner_carry, inner_input):
            i_iter = inner_input
            state, n_acc, _ = inner_carry
            step_key = jax.random.fold_in(key, i_iter)

            next_state, accepted = self.__call__(
                key=step_key,
                iter_state=state,
                observations=observations,
                step_size=step_size,
            )
            n_acc = n_acc + accepted.astype(jnp.int32)

            carry = (next_state, n_acc, accepted)
            history_entry = None
            return carry, history_entry

        def scan_fn(carry, _):
            current_state, i_outer = carry

            inner_iter = i_outer + jnp.arange(keep_interval, dtype=jnp.int32)
            inner_input = inner_iter
            inner_carry = (current_state, jnp.int32(0), jnp.array(False))

            scan_inner = _resolve_scan_tqdm_fn(
                inner_scan_fn,
                progress=progress,
                total_steps=n,
                desc="Sampling",
                jax_tqdm_kwargs=jax_tqdm_kwargs,
            )
            (final_state, n_accepted, last_accepted), _ = _scan_with_optional_progress(
                scan_inner,
                inner_carry,
                inner_input,
                progress=progress,
            )

            i_next_outer = i_outer + keep_interval
            carry = (final_state, i_next_outer)
            history_output = (i_next_outer, final_state, n_accepted, last_accepted)
            return carry, history_output

        initial_carry = (iter_state, jnp.int32(0))
        scan_inputs = jnp.arange(n_saved)

        carry_final, history = jax.lax.scan(scan_fn, initial_carry, scan_inputs)
        final_state, _ = carry_final

        saved_states = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), history[1]
        )
        saved_iterand_n_accepted = jnp.expand_dims(history[2], axis=0)
        saved_states_accepted = jnp.expand_dims(history[3], axis=0)

        history_obj = History(
            iterations=history[0],
            states=saved_states,
            iterand_n_accepted=saved_iterand_n_accepted,
            states_accepted=saved_states_accepted,
            states_swap_accepted=None,
            iterand_n_swap_accepted=None,
            betas=jnp.atleast_1d(self.beta),
            step_sizes=jnp.atleast_1d(self.proposal_dist.step_size),
            varying_step_sizes=False,
            varying_betas=False,
        )
        return final_state, history_obj

    @eqx.filter_jit
    def tune(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_state: IterationState,
        observations: ObservationsT,
        initial_step_size: Float[Array, ""] | None = None,
        target_acceptance_rate: float = 0.25,
        learning_rate: float = 1.0,
        learning_rate_decay: float = 0.5,
        keep_interval: int = 1,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        if keep_interval <= 0:
            raise ValueError(
                f"keep_interval ({keep_interval}) must be a positive integer"
            )
        if tune_interval <= 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a positive integer"
            )
        if tune_interval % keep_interval != 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a multiple of keep_interval ({keep_interval})"
            )
        if n_steps_tune % tune_interval != 0:
            raise ValueError(
                f"n_steps_tune ({n_steps_tune}) must be a multiple of tune_interval ({tune_interval})"
            )

        n_tune_intervals = n_steps_tune // tune_interval
        saves_per_tune = tune_interval // keep_interval
        n_saved = n_steps_tune // keep_interval
        chain_template = self

        if initial_step_size is not None:
            step_size_init = jnp.asarray(initial_step_size)
        else:
            step_size_init = chain_template.proposal_dist.step_size

        dtype = step_size_init.dtype
        learning_rates = learning_rate * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_decay)

        # Over n_tune_intervals
        def outer_scan_fn(outer_carry, scan_input):
            i_tune, lr = scan_input
            step_sz, st = outer_carry

            # Over keep_interval
            def inner_step_fn(inner_carry, inner_input):
                i_iter = inner_input
                s, n_acc = inner_carry

                step_key = jax.random.fold_in(key, i_iter)
                nst, accepted = chain_template.__call__(
                    key=step_key,
                    iter_state=s,
                    observations=observations,
                    step_size=step_sz,
                )

                return (nst, n_acc + accepted.astype(jnp.int32)), None

            scan_inner = _resolve_scan_tqdm_fn(
                inner_step_fn,
                progress=progress,
                total_steps=n_steps_tune,
                desc="Tuning",
                jax_tqdm_kwargs=jax_tqdm_kwargs,
            )

            # Over saves_per_tune = tune_interval // keep_interval
            def middle_scan_fn(mid_carry, mid_input):
                i_save_start = mid_input
                current_st, total_acc = mid_carry

                inner_init = (current_st, jnp.int32(0))
                inner_inputs = i_save_start + jnp.arange(keep_interval, dtype=jnp.int32)

                (next_st, chunk_acc), _ = _scan_with_optional_progress(
                    scan_inner,
                    inner_init,
                    inner_inputs,
                    progress=progress,
                )

                next_total_acc = total_acc + chunk_acc
                current_iteration = i_save_start + keep_interval

                # Pass chunk_acc over instead of chunk_acc_rate for the plot accuracy
                history_entry = (current_iteration, next_st, step_sz, chunk_acc)

                return (next_st, next_total_acc), history_entry

            mid_init = (st, jnp.int32(0))
            mid_inputs = (
                i_tune * tune_interval
                + jnp.arange(saves_per_tune, dtype=jnp.int32) * keep_interval
            )

            (final_st, total_tune_acc), chunk_histories = jax.lax.scan(
                middle_scan_fn, mid_init, mid_inputs
            )

            acceptance_rate = total_tune_acc / tune_interval
            next_step_size = step_sz * jnp.exp(
                lr * (acceptance_rate - target_acceptance_rate)
            )
            next_step_size = jnp.clip(next_step_size, 1e-5, 1.0)

            return (next_step_size, final_st), chunk_histories

        scan_inputs = (jnp.arange(n_tune_intervals, dtype=jnp.int32), learning_rates)
        initial_carry = (step_size_init, iter_state)

        final_carry, history = jax.lax.scan(outer_scan_fn, initial_carry, scan_inputs)
        final_step_size, final_iter_state = final_carry

        flat_history = jax.tree_util.tree_map(
            lambda x: x.reshape((n_saved,) + x.shape[2:]), history
        )

        saved_states = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), flat_history[1]
        )
        saved_step_sizes = jnp.expand_dims(flat_history[2], axis=0)
        saved_iterand_n_accepted = jnp.expand_dims(flat_history[3], axis=0)

        final_chain = eqx.tree_at(
            lambda c: c.proposal_dist.step_size, chain_template, final_step_size
        )

        history_obj = History(
            iterations=flat_history[0],
            states=saved_states,
            iterand_n_accepted=saved_iterand_n_accepted,
            states_accepted=jnp.zeros_like(saved_iterand_n_accepted, dtype=jnp.bool_),
            states_swap_accepted=None,
            iterand_n_swap_accepted=None,
            betas=jnp.atleast_1d(final_chain.beta),
            step_sizes=saved_step_sizes,
            varying_step_sizes=True,
            varying_betas=False,
        )

        return final_chain, final_iter_state, history_obj


class ParallelTemperingSampler(eqx.Module):
    chains: ExtendedMetropolisChain
    n_chains: int = eqx.field(static=True)
    chain_axes_spec: jax.tree_util.PyTreeDef = eqx.field(static=True)

    def __init__(
        self,
        chains: ExtendedMetropolisChain,
        chain_axes_spec: jax.tree_util.PyTreeDef | None = None,
    ):
        self.chains = chains
        self.n_chains = len(chains.beta)

        if chain_axes_spec is None:
            self.chain_axes_spec = make_pytree_spec(
                chains,
                {
                    "beta": 0,
                    "proposal_dist.step_size": 0,
                    "*": None,
                },
            )
        else:
            self.chain_axes_spec = chain_axes_spec

    def _swap_adjacent(
        self, key: Key, iter_states: IterationState, offset: Literal[0, 1]
    ):
        idx1 = jnp.arange(self.n_chains - 1)
        idx2 = idx1 + 1

        log_L1 = iter_states.log_likelihood[idx1]
        log_L2 = iter_states.log_likelihood[idx2]
        beta1 = self.chains.beta[idx1]
        beta2 = self.chains.beta[idx2]

        delta_log_L = log_L2 - log_L1
        delta_beta = beta1 - beta2
        log_alpha = jnp.minimum(0.0, delta_log_L * delta_beta)

        u = jax.random.uniform(key, shape=log_alpha.shape)
        accept_raw = jnp.log(u) < log_alpha
        eligible = (idx1 % 2) == offset
        accept = accept_raw & eligible

        pos = jnp.arange(self.n_chains)
        accept_left = jnp.concatenate([accept, jnp.array([False])])
        accept_right = jnp.concatenate([jnp.array([False]), accept])

        perm = jnp.where(accept_left, pos + 1, pos)
        perm = jnp.where(accept_right, pos - 1, perm)

        new_iter_states = IterationState(
            state=iter_states.state[perm],
            log_likelihood=iter_states.log_likelihood[perm],
        )
        return new_iter_states, accept

    @staticmethod
    def _update_betas(
        betas: Float[Array, "n_chains"],
        swap_acceptances: Float[Array, "..."],
        learning_rate: float,
        target_acceptance: float = 0.25,
    ) -> Float[Array, "n_chains"]:
        if betas.shape[0] <= 1:
            return betas

        dtype = betas.dtype
        eps = jnp.asarray(10.0 * jnp.finfo(dtype).eps, dtype=dtype)

        beta_0 = betas[0]
        spacing = -jnp.diff(betas)

        log_update = learning_rate * (swap_acceptances - target_acceptance)
        spacing_new = spacing * jnp.exp(log_update)
        beta_tail = beta_0 - jnp.cumsum(spacing_new)

        min_beta_tail = eps * (jnp.arange(beta_tail.size)[::-1] + 1)
        beta_tail = jnp.maximum(beta_tail, min_beta_tail)

        beta_new = jnp.concatenate([jnp.array([beta_0], dtype=dtype), beta_tail])
        return beta_new

    @staticmethod
    def _update_betas_czyz(
        betas: Float[Array, "n_chains"],
        swap_acceptances: Float[Array, "n_chains-1"],
        learning_rate: float,
    ) -> Float[Array, "n_chains"]:
        if betas.shape[0] <= 1:
            return betas

        dtype = betas.dtype
        # Epsilon enforces strict monotonicity when computing the cumulative communication
        # barrier, guaranteeing that the subsequent linear inversion is strictly well-posed.
        eps = jnp.asarray(10.0 * jnp.finfo(dtype).eps, dtype=dtype)

        # In the thermodynamic limit of dense chains, the bottleneck is governed by the
        # instantaneous rejection rate λ(β). We estimate the discrete communication barrier
        # Λ_i ≈ ∑ ρ(β_j, β_{j+1}), where ρ is the empirical rejection probability.
        rejection_rates = jnp.maximum(1.0 - swap_acceptances, eps)
        lambdas = jnp.concatenate(
            [jnp.zeros(1, dtype=dtype), jnp.cumsum(rejection_rates)]
        )

        # The optimal non-reversible parallel tempering schedule distributes the communication
        # barrier uniformly across all adjacent chains, enforcing a constant ΔΛ.
        target_lambdas = jnp.linspace(0.0, lambdas[-1], betas.shape[0], dtype=dtype)

        # Project the uniform Λ grid back into the β domain. Because Λ is strictly increasing,
        # we can invert the mapping numerically via piecewise linear interpolation.
        optimal_betas = jnp.interp(target_lambdas, lambdas, betas)

        # The convex combination acts as a Robbins-Monro update, allowing the schedule to
        # converge steadily and preventing chain collapse from early stochastic noise in Λ(β).
        beta_new = (1.0 - learning_rate) * betas + learning_rate * optimal_betas

        return beta_new

    @eqx.filter_jit
    def _step_with_swap(
        self,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        swap_offset: Literal[0, 1] = 0,
    ):
        step_key, swap_key = jax.random.split(key)

        @partial(jax.vmap, in_axes=(self.chain_axes_spec, 0, 0, None))
        def _step_local(chain_b, k, iter_state_b, obs):
            return chain_b(k, iter_state_b, obs)

        keys = jax.random.split(step_key, self.n_chains)
        new_iter_states, local_accepted = _step_local(
            self.chains, keys, iter_states, observations
        )
        swapped_iter_states, swap_accepted = self._swap_adjacent(
            swap_key, new_iter_states, swap_offset
        )

        return swapped_iter_states, swap_accepted, local_accepted

    @eqx.filter_jit
    def __call__(
        self,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        swap_offset: Literal[0, 1] = 0,
    ):
        new_iter_state, swap_accepted, local_accepted = self._step_with_swap(
            key=key,
            iter_states=iter_states,
            observations=observations,
            swap_offset=swap_offset,
        )
        return new_iter_state, swap_accepted, local_accepted

    @eqx.filter_jit
    def step_n(
        self,
        n: int,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        keep_interval: int = 1,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        if keep_interval <= 0:
            raise ValueError(
                f"keep_interval ({keep_interval}) must be a positive integer"
            )
        if n % keep_interval != 0:
            raise ValueError(
                f"n ({n}) must be a multiple of keep_interval ({keep_interval})"
            )

        n_saved = n // keep_interval

        def inner_step_fn(inner_carry, inner_input):
            st, sw_n_acc, loc_n_acc, _, _ = inner_carry
            i_iter = inner_input

            step_key = jax.random.fold_in(key, i_iter)
            offset = i_iter % 2

            next_st, swap_acc, local_acc = self.__call__(
                key=step_key,
                iter_states=st,
                observations=observations,
                swap_offset=offset,
            )

            next_sw_n_acc = sw_n_acc + swap_acc.astype(jnp.int32)
            next_loc_n_acc = loc_n_acc + local_acc.astype(jnp.int32)

            next_carry = (next_st, next_sw_n_acc, next_loc_n_acc, swap_acc, local_acc)
            return next_carry, None

        inner_scan = _resolve_scan_tqdm_fn(
            inner_step_fn,
            progress=progress,
            total_steps=n,
            desc="PT Sampling",
            jax_tqdm_kwargs=jax_tqdm_kwargs,
        )

        def outer_scan_fn(carry, scan_input):
            states = carry
            i_saved = scan_input

            inner_init = (
                states,
                jnp.zeros(self.n_chains - 1, dtype=jnp.int32),
                jnp.zeros(self.n_chains, dtype=jnp.int32),
                jnp.zeros(self.n_chains - 1, dtype=jnp.bool_),
                jnp.zeros(self.n_chains, dtype=jnp.bool_),
            )
            inner_inputs = i_saved * keep_interval + jnp.arange(
                keep_interval, dtype=jnp.int32
            )

            (final_states, total_sw_acc, total_loc_acc, sw_last, loc_last), _ = (
                _scan_with_optional_progress(
                    inner_scan,
                    inner_init,
                    inner_inputs,
                    progress=progress,
                )
            )

            current_iteration = (i_saved + 1) * keep_interval
            history_entry = (
                current_iteration,
                final_states,
                total_sw_acc,
                total_loc_acc,
                sw_last,
                loc_last,
            )

            return final_states, history_entry

        initial_carry = iter_states
        scan_inputs = jnp.arange(n_saved, dtype=jnp.int32)

        final_states, history = jax.lax.scan(outer_scan_fn, initial_carry, scan_inputs)

        # This has produced ordering (n_saved, n_chains), but we want (n_chains, n_saved)
        # Swap (n_saved, n_chains) -> (n_chains, n_saved)
        def _swap_chain_saved(arr):
            return jnp.swapaxes(arr, 0, 1)

        history_obj = History(
            iterations=history[0],  # Shape: (n_saved,) - No transpose needed
            states=jax.tree_util.tree_map(_swap_chain_saved, history[1]),
            iterand_n_accepted=_swap_chain_saved(history[3]),
            states_accepted=_swap_chain_saved(history[5]),
            states_swap_accepted=_swap_chain_saved(history[4]),
            iterand_n_swap_accepted=_swap_chain_saved(history[2]),
            betas=self.chains.beta,
            step_sizes=self.chains.proposal_dist.step_size,
            varying_step_sizes=False,
            varying_betas=False,
        )
        return final_states, history_obj

    @eqx.filter_jit
    def tune_jointly(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        initial_step_size: Float[Array, "..."] | None = None,
        initial_betas: Float[Array, "n_chains"] | None = None,
        target_chain_acceptance_rate: float = 0.25,
        target_swap_acceptance_rate: float = 0.25,
        learning_rate_step_size: float = 1.0,
        learning_rate_step_size_decay: float = 0.5,
        learning_rate_beta: float = 1.0,
        learning_rate_beta_decay: float = 0.5,
        keep_interval: int = 1,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        if keep_interval <= 0:
            raise ValueError(
                f"keep_interval ({keep_interval}) must be a positive integer"
            )
        if tune_interval <= 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a positive integer"
            )
        if tune_interval % keep_interval != 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a multiple of keep_interval ({keep_interval})"
            )
        if n_steps_tune % tune_interval != 0:
            raise ValueError(
                f"n_steps_tune ({n_steps_tune}) must be a multiple of tune_interval ({tune_interval})"
            )

        n_tune_intervals = n_steps_tune // tune_interval
        saves_per_tune = tune_interval // keep_interval
        n_saved = n_steps_tune // keep_interval
        sampler = self

        if initial_step_size is not None:
            sampler = eqx.tree_at(
                lambda s: s.chains.proposal_dist.step_size, sampler, initial_step_size
            )
        if initial_betas is not None:
            sampler = eqx.tree_at(lambda s: s.chains.beta, sampler, initial_betas)

        dtype = jnp.asarray(sampler.chains.beta).dtype
        learning_rates_step = learning_rate_step_size * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_step_size_decay)
        learning_rates_beta = learning_rate_beta * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_beta_decay)

        def outer_scan_fn(outer_carry, scan_input):
            i_tune, lr_step, lr_beta = scan_input
            samp, st = outer_carry

            def inner_step_fn(inner_carry, inner_input):
                i_iter = inner_input
                s, acc_sw, acc_loc, att_sw = inner_carry

                step_key = jax.random.fold_in(key, i_iter)
                offset = i_iter % 2

                nst, sw_acc, loc_acc = samp._step_with_swap(
                    key=step_key,
                    iter_states=s,
                    observations=observations,
                    swap_offset=offset,
                )

                idx1 = jnp.arange(samp.n_chains - 1)
                eligibility = (idx1 % 2) == offset

                acc_sw = acc_sw + sw_acc.astype(jnp.int32)
                acc_loc = acc_loc + loc_acc.astype(jnp.int32)
                att_sw = att_sw + eligibility.astype(jnp.int32)

                return (nst, acc_sw, acc_loc, att_sw), None

            scan_inner = _resolve_scan_tqdm_fn(
                inner_step_fn,
                progress=progress,
                total_steps=n_steps_tune,
                desc="PT Tuning",
                jax_tqdm_kwargs=jax_tqdm_kwargs,
            )

            def middle_scan_fn(mid_carry, mid_input):
                i_save_start = mid_input
                current_st, total_sw_acc, total_loc_acc, total_sw_att = mid_carry

                inner_init = (
                    current_st,
                    jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                    jnp.zeros(samp.n_chains, dtype=jnp.int32),
                    jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                )
                inner_inputs = i_save_start + jnp.arange(keep_interval, dtype=jnp.int32)

                (next_st, chunk_sw_acc, chunk_loc_acc, chunk_sw_att), _ = (
                    _scan_with_optional_progress(
                        scan_inner,
                        inner_init,
                        inner_inputs,
                        progress=progress,
                    )
                )

                next_total_sw_acc = total_sw_acc + chunk_sw_acc
                next_total_loc_acc = total_loc_acc + chunk_loc_acc
                next_total_sw_att = total_sw_att + chunk_sw_att

                current_iteration = i_save_start + keep_interval

                history_entry = (
                    current_iteration,
                    next_st,
                    samp.chains.proposal_dist.step_size,
                    chunk_loc_acc,
                    samp.chains.beta,
                    chunk_sw_acc,
                )

                return (
                    next_st,
                    next_total_sw_acc,
                    next_total_loc_acc,
                    next_total_sw_att,
                ), history_entry

            mid_init = (
                st,
                jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                jnp.zeros(samp.n_chains, dtype=jnp.int32),
                jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
            )
            mid_inputs = (
                i_tune * tune_interval
                + jnp.arange(saves_per_tune, dtype=jnp.int32) * keep_interval
            )

            (
                (final_st, total_tune_sw_acc, total_tune_loc_acc, total_tune_sw_att),
                chunk_histories,
            ) = jax.lax.scan(middle_scan_fn, mid_init, mid_inputs)

            local_acceptance_rate = total_tune_loc_acc / tune_interval
            swap_acceptance_rate = jnp.where(
                total_tune_sw_att > 0, total_tune_sw_acc / total_tune_sw_att, 0.0
            )

            step_size = jnp.asarray(samp.chains.proposal_dist.step_size)
            if step_size.ndim == 0:
                step_size = jnp.broadcast_to(step_size, (samp.n_chains,))

            new_step_size = step_size * jnp.exp(
                lr_step * (local_acceptance_rate - target_chain_acceptance_rate)
            )
            new_step_size = jnp.clip(new_step_size, 1e-5, 1.0)

            new_betas = samp._update_betas(
                samp.chains.beta,
                swap_acceptance_rate,
                learning_rate=lr_beta,
                target_acceptance=target_swap_acceptance_rate,
            )

            new_chains = eqx.tree_at(
                lambda c: (c.proposal_dist.step_size, c.beta),
                samp.chains,
                (new_step_size, new_betas),
            )
            new_samp = eqx.tree_at(lambda s: s.chains, samp, new_chains)

            return (new_samp, final_st), chunk_histories

        scan_inputs = (
            jnp.arange(n_tune_intervals, dtype=jnp.int32),
            learning_rates_step,
            learning_rates_beta,
        )
        initial_carry = (sampler, iter_states)

        final_carry, history = jax.lax.scan(outer_scan_fn, initial_carry, scan_inputs)
        tuned_sampler, final_iter_states = final_carry

        flat_history = jax.tree_util.tree_map(
            lambda x: x.reshape((n_saved,) + x.shape[2:]), history
        )

        def _swap_chain_saved(arr):
            return jnp.swapaxes(arr, 0, 1) if arr.ndim >= 2 else arr

        history_obj = History(
            iterations=flat_history[0],
            states=jax.tree_util.tree_map(_swap_chain_saved, flat_history[1]),
            step_sizes=_swap_chain_saved(flat_history[2]),
            iterand_n_accepted=_swap_chain_saved(flat_history[3]),
            betas=_swap_chain_saved(flat_history[4]),
            iterand_n_swap_accepted=_swap_chain_saved(flat_history[5]),
            states_accepted=jnp.zeros_like(
                _swap_chain_saved(flat_history[3]), dtype=jnp.bool_
            ),
            states_swap_accepted=jnp.zeros_like(
                _swap_chain_saved(flat_history[5]), dtype=jnp.bool_
            ),
            varying_step_sizes=True,
            varying_betas=True,
        )

        return tuned_sampler, final_iter_states, history_obj

    @eqx.filter_jit
    def _tune_single_parameter(
        self,
        target_parameter: Literal["step_size", "beta"],
        method: Literal["robbins_monro", "czyz"],
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        target_acceptance_rate: float,
        learning_rate: float,
        learning_rate_decay: float,
        keep_interval: int = 1,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        if keep_interval <= 0:
            raise ValueError(
                f"keep_interval ({keep_interval}) must be a positive integer"
            )
        if tune_interval <= 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a positive integer"
            )
        if tune_interval % keep_interval != 0:
            raise ValueError(
                f"tune_interval ({tune_interval}) must be a multiple of keep_interval ({keep_interval})"
            )
        if n_steps_tune % tune_interval != 0:
            raise ValueError(
                f"n_steps_tune ({n_steps_tune}) must be a multiple of tune_interval ({tune_interval})"
            )

        n_tune_intervals = n_steps_tune // tune_interval
        saves_per_tune = tune_interval // keep_interval
        n_saved = n_steps_tune // keep_interval
        sampler = self

        dtype = jnp.asarray(sampler.chains.beta).dtype
        learning_rates = learning_rate * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_decay)

        def outer_scan_fn(outer_carry, scan_input):
            i_tune, lr = scan_input
            samp, st = outer_carry

            def inner_step_fn(inner_carry, inner_input):
                i_iter = inner_input
                s, acc_sw, acc_loc, att_sw = inner_carry

                step_key = jax.random.fold_in(key, i_iter)
                offset = i_iter % 2

                nst, sw_acc, loc_acc = samp._step_with_swap(
                    key=step_key,
                    iter_states=s,
                    observations=observations,
                    swap_offset=offset,
                )

                idx1 = jnp.arange(samp.n_chains - 1)
                eligibility = (idx1 % 2) == offset

                acc_sw = acc_sw + sw_acc.astype(jnp.int32)
                acc_loc = acc_loc + loc_acc.astype(jnp.int32)
                att_sw = att_sw + eligibility.astype(jnp.int32)

                return (nst, acc_sw, acc_loc, att_sw), None

            desc = (
                "PT Step Tuning"
                if target_parameter == "step_size"
                else "PT Beta Tuning"
            )
            scan_inner = _resolve_scan_tqdm_fn(
                inner_step_fn,
                progress=progress,
                total_steps=n_steps_tune,
                desc=desc,
                jax_tqdm_kwargs=jax_tqdm_kwargs,
            )

            def middle_scan_fn(mid_carry, mid_input):
                i_save_start = mid_input
                current_st, total_sw_acc, total_loc_acc, total_sw_att = mid_carry

                inner_init = (
                    current_st,
                    jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                    jnp.zeros(samp.n_chains, dtype=jnp.int32),
                    jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                )
                inner_inputs = i_save_start + jnp.arange(keep_interval, dtype=jnp.int32)

                (next_st, chunk_sw_acc, chunk_loc_acc, chunk_sw_att), _ = (
                    _scan_with_optional_progress(
                        scan_inner,
                        inner_init,
                        inner_inputs,
                        progress=progress,
                    )
                )

                next_total_sw_acc = total_sw_acc + chunk_sw_acc
                next_total_loc_acc = total_loc_acc + chunk_loc_acc
                next_total_sw_att = total_sw_att + chunk_sw_att

                current_iteration = i_save_start + keep_interval

                # We save both into the history so the History object can pull
                # the respective series for the tuned parameter.
                history_entry = (
                    current_iteration,
                    next_st,
                    samp.chains.proposal_dist.step_size,
                    chunk_loc_acc,
                    samp.chains.beta,
                    chunk_sw_acc,
                )

                return (
                    next_st,
                    next_total_sw_acc,
                    next_total_loc_acc,
                    next_total_sw_att,
                ), history_entry

            mid_init = (
                st,
                jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
                jnp.zeros(samp.n_chains, dtype=jnp.int32),
                jnp.zeros(samp.n_chains - 1, dtype=jnp.int32),
            )
            mid_inputs = (
                i_tune * tune_interval
                + jnp.arange(saves_per_tune, dtype=jnp.int32) * keep_interval
            )

            (
                (final_st, total_tune_sw_acc, total_tune_loc_acc, total_tune_sw_att),
                chunk_histories,
            ) = jax.lax.scan(middle_scan_fn, mid_init, mid_inputs)

            match target_parameter, method:
                case "step_size", "robbins_monro":
                    local_acceptance_rate = total_tune_loc_acc / tune_interval

                    step_size = jnp.asarray(samp.chains.proposal_dist.step_size)
                    if step_size.ndim == 0:
                        step_size = jnp.broadcast_to(step_size, (samp.n_chains,))

                    new_step_size = step_size * jnp.exp(
                        lr * (local_acceptance_rate - target_acceptance_rate)
                    )
                    new_step_size = jnp.clip(new_step_size, 1e-5, 1.0)

                    new_chains = eqx.tree_at(
                        lambda c: c.proposal_dist.step_size,
                        samp.chains,
                        new_step_size,
                    )

                case "beta", "robbins_monro":
                    swap_acceptance_rate = jnp.where(
                        total_tune_sw_att > 0,
                        total_tune_sw_acc / total_tune_sw_att,
                        0.0,
                    )

                    new_betas = samp._update_betas(
                        samp.chains.beta,
                        swap_acceptance_rate,
                        learning_rate=lr,
                        target_acceptance=target_acceptance_rate,
                    )

                    new_chains = eqx.tree_at(
                        lambda c: c.beta,
                        samp.chains,
                        new_betas,
                    )

                case "beta", "czyz":
                    # The geometric optimization depends exclusively on the empirical swap
                    # probabilities to approximate Λ, operating independently of an arbitrary target rate.
                    swap_acceptance_rate = jnp.where(
                        total_tune_sw_att > 0,
                        total_tune_sw_acc / total_tune_sw_att,
                        0.0,
                    )

                    new_betas = samp._update_betas_czyz(
                        samp.chains.beta,
                        swap_acceptance_rate,
                        learning_rate=lr,
                    )

                    new_chains = eqx.tree_at(
                        lambda c: c.beta,
                        samp.chains,
                        new_betas,
                    )

                case _:
                    raise ValueError(
                        f"Unsupported combination of target_parameter and method: {target_parameter}, {method}"
                    )

            new_samp = eqx.tree_at(lambda s: s.chains, samp, new_chains)

            return (new_samp, final_st), chunk_histories

        scan_inputs = (
            jnp.arange(n_tune_intervals, dtype=jnp.int32),
            learning_rates,
        )
        initial_carry = (sampler, iter_states)

        final_carry, history = jax.lax.scan(outer_scan_fn, initial_carry, scan_inputs)
        tuned_sampler, final_iter_states = final_carry

        flat_history = jax.tree_util.tree_map(
            lambda x: x.reshape((n_saved,) + x.shape[2:]), history
        )

        def _swap_chain_saved(arr):
            return jnp.swapaxes(arr, 0, 1) if arr.ndim >= 2 else arr

        tracked_step_sizes = _swap_chain_saved(flat_history[2])
        tracked_betas = _swap_chain_saved(flat_history[4])

        step_sizes = (
            tracked_step_sizes
            if target_parameter == "step_size"
            else tuned_sampler.chains.proposal_dist.step_size
        )
        betas = (
            tracked_betas if target_parameter == "beta" else tuned_sampler.chains.beta
        )

        history_obj = History(
            iterations=flat_history[0],
            states=jax.tree_util.tree_map(_swap_chain_saved, flat_history[1]),
            step_sizes=step_sizes,
            iterand_n_accepted=_swap_chain_saved(flat_history[3]),
            betas=betas,
            iterand_n_swap_accepted=_swap_chain_saved(flat_history[5]),
            states_accepted=jnp.zeros_like(
                _swap_chain_saved(flat_history[3]), dtype=jnp.bool_
            ),
            states_swap_accepted=jnp.zeros_like(
                _swap_chain_saved(flat_history[5]), dtype=jnp.bool_
            ),
            varying_step_sizes=(target_parameter == "step_size"),
            varying_betas=(target_parameter == "beta"),
        )

        return tuned_sampler, final_iter_states, history_obj

    @eqx.filter_jit
    def tune_step_sizes(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        initial_step_size: Float[Array, "..."] | None = None,
        target_chain_acceptance_rate: float = 0.25,
        learning_rate: float = 1.0,
        learning_rate_decay: float = 0.5,
        keep_interval: int = 1,
        method: Literal["robbins_monro"] | EllipsisType = ...,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        sampler = self
        if initial_step_size is not None:
            sampler = eqx.tree_at(
                lambda s: s.chains.proposal_dist.step_size, sampler, initial_step_size
            )

        if method is ...:
            method = "robbins_monro"

        return sampler._tune_single_parameter(
            target_parameter="step_size",
            method=method,
            n_steps_tune=n_steps_tune,
            tune_interval=tune_interval,
            key=key,
            iter_states=iter_states,
            observations=observations,
            target_acceptance_rate=target_chain_acceptance_rate,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            keep_interval=keep_interval,
            progress=progress,
            jax_tqdm_kwargs=jax_tqdm_kwargs,
        )

    @eqx.filter_jit
    def tune_betas(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_states: IterationState,
        observations: ObservationsT,
        initial_betas: Float[Array, "n_chains"] | None = None,
        target_swap_acceptance_rate: float | EllipsisType = ...,
        learning_rate: float = 1.0,
        learning_rate_decay: float = 0.5,
        keep_interval: int = 1,
        method: Literal["robbins_monro", "czyz"] | EllipsisType = ...,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        sampler = self
        if initial_betas is not None:
            sampler = eqx.tree_at(lambda s: s.chains.beta, sampler, initial_betas)

        if method == "czyz" and target_swap_acceptance_rate is not Ellipsis:
            raise ValueError(
                "target_swap_acceptance_rate is not supported when using the 'czyz' method."
            )

        if target_swap_acceptance_rate is ...:
            target_swap_acceptance_rate = 0.25

        if method is ...:
            method = "czyz"

        return sampler._tune_single_parameter(
            target_parameter="beta",
            method=method,
            n_steps_tune=n_steps_tune,
            tune_interval=tune_interval,
            key=key,
            iter_states=iter_states,
            observations=observations,
            target_acceptance_rate=target_swap_acceptance_rate,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            keep_interval=keep_interval,
            progress=progress,
            jax_tqdm_kwargs=jax_tqdm_kwargs,
        )
