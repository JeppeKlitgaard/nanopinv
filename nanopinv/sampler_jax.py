from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import checkify
from jax_tqdm import scan_tqdm

from nanopinv._typing import Array, Bool, Float, Int, Key, Shaped
from nanopinv.distribution import DistributionBase
from nanopinv.types import Observations
from nanopinv.utils import make_pytree_spec


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

        # Implement the logic to propose a new state based on the current state
        new_realisation = self.dist(key)

        # Do Precondition Crank-Nicolson to combine the current state and the new realisation
        state_proposal = (
            self.mean
            + jnp.sqrt(1.0 - step_size**2) * (state_current - self.mean)
            + step_size * (new_realisation - self.mean)
        )

        return state_proposal


class IterationState(eqx.Module):
    state: Float[Array, "*grid"]
    log_likelihood: Float[Array, ""]


class ExtendedMetropolisChain(eqx.Module):
    temperature: Float[Array, ""]

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
        temperature: float | Float[Array, ""] = 1.0,
    ):
        self.proposal_dist = proposal_dist
        self.forward_model = forward_model
        self.log_likelihood_fn = log_likelihood_fn

        self.temperature = jnp.asarray(temperature)

    @eqx.filter_jit
    def log_likelihood(self, data: Float[Array, "N_data"], obs: Observations) -> Float[Array, ""]:
        return self.log_likelihood_fn(data, obs.data_obs, obs.data_std)

    @eqx.filter_jit
    def get_iteration_state(self, state: Float[Array, "*grid"], obs: Observations) -> IterationState:
        """
        This is used internally and may also be used to obtain the initial state easily.
        """
        data = self.forward_model(state)
        log_likelihood = self.log_likelihood(data, obs)
        return IterationState(state=state, log_likelihood=log_likelihood)

    @eqx.filter_jit
    def __call__(
        self,
        key: Key,
        iter_state: IterationState,
        observations: Observations,
        step_size: Float[Array, ""] | None = None,
    ) -> tuple[IterationState, Bool[Array, ""]]:
        proposal_key, accept_key = jax.random.split(key)

        state_proposal = self.proposal_dist.propose(
            iter_state.state, proposal_key, step_size=step_size
        )
        iter_state_proposal = self.get_iteration_state(state_proposal, observations)

        # Compute acceptance probability
        log_ratio = (iter_state_proposal.log_likelihood - iter_state.log_likelihood) * (1 / self.temperature)
        log_P_accept = jnp.minimum(0.0, log_ratio)
        P_accept = jnp.exp(log_P_accept)

        accept = jax.random.uniform(accept_key) < P_accept

        # Select next iteration state based on acceptance
        iter_state_next = jax.lax.cond(accept, lambda: iter_state_proposal, lambda: iter_state)

        return iter_state_next, accept

    @eqx.filter_jit
    def step_n(
        self,
        n: int,
        key: Key,
        iter_state: IterationState,
        observations: Observations,
        step_size: Float[Array, ""] | None = None,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):

        def scan_fn(carry, scan_input):
            # We keep i_iter to work with jax-tqdm, neglible overhead
            i_iter = scan_input
            iter_state = carry
            step_key = jax.random.fold_in(key, i_iter)

            iter_state, accepted = self.__call__(
                key=step_key,
                iter_state=iter_state,
                observations=observations,
                step_size=step_size,
            )

            new_carry = iter_state
            history_output = (iter_state, accepted)

            return new_carry, history_output

        initial_carry = iter_state
        scan_inputs = jnp.arange(n)

        if progress:
            if jax_tqdm_kwargs is None:
                jax_tqdm_kwargs = {
                    "desc": "Sampling",
                    "print_rate": 10,
                    "tqdm_type": "auto",
                }

            scan_fn = scan_tqdm(n, **(jax_tqdm_kwargs))(scan_fn)

        carry_final, history = jax.lax.scan(scan_fn, initial_carry, scan_inputs)
        return carry_final, history

    @eqx.filter_jit
    def tune(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_state: IterationState,
        observations: Observations,
        initial_step_size: Float[Array, ""] | None = None,
        target_acceptance_rate: float = 0.25,
        learning_rate: float = 1.0,
        learning_rate_decay: float = 0.5,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        checkify.check(
            n_steps_tune % tune_interval == 0,
            "n_steps_tune must be a multiple of tune_interval",
        )

        chain_template = self
        if initial_step_size is not None:
            step_size_init = jnp.asarray(initial_step_size)
        else:
            step_size_init = chain_template.proposal_dist.step_size

        dtype = jnp.asarray(step_size_init).dtype
        n_tune_intervals = n_steps_tune // tune_interval

        # Set up inputs for scan/for-loop
        # Based on Robbins-Monro algorithm: https://en.wikipedia.org/w/index.php?title=Stochastic_approximation&oldid=1320073982#Robbins%E2%80%93Monro_algorithm
        learning_rates = learning_rate * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_decay)

        def scan_fn(carry, scan_input):
            i_interval, learning_rate = scan_input
            iter_state_current, step_size = carry
            interval_key = jax.random.fold_in(key, i_interval)

            iter_state_current, history = chain_template.step_n(
                tune_interval,
                key=interval_key,
                iter_state=iter_state_current,
                observations=observations,
                step_size=step_size,
            )

            acceptance_rate = jnp.mean(history[1])
            acceptance_rate_error = acceptance_rate - target_acceptance_rate
            step_size = step_size * jnp.exp(learning_rate * acceptance_rate_error)
            step_size = jnp.clip(step_size, 1e-5, 1.0)  # Must be in (0, 1] for pCN

            new_carry = (iter_state_current, step_size)
            history_entry = (iter_state_current, step_size, acceptance_rate)

            return new_carry, history_entry

        if progress:
            if jax_tqdm_kwargs is None:
                jax_tqdm_kwargs = {
                    "desc": "Tuning",
                    "print_rate": 1,
                    "tqdm_type": "auto",
                }

            jax_tqdm_kwargs = {
                "unit_scale": tune_interval,
            } | jax_tqdm_kwargs

            scan_fn = scan_tqdm(n_tune_intervals, **(jax_tqdm_kwargs))(scan_fn)

        scan_inputs = (jnp.arange(n_tune_intervals), learning_rates)
        initial_carry = (iter_state, step_size_init)

        final_carry, history = jax.lax.scan(scan_fn, initial_carry, scan_inputs)
        final_iter_state, final_step_size = final_carry
        final_chain = eqx.tree_at(
            lambda c: c.proposal_dist.step_size, chain_template, final_step_size
        )
        return final_chain, final_iter_state, history


class ParallelTemperingSampler(eqx.Module):
    chains: ExtendedMetropolisChain
    n_chains: int = eqx.field(static=True)

    # Either generated or passed in by user.
    # This is the specification for how to vectorise over the chains.
    chain_axes_spec: jax.tree_util.PyTreeDef = eqx.field(static=True)

    def __init__(
        self,
        chains: ExtendedMetropolisChain,
        chain_axes_spec: jax.tree_util.PyTreeDef | None = None,
    ):
        self.chains = chains
        self.n_chains = len(chains.temperature)

        if chain_axes_spec is None:
            # Assume user has taken care to make chain components such that
            # vectorised axes are only:
            # - the leading axis of temperature
            # - the leading axis of proposal_dist.step_size
            # If you are user: See examples in notebooks
            self.chain_axes_spec = make_pytree_spec(
                chains,
                {
                    "temperature": 0,
                    "proposal_dist.step_size": 0,
                    "*": None,
                },
            )
        else:
            self.chain_axes_spec = chain_axes_spec

    def _swap_adjacent(self, key: Key, iter_states: IterationState, offset: Literal[0, 1]):
        """
        Attempts to swap states and log-likelihoods between adjacent chains.

        `offset=0` permits swaps on `(0,1), (2,3)`.
        `offset=1` permits swaps on `(1,2), (3,4)`.
        """
        idx1 = jnp.arange(self.n_chains - 1)
        idx2 = idx1 + 1

        log_L1 = iter_states.log_likelihood[idx1]
        log_L2 = iter_states.log_likelihood[idx2]
        T1 = self.chains.temperature[idx1]
        T2 = self.chains.temperature[idx2]

        # PT swap acceptance log-probability
        delta_log_L = log_L2 - log_L1
        delta_inv_T = (1.0 / T1) - (1.0 / T2)
        log_alpha = jnp.minimum(0.0, delta_log_L * delta_inv_T)

        u = jax.random.uniform(key, shape=log_alpha.shape)

        # Candidate swap acceptances, then parity mask from traced offset.
        accept_raw = jnp.log(u) < log_alpha
        eligible = (idx1 % 2) == offset
        accept = accept_raw & eligible

        # Build permutation without dynamic indexing.
        pos = jnp.arange(self.n_chains)
        accept_left = jnp.concatenate([accept, jnp.array([False])])       # position i starts a swap
        accept_right = jnp.concatenate([jnp.array([False]), accept])      # position i is right partner

        perm = jnp.where(accept_left, pos + 1, pos)
        perm = jnp.where(accept_right, pos - 1, perm)

        new_iter_states = IterationState(
            state=iter_states.state[perm],
            log_likelihood=iter_states.log_likelihood[perm],
        )

        return new_iter_states, accept

    @staticmethod
    def _update_temperatures(
        temperatures: Float[Array, "n_chains"],
        swap_acceptances: Float[Array, "..."],
        learning_rate: float,
        target_acceptance: float = 0.25,
    ) -> Float[Array, "n_chains"]:
        """Update temperatures through inverse-temperature spacing reparameterization."""
        if temperatures.shape[0] <= 1:
            return temperatures

        beta = 1.0 / temperatures
        beta_0 = beta[0]

        # S_i = beta_i - beta_{i+1} > 0
        spacing = -jnp.diff(beta)

        # Robbins-Monro update on spacing
        log_update = learning_rate * (swap_acceptances - target_acceptance)
        log_update = jnp.clip(log_update, -20.0, 20.0)
        spacing_new = spacing * jnp.exp(log_update)

        # Ensure the sum of spacings does not exceed beta_0 (leaving an epsilon buffer)
        eps = jnp.asarray(1e-12, dtype=beta.dtype)
        max_allowed_spacing = jnp.maximum(beta_0 - eps, eps)
        current_total_spacing = jnp.sum(spacing_new)

        # If the total spacing is too large, scale it back uniformly, otherwise 1.0.
        scale_factor = jnp.minimum(1.0, max_allowed_spacing / current_total_spacing)
        spacing_safe = spacing_new * scale_factor

        # Reconstruct ladder anchored at beta_0
        beta_tail = beta_0 - jnp.cumsum(spacing_safe)
        beta_new = jnp.concatenate([jnp.array([beta_0], dtype=beta.dtype), beta_tail])

        return 1.0 / beta_new

    @eqx.filter_jit
    def _step_with_swap(
        self,
        key: Key,
        iter_states: IterationState,
        observations: Observations,
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
        observations: Observations,
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
        observations: Observations,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        def scan_fn(carry, scan_input):
            i_iter = scan_input
            iter_states, = carry
            step_key = jax.random.fold_in(key, i_iter)

            # Swap every step and alternate pairings (0,1),(2,3) <-> (1,2),(3,4)
            offset = i_iter % 2

            next_iter_state, swap_accepted, local_accepted = self.__call__(
                key=step_key,
                iter_states=iter_states,
                observations=observations,
                swap_offset=offset
            )
            new_carry = (next_iter_state, )
            history_entry = (next_iter_state, swap_accepted, local_accepted)

            return new_carry, history_entry

        initial_carry = (iter_states, )
        scan_inputs = jnp.arange(n)

        if progress:
            if jax_tqdm_kwargs is None:
                jax_tqdm_kwargs = {
                    "desc": "PT Sampling",
                    "print_rate": 10,
                    "tqdm_type": "auto",
                }
            scan_fn = scan_tqdm(n, **(jax_tqdm_kwargs))(scan_fn)

        carry_final, history = jax.lax.scan(scan_fn, initial_carry, scan_inputs)
        return carry_final, history

    @eqx.filter_jit
    def tune(
        self,
        n_steps_tune: int,
        tune_interval: int,
        key: Key,
        iter_states: IterationState,
        observations: Observations,
        initial_step_size: Float[Array, "..."] | None = None,
        initial_temperatures: Float[Array, "n_chains"] | None = None,
        target_chain_acceptance_rate: float = 0.25,
        target_swap_acceptance_rate: float = 0.25,
        learning_rate_step_size: float = 1.0,
        learning_rate_step_size_decay: float = 0.5,
        learning_rate_temperature: float = 1.0,
        learning_rate_temperature_decay: float = 0.5,
        progress: bool = False,
        jax_tqdm_kwargs: dict[str, Any] | None = None,
    ):
        checkify.check(
            n_steps_tune % tune_interval == 0,
            "n_steps_tune must be a multiple of tune_interval",
        )

        sampler = self
        if initial_step_size is not None:
            sampler = eqx.tree_at(
                lambda s: s.chains.proposal_dist.step_size, sampler, initial_step_size
            )
        if initial_temperatures is not None:
            sampler = eqx.tree_at(
                lambda s: s.chains.temperature, sampler, initial_temperatures
            )

        checkify.check(
            jnp.all(jnp.diff(sampler.chains.temperature) > 0),
            "Temperatures must be strictly monotonically increasing",
        )

        n_tune_intervals = n_steps_tune // tune_interval
        dtype = jnp.asarray(sampler.chains.temperature).dtype

        learning_rates_step = learning_rate_step_size * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_step_size_decay)
        learning_rates_temp = learning_rate_temperature * (
            jnp.arange(n_tune_intervals, dtype=dtype) + 1
        ) ** (-learning_rate_temperature_decay)

        def scan_fn(carry, scan_input):
            i_interval, lr_step, lr_temp = scan_input
            sampler, iter_states = carry
            interval_key = jax.random.fold_in(key, i_interval)

            def interval_step_fn(carry_step, step_input):
                i_iter = step_input
                iter_states = carry_step
                step_key = jax.random.fold_in(interval_key, i_iter)

                offset = i_iter % 2
                next_iter_states, swap_accepted, local_accepted = (
                    sampler._step_with_swap(
                        key=step_key,
                        iter_states=iter_states,
                        observations=observations,
                        swap_offset=offset,
                    )
                )
                history_step = (swap_accepted, local_accepted)
                return next_iter_states, history_step

            iter_states, history_interval = jax.lax.scan(
                interval_step_fn,
                iter_states,
                jnp.arange(tune_interval),
            )

            swap_accept_history, local_accept_history = history_interval

            local_acceptance_rate = jnp.mean(local_accept_history, axis=0)

            idx1 = jnp.arange(sampler.n_chains - 1)
            i_iters = jnp.arange(tune_interval)
            offsets = i_iters % 2

            eligibility_mask = (idx1[None, :] % 2) == offsets[:, None]
            attempts = jnp.sum(eligibility_mask, axis=0)
            accepts = jnp.sum(swap_accept_history, axis=0)

            # Protect against zero division just in case
            swap_acceptance_rate = jnp.where(
                attempts > 0,
                accepts / attempts,
                0.0
            )

            step_size = jnp.asarray(sampler.chains.proposal_dist.step_size)
            if step_size.ndim == 0:
                step_size = jnp.broadcast_to(step_size, (sampler.n_chains,))

            new_step_size = step_size * jnp.exp(
                lr_step * (local_acceptance_rate - target_chain_acceptance_rate)
            )
            new_step_size = jnp.clip(new_step_size, 1e-5, 1.0)

            new_temperatures = sampler._update_temperatures(
                sampler.chains.temperature,
                swap_acceptance_rate,
                learning_rate=lr_temp,
                target_acceptance=target_swap_acceptance_rate,
            )

            new_chains = eqx.tree_at(
                lambda c: (c.proposal_dist.step_size, c.temperature),
                sampler.chains,
                (new_step_size, new_temperatures),
            )
            sampler = eqx.tree_at(lambda s: s.chains, sampler, new_chains)

            new_carry = (sampler, iter_states)
            history_entry = (
                iter_states,
                new_step_size,
                local_acceptance_rate,
                new_temperatures,
                swap_acceptance_rate,
            )
            return new_carry, history_entry

        if progress:
            if jax_tqdm_kwargs is None:
                jax_tqdm_kwargs = {
                    "desc": "PT Tuning",
                    "print_rate": 1,
                    "tqdm_type": "auto",
                }
            jax_tqdm_kwargs = {
                "unit_scale": tune_interval,
            } | jax_tqdm_kwargs
            scan_fn = scan_tqdm(n_tune_intervals, **(jax_tqdm_kwargs))(scan_fn)

        scan_inputs = (
            jnp.arange(n_tune_intervals),
            learning_rates_step,
            learning_rates_temp,
        )
        initial_carry = (sampler, iter_states)
        final_carry, history = jax.lax.scan(scan_fn, initial_carry, scan_inputs)

        tuned_sampler, final_iter_states = final_carry
        return tuned_sampler, final_iter_states, history

