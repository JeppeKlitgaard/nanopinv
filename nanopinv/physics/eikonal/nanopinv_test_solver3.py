from itertools import product

import jax
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float
from nanopinv.physics.eikonal.nanopinv_fsm import (
    _get_axis_min_times,
    _vectorized_godunov_jax,
)


def hyperplane_fsm_single_source(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_sweeps: int,
    tolerance: float,
) -> Float[Array, "*grid"]:
    ndim = phi.ndim
    shape = phi.shape

    phi_flat = phi.reshape(-1)
    fixed_flat = phi_flat <= 0.0
    fallback_fixed = jnp.zeros_like(fixed_flat).at[jnp.argmin(phi_flat)].set(True)
    fixed_flat = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed)
    fixed_mask = fixed_flat.reshape(shape)

    T_init = jnp.where(
        fixed_mask,
        jnp.asarray(0.0, dtype=speed.dtype),
        jnp.asarray(jnp.inf, dtype=speed.dtype),
    )

    spacing_sq_inv = 1.0 / (dr * dr)
    inv_speed = 1.0 / speed
    tol_j = jnp.asarray(tolerance, dtype=speed.dtype)

    grid_indices = jnp.indices(shape)
    sum_indices = jnp.sum(grid_indices, axis=0)
    max_k = sum(s - 1 for s in shape)

    def apply_flips(arr, flips):
        for ax, f in enumerate(flips):
            if f == -1:
                arr = jnp.flip(arr, axis=ax)
        return arr

    def single_sweep(T_current, flips):
        T_f = apply_flips(T_current, flips)
        F_f = apply_flips(inv_speed, flips)
        fix_f = apply_flips(fixed_mask, flips)

        def step(T_carry, k):
            mask = sum_indices == k
            axis_mins = _get_axis_min_times(T_carry, ndim)
            T_cand = _vectorized_godunov_jax(axis_mins, spacing_sq_inv, F_f, dr)
            T_new = jnp.where(mask, jnp.minimum(T_carry, T_cand), T_carry)
            T_new = jnp.where(fix_f, 0.0, T_new)
            return T_new, None

        T_f_out, _ = lax.scan(step, T_f, jnp.arange(max_k + 1))
        return apply_flips(T_f_out, flips)

    directions = list(product((1, -1), repeat=ndim))

    def full_sweep_cycle(T_carry):
        T_curr = T_carry
        for dirs in directions:
            T_curr = single_sweep(T_curr, dirs)
        return T_curr

    def convergence_body(i, state):
        T, converged = state

        def do_sweep(_T):
            T_next = full_sweep_cycle(_T)
            diff = jnp.max(
                jnp.abs(
                    jnp.where(
                        jnp.isfinite(_T),
                        _T - T_next,
                        jnp.where(jnp.isfinite(T_next), jnp.inf, 0.0),
                    )
                )
            )
            return T_next, diff <= tol_j

        return lax.cond(converged, lambda _: state, lambda _: do_sweep(T), operand=None)

    init_state = (T_init, jnp.asarray(False))
    T_final, _ = lax.fori_loop(0, max_sweeps, convergence_body, init_state)
    return T_final


def hyperplane_fsm_multi_source(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_sweeps: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: hyperplane_fsm_single_source(
            phi=phi, speed=speed, dr=dr, max_sweeps=max_sweeps, tolerance=tolerance
        ),
        in_axes=0,
    )(phis)
