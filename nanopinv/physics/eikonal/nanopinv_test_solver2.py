"""
Solver based on IFIM:

Huang, Y. Improved Fast Iterative Algorithm for Eikonal Equation for GPU Computing.
Preprint at https://doi.org/10.48550/arXiv.2106.15869 (2021).

Largely generated with Gemini Pro 3.1.
"""

import jax
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float
from nanopinv.physics.eikonal._common import get_axis_min_times, vectorized_godunov


def _dilate_mask(mask, ndim):
    res = jnp.zeros_like(mask)
    for ax in range(ndim):
        res = res | jnp.roll(mask, 1, axis=ax) | jnp.roll(mask, -1, axis=ax)
    return res


def ifim_single_source(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
) -> Float[Array, "*grid"]:
    ndim = phi.ndim

    phi_flat = phi.reshape(-1)
    fixed_flat = phi_flat <= 0.0
    fallback_fixed = jnp.zeros_like(fixed_flat).at[jnp.argmin(phi_flat)].set(True)
    source_mask = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed).reshape(
        phi.shape
    )

    T_init = jnp.where(
        source_mask,
        jnp.asarray(0.0, dtype=speed.dtype),
        jnp.asarray(jnp.inf, dtype=speed.dtype),
    )

    spacing_sq_inv = 1.0 / (dr * dr)
    inv_speed = 1.0 / speed
    tol_j = jnp.asarray(tolerance, dtype=speed.dtype)
    active_mask = _dilate_mask(source_mask, ndim) & ~source_mask

    def update_step(state):
        T, source, active, active_empty = state
        axis_min_times = get_axis_min_times(T, ndim)
        T_cand = vectorized_godunov(axis_min_times, spacing_sq_inv, inv_speed, dr)

        converged = active & (jnp.abs(T_cand - T) <= tol_j)
        new_source = source | converged
        T_new = jnp.where(active & ~converged, T_cand, T)

        new_active = (active & ~converged) | (
            _dilate_mask(converged, ndim) & ~(new_source | active)
        )
        return T_new, new_source, new_active, ~jnp.any(new_active)

    def update_cond_body(i, state):
        return lax.cond(state[3], lambda s: s, update_step, state)

    state = (T_init, source_mask, active_mask, ~jnp.any(active_mask))
    T_after_update, final_source, _, _ = lax.fori_loop(
        0, max_iter, update_cond_body, state
    )

    axis_min_times = get_axis_min_times(T_after_update, ndim)
    T_cand_rem = vectorized_godunov(axis_min_times, spacing_sq_inv, inv_speed, dr)
    remedy_mask = (jnp.abs(T_cand_rem - T_after_update) > tol_j) & ~final_source

    def remedy_step(state):
        T, remedy, remedy_empty = state
        axis_min_times = get_axis_min_times(T, ndim)
        T_cand = vectorized_godunov(axis_min_times, spacing_sq_inv, inv_speed, dr)

        improved = remedy & (T_cand < T - tol_j)
        T_new = jnp.where(improved, T_cand, T)

        new_remedy = _dilate_mask(improved, ndim) & ~final_source
        return T_new, new_remedy, ~jnp.any(new_remedy)

    def remedy_cond_body(i, state):
        return lax.cond(state[2], lambda s: s, remedy_step, state)

    rem_state = (T_after_update, remedy_mask, ~jnp.any(remedy_mask))
    T_final, _, _ = lax.fori_loop(0, max_iter, remedy_cond_body, rem_state)

    return T_final


def ifim_multi_source(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: ifim_single_source(
            phi=phi, speed=speed, dr=dr, max_iter=max_iter, tolerance=tolerance
        ),
        in_axes=0,
    )(phis)
