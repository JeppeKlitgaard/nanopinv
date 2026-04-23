from itertools import product

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float


def build_fsm_stencils(grid_shape: tuple[int, ...]):
    ndim = len(grid_shape)
    n_nodes = int(np.prod(grid_shape))

    neighbors_minus = np.full((ndim, n_nodes), -1, dtype=np.int32)
    neighbors_plus = np.full((ndim, n_nodes), -1, dtype=np.int32)

    for lin in range(n_nodes):
        idx = np.unravel_index(lin, grid_shape)
        for axis in range(ndim):
            if idx[axis] > 0:
                idx_minus = list(idx)
                idx_minus[axis] -= 1
                neighbors_minus[axis, lin] = np.ravel_multi_index(
                    tuple(idx_minus), grid_shape
                )

            if idx[axis] < grid_shape[axis] - 1:
                idx_plus = list(idx)
                idx_plus[axis] += 1
                neighbors_plus[axis, lin] = np.ravel_multi_index(
                    tuple(idx_plus), grid_shape
                )

    n_directions = 2**ndim
    sweep_orders = np.empty((n_directions, n_nodes), dtype=np.int32)

    for i_dir, direction in enumerate(product((1, -1), repeat=ndim)):
        ranges = [
            range(n) if d > 0 else range(n - 1, -1, -1)
            for n, d in zip(grid_shape, direction)
        ]
        sweep_orders[i_dir] = np.fromiter(
            (np.ravel_multi_index(idx, grid_shape) for idx in product(*ranges)),
            dtype=np.int32,
            count=n_nodes,
        )

    return sweep_orders, neighbors_minus, neighbors_plus


def _fsm_local_update_jax(
    axis_min_times: Float[Array, "ndim"],
    spacing_sq_inv: Float[Array, "ndim"],
    inv_speed: Float[Array, ""],
    dr: Float[Array, "ndim"],
) -> Float[Array, ""]:
    ndim = axis_min_times.shape[0]
    rhs2 = inv_speed * inv_speed

    order = jnp.argsort(axis_min_times)
    a_sorted = axis_min_times[order]
    w_sorted = spacing_sq_inv[order]

    next_sorted = jnp.concatenate(
        [a_sorted[1:], jnp.asarray([jnp.inf], dtype=a_sorted.dtype)]
    )

    def body(i, state):
        A, B, C, candidate, found = state

        a = a_sorted[i]
        w = w_sorted[i]
        active = (~found) & jnp.isfinite(a)

        A_new = A + jnp.where(active, w, 0.0)
        B_new = B + jnp.where(active, w * a, 0.0)
        C_new = C + jnp.where(active, w * a * a, 0.0)

        discriminant = B_new * B_new - A_new * (C_new - rhs2)
        sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        t_new = (B_new + sqrt_discriminant) / jnp.where(A_new > 0.0, A_new, 1.0)

        valid = active & (discriminant >= 0.0) & (t_new <= next_sorted[i] + 1e-12)
        candidate_new = jnp.where(valid, t_new, candidate)
        found_new = found | valid

        return A_new, B_new, C_new, candidate_new, found_new

    init_state = (
        jnp.asarray(0.0, dtype=axis_min_times.dtype),
        jnp.asarray(0.0, dtype=axis_min_times.dtype),
        jnp.asarray(0.0, dtype=axis_min_times.dtype),
        jnp.asarray(jnp.inf, dtype=axis_min_times.dtype),
        jnp.asarray(False),
    )
    _, _, _, candidate, found = lax.fori_loop(0, ndim, body, init_state)

    fallback = jnp.min(
        jnp.where(
            jnp.isfinite(axis_min_times), axis_min_times + dr * inv_speed, jnp.inf
        )
    )
    return jnp.where(found, candidate, fallback)


def fast_sweeping_single_source(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    sweep_orders: jnp.ndarray,
    neighbors_minus: jnp.ndarray,
    neighbors_plus: jnp.ndarray,
    max_sweeps: int,
    tolerance: float,
) -> Float[Array, "*grid"]:
    ndim = phi.ndim

    phi_flat = phi.reshape(-1)
    fixed_flat = phi_flat <= 0.0
    fallback_fixed = jnp.zeros_like(fixed_flat).at[jnp.argmin(phi_flat)].set(True)
    fixed_flat = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed)

    speed_flat = speed.reshape(-1)
    travel_init = jnp.where(
        fixed_flat,
        jnp.asarray(0.0, dtype=speed.dtype),
        jnp.asarray(jnp.inf, dtype=speed.dtype),
    )

    spacing_sq_inv = 1.0 / (dr * dr)
    tolerance_j = jnp.asarray(tolerance, dtype=speed.dtype)

    def update_one_cell(travel_flat, lin):
        is_fixed = fixed_flat[lin]

        def keep_current(t):
            return t

        def update_current(t):
            axis_min_vals = []
            for axis in range(ndim):
                idx_minus = neighbors_minus[axis, lin]
                idx_plus = neighbors_plus[axis, lin]

                val_minus = jnp.where(idx_minus >= 0, t[idx_minus], jnp.inf)
                val_plus = jnp.where(idx_plus >= 0, t[idx_plus], jnp.inf)
                axis_min_vals.append(jnp.minimum(val_minus, val_plus))

            axis_min_times = jnp.stack(axis_min_vals)
            has_neighbor = jnp.any(jnp.isfinite(axis_min_times))

            def solve_and_update(t_inner):
                candidate = _fsm_local_update_jax(
                    axis_min_times=axis_min_times,
                    spacing_sq_inv=spacing_sq_inv,
                    inv_speed=1.0 / speed_flat[lin],
                    dr=dr,
                )
                new_value = jnp.minimum(t_inner[lin], candidate)
                return t_inner.at[lin].set(new_value)

            return lax.cond(has_neighbor, solve_and_update, lambda t_inner: t_inner, t)

        return lax.cond(is_fixed, keep_current, update_current, travel_flat)

    def sweep_one_direction(travel_flat, order_flat):
        def body(i, carry):
            lin = order_flat[i]
            return update_one_cell(carry, lin)

        return lax.fori_loop(0, order_flat.shape[0], body, travel_flat)

    def full_sweep(travel_flat):
        def body(i_dir, carry):
            return sweep_one_direction(carry, sweep_orders[i_dir])

        return lax.fori_loop(0, sweep_orders.shape[0], body, travel_flat)

    def sweep_body(_, state):
        current, max_delta, converged = state

        def do_sweep(cur):
            updated = full_sweep(cur)
            delta = jnp.where(
                jnp.isfinite(cur),
                cur - updated,
                jnp.where(jnp.isfinite(updated), jnp.inf, 0.0),
            )
            new_max_delta = jnp.max(delta)
            new_converged = new_max_delta <= tolerance_j
            return updated, new_max_delta, new_converged

        return lax.cond(
            converged,
            lambda _: (current, max_delta, converged),
            lambda _: do_sweep(current),
            operand=None,
        )

    initial_state = (
        travel_init,
        jnp.asarray(jnp.inf, dtype=speed.dtype),
        jnp.asarray(False),
    )
    travel_final, _, _ = lax.fori_loop(0, max_sweeps, sweep_body, initial_state)

    return travel_final.reshape(phi.shape)


def fast_sweeping_multi_source(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    sweep_orders: jnp.ndarray,
    neighbors_minus: jnp.ndarray,
    neighbors_plus: jnp.ndarray,
    max_sweeps: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: fast_sweeping_single_source(
            phi=phi,
            speed=speed,
            dr=dr,
            sweep_orders=sweep_orders,
            neighbors_minus=neighbors_minus,
            neighbors_plus=neighbors_plus,
            max_sweeps=max_sweeps,
            tolerance=tolerance,
        ),
        in_axes=0,
    )(phis)
