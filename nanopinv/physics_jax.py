"""
This module provides JAX-compatible travel-time calculations.

It supports six eikonal backends:
- `skfmm` through `jax.pure_callback`
- `fsm`: a pure-JAX first-order Fast Sweeping Method
- `solver3`: a pure-JAX Parallel Godunov-Jacobi solver
- `solver4`: Improved Fast Iterative Method (iFIM) adapted for dense JAX tensors
- `solver5`: Hyperplane (Diagonal) Fast Sweeping Method for massive GPU parallelism
- `solver6`: User-provided 2D specific nested-loop Fast Sweeping Method
"""

from collections.abc import Sequence
from functools import partial
from itertools import product
from typing import Literal

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from jax.scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

from nanopinv._typing import Array, Float

try:
    import skfmm
except ImportError:  # pragma: no cover - optional dependency
    skfmm = None


_SolverName = Literal["skfmm", "fsm", "solver3", "solver4", "solver5", "solver6"]


def _skfmm_caller(
    phi: Float[Array, "*grid"],
    speeds: Float[Array, "*grid"],
    dr: Float[Array, "{len(phi.shape)}"],
    order: int,
):
    if skfmm is None:
        raise ImportError(
            "solver='skfmm' requested but scikit-fmm is not installed. "
            "Install scikit-fmm or use a pure JAX solver."
        )

    return skfmm.travel_time(phi, speeds, dx=dr, order=order)


_PARALLEL_ARGS = {
    "n_jobs": -1,
    "prefer": "threads",
    "batch_size": "auto",
    "return_as": "generator",
}


def _batched_skfmm_mp_caller(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*batch *grid"],
    dx: Float[Array, "{len(phis.shape)-1}"],
    order: int = 2,
):
    phis_np = np.asarray(phis)
    speed_np = np.asarray(speed)
    dx_np = np.asarray(dx)

    ndim = len(dx)
    grid_shape = phis_np.shape[1:]
    N_sources = phis_np.shape[0]

    batch_shape = speed_np.shape[:-ndim]
    N_models = int(np.prod(batch_shape)) if batch_shape else 1

    speed_flat = speed_np.reshape((N_models, *grid_shape))

    tasks_phi = []
    tasks_speed = []
    for m_idx in range(N_models):
        for s_idx in range(N_sources):
            tasks_phi.append(phis_np[s_idx])
            tasks_speed.append(speed_flat[m_idx])

    results = Parallel(**_PARALLEL_ARGS)(
        delayed(_skfmm_caller)(p, s, dx_np, order)
        for p, s in zip(tasks_phi, tasks_speed)
    )

    res = np.empty((N_models, N_sources, *grid_shape), dtype=phis_np.dtype)
    for i, r in enumerate(results):
        m_idx = i // N_sources
        s_idx = i % N_sources
        res[m_idx, s_idx] = r

    result = res.reshape((*batch_shape, N_sources, *grid_shape))
    return result


@custom_vmap
def _fmm_jax_caller(phis, speed, dr, order):
    ndim = len(dr)
    batch_shape = speed.shape[:-ndim]
    grid_shape = speed.shape[-ndim:]

    result_shape = jax.ShapeDtypeStruct(
        (*batch_shape, phis.shape[0], *grid_shape), phis.dtype
    )

    return jax.pure_callback(
        _batched_skfmm_mp_caller, result_shape, phis, speed, dr, order
    )


@_fmm_jax_caller.def_vmap
def _fmm_jax_caller_vmap(axis_size, in_batched, phis, speed, dr, order):
    ndim = len(dr)
    batch_shape = speed.shape[:-ndim]
    grid_shape = speed.shape[-ndim:]

    result_shape = jax.ShapeDtypeStruct(
        (*batch_shape, phis.shape[0], *grid_shape), phis.dtype
    )

    travel_times_batched = jax.pure_callback(
        _batched_skfmm_mp_caller,
        result_shape,
        phis,
        speed,
        dr,
        order,
    )

    return travel_times_batched, True


def _build_fsm_stencils(grid_shape: tuple[int, ...]):
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
        jnp.where(jnp.isfinite(axis_min_times), axis_min_times + dr * inv_speed, jnp.inf)
    )
    return jnp.where(found, candidate, fallback)


def _fast_sweeping_single_source_jax(
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


def _fast_sweeping_multi_source_jax(
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
        lambda phi: _fast_sweeping_single_source_jax(
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


def _vectorized_godunov_jax(
    axis_min_times: jnp.ndarray,
    spacing_sq_inv: jnp.ndarray,
    inv_speed: jnp.ndarray,
    dr: jnp.ndarray,
) -> jnp.ndarray:
    """Fully vectorized Godunov upwind scheme for the entire grid simultaneously."""
    ndim = axis_min_times.shape[0]
    grid_shape = axis_min_times.shape[1:]
    rhs2 = inv_speed * inv_speed

    order = jnp.argsort(axis_min_times, axis=0)
    a_sorted = jnp.take_along_axis(axis_min_times, order, axis=0)

    # Avoid broadcast_to + take_along_axis by directly indexing
    w_sorted = spacing_sq_inv[order]

    next_sorted = jnp.concatenate(
        [a_sorted[1:], jnp.full((1, *grid_shape), jnp.inf, dtype=a_sorted.dtype)],
        axis=0,
    )

    A = jnp.zeros(grid_shape, dtype=axis_min_times.dtype)
    B = jnp.zeros(grid_shape, dtype=axis_min_times.dtype)
    C = jnp.zeros(grid_shape, dtype=axis_min_times.dtype)
    candidate = jnp.full(grid_shape, jnp.inf, dtype=axis_min_times.dtype)
    found = jnp.zeros(grid_shape, dtype=bool)

    # Unroll the loop manually since ndim is tiny (usually 2 or 3).
    # Allows XLA to fully fuse operations avoiding JAX control-flow overhead.
    for i in range(ndim):
        a = a_sorted[i]
        w = w_sorted[i]
        active = (~found) & jnp.isfinite(a)

        A = A + jnp.where(active, w, 0.0)
        B = B + jnp.where(active, w * a, 0.0)
        C = C + jnp.where(active, w * a * a, 0.0)

        discriminant = B * B - A * (C - rhs2)
        sqrt_discriminant = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        t_new = (B + sqrt_discriminant) / jnp.where(A > 0.0, A, 1.0)

        valid = active & (discriminant >= 0.0) & (t_new <= next_sorted[i] + 1e-12)
        candidate = jnp.where(valid, t_new, candidate)
        found = found | valid

    dr_expanded = dr.reshape([ndim] + [1] * len(grid_shape))
    fallback = jnp.min(
        jnp.where(
            jnp.isfinite(axis_min_times),
            axis_min_times + dr_expanded * inv_speed,
            jnp.inf,
        ),
        axis=0,
    )
    return jnp.where(found, candidate, fallback)


def _get_axis_min_times(T, ndim):
    axis_mins = []
    for ax in range(ndim):
        pad_width = [(0, 0)] * ndim
        pad_width[ax] = (1, 1)
        T_padded = jnp.pad(T, pad_width, mode="constant", constant_values=jnp.inf)

        slice_left = [slice(None)] * ndim
        slice_left[ax] = slice(0, -2)
        slice_right = [slice(None)] * ndim
        slice_right[ax] = slice(2, None)

        axis_mins.append(
            jnp.minimum(T_padded[tuple(slice_left)], T_padded[tuple(slice_right)])
        )
    return jnp.stack(axis_mins, axis=0)


def _dilate_mask(mask, ndim):
    res = jnp.zeros_like(mask)
    for ax in range(ndim):
        res = res | jnp.roll(mask, 1, axis=ax) | jnp.roll(mask, -1, axis=ax)
    return res


def _jacobi_single_source_jax(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
) -> Float[Array, "*grid"]:
    ndim = phi.ndim

    fixed_mask = phi <= 0.0
    min_idx = jnp.unravel_index(jnp.argmin(phi), phi.shape)
    fallback_mask = jnp.zeros_like(fixed_mask).at[min_idx].set(True)
    fixed_mask = jnp.where(jnp.any(fixed_mask), fixed_mask, fallback_mask)

    T_init = jnp.where(
        fixed_mask,
        jnp.asarray(0.0, dtype=speed.dtype),
        jnp.asarray(jnp.inf, dtype=speed.dtype),
    )

    spacing_sq_inv = 1.0 / (dr * dr)
    inv_speed = 1.0 / speed
    tolerance_j = jnp.asarray(tolerance, dtype=speed.dtype)

    def cond_fun(state):
        T, diff, i = state
        return (diff > tolerance_j) & (i < max_iter)

    def body_fun(state):
        T, _, i = state
        axis_min_times = _get_axis_min_times(T, ndim)
        T_candidate = _vectorized_godunov_jax(
            axis_min_times, spacing_sq_inv, inv_speed, dr
        )

        T_new = jnp.minimum(T, T_candidate)
        T_new = jnp.where(fixed_mask, 0.0, T_new)

        # T_new <= T ensures T - T_new >= 0. Abs is unnecessary.
        diff = jnp.max(
            jnp.where(
                jnp.isfinite(T),
                T - T_new,
                jnp.where(jnp.isfinite(T_new), jnp.inf, 0.0),
            )
        )

        return T_new, diff, i + 1

    init_state = (T_init, jnp.asarray(jnp.inf, dtype=speed.dtype), jnp.array(0, jnp.int32))

    # while_loop terminates exactly when convergence tolerance is met
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    return final_state[0]


def _jacobi_multi_source_jax(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: _jacobi_single_source_jax(
            phi=phi, speed=speed, dr=dr, max_iter=max_iter, tolerance=tolerance
        ),
        in_axes=0,
    )(phis)


def _ifim_single_source_jax(
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
    source_mask = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed).reshape(phi.shape)

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
        axis_min_times = _get_axis_min_times(T, ndim)
        T_cand = _vectorized_godunov_jax(axis_min_times, spacing_sq_inv, inv_speed, dr)

        converged = active & (jnp.abs(T_cand - T) <= tol_j)
        new_source = source | converged
        T_new = jnp.where(active & ~converged, T_cand, T)

        new_active = (active & ~converged) | (_dilate_mask(converged, ndim) & ~(new_source | active))
        return T_new, new_source, new_active, ~jnp.any(new_active)

    def update_cond_body(i, state):
        return lax.cond(state[3], lambda s: s, update_step, state)

    state = (T_init, source_mask, active_mask, ~jnp.any(active_mask))
    T_after_update, final_source, _, _ = lax.fori_loop(0, max_iter, update_cond_body, state)

    axis_min_times = _get_axis_min_times(T_after_update, ndim)
    T_cand_rem = _vectorized_godunov_jax(axis_min_times, spacing_sq_inv, inv_speed, dr)
    remedy_mask = (jnp.abs(T_cand_rem - T_after_update) > tol_j) & ~final_source

    def remedy_step(state):
        T, remedy, remedy_empty = state
        axis_min_times = _get_axis_min_times(T, ndim)
        T_cand = _vectorized_godunov_jax(axis_min_times, spacing_sq_inv, inv_speed, dr)

        improved = remedy & (T_cand < T - tol_j)
        T_new = jnp.where(improved, T_cand, T)

        new_remedy = _dilate_mask(improved, ndim) & ~final_source
        return T_new, new_remedy, ~jnp.any(new_remedy)

    def remedy_cond_body(i, state):
        return lax.cond(state[2], lambda s: s, remedy_step, state)

    rem_state = (T_after_update, remedy_mask, ~jnp.any(remedy_mask))
    T_final, _, _ = lax.fori_loop(0, max_iter, remedy_cond_body, rem_state)

    return T_final


def _ifim_multi_source_jax(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: _ifim_single_source_jax(
            phi=phi, speed=speed, dr=dr, max_iter=max_iter, tolerance=tolerance
        ),
        in_axes=0,
    )(phis)


def _hyperplane_fsm_single_source_jax(
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


def _hyperplane_fsm_multi_source_jax(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_sweeps: int,
    tolerance: float,
):
    return jax.vmap(
        lambda phi: _hyperplane_fsm_single_source_jax(
            phi=phi, speed=speed, dr=dr, max_sweeps=max_sweeps, tolerance=tolerance
        ),
        in_axes=0,
    )(phis)


def _user_fsm_2d_single_source_jax(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    iterations: int,
) -> Float[Array, "*grid"]:
    """User-provided 2D Fast Sweeping Method (solver6)."""
    nx, ny = phi.shape

    phi_flat = phi.reshape(-1)
    fixed_flat = phi_flat <= 0.0
    fallback_fixed = jnp.zeros_like(fixed_flat).at[jnp.argmin(phi_flat)].set(True)
    fixed_cells = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed).reshape(nx, ny)

    # Initialize travel time grid
    large_val = jnp.inf
    grid = jnp.where(fixed_cells, 0.0, large_val)

    # In the original snippet f was slowness
    f = 1.0 / speed
    dh = dr[0]

    sweep_dirs = [
        (0, nx, 1, 0, ny, 1),  # Top-left to bottom-right
        (nx - 1, -1, -1, 0, ny, 1),  # Top-right to bottom-left
        (nx - 1, -1, -1, ny - 1, -1, -1),  # Bottom-right to top-left
        (0, nx, 1, ny - 1, -1, -1),  # Bottom-left to top-right
    ]

    # Note: no obstacle map specified from the module interface, using zeros
    obstacle = jnp.zeros_like(fixed_cells)
    frozen = jnp.logical_or(fixed_cells, obstacle)
    padded = jnp.pad(grid, pad_width=1, mode="constant", constant_values=large_val)

    def run_sweep(sweep_dir, _grid):
        x_start, x_end, x_step, y_start, y_end, y_step = sweep_dir

        def y_loop_body(iy, curr_grid):
            def x_loop_body(ix, _curr_grid):
                piy, pix = iy + 1, ix + 1
                a = jnp.minimum(_curr_grid[piy, pix - 1], _curr_grid[piy, pix + 1])
                b = jnp.minimum(_curr_grid[piy - 1, pix], _curr_grid[piy + 1, pix])

                updated_val = jnp.where(
                    frozen[iy, ix],
                    _curr_grid[piy, pix],  # no change if frozen
                    jnp.minimum(  # min of curr and updated val
                        _curr_grid[piy, pix],
                        jnp.where(  # eqn 2.4
                            jnp.abs(a - b) >= f[iy, ix] * dh,
                            jnp.minimum(a, b) + f[iy, ix] * dh,
                            (a + b + jnp.sqrt(2 * (f[iy, ix] * dh) ** 2 - (a - b) ** 2)) / 2,
                        ),
                    ),
                )
                return _curr_grid.at[piy, pix].set(updated_val)

            x_indices = jnp.arange(x_start, x_end, x_step)
            return jax.lax.fori_loop(
                0,
                len(x_indices),
                lambda ix, val: x_loop_body(x_indices[ix], val),
                curr_grid,
            )

        y_indices = jnp.arange(y_start, y_end, y_step)
        return jax.lax.fori_loop(
            0,
            len(y_indices),
            lambda iy, val: y_loop_body(y_indices[iy], val),
            _grid,
        )

    def iteration_body(_, cur_grid):
        grid_s1 = run_sweep(sweep_dirs[0], cur_grid)
        grid_s2 = run_sweep(sweep_dirs[1], grid_s1)
        grid_s3 = run_sweep(sweep_dirs[2], grid_s2)
        grid_s4 = run_sweep(sweep_dirs[3], grid_s3)
        return grid_s4

    final_grid = jax.lax.fori_loop(0, iterations, iteration_body, padded)
    return final_grid[1:-1, 1:-1]


def _user_fsm_2d_multi_source_jax(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    iterations: int,
):
    return jax.vmap(
        lambda phi: _user_fsm_2d_single_source_jax(
            phi=phi, speed=speed, dr=dr, iterations=iterations
        ),
        in_axes=0,
    )(phis)


@jax.jit(
    static_argnames=[
        "window",
    ]
)
def compute_phi(
    source: Float[Array, "{len(r)}"],
    r: Sequence[Float[Array, "ax"]],
    window: int,
):
    ndim = len(r)
    shape_grid = tuple(len(arr) for arr in r)
    phi = jnp.ones(shape=shape_grid, dtype=source.dtype)

    nearest_idx = [jnp.argmin(jnp.abs(r[i] - source[i])) for i in range(ndim)]

    start_indices = []
    w_sizes = []
    local_coords = []

    for i in range(ndim):
        w_size = min(2 * window + 1, shape_grid[i])
        w_sizes.append(w_size)
        start_idx = jnp.clip(nearest_idx[i] - window, 0, shape_grid[i] - w_size)
        start_indices.append(start_idx)
        local_c = lax.dynamic_slice_in_dim(r[i], start_idx, w_size)
        local_coords.append(local_c)

    local_phi = jnp.zeros(tuple(w_sizes))
    for i in range(ndim):
        shape = [1] * ndim
        shape[i] = w_sizes[i]
        c_reshaped = local_coords[i].reshape(shape)
        local_phi += (c_reshaped - source[i]) ** 2

    local_phi = jnp.sqrt(local_phi)

    radius = jnp.min(local_phi) + 1e-10
    local_phi -= radius

    phi = lax.dynamic_update_slice(phi, local_phi, tuple(start_indices))
    return phi


def build_travel_time_points(
    sources,
    receivers,
    *grid_axes,
    order: int = 2,
    window: int = 1,
    solver: _SolverName = "skfmm",
    fsm_max_sweeps: int = 64,
    fsm_tolerance: float = 1e-8,
    jacobi_max_iter: int = 2000,
):
    valid_solvers = {"skfmm", "fsm", "solver3", "solver4", "solver5", "solver6"}
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {valid_solvers}")

    unique_sources, inverse_sources = np.unique(sources, axis=0, return_inverse=True)

    unique_sources_j = jnp.asarray(unique_sources)
    inverse_j = jnp.asarray(inverse_sources)

    r = list(grid_axes)
    dr = jnp.array([jnp.diff(arr)[0] for arr in r])

    for i, arr in enumerate(r):
        if len(arr) < 2:
            raise ValueError(
                f"Grid axis {i} must contain at least two points to determine spacing"
            )

        d_arr = jnp.diff(arr)

        eps = jnp.finfo(arr.dtype).eps
        rtol = eps * 100
        atol = eps * 100

        if not jnp.allclose(d_arr, d_arr[0], rtol=rtol, atol=atol):
            raise ValueError(f"Grid axis {i} must be equidistant")

    shape_grid = tuple(len(arr) for arr in r)

    vmap_compute_phi = jax.vmap(compute_phi, in_axes=(0, None, None))
    phis_batched = vmap_compute_phi(unique_sources_j, r, window)

    if solver == "solver6":
        if len(shape_grid) != 2:
            raise ValueError("solver='solver6' is strictly implemented for 2D grids.")
        if not jnp.allclose(dr, dr[0]):
            raise ValueError("solver='solver6' requires uniform grid spacing (dx == dy).")

        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _user_fsm_2d_multi_source_jax(
                phis=phis_batched,
                speed=m_single,
                dr=dr,
                iterations=fsm_max_sweeps,
            )

    elif solver == "solver5":
        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _hyperplane_fsm_multi_source_jax(
                phis=phis_batched,
                speed=m_single,
                dr=dr,
                max_sweeps=fsm_max_sweeps,
                tolerance=fsm_tolerance,
            )

    elif solver == "solver4":
        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _ifim_multi_source_jax(
                phis=phis_batched,
                speed=m_single,
                dr=dr,
                max_iter=jacobi_max_iter,
                tolerance=fsm_tolerance,
            )

    elif solver == "solver3":
        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _jacobi_multi_source_jax(
                phis=phis_batched,
                speed=m_single,
                dr=dr,
                max_iter=jacobi_max_iter,
                tolerance=fsm_tolerance,
            )

    elif solver == "fsm":
        sweep_orders_np, neighbors_minus_np, neighbors_plus_np = _build_fsm_stencils(
            shape_grid
        )
        sweep_orders = jnp.asarray(sweep_orders_np)
        neighbors_minus = jnp.asarray(neighbors_minus_np)
        neighbors_plus = jnp.asarray(neighbors_plus_np)

        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _fast_sweeping_multi_source_jax(
                phis=phis_batched,
                speed=m_single,
                dr=dr,
                sweep_orders=sweep_orders,
                neighbors_minus=neighbors_minus,
                neighbors_plus=neighbors_plus,
                max_sweeps=fsm_max_sweeps,
                tolerance=fsm_tolerance,
            )

    else:
        @jax.jit
        def solve_eikonal(m_single: Float[Array, "*grid"]):
            return _fmm_jax_caller(phis_batched, m_single, dr, order)

    @jax.jit(donate_argnums=[0])
    def single_model(m_single: Float[Array, "*grid"]):
        travel_times = solve_eikonal(m_single)
        tt_expanded = jnp.take(travel_times, inverse_j, axis=0)

        def interp_fn(grid, rec):
            return RegularGridInterpolator(
                points=r, values=grid, bounds_error=False, fill_value=jnp.inf
            )(rec).squeeze()

        vmap_data = jax.vmap(interp_fn, in_axes=(0, 0))
        return vmap_data(tt_expanded, receivers)

    @jax.jit(donate_argnums=[0])
    def batched_model(m: Float[Array, "*batch *grid"]):
        batch_ndim = m.ndim - len(shape_grid)

        f = single_model
        for _ in range(batch_ndim):
            f = jax.vmap(f)

        return f(m)

    return batched_model
