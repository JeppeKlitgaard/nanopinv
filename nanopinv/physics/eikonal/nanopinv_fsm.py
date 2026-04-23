import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float


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


def jacobi_single_source(
    phi: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
    debug: bool,
    debug_verbosity: int,
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
        """
        Returns True if the iteration should continue, False if it should stop.
        """
        T, diff, i = state
        should_continue = (diff > tolerance_j) & (i < max_iter)

        if debug and debug_verbosity >= 2:
            if debug_verbosity >= 2:
                jax.debug.print(
                    "ITER: Iteration: {:d}, Max Diff: {:.12f}, Tolerance: {:.12f}",
                    i,
                    diff,
                    tolerance_j,
                )

        return should_continue

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

    init_state = (
        T_init,
        jnp.asarray(jnp.inf, dtype=speed.dtype),
        jnp.array(0, jnp.int32),
    )

    # Terminate at max_iter or tolerance
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    T_final, diff_final, i_final = final_state

    if debug and debug_verbosity >= 1:
        terminated_because_tolerance = diff_final <= tolerance_j
        terminated_because_iteration = i_final >= max_iter

        jax.debug.print(
            "CONV: Tol: {:.12f}, Diff: {:.12f}, Iteration: {:d}, Tolerance Termination: {}, Iteration Termination: {}, Should Continue: False",
            tolerance_j,
            diff_final,
            i_final,
            terminated_because_tolerance,
            terminated_because_iteration,
        )

    return T_final


@jax.jit(static_argnames=("max_iter", "tolerance", "debug", "debug_verbosity"))
def jacobi_multi_source(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
    debug: bool,
    debug_verbosity: int,
):
    return jax.vmap(
        lambda phi: jacobi_single_source(
            phi=phi,
            speed=speed,
            dr=dr,
            max_iter=max_iter,
            tolerance=tolerance,
            debug=debug,
            debug_verbosity=debug_verbosity,
        ),
        in_axes=0,
    )(phis)
