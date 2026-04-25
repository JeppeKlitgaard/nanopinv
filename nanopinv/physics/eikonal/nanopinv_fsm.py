import jax
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float


def _get_upwind_stencil(
    T: jnp.ndarray, ndim: int, order: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the target value A and scaling factor lambda for the FSM.
    Evaluates at JIT-compile time whether to build the 1st-order or 2nd-order stencil.
    """
    if order not in (1, 2):
        raise ValueError(f"FSM order must be 1 or 2, got {order}.")

    A_list = []
    lambda_list = []

    for ax in range(ndim):
        if order == 1:
            # 1st-order: Simple 1-point upwind differences
            pad_width = [(0, 0)] * ndim
            pad_width[ax] = (1, 1)
            T_padded = jnp.pad(T, pad_width, mode="constant", constant_values=jnp.inf)

            s_im1 = [slice(None)] * ndim; s_im1[ax] = slice(0, -2)
            s_ip1 = [slice(None)] * ndim; s_ip1[ax] = slice(2, None)

            A = jnp.minimum(T_padded[tuple(s_im1)], T_padded[tuple(s_ip1)])
            lam = jnp.ones_like(A)

        elif order == 2:
            # 2nd-order: Switch-based 2-point upwind differences (Tro et al. 2023)
            pad_width = [(0, 0)] * ndim
            pad_width[ax] = (2, 2)
            T_padded = jnp.pad(T, pad_width, mode="constant", constant_values=jnp.inf)

            s_im2 = [slice(None)] * ndim; s_im2[ax] = slice(0, -4)
            s_im1 = [slice(None)] * ndim; s_im1[ax] = slice(1, -3)
            s_ip1 = [slice(None)] * ndim; s_ip1[ax] = slice(3, -1)
            s_ip2 = [slice(None)] * ndim; s_ip2[ax] = slice(4, None)

            u_im2 = T_padded[tuple(s_im2)]
            u_im1 = T_padded[tuple(s_im1)]
            u_ip1 = T_padded[tuple(s_ip1)]
            u_ip2 = T_padded[tuple(s_ip2)]

            cond1 = u_im2 < u_im1
            cond2 = u_ip2 >= u_ip1
            cond3 = u_ip2 < u_ip1
            cond4 = u_im2 >= u_im1

            # Prevent inf - inf = NaN propagation in arithmetic comparisons
            both_inf_p = jnp.isinf(u_ip1) & jnp.isinf(u_ip2)
            both_inf_m = jnp.isinf(u_im1) & jnp.isinf(u_im2)

            val_p_cmp = jnp.where(both_inf_p, jnp.inf, -4.0 * u_ip1 + u_ip2)
            val_m_cmp = jnp.where(both_inf_m, jnp.inf, -4.0 * u_im1 + u_im2)

            cond5 = val_p_cmp >= val_m_cmp
            cond6 = val_p_cmp < val_m_cmp

            sw_m_bool = (cond1 & cond2) | (cond1 & cond3 & cond6)
            sw_p_bool = (cond3 & cond4) | (cond1 & cond3 & cond5)

            # Strictly enforce that 2nd-order is only used if BOTH upwind points are computed (finite).
            # This implicitly manages domain boundaries as padded infs inherently fail this check.
            sw_m_bool = sw_m_bool & jnp.isfinite(u_im1) & jnp.isfinite(u_im2)
            sw_p_bool = sw_p_bool & jnp.isfinite(u_ip1) & jnp.isfinite(u_ip2)

            sw_m = sw_m_bool.astype(T.dtype)
            sw_p = sw_p_bool.astype(T.dtype)

            max_sw = jnp.maximum(sw_p, sw_m)

            # lambda = 1 if fallback to 1st order, 9/4 = 2.25 if using 2nd order
            lam = (1.0 - max_sw) + 2.25 * max_sw

            # Prevent inf - inf = NaN in the calculation terms
            term_p = jnp.where(both_inf_p, jnp.inf, 4.0 * u_ip1 - u_ip2)
            term_m = jnp.where(both_inf_m, jnp.inf, 4.0 * u_im1 - u_im2)

            # Nested jnp.where safely sidesteps 0.0 * inf = NaN evaluations
            A = jnp.where(
                max_sw == 0.0,
                jnp.minimum(u_ip1, u_im1),
                jnp.where(sw_p > 0.5, term_p / 3.0, term_m / 3.0)
            )

        A_list.append(A)
        lambda_list.append(lam)

    return jnp.stack(A_list, axis=0), jnp.stack(lambda_list, axis=0)


def _vectorized_godunov(
    A_stack: jnp.ndarray,
    lambda_stack: jnp.ndarray,
    spacing_sq_inv: jnp.ndarray,
    inv_speed: jnp.ndarray,
    dr: jnp.ndarray,
) -> jnp.ndarray:
    """Fully vectorized Godunov upwind scheme generalized for arbitrary spatial weightings."""
    ndim = A_stack.shape[0]
    grid_shape = A_stack.shape[1:]
    rhs2 = inv_speed * inv_speed

    order = jnp.argsort(A_stack, axis=0)
    a_sorted = jnp.take_along_axis(A_stack, order, axis=0)

    # Dynamic weighting incorporating lambda for the generalized quadratic scheme
    reshape_dims = [ndim] + [1] * len(grid_shape)
    w_base = spacing_sq_inv.reshape(reshape_dims)
    w_stack = lambda_stack * w_base
    w_sorted = jnp.take_along_axis(w_stack, order, axis=0)

    next_sorted = jnp.concatenate(
        [a_sorted[1:], jnp.full((1, *grid_shape), jnp.inf, dtype=a_sorted.dtype)],
        axis=0,
    )

    A = jnp.zeros(grid_shape, dtype=A_stack.dtype)
    B = jnp.zeros(grid_shape, dtype=A_stack.dtype)
    C = jnp.zeros(grid_shape, dtype=A_stack.dtype)
    candidate = jnp.full(grid_shape, jnp.inf, dtype=A_stack.dtype)
    found = jnp.zeros(grid_shape, dtype=bool)

    # Unroll the loop manually since ndim is tiny (usually 2 or 3).
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

    dr_expanded = dr.reshape(reshape_dims)
    fallback = jnp.min(
        jnp.where(
            jnp.isfinite(A_stack),
            A_stack + dr_expanded * inv_speed / jnp.sqrt(lambda_stack),
            jnp.inf,
        ),
        axis=0,
    )
    return jnp.where(found, candidate, fallback)


def jacobi_single_source(
    phi: Float[Array, "*grid"],
    distance: Float[Array, "*grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
    order: int,
    debug: bool,
    debug_verbosity: int,
) -> Float[Array, "*grid"]:
    ndim = phi.ndim

    # The nodes inside the analytical radius are permanently fixed
    fixed_mask = phi <= 0.0

    # Ensure at least the source node is caught if the radius is somehow too small
    fallback_mask = distance == jnp.min(distance)
    fixed_mask = jnp.where(jnp.any(fixed_mask), fixed_mask, fallback_mask)

    # Initialize exact analytical time: Time = Distance / Velocity
    v_source = jnp.max(jnp.where(distance == jnp.min(distance), speed, 0.0))

    T_init = jnp.where(
        fixed_mask,
        jnp.asarray(distance / v_source, dtype=speed.dtype),
        jnp.asarray(jnp.inf, dtype=speed.dtype),
    )

    spacing_sq_inv = 1.0 / (dr * dr)
    inv_speed = 1.0 / speed
    tolerance_j = jnp.asarray(tolerance, dtype=speed.dtype)

    def cond_fun(state):
        T, diff, i = state
        should_continue = (diff > tolerance_j) & (i < max_iter)

        if debug and debug_verbosity >= 2:
            jax.debug.print(
                "ITER: Iteration: {:d}, Max Diff: {:.16f}, Tolerance: {:.16f}",
                i, diff, tolerance_j,
            )

        return should_continue

    def body_fun(state):
        T, _, i = state
        A_stack, lambda_stack = _get_upwind_stencil(T, ndim, order)
        T_candidate = _vectorized_godunov(
            A_stack, lambda_stack, spacing_sq_inv, inv_speed, dr
        )

        T_new = jnp.minimum(T, T_candidate)
        T_new = jnp.where(fixed_mask, T_init, T_new)

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

    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    T_final, diff_final, i_final = final_state

    if debug and debug_verbosity >= 1:
        terminated_because_tolerance = diff_final <= tolerance_j
        terminated_because_iteration = i_final >= max_iter

        jax.debug.print(
            "CONV: Tol: {:.16f}, Diff: {:.16f}, Iteration: {:d}, Tolerance Termination: {}, Iteration Termination: {}, Should Continue: False",
            tolerance_j, diff_final, i_final, terminated_because_tolerance, terminated_because_iteration,
        )

    return T_final


@jax.jit(static_argnames=("max_iter", "tolerance", "order", "debug", "debug_verbosity"))
def jacobi_multi_source(
    phis: Float[Array, "N_sources *grid"],
    distances: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    max_iter: int,
    tolerance: float,
    order: int,
    debug: bool,
    debug_verbosity: int,
):
    return jax.vmap(
        lambda phi, distance: jacobi_single_source(
            phi=phi,
            distance=distance,
            speed=speed,
            dr=dr,
            max_iter=max_iter,
            tolerance=tolerance,
            order=order,
            debug=debug,
            debug_verbosity=debug_verbosity,
        ),
        in_axes=(0, 0),
    )(phis, distances)
