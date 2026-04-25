from jax import numpy as jnp


def vectorized_godunov(
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


def get_axis_min_times(T, ndim):
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
