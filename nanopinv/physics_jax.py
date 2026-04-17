"""
This module provides a JAX-compatible implementation of the travel-time calculation.

Note that this is mostly just a wrapper around `skfmm` using `jax.pure_callback`
and appropriate shape handling for use with `jax.vmap` and `jax.jit`.

The calls top `skfmm.travel_time` are processed in parallel, which allows for a decent speedup.
"""

from collections.abc import Sequence

import jax
import numpy as np
import skfmm
from jax import lax
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import print_bindings
from joblib import Parallel, cpu_count, delayed
from numpy.typing import DTypeLike

from nanopinv._typing import Array, Float, typecheck


def _skfmm_caller(
    phi: Float[Array, "*grid"],
    speeds: Float[Array, "*grid"],
    dr: Float[Array, "{len(phi.shape)}"],
    order: int,
):
    return skfmm.travel_time(phi, speeds, dx=dr, order=order)


_PARALLEL_ARGS = {
    # "n_jobs": cpu_count(only_physical_cores=True),
    "n_jobs": -1,
    "prefer": "processes",
    "batch_size": "auto",
    "return_as": "generator",
}


# @typecheck
def _batched_skfmm_mp_caller(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*batch *grid"],
    dx: Float[Array, "{len(phis.shape)-1}"],  # len(dx) == ndim == len(phis.shape) - 1
    order: int = 2,
):
    """
    Batched wrapper for calling `skfmm.travel_time` in parallel across multiple sources and speed models.
    """
    ndim = len(dx)
    grid_shape = phis.shape[1:]
    N_sources = phis.shape[0]

    # Isolate batch dimensions (e.g., handles flat vector, single grids, or batch cubes)
    batch_shape = speed.shape[:-ndim]
    N_models = int(np.prod(batch_shape)) if batch_shape else 1

    # Reshape all speed models into a unified flat batch dimension: (N_models, *grid_shape)
    speed_flat = speed.reshape((N_models, *grid_shape))

    # Build exact tasks for CPU distribution (N_models * N_sources)
    tasks_phi = []
    tasks_speed = []
    for m_idx in range(N_models):
        for s_idx in range(N_sources):
            tasks_phi.append(phis[s_idx])
            tasks_speed.append(speed_flat[m_idx])

    results = Parallel(**_PARALLEL_ARGS)(
        delayed(_skfmm_caller)(p, s, dx, order) for p, s in zip(tasks_phi, tasks_speed)
    )

    # Repack results perfectly into (N_models, N_sources, *grid_shape)
    res = np.empty((N_models, N_sources, *grid_shape), dtype=phis.dtype)
    for i, r in enumerate(results):
        m_idx = i // N_sources
        s_idx = i % N_sources
        res[m_idx, s_idx] = r

    # Reshape back into original batch dimensions
    result = res.reshape((*batch_shape, N_sources, *grid_shape))
    return result


# Custom vmap wrapper around pure_callback to allow vmapping across the speed input without broadcasting other inputs as well.
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

    # Safely reconstruct the shape inside the vmap trace as well
    result_shape = jax.ShapeDtypeStruct(
        (*batch_shape, phis.shape[0], *grid_shape), phis.dtype
    )

    travel_times_batched = jax.pure_callback(
        _batched_skfmm_mp_caller,
        result_shape,
        phis,  # Will statically pass (N_sources, *grid_shape)
        speed,  # Batched on leading axis, shape (new_batch *old_batch_shape, *grid_shape)
        dr,  # Statically passed
        order,  # Statically passed
    )

    return travel_times_batched, True


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
    """
    `vmap`-able function to compute the distance phi map for a single source point.

    This constructs the `phi` array used in the `skfmm.travel_time` call
    by encoding the source location as a zero-contour.
    """

    ndim = len(r)
    shape_grid = tuple(len(arr) for arr in r)
    phi = jnp.ones(shape=shape_grid, dtype=source.dtype)

    # Find nearest grid point to the source
    nearest_idx = [jnp.argmin(jnp.abs(r[i] - source[i])) for i in range(ndim)]

    # Build a local window in which we construct zero-contour to set source
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

    # Ensure we have a zero-contour:
    # The radius must encapsulate at least the nearest node
    # Add a tiny epsilon so the nearest node becomes strictly <= 0
    radius = jnp.min(local_phi) + 1e-10

    local_phi -= radius

    # Update the global phi with the local window
    phi = lax.dynamic_update_slice(phi, local_phi, tuple(start_indices))
    return phi


def build_time_travel_points(
    sources, receivers, *grid_axes, order: int = 2, window: int = 1
):
    # Use numpy here since jax does support unique without known size
    unique_sources, inverse_sources = np.unique(sources, axis=0, return_inverse=True)

    unique_sources_j = jnp.asarray(unique_sources)
    inverse_j = jnp.asarray(inverse_sources)

    r = list(grid_axes)
    dr = jnp.array([jnp.diff(arr)[0] for arr in r])

    # Check that grids are equidistant and have at least 2 points
    for i, arr in enumerate(r):
        if len(arr) < 2:
            raise ValueError(
                f"Grid axis {i} must contain at least two points to determine spacing"
            )

        d_arr = jnp.diff(arr)
        if not jnp.allclose(d_arr, d_arr[0]):
            raise ValueError(f"Grid axis {i} must be equidistant")

    shape_grid = tuple(len(arr) for arr in r)

    # Pre-compute phi, does not depend on the model m.
    vmap_compute_phi = jax.vmap(compute_phi, in_axes=(0, None, None))
    phis_batched = vmap_compute_phi(unique_sources_j, r, window)

    # We create a version that takes a single model
    @jax.jit
    def single_model(m_single: Float[Array, "*grid"]):
        travel_times = _fmm_jax_caller(phis_batched, m_single, dr, order)
        tt_expanded = jnp.take(travel_times, inverse_j, axis=0)

        def interp_fn(grid, rec):
            return RegularGridInterpolator(
                points=r, values=grid, bounds_error=False, fill_value=jnp.inf
            )(rec).squeeze()

        vmap_data = jax.vmap(interp_fn, in_axes=(0, 0))
        return vmap_data(tt_expanded, receivers)

    # We recursively map the single model implementation upward
    # to account for an arbitrary number of batch dimensions that `speed` may have
    @jax.jit
    def batched_model(m: Float[Array, "*batch *grid"]):
        batch_ndim = m.ndim - len(shape_grid)

        f = single_model
        for _ in range(batch_ndim):
            f = jax.vmap(f)

        return f(m)

    return batched_model
