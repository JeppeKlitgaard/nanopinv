from collections.abc import Sequence
from itertools import product
from typing import Any, Literal

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from nanopinv._typing import Array, Float
from nanopinv.physics.eikonal.nanopinv_fsm import jacobi_multi_source
from nanopinv.physics.eikonal.nanopinv_test_solver1 import (
    build_fsm_stencils,
    fast_sweeping_multi_source,
)
from nanopinv.physics.eikonal.nanopinv_test_solver2 import ifim_multi_source
from nanopinv.physics.eikonal.nanopinv_test_solver3 import hyperplane_fsm_multi_source
from nanopinv.physics.eikonal.nanopinv_test_solver4 import user_fsm_2d_multi_source
from nanopinv.physics.eikonal.skfmm_fmm import make_skfmm_jax_caller

_SOLVER_NAMES = (
    "skfmm:fmm",
    "nanopinv:fsm",
    "nanopinv:test_solver1",
    "nanopinv:test_solver2",
    "nanopinv:test_solver3",
    "nanopinv:test_solver4",
)

_SolverNameT = Literal[
    "skfmm:fmm",
    "nanopinv:fsm",
    "nanopinv:test_solver1",
    "nanopinv:test_solver2",
    "nanopinv:test_solver3",
    "nanopinv:test_solver4",
]


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


@jax.jit(static_argnames=["radius_multiplier"])
def compute_phi_and_distance(
    source: Float[Array, "ndim"],
    r: Sequence[Float[Array, "ax"]],
    dr: Float[Array, "ndim"],
    radius_multiplier: float,
):
    ndim = len(r)

    # 1. Compute global Euclidean distance smoothly across the whole grid
    mesh = jnp.meshgrid(*r, indexing="ij")
    dist_sq = jnp.zeros_like(mesh[0])
    for i in range(ndim):
        dist_sq += (mesh[i] - source[i]) ** 2
    distance = jnp.sqrt(dist_sq)

    # 2. Define an analytical radius to bypass the singularity
    # We use 2.5x the smallest grid spacing to ensure a healthy ring of nodes
    radius = radius_multiplier * jnp.min(dr)

    # 3. The level-set is the distance shifted by the radius
    phi = distance - radius

    return phi, distance, radius


def build_travel_time_points(
    sources,
    receivers,
    *grid_axes,
    solver: _SolverNameT,
    radius_multiplier: float = 1.5,
    solver_kwargs: dict | None = None,
    debug: bool = False,
    debug_verbosity: int = 1,
):
    if solver not in _SOLVER_NAMES:
        raise ValueError(f"solver must be one of {_SOLVER_NAMES}")

    solver_kwargs = {} if solver_kwargs is None else solver_kwargs

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

    vmap_compute_phi_and_distance = jax.vmap(
        compute_phi_and_distance, in_axes=(0, None, None, None), out_axes=(0, 0, None)
    )
    phis_batched, distances_batched, radius = vmap_compute_phi_and_distance(
        unique_sources_j, r, dr, radius_multiplier
    )

    match solver:
        case "nanopinv:fsm":
            max_iter: int = solver_kwargs.get("max_iter", 2000)
            tolerance: float = solver_kwargs.get("tolerance", 1e-8)

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return jacobi_multi_source(
                    phis=phis_batched,
                    distances=distances_batched,
                    speed=m_single,
                    dr=dr,
                    max_iter=max_iter,
                    tolerance=tolerance,
                    debug=debug,
                    debug_verbosity=debug_verbosity,
                )

        case "skfmm:fmm":
            order: int = solver_kwargs.get("order", 2)
            chunk_size: int = solver_kwargs.get("chunk_size", 64)
            parallel_args: dict[str, Any] = solver_kwargs.get("parallel_args", {})
            jax_caller = make_skfmm_jax_caller(
                radius=float(radius),  # Must convert to Python float, we get JAX array back
                order=order,
                chunk_size=chunk_size,
                parallel_args=parallel_args,
            )

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return jax_caller(
                    phis=phis_batched,
                    distances=distances_batched,
                    dr=dr,
                    speed=m_single,
                )

        case "nanopinv:test_solver1":
            sweep_orders_np, neighbors_minus_np, neighbors_plus_np = build_fsm_stencils(
                shape_grid
            )
            sweep_orders = jnp.asarray(sweep_orders_np)
            neighbors_minus = jnp.asarray(neighbors_minus_np)
            neighbors_plus = jnp.asarray(neighbors_plus_np)

            max_sweeps: int = solver_kwargs.get("max_sweeps", 20)
            tolerance: float = solver_kwargs.get("tolerance", 1e-8)

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return fast_sweeping_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    sweep_orders=sweep_orders,
                    neighbors_minus=neighbors_minus,
                    neighbors_plus=neighbors_plus,
                    max_sweeps=max_sweeps,
                    tolerance=tolerance,
                )

        case "nanopinv:test_solver2":
            max_iter: int = solver_kwargs.get("max_iter", 2000)
            tolerance: float = solver_kwargs.get("tolerance", 1e-8)

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):

                return ifim_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    max_iter=max_iter,
                    tolerance=tolerance,
                )

        case "nanopinv:test_solver3":
            max_sweeps: int = solver_kwargs.get("max_sweeps", 20)
            tolerance: float = solver_kwargs.get("tolerance", 1e-8)

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return hyperplane_fsm_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    max_sweeps=max_sweeps,
                    tolerance=tolerance,
                )

        case "nanopinv:test_solver4":
            max_sweeps: int = solver_kwargs.get("max_sweeps", 20)

            if len(shape_grid) != 2:
                raise ValueError(
                    "Solver 'nanopinv:test_solver4' is strictly implemented for 2D grids."
                )
            if not jnp.allclose(dr, dr[0]):
                raise ValueError(
                    "Solver 'nanopinv:test_solver4' requires uniform grid spacing (dx == dy)."
                )

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return user_fsm_2d_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    iterations=max_sweeps,
                )

        case _:
            raise RuntimeError("This should never be reached. This is a bug!")

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
