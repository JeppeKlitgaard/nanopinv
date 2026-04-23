from collections.abc import Sequence
from itertools import product
from typing import Literal

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from nanopinv._typing import Array, Float
from nanopinv.physics.eikonal.nanopinv_fsm import jacobi_multi_source
from nanopinv.physics.eikonal.nanopinv_test_solver3 import hyperplane_fsm_multi_source
from nanopinv.physics.eikonal.nanopinv_test_solver4 import user_fsm_2d_multi_source
from nanopinv.physics.eikonal.skfmm_fmm import skfmm_jax_caller
from nanopinv.physics.eikonal.nanopinv_test_solver1 import fast_sweeping_multi_source, build_fsm_stencils
from nanopinv.physics.eikonal.nanopinv_test_solver2 import ifim_multi_source

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


def build_travel_time_points(
    sources,
    receivers,
    *grid_axes,
    solver: _SolverNameT,
    order: int = 2,
    window: int = 1,
    fsm_max_sweeps: int = 64,
    fsm_tolerance: float = 1e-8,
    jacobi_max_iter: int = 2000,
    debug: bool = False,
    debug_verbosity: int = 1,
):
    if solver not in _SOLVER_NAMES:
        raise ValueError(f"solver must be one of {_SOLVER_NAMES}")

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

    match solver:
        case "nanopinv:fsm":

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return jacobi_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    max_iter=jacobi_max_iter,
                    tolerance=fsm_tolerance,
                    debug=debug,
                    debug_verbosity=debug_verbosity,
                )

        case "skfmm:fmm":

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return skfmm_jax_caller(phis_batched, m_single, dr, order)

        case "nanopinv:test_solver1":
            sweep_orders_np, neighbors_minus_np, neighbors_plus_np = (
                build_fsm_stencils(shape_grid)
            )
            sweep_orders = jnp.asarray(sweep_orders_np)
            neighbors_minus = jnp.asarray(neighbors_minus_np)
            neighbors_plus = jnp.asarray(neighbors_plus_np)

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return fast_sweeping_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    sweep_orders=sweep_orders,
                    neighbors_minus=neighbors_minus,
                    neighbors_plus=neighbors_plus,
                    max_sweeps=fsm_max_sweeps,
                    tolerance=fsm_tolerance,
                )

        case "nanopinv:test_solver2":

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return ifim_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    max_iter=jacobi_max_iter,
                    tolerance=fsm_tolerance,
                )

        case "nanopinv:test_solver3":

            @jax.jit
            def solve_eikonal(m_single: Float[Array, "*grid"]):
                return hyperplane_fsm_multi_source(
                    phis=phis_batched,
                    speed=m_single,
                    dr=dr,
                    max_sweeps=fsm_max_sweeps,
                    tolerance=fsm_tolerance,
                )

        case "nanopinv:test_solver4":
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
                    iterations=fsm_max_sweeps,
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
