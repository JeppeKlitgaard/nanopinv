import jax
from jax import lax
from jax import numpy as jnp

from nanopinv._typing import Array, Float


def user_fsm_2d_single_source_jax(
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
    fixed_cells = jnp.where(jnp.any(fixed_flat), fixed_flat, fallback_fixed).reshape(
        nx, ny
    )

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
                            (a + b + jnp.sqrt(2 * (f[iy, ix] * dh) ** 2 - (a - b) ** 2))
                            / 2,
                        ),
                    ),
                )
                return _curr_grid.at[piy, pix].set(updated_val)

            x_indices = jnp.arange(x_start, x_end, x_step)
            return lax.fori_loop(
                0,
                len(x_indices),
                lambda ix, val: x_loop_body(x_indices[ix], val),
                curr_grid,
            )

        y_indices = jnp.arange(y_start, y_end, y_step)
        return lax.fori_loop(
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

    final_grid = lax.fori_loop(0, iterations, iteration_body, padded)
    return final_grid[1:-1, 1:-1]


def user_fsm_2d_multi_source(
    phis: Float[Array, "N_sources *grid"],
    speed: Float[Array, "*grid"],
    dr: Float[Array, "ndim"],
    iterations: int,
):
    return jax.vmap(
        lambda phi: user_fsm_2d_single_source(
            phi=phi, speed=speed, dr=dr, iterations=iterations
        ),
        in_axes=0,
    )(phis)
