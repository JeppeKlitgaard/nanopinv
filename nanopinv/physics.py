from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import skfmm
from numba import njit, prange
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from nanopinv._typing import Array, Float, typecheck


@typecheck
def travel_time_points(
    r: Sequence[Float[Array, "ax"]],
    speeds: Float[Array, "{', '.join([f'ax_{i}' for i in range(len(r))])}"],
    source: Float[Array, "{len(r)}"],
    receivers: Float[Array, "receivers, {len(r)}"],
    window: int = 2,
    order: int = 2,
    check: bool = True,
):
    # -> Float[Array, "receivers"]:
    """
    Calculates the travel time between point `source` and point `receiver` in an ND grid `r` given the velocity field `velocities`.

    Note that the grid must be equidistant along each axis.

    Arguments:
        r: Grid points
            Must be a sequence of 1D arrays, one for each axis.
            The shape of the grid is determined by the lengths of these arrays.
            Must be sorted in ascending order.

        check: if True, does some basic checks on the inputs and raises ValueError if they are not valid. Default is True.
    """
    ndim = len(r)
    shape_grid = tuple(len(arr) for arr in r)
    # shape_out = ()

    if check and not np.all(speeds > 0.0):
        raise ValueError("Speeds must be positive")

    # Calculate and check the grid spacing
    dr = []
    for arr in r:
        if check and len(arr) < 2:
            raise ValueError(
                "Each grid axis must contain at least two points to determine spacing"
            )

        d_arr = np.diff(arr)
        if check and not np.allclose(d_arr, d_arr[0]):
            raise ValueError("Grid must be equidistant along each axis")

        dr.append(float(d_arr[0]))

    # scikit-fmm requires us to pass in a level set function, phi, where the source boundary is understood to be the zero contour of phi.
    # This can be in-between grid points.
    # We can calculate phi as the signed distance from the source point to nearby grid points
    phi = np.ones(shape=shape_grid)

    nearest_idx = [np.argmin(np.abs(r[i] - source[i])) for i in range(ndim)]
    slices_nearby = []
    axes_nearby = []
    for i in range(ndim):
        idx = nearest_idx[i]

        start = max(0, idx - window)
        end = min(shape_grid[i], idx + window + 1)

        slice_ = slice(start, end)
        slices_nearby.append(slice_)
        axes_nearby.append(r[i][slice_])

    # Compute distances
    grids_nearby = np.ix_(*axes_nearby)
    dists_nearby = np.sqrt(sum((grids_nearby[i] - source[i]) ** 2 for i in range(ndim)))
    radius = np.min(dists_nearby) + 1e-10
    dists_nearby -= radius

    # Update phi to get zero contour matrix
    phi[tuple(slices_nearby)] = dists_nearby

    # Carry out fast marching method
    travel_times_full = skfmm.travel_time(phi=phi, speed=speeds, dx=dr, order=order)

    # Get the interpolator for the travel time field
    interpolator = RegularGridInterpolator(
        points=r, values=travel_times_full, bounds_error=False, fill_value=np.inf
    )
    receiver_travel_times = interpolator(receivers)

    return receiver_travel_times


def build_time_travel_points(
    sources, receivers, *grid_axes, order: int = 2, window: int = 1
):
    """
    Factory function to build a forward model evaluator for sequential Numpy arrays.

    Pre-computes the unique sources and inverse mapping to minimize overhead.
    Returns a callable `forward_model_impl(m)` that evaluates against a given velocity field `m`.
    """
    unique_sources, inverse_indices = np.unique(sources, axis=0, return_inverse=True)
    N_data = len(sources)
    r = list(grid_axes)

    def forward_model_impl(m):
        # NOTE: We want to support batched `m` matching the JAX signature `*batch *grid`:
        shape_grid = tuple(len(arr) for arr in r)
        batch_shape = m.shape[: -len(shape_grid)]
        N_models = int(np.prod(batch_shape)) if batch_shape else 1

        m_flat = m.reshape((N_models, *shape_grid))
        out_flat = np.empty((N_models, N_data), dtype=np.float64)

        for m_idx in range(N_models):
            for i, u_src in enumerate(unique_sources):
                mask = inverse_indices == i
                associated_receivers = receivers[mask]

                # Call the core travel time calculator for this specific source
                tt = travel_time_points(
                    r=r,
                    speeds=m_flat[m_idx],
                    source=u_src,
                    receivers=associated_receivers,
                    window=window,
                    order=order,
                    check=False,  # Set to false to avoid repeating diff computations
                )

                out_flat[m_idx, mask] = tt

        return out_flat.reshape((*batch_shape, N_data))

    return forward_model_impl
