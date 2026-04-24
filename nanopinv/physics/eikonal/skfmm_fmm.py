from typing import Any

import jax
import numpy as np
import skfmm
from jax import lax
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from joblib import Parallel, delayed

from nanopinv._typing import Array, Float

_PARALLEL_ARGS_DEFAULT = {
    "n_jobs": -1,
    "prefer": "processes",
    "batch_size": "auto",
    "return_as": "generator",
}


def make_skfmm_jax_caller(
    radius: float,
    order: int,
    chunk_size: int,
    parallel_args: dict[str, Any] | None,
):
    assert isinstance(radius, float), (
        f"Expected radius to be a float, got {type(radius)}"
    )
    assert isinstance(order, int), f"Expected order to be an integer, got {type(order)}"
    assert isinstance(chunk_size, int), (
        f"Expected chunk_size to be an integer, got {type(chunk_size)}"
    )

    parallel_args = {} if parallel_args is None else parallel_args
    parallel_args = _PARALLEL_ARGS_DEFAULT | parallel_args

    def _skfmm_caller(
        phi: np.ndarray,
        distance: np.ndarray,
        speeds: np.ndarray,
        dx: np.ndarray,
    ):
        tau_skfmm = skfmm.travel_time(phi, speeds, dx=dx, order=order)

        if radius <= 1e-12:
            return tau_skfmm

        source_idx = np.unravel_index(np.argmin(distance), distance.shape)
        v_source = speeds[source_idx]

        tau_total = tau_skfmm + (radius / v_source)

        inside_mask = phi <= 0.0
        tau_total[inside_mask] = distance[inside_mask] / v_source

        return tau_total

    def _batched_skfmm_mp_caller(
        phis: Float[Array, "N_sources *grid"],
        distances: Float[Array, "N_sources *grid"],
        speed: Float[Array, "*batch *grid"],
        dx: Float[Array, "{len(phis.shape)-1}"],
    ):
        phis_np = np.asarray(phis)
        distances_np = np.asarray(distances)
        speed_np = np.asarray(speed)
        dx_np = np.asarray(dx)

        ndim = len(dx_np)
        grid_shape = phis_np.shape[1:]
        N_sources = phis_np.shape[0]

        batch_shape = speed_np.shape[:-ndim]
        N_models = int(np.prod(batch_shape)) if batch_shape else 1

        speed_flat = speed_np.reshape((N_models, *grid_shape))

        tasks_phi = []
        tasks_dist = []
        tasks_speed = []
        for m_idx in range(N_models):
            for s_idx in range(N_sources):
                tasks_phi.append(phis_np[s_idx])
                tasks_dist.append(distances_np[s_idx])
                tasks_speed.append(speed_flat[m_idx])

        results = Parallel(**parallel_args)(
            delayed(_skfmm_caller)(p, d, s, dx_np)
            for p, d, s in zip(tasks_phi, tasks_dist, tasks_speed)
        )

        res = np.empty((N_models, N_sources, *grid_shape), dtype=phis_np.dtype)
        for i, r in enumerate(results):
            m_idx = i // N_sources
            s_idx = i % N_sources
            res[m_idx, s_idx] = r

        return res.reshape((*batch_shape, N_sources, *grid_shape))

    @custom_vmap
    def skfmm_jax_caller(
        phis,
        distances,
        speed,
        dr,
    ):
        ndim = len(dr)
        batch_shape = speed.shape[:-ndim]
        grid_shape = speed.shape[-ndim:]

        result_shape = jax.ShapeDtypeStruct(
            (*batch_shape, phis.shape[0], *grid_shape), phis.dtype
        )

        return jax.pure_callback(
            _batched_skfmm_mp_caller,
            result_shape,
            phis=phis,
            distances=distances,
            speed=speed,
            dx=dr,
            # Pass all in one big batch, this gets intercepted by custom_vmap below
            # and split into chunks
            vmap_method="broadcast_all",
        )

    @skfmm_jax_caller.def_vmap
    def skfmm_jax_caller_vmap(
        axis_size,
        in_batched,
        phis,
        distances,
        speed,
        dr,
    ):
        """
        Arguments:
            chunk_size: Size of the chunk that gets passed on to the CPU for parallel processing
        """

        ndim = len(dr)
        batch_shape = speed.shape[:-ndim]
        grid_shape = speed.shape[-ndim:]
        n_models = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

        flat_speed = speed.reshape((n_models, *grid_shape))

        # Pad batch to cleanly divide by chunk_size.
        # We repeat the last model instead of appending zeros to avoid Eikonal singularities.
        pad_size = (chunk_size - (n_models % chunk_size)) % chunk_size
        if pad_size > 0:
            pad_speeds = jnp.repeat(flat_speed[-1:], pad_size, axis=0)
            flat_speed = jnp.concatenate([flat_speed, pad_speeds], axis=0)

        # Reshape into fixed chunks
        chunked_speed = flat_speed.reshape((-1, chunk_size, *grid_shape))

        # lax.map passes inputs purely; no JAX vectorization hits this callback,
        def process_chunk(speed_chunk):
            res_shape = jax.ShapeDtypeStruct(
                (chunk_size, phis.shape[0], *grid_shape), phis.dtype
            )
            return jax.pure_callback(
                _batched_skfmm_mp_caller,
                res_shape,
                phis=phis,
                distances=distances,
                speed=speed_chunk,
                dx=dr,
                vmap_method="broadcast_all",
            )

        # Process the chunks sequentially on the GPU, parallelized on the CPU host.
        chunked_results = lax.map(process_chunk, chunked_speed)

        # Strip padding and restore original batch geometry.
        flat_results = chunked_results.reshape((-1, phis.shape[0], *grid_shape))
        valid_results = flat_results[:n_models]
        travel_times_batched = valid_results.reshape(
            (*batch_shape, phis.shape[0], *grid_shape)
        )

        return travel_times_batched, True

    return skfmm_jax_caller
