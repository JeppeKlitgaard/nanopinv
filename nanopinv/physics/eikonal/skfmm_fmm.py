
import jax
import numpy as np
import skfmm
from jax import lax
from jax import numpy as jnp
from jax.custom_batching import custom_vmap
from joblib import Parallel, delayed

from nanopinv._typing import Array, Float


def _skfmm_caller(
    phi: Float[Array, "*grid"],
    speeds: Float[Array, "*grid"],
    dr: Float[Array, "{len(phi.shape)}"],
    order: int,
):
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
    dx: Float[Array, "{len(phis.shape)-1}"],  # len(dx) == ndim == len(phis.shape) - 1
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
def skfmm_jax_caller(phis, speed, dr, order):
    ndim = len(dr)
    batch_shape = speed.shape[:-ndim]
    grid_shape = speed.shape[-ndim:]

    result_shape = jax.ShapeDtypeStruct(
        (*batch_shape, phis.shape[0], *grid_shape), phis.dtype
    )

    return jax.pure_callback(
        _batched_skfmm_mp_caller, result_shape, phis, speed, dr, order
    )


@skfmm_jax_caller.def_vmap
def skfmm_jax_caller_vmap(axis_size, in_batched, phis, speed, dr, order):
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
