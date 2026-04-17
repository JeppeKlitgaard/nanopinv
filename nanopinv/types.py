import equinox as eqx

from nanopinv._typing import Array, Float


class Observations(eqx.Module):
    data_obs: Float[Array, "N_data"]
    data_std: Float[Array, "N_data"]
