from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import PyTree

from ...utils import PhysicalConstants
from .stats import AbstractStandardStatsModel


from simple_pytree import Pytree
from jaxtyping import Array


@dataclass
class MinimalMixedLayerInitConds(Pytree):
    """Minimal mixed layer model initial state."""

    # the following variables are expected to be initialized by the user
    h_abl: Array
    """Initial ABL height [m]."""
    surf_pressure: Array
    """Surface pressure [Pa]."""
    theta: Array
    """Initial mixed-layer potential temperature [K]."""
    deltatheta: Array
    """Initial temperature jump at h [K]."""
    wtheta: Array
    """Surface kinematic heat flux [K m/s]."""
    q: Array
    """Initial mixed-layer specific humidity [kg/kg]."""
    dq: Array
    """Initial specific humidity jump at h [kg/kg]."""
    wq: Array
    """Surface kinematic moisture flux [kg/kg m/s]."""
    co2: Array
    """Initial mixed-layer CO2 [ppm]."""
    deltaCO2: Array
    """Initial CO2 jump at h [ppm]."""
    wCO2: Array
    """Surface kinematic CO2 flux [mgC/m²/s]."""
    u: Array
    """Initial mixed-layer u-wind speed [m/s]."""
    v: Array
    """Initial mixed-layer v-wind speed [m/s]."""
    dz_h: Array
    """Transition layer thickness [-]."""

    # the following variables are initialized as zero
    wstar: Array = 1e-6
    """Convective velocity scale [m s-1]."""
    wqe: Array = 0.0
    """Entrainment moisture flux [kg kg-1 m s-1]."""
    wCO2A: Array = 0.0
    """Surface assimilation CO2 flux [mgC m-2 s]."""
    wCO2R: Array = 0.0
    """Surface respiration CO2 flux [mgC m-2 s]."""
    wCO2M: Array = 0.0
    """CO2 mass flux [mgC m-2 s]."""
    wCO2e: Array = 0.0
    """Entrainment CO2 flux [mgC m-2 s]."""

    # the following variables are expected to be assigned during warmup
    thetav: Array = jnp.nan
    """Mixed-layer potential temperature [K]."""
    wthetav: Array = jnp.nan
    """Surface kinematic virtual heat flux [K m s-1]."""
    qsat: Array = jnp.nan
    """Saturation specific humidity [kg/kg]."""
    e: Array = jnp.nan
    """Vapor pressure [Pa]."""
    esat: Array = jnp.nan
    """Saturation vapor pressure [Pa]."""
    lcl: Array = jnp.nan
    """Lifting condensation level [m]."""
    deltathetav: Array = jnp.nan
    """Virtual temperature jump at h [K]."""
    top_p: Array = jnp.nan
    """Pressure at top of mixed layer [Pa]."""
    top_T: Array = jnp.nan
    """Temperature at top of mixed layer [K]."""
    top_rh: Array = jnp.nan
    """Relative humidity at top of mixed layer [-]."""


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    """Minimal mixed layer model with constant properties."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """Pass."""
        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        """Pass."""
        return state
