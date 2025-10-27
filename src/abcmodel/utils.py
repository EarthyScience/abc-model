from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array


def get_esat(temp: Array) -> Array:
    """Calculate saturated vapor pressure using Tetens formula.

    Args:
        temp: temperature [K].

    Returns
        Saturated vapor pressure [Pa].
    """
    temp_celsius = temp - 273.16
    denominator = temp - 35.86
    return 0.611e3 * jnp.exp(17.2694 * temp_celsius / denominator)


def get_qsat(temp: Array, pressure: Array) -> Array:
    """Calculate saturated specific humidity.

    Args:
        temp: temperature [K].
        pressure: pressure [Pa].

    Returns
        Saturated specific humidity [kg/kg].
    """
    esat = get_esat(temp)
    return 0.622 * esat / pressure


def get_psim(zeta: Array) -> Array:
    """Calculate momentum stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L [-].

    Returns:
        Momentum stability correction [-].
    """
    # Constants for stable conditions
    alpha = 0.35
    beta = 5.0 / alpha
    gamma = (10.0 / 3.0) / alpha
    pi_half = jnp.pi / 2.0

    # unstable conditions (zeta <= 0)
    x = (1.0 - 16.0 * zeta) ** 0.25
    arctan_term = 2.0 * jnp.arctan(x)
    log_numerator = (1.0 + x) ** 2.0 * (1.0 + x**2.0)
    log_term = jnp.log(log_numerator / 8.0)
    psim_unstable = pi_half - arctan_term + log_term

    # stable conditions (zeta > 0)
    exponential_term = (zeta - beta) * jnp.exp(-alpha * zeta)
    psim_stable = -2.0 / 3.0 * exponential_term - zeta - gamma

    # select based on stability condition
    psim = jnp.where(zeta <= 0, psim_unstable, psim_stable)

    return psim


def get_psih(zeta: Array) -> Array:
    """Calculate scalar stability function from Monin-Obukhov similarity theory.

    Args:
        zeta: stability parameter z/L [-].

    Returns:
        Scalar stability correction [-].
    """
    # constants for stable conditions
    alpha = 0.35
    beta = 5.0 / alpha
    gamma = (10.0 / 3.0) / alpha

    # unstable conditions (zeta <= 0)
    x = (1.0 - 16.0 * zeta) ** 0.25
    log_argument = (1.0 + x * x) / 2.0
    psih_unstable = 2.0 * jnp.log(log_argument)

    # stable conditions (zeta > 0)
    exponential_term = (zeta - beta) * jnp.exp(-alpha * zeta)
    power_term = (1.0 + (2.0 / 3.0) * zeta) ** 1.5
    psih_stable = -2.0 / 3.0 * exponential_term - power_term - gamma + 1.0

    # select based on stability condition
    psih = jnp.where(zeta <= 0, psih_unstable, psih_stable)

    return psih

@dataclass
class PhysicalConstants:
    """Container for physical constants used throughout the model."""
    lv = 2.5e6
    """Heat of vaporization [J kg-1]."""
    cp = 1005.0
    """Specific heat of dry air [J kg-1 K-1]."""
    rho = 1.2
    """Density of air [kg m-3]."""
    g = 9.81
    """Gravity acceleration [m s-2]."""
    rd = 287.0
    """Gas constant for dry air [J kg-1 K-1]."""
    rv = 461.5
    """Gas constant for moist air [J kg-1 K-1]."""
    rhow = 1000.0
    """Density of water [kg m-3]."""
    k = 0.4
    """Von Karman constant [-]."""
    bolz = 5.67e-8
    """Boltzmann constant [-]."""
    solar_in = 1368.0
    """Solar constant [W m-2]"""
    mco2 = 44.0
    """Molecular weight CO2 [g mol-1]."""
    mair = 28.9
    """Molecular weight air [g mol-1]."""
    nuco2q = 1.6
    """Ratio molecular viscosity water to carbon dioxide."""
