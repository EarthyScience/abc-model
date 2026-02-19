from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ..abstracts import AbstractCoupledState, AbstractLandModel, AbstractLandState
from ..utils import compute_esat, compute_qsat


# limamau: this could be much simpler!
@dataclass
class MinimalLandSurfaceState(AbstractLandState):
    """Minimal land surface model state."""

    alpha: Array = field(
        metadata={
            "label": r"$\alpha$",
            "unit": "-",
            "description": "Surface albedo",
        }
    )
    """surface albedo [-], range 0 to 1."""
    surf_temp: Array = field(
        metadata={
            "label": r"$T_{surf}$",
            "unit": "K",
            "description": "Surface temperature",
        }
    )
    """Surface temperature [K]."""
    rs: Array = field(
        metadata={
            "label": r"$r_s$",
            "unit": "s m^{-1}",
            "description": "Surface resistance",
        }
    )
    """Surface resistance [s m-1]."""
    wg: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$w_g$",
            "unit": "m^3 m^{-3}",
            "description": "Soil moisture content",
        },
    )
    """No moisture content in the root zone [m3 m-3]."""
    wl: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$W_l$",
            "unit": "m",
            "description": "Canopy water content",
        },
    )
    """No water content in the canopy [m]."""

    # the following variables are assigned during warmup/timestep
    esat: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$e_{sat}$",
            "unit": "Pa",
            "description": "Saturation vapor pressure",
        },
    )
    """Saturation vapor pressure [Pa]."""
    qsat: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$q_{sat}$",
            "unit": "kg kg^{-1}",
            "description": "Saturation specific humidity",
        },
    )
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$dq_{sat}/dT$",
            "unit": "kg kg^{-1} K^{-1}",
            "description": "Derivative of saturation specific humidity",
        },
    )
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$e$",
            "unit": "Pa",
            "description": "Vapor pressure",
        },
    )
    """Vapor pressure [Pa]."""
    qsatsurf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$q_{sat}(T_s)$",
            "unit": "kg kg^{-1}",
            "description": "Saturation specific humidity at surface",
        },
    )
    """Saturation specific humidity at surface temperature [kg/kg]."""
    wtheta: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'\theta')_{s}$",
            "unit": "K m s^{-1}",
            "description": "Kinematic heat flux",
        },
    )
    """Kinematic heat flux [K m/s]."""
    wq: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'q')_{s}$",
            "unit": "kg kg^{-1} m s^{-1}",
            "description": "Kinematic moisture flux",
        },
    )
    """Kinematic moisture flux [kg/kg m/s]."""
    wCO2: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$w'CO_2'$",
            "unit": "kg kg^{-1} m s^{-1}",
            "description": "Kinematic CO2 flux",
        },
    )
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""


# alias
class MinimalLandSurfaceModel(AbstractLandModel):
    """Minimal land surface model with fixed surface properties."""

    def __init__(self):
        self.d1 = 0.0

    def init_state(
        self,
        alpha: float = 0.25,
        surf_temp: float = 290.0,
        rs: float = 1.0e6,
        wg: float = 0.0,
        wl: float = 0.0,
        wtheta: float = 0.0,
    ) -> MinimalLandSurfaceState:
        """Initialize the model state.

        Args:
            alpha: surface albedo [-], range 0 to 1. Default is 0.25.
            surf_temp: Surface temperature [K]. Default is 290.0.
            rs: Surface resistance [s m-1]. Default is 1.0e6.
            wg: Volumetric soil moisture [m3 m-3]. Default is 0.0.
            wl: Canopy water content [m]. Default is 0.0.
            wtheta: Kinematic heat flux [K m/s]. Default is 0.0.

        Returns:
            The initial land state.
        """
        return MinimalLandSurfaceState(
            alpha=jnp.array(alpha),
            surf_temp=jnp.array(surf_temp),
            rs=jnp.array(rs),
            wg=jnp.array(wg),
            wl=jnp.array(wl),
            wtheta=jnp.array(wtheta),
        )

    def run(
        self,
        state: AbstractCoupledState,
    ) -> MinimalLandSurfaceState:
        """Run the model.

        Args:
            state: CoupledState.

        Returns:
            The updated land state object.
        """
        land_state = state.land
        atmos = state.atmos
        esat = compute_esat(atmos.theta)
        qsat = compute_qsat(atmos.theta, atmos.surf_pressure)
        dqsatdT = self.compute_dqsatdT(esat, atmos.theta, atmos.surf_pressure)
        e = self.compute_e(atmos.q, atmos.surf_pressure)
        return replace(land_state, esat=esat, qsat=qsat, dqsatdT=dqsatdT, e=e)

    def compute_dqsatdT(self, esat: Array, theta: float, surf_pressure: float) -> Array:
        """Compute the derivative of saturation vapor pressure with respect to temperature ``dqsatdT``.

        Notes:
            Using :meth:`~abcmodel.utils.compute_esat`, the derivative of the saturated vapor pressure
            :math:`e_\\text{sat}` with respect to temperature :math:`T` is given by

            .. math::
                \\frac{\\text{d}e_\\text{sat}}{\\text{d} T} =
                e_\\text{sat}\\frac{17.2694(T-237.16)}{(T-35.86)^2},

            which combined with :meth:`~abcmodel.utils.compute_qsat` can be used to get

            .. math::
                \\frac{\\text{d}q_{\\text{sat}}}{\\text{d} T} \\approx \\epsilon \\frac{\\frac{\\text{d}e_\\text{sat}}{\\text{d} T}}{p}.
        """
        num = 17.2694 * (theta - 273.16)
        den = (theta - 35.86) ** 2.0
        mult = num / den
        desatdT = esat * mult
        return 0.622 * desatdT / surf_pressure

    def compute_e(self, q: Array, surf_pressure: Array) -> Array:
        """Compute the vapor pressure ``e``.

        Notes:
            This function uses the same formula used in :meth:`~abcmodel.utils.compute_esat`,
            but now factoring the vapor pressure :math:`e` as a function of specific humidity :math:`q`
            and surface pressure :math:`p`, which give us

            .. math::
                e = q \\cdot p / 0.622.
        """
        return q * surf_pressure / 0.622

    def integrate(
        self, state: MinimalLandSurfaceState, dt: float
    ) -> MinimalLandSurfaceState:
        """Integrate the model forward in time.

        Args:
            state: the state object carrying all variables.
            dt: the time step.

        Returns:
            The updated state object.
        """
        return state
