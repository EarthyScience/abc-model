from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractState


@dataclass
class ResidualLayerState(AbstractState):
    """Residual layer state — the "frozen" daytime mixed layer aloft at night.

    Captured from the convective mixed layer at sunset, preserved
    unchanged through the night, and used to re-arm the mixed layer
    at the next sunrise.
    """

    theta: Array = field(
        metadata={
            "label": r"$\theta_{res}$",
            "unit": "K",
            "description": "Residual layer potential temperature",
        }
    )
    q: Array = field(
        metadata={
            "label": r"$q_{res}$",
            "unit": "kg/kg",
            "description": "Residual layer specific humidity",
        }
    )
    co2: Array = field(
        metadata={
            "label": r"$CO_{2,res}$",
            "unit": "ppm",
            "description": "Residual layer CO2 concentration",
        }
    )
    u: Array = field(
        metadata={
            "label": r"$u_{res}$",
            "unit": "m/s",
            "description": "Residual layer zonal wind",
        }
    )
    v: Array = field(
        metadata={
            "label": r"$v_{res}$",
            "unit": "m/s",
            "description": "Residual layer meridional wind",
        }
    )
    h: Array = field(
        metadata={
            "label": r"$h_{res}$",
            "unit": "m",
            "description": "Residual layer depth (previous day max h_abl)",
        }
    )
    delta_theta: Array = field(
        metadata={
            "label": r"$\Delta\theta_{res}$",
            "unit": "K",
            "description": "Potential temperature jump at top of residual layer",
        }
    )
    delta_q: Array = field(
        metadata={
            "label": r"$\Delta q_{res}$",
            "unit": "kg/kg",
            "description": "Specific humidity jump at top of residual layer",
        }
    )
    delta_co2: Array = field(
        metadata={
            "label": r"$\Delta CO_{2,res}$",
            "unit": "ppm",
            "description": "CO2 jump at top of residual layer",
        }
    )
    dz_h: Array = field(
        metadata={
            "label": r"$dz_{h,res}$",
            "unit": "m",
            "description": "Transition layer thickness at top of residual layer",
        }
    )
    thetav: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\theta_{v,res}$",
            "unit": "K",
            "description": "Residual layer virtual potential temperature",
        },
    )
    deltathetav: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\Delta\theta_{v,res}$",
            "unit": "K",
            "description": "Virtual temperature jump at top of residual layer",
        },
    )


class ResidualLayerModel:
    """Residual layer model — a passive container for the frozen daytime mixed layer.

    The residual layer is captured from the convective mixed layer at
    the day→night transition and released back at the night→day transition.
    """

    def init_state(
        self,
        theta: float = 288.0,
        q: float = 0.008,
        co2: float = 422.0,
        u: float = 6.0,
        v: float = -4.0,
        h: float = 200.0,
        delta_theta: float = 1.0,
        delta_q: float = -0.001,
        delta_co2: float = -44.0,
        dz_h: float = 150.0,
    ) -> ResidualLayerState:
        """Initialize the residual layer state.

        Args:
            theta: Residual layer potential temperature [K].
            q: Residual layer specific humidity [kg/kg].
            co2: Residual layer CO2 [ppm].
            u: Residual layer zonal wind [m/s].
            v: Residual layer meridional wind [m/s].
            h: Residual layer depth [m].
            delta_theta: Potential temperature jump at top [K].
            delta_q: Specific humidity jump at top [kg/kg].
            delta_co2: CO2 jump at top [ppm].
            dz_h: Transition layer thickness [m].

        Returns:
            The initial residual layer state.
        """
        return ResidualLayerState(
            theta=jnp.array(theta),
            q=jnp.array(q),
            co2=jnp.array(co2),
            u=jnp.array(u),
            v=jnp.array(v),
            h=jnp.array(h),
            delta_theta=jnp.array(delta_theta),
            delta_q=jnp.array(delta_q),
            delta_co2=jnp.array(delta_co2),
            dz_h=jnp.array(dz_h),
        )

    def compute_thetav(self, theta: Array, q: Array) -> Array:
        """Compute virtual potential temperature."""
        return theta * (1.0 + 0.61 * q)

    def compute_deltathetav(
        self,
        theta: Array,
        delta_theta: Array,
        q: Array,
        delta_q: Array,
    ) -> Array:
        """Compute virtual potential temperature jump at top."""
        return (theta + delta_theta) * (1.0 + 0.61 * (q + delta_q)) - theta * (
            1.0 + 0.61 * q
        )
