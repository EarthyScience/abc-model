from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState, AbstractState


@dataclass
class SBLState(AbstractState):
    """Stable boundary layer (SBL) state.

    Fields with `default_factory` are diagnostic and are populated during ``run()``;
    user-provided fields are set at initialization.
    """

    h_sbl: Array = field(
        metadata={
            "label": r"$h_{sbl}$",
            "unit": "m",
            "description": "Stable boundary layer height",
        }
    )
    theta: Array = field(
        metadata={
            "label": r"$\theta$",
            "unit": "K",
            "description": "SBL potential temperature",
        }
    )
    q: Array = field(
        metadata={
            "label": r"$q$",
            "unit": "kg/kg",
            "description": "SBL specific humidity",
        }
    )
    co2: Array = field(
        metadata={
            "label": r"$CO_2$",
            "unit": "ppm",
            "description": "SBL CO2 concentration",
        }
    )
    u: Array = field(
        metadata={
            "label": r"$u$",
            "unit": "m/s",
            "description": "SBL zonal wind",
        }
    )
    v: Array = field(
        metadata={
            "label": r"$v$",
            "unit": "m/s",
            "description": "SBL meridional wind",
        }
    )
    surf_pressure: Array = field(
        metadata={
            "label": r"$p_{surf}$",
            "unit": "Pa",
            "description": "Surface pressure",
        }
    )

    thetav: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\theta_v$",
            "unit": "K",
            "description": "Virtual potential temperature",
        },
    )
    ustar: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$u_*$",
            "unit": "m/s",
            "description": "Friction velocity",
        },
    )
    obukhov: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$L$",
            "unit": "m",
            "description": "Obukhov length",
        },
    )
    wtheta: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$w'\\theta'$",
            "unit": "K m/s",
            "description": "Surface kinematic heat flux",
        },
    )
    wq: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$w'q'$",
            "unit": "kg/kg m/s",
            "description": "Surface kinematic moisture flux",
        },
    )
    wCO2: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$w'CO_2'$",
            "unit": "mgC/m²/s",
            "description": "Surface kinematic CO2 flux",
        },
    )
    thetatend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\\partial \\theta / \\partial t$",
            "unit": "K s^{-1}",
            "description": "SBL potential temperature tendency",
        },
    )
    qtend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\\partial q / \\partial t$",
            "unit": "kg/kg s^{-1}",
            "description": "SBL specific humidity tendency",
        },
    )
    co2tend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\\partial CO_2 / \\partial t$",
            "unit": "ppm s^{-1}",
            "description": "SBL CO2 tendency",
        },
    )
    utend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\\partial u / \\partial t$",
            "unit": "m s^{-2}",
            "description": "SBL zonal wind tendency",
        },
    )
    vtend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\\partial v / \\partial t$",
            "unit": "m s^{-2}",
            "description": "SBL meridional wind tendency",
        },
    )


class SBLModel:
    """Stable boundary layer model.

    Computes the SBL height using the Zilitinkevich (1972) formula
    and provides tendencies for thermodynamic variables.

    Args:
        coriolis_param: Coriolis parameter [s-1]. Default is 1e-4.
        zilitinkevich_const: Zilitinkevich constant [-]. Default is 0.4.
        is_wind_prog: Prognostic wind switch. Default is True.
        gamma_theta: Free-atmosphere theta lapse rate [K/m]. Default is 0.006.
        gamma_q: Free-atmosphere q lapse rate [kg/kg/m]. Default is 0.0.
        gamma_co2: Free-atmosphere CO2 lapse rate [ppm/m]. Default is 0.0.
        gammau: Free-atmosphere u lapse rate [s-1]. Default is 0.0.
        gammav: Free-atmosphere v lapse rate [s-1]. Default is 0.0.
    """

    def __init__(
        self,
        coriolis_param: float = 1e-4,
        zilitinkevich_const: float = 0.4,
        is_wind_prog: bool = True,
        gamma_theta: float = 0.006,
        gamma_q: float = 0.0,
        gamma_co2: float = 0.0,
        gammau: float = 0.0,
        gammav: float = 0.0,
    ):
        self.coriolis_param = coriolis_param
        self.zilitinkevich_const = zilitinkevich_const
        self.is_wind_prog = is_wind_prog
        self.gamma_theta = gamma_theta
        self.gamma_q = gamma_q
        self.gamma_co2 = gamma_co2
        self.gammau = gammau
        self.gammav = gammav

    def init_state(
        self,
        h_sbl: float = 100.0,
        theta: float = 288.0,
        q: float = 0.008,
        co2: float = 422.0,
        u: float = 6.0,
        v: float = -4.0,
        surf_pressure: float = 101300.0,
    ) -> SBLState:
        """Initialize the SBL state.

        Args:
            h_sbl: SBL height [m]. Default 100.0.
            theta: SBL potential temperature [K]. Default 288.0.
            q: SBL specific humidity [kg/kg]. Default 0.008.
            co2: SBL CO2 [ppm]. Default 422.0.
            u: SBL zonal wind [m/s]. Default 6.0.
            v: SBL meridional wind [m/s]. Default -4.0.
            surf_pressure: Surface pressure [Pa]. Default 101300.0.
        """
        return SBLState(
            h_sbl=jnp.array(h_sbl),
            theta=jnp.array(theta),
            q=jnp.array(q),
            co2=jnp.array(co2),
            u=jnp.array(u),
            v=jnp.array(v),
            surf_pressure=jnp.array(surf_pressure),
        )

    def run(
        self,
        state: AbstractCoupledState,
        h_residual: Array,
    ) -> SBLState:
        """Run the SBL model.

        Computes SBL height via Zilitinkevich and tendencies.

        Args:
            state: The coupled state.
            h_residual: Residual layer height [m] (upper bound for SBL).

        Returns:
            Updated SBL state.
        """
        sbl_state = state.atmos.active_bl.sbl
        wtheta = state.land.wtheta
        wq = state.land.wq
        wCO2 = state.land.wCO2
        ustar = state.atmos.ustar
        obukhov = state.atmos.surface.obukhov_length
        du = state.atmos.active_bl.mixed.u - sbl_state.u
        dv = state.atmos.active_bl.mixed.v - sbl_state.v
        uw = state.atmos.uw
        vw = state.atmos.vw
        h_sbl = self.compute_sbl_height(ustar, obukhov, h_residual)
        thetatend = self.compute_thetatend(h_sbl, wtheta)
        qtend = self.compute_qtend(h_sbl, wq)
        co2tend = self.compute_co2tend(h_sbl, wCO2)
        utend = self.compute_utend(h_sbl, uw, du, dv)
        vtend = self.compute_vtend(h_sbl, vw, du, dv)

        return sbl_state.replace(
            h_sbl=h_sbl,
            ustar=ustar,
            obukhov=obukhov,
            wtheta=wtheta,
            wq=wq,
            wCO2=wCO2,
            thetatend=thetatend,
            qtend=qtend,
            co2tend=co2tend,
            utend=utend,
            vtend=vtend,
        )

    def statistics(
        self,
        state: AbstractCoupledState,
    ) -> SBLState:
        """Compute SBL diagnostic statistics.

        Args:
            state: Coupled state with SBL.

        Returns:
            Updated SBL state with diagnostic fields.
        """
        sbl_state = state.atmos.active_bl.sbl
        thetav = self.compute_thetav(sbl_state.theta, sbl_state.q)
        return sbl_state.replace(thetav=thetav)

    def integrate(self, state: SBLState, dt: float) -> SBLState:
        """Integrate SBL state forward in time.

        Args:
            state: SBL state.
            dt: Time step [s].

        Returns:
            Updated SBL state.
        """
        theta = state.theta + dt * state.thetatend
        q = state.q + dt * state.qtend
        co2 = state.co2 + dt * state.co2tend
        u = jnp.where(self.is_wind_prog, state.u + dt * state.utend, state.u)
        v = jnp.where(self.is_wind_prog, state.v + dt * state.vtend, state.v)
        return state.replace(
            theta=theta,
            q=q,
            co2=co2,
            u=u,
            v=v,
        )

    def compute_sbl_height(
        self,
        ustar: Array,
        obukhov: Array,
        h_residual: Array,
    ) -> Array:
        """Compute SBL height using the Zilitinkevich (1972) formula.

        .. math::
            h_{sbl} = C_z \\, \\sqrt{\\frac{u_* \\, L}{f}}

        where :math:`C_z` is the Zilitinkevich constant, :math:`u_*` is the
        friction velocity, :math:`L` is the Obukhov length (positive in
        stable conditions), and :math:`f` is the Coriolis parameter.

        The result is clamped between 10 m and ``h_residual``.

        Args:
            ustar: Friction velocity [m/s].
            obukhov: Obukhov length [m] (positive in stable conditions).
            h_residual: Residual layer height [m] (upper bound).

        Returns:
            SBL height [m].
        """
        L_pos = jnp.maximum(obukhov, 0.1)
        product = ustar * L_pos
        safe_product = jnp.maximum(product, 1e-4)
        f_safe = jnp.maximum(self.coriolis_param, 1e-8)
        h_raw = self.zilitinkevich_const * jnp.sqrt(safe_product / f_safe)
        h_res = jnp.maximum(h_residual, 10.0)
        return jnp.clip(h_raw, 10.0, h_res)

    def compute_thetatend(
        self,
        h_sbl: Array,
        wtheta: Array,
    ) -> Array:
        """Compute potential temperature tendency for SBL.

        .. math::
            \\frac{d\\theta}{dt} = \\frac{w'\\theta'}{h_{sbl}}

        where :math:`w'\\theta'` is the surface kinematic heat flux
        (negative at night, cooling the SBL).

        Args:
            h_sbl: SBL height [m].
            wtheta: Surface kinematic heat flux [K m/s].

        Returns:
            Potential temperature tendency [K/s].
        """
        return wtheta / h_sbl

    def compute_qtend(
        self,
        h_sbl: Array,
        wq: Array,
    ) -> Array:
        """Compute specific humidity tendency for SBL.

        .. math::
            \\frac{dq}{dt} = \\frac{w'q'}{h_{sbl}}

        Args:
            h_sbl: SBL height [m].
            wq: Surface kinematic moisture flux [kg/kg m/s].

        Returns:
            Specific humidity tendency [kg/kg/s].
        """
        return wq / h_sbl

    def compute_co2tend(
        self,
        h_sbl: Array,
        wCO2: Array,
    ) -> Array:
        """Compute CO2 tendency for SBL.

        .. math::
            \\frac{dCO_2}{dt} = \\frac{w'CO_2'}{h_{sbl}}

        Args:
            h_sbl: SBL height [m].
            wCO2: Surface kinematic CO2 flux [mgC/m²/s].

        Returns:
            CO2 tendency [ppm/s].
        """
        return wCO2 / h_sbl

    def compute_utend(
        self,
        h_sbl: Array,
        uw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute zonal wind tendency for the SBL.

        .. math::
            \\frac{du}{dt} = -f \\, dv - \\frac{\\overline{u'w'}}{h_{sbl}}

        where :math:`\\overline{u'w'}` is the surface momentum flux and
        :math:`f` is the Coriolis parameter.

        Args:
            h_sbl: SBL height [m].
            uw: Surface zonal momentum flux [m²/s²].
            du: Zonal wind jump at SBL top [m/s].
            dv: Meridional wind jump at SBL top [m/s].

        Returns:
            Zonal wind tendency [m/s²].
        """
        coriolis = -self.coriolis_param * dv
        drag = uw / h_sbl
        return jnp.where(self.is_wind_prog, coriolis + drag, 0.0)

    def compute_vtend(
        self,
        h_sbl: Array,
        vw: Array,
        du: Array,
        dv: Array,
    ) -> Array:
        """Compute meridional wind tendency for the SBL.

        .. math::
            \\frac{dv}{dt} = f \\, du - \\frac{\\overline{v'w'}}{h_{sbl}}

        Args:
            h_sbl: SBL height [m].
            vw: Surface meridional momentum flux [m²/s²].
            du: Zonal wind jump at SBL top [m/s].
            dv: Meridional wind jump at SBL top [m/s].

        Returns:
            Meridional wind tendency [m/s²].
        """
        coriolis = self.coriolis_param * du
        drag = vw / h_sbl
        return jnp.where(self.is_wind_prog, coriolis + drag, 0.0)

    def compute_thetav(self, theta: Array, q: Array) -> Array:
        """Compute virtual potential temperature.

        .. math::
            \\theta_v = \\theta (1 + 0.61 q)

        Args:
            theta: Potential temperature [K].
            q: Specific humidity [kg/kg].

        Returns:
            Virtual potential temperature [K].
        """
        return theta * (1.0 + 0.61 * q)
