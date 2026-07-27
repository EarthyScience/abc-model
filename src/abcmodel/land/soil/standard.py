from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState
from ...utils import PhysicalConstants as cst
from ..abstracts import AbstractSoilModel, AbstractSoilState


@dataclass
class StandardSoilState(AbstractSoilState):
    """Standard soil state."""

    wg: Array = field(
        metadata={
            "label": r"$w_g$",
            "unit": "m^3 m^{-3}",
            "description": "Soil moisture content",
        }
    )
    """Soil moisture content in the root zone [m3 m-3]."""
    temp_soil: Array = field(
        metadata={
            "label": r"$T_{soil}$",
            "unit": "K",
            "description": "Soil temperature",
        }
    )
    """Soil temperature [K]."""
    temp2: Array = field(
        metadata={
            "label": r"$T_{soil,2}$",
            "unit": "K",
            "description": "Deep soil temperature",
        }
    )
    """Deep soil temperature [K]."""
    rssoil: Array = field(
        default_factory=lambda: jnp.array(1.0e6),
        metadata={
            "label": r"$r_{soil}$",
            "unit": "s m^{-1}",
            "description": "Soil resistance",
        },
    )
    """Soil resistance [m s-1]."""
    temp_soil_tend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\partial T_{soil} / \partial t$",
            "unit": "K s^{-1}",
            "description": "Soil temperature tendency",
        },
    )
    """Soil temperature tendency [K s-1]."""
    wgtend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\partial w_g / \partial t$",
            "unit": "m^3 m^{-3} s^{-1}",
            "description": "Soil moisture tendency",
        },
    )
    """Soil moisture tendency [m3 m-3 s-1]."""


class StandardSoilModel(AbstractSoilModel[StandardSoilState]):
    """Standard soil model with moisture and temperature dynamics.

    Args:
        a: Clapp and Hornberger (1978) retention curve parameter. Default is 0.219.
        b: Clapp and Hornberger (1978) retention curve parameter. Default is 4.90.
        p: Clapp and Hornberger (1978) retention curve parameter. Default is 4.0.
        cgsat: saturated soil heat capacity [J m-3 K-1]. Default is 3.56e-6.
        wsat: saturated soil moisture content [m3 m-3]. Default is 0.472.
        wfc: soil moisture content at field capacity [m3 m-3]. Default is 0.323.
        wwilt: soil moisture content at wilting point [m3 m-3]. Default is 0.171.
        w2: soil moisture content at the second layer [m3 m-3]. Default is 0.21.
        d1: depth of the top soil layer [m]. Default is 0.1.
        c1sat: saturated soil hydraulic conductivity parameter [-]. Default is 0.132.
        c2ref: reference soil hydraulic conductivity parameter [-]. Default is 1.8.
        rssoilmin: minimum soil resistance [s m-1]. Default is 50.0.
    """

    def __init__(
        self,
        a: float = 0.219,
        b: float = 4.90,
        p: float = 4.0,
        cgsat: float = 3.56e-6,
        wsat: float = 0.472,
        wfc: float = 0.323,
        wwilt: float = 0.171,
        w2: float = 0.21,
        d1: float = 0.1,
        c1sat: float = 0.132,
        c2ref: float = 1.8,
        rssoilmin: float = 50.0,
    ):
        self.a = a
        self.b = b
        self.p = p
        self.cgsat = cgsat
        self.wsat = wsat
        self.wfc = wfc
        self.wwilt = wwilt
        self.w2 = w2
        self.d1 = d1
        self.c1sat = c1sat
        self.c2ref = c2ref
        self.rssoilmin = rssoilmin

    def init_state(
        self,
        wg: float = 0.21,
        temp_soil: float = 285.0,
        temp2: float = 286.0,
        rssoil: float = 1.0e6,
    ) -> StandardSoilState:
        """Initialize the soil state.

        Args:
            wg: Volumetric soil moisture [m3 m-3]. Default is 0.21.
            temp_soil: Soil temperature [K]. Default is 285.0.
            temp2: Deep soil temperature [K]. Default is 286.0.
            rssoil: Soil resistance [s m-1]. Default is 1.0e6.

        Returns:
            The initialized StandardSoilState.
        """
        return StandardSoilState(
            wg=jnp.array(wg),
            temp_soil=jnp.array(temp_soil),
            temp2=jnp.array(temp2),
            rssoil=jnp.array(rssoil),
        )

    def run(self, state: AbstractCoupledState) -> StandardSoilState:
        """Compute soil surface resistance."""
        rssoil = self.compute_soil_resistance(state.land.wg)
        return replace(state.land.soil, rssoil=rssoil)

    def compute_soil_resistance(self, wg: Array) -> Array:
        """Compute the soil resistance ``rssoil``.

        Notes:
            The soil resistance is calculated as

            .. math::
                r_\\text{soil} = r_\\text{soil,min} \\cdot f_2,

            where the parameter :math:`r_\\text{soil,min}` is the minimum surface resistance and
            the correction function :math:`f_2` is given by

            .. math::
                f_2 =
                    \\begin{cases}
                        \\frac{w_\\text{fc} - w_\\text{wilt}}{w_g - w_\\text{wilt}}, & \\text{if } w_g > w_\\text{wilt} \\\\
                        10^8, & \\text{otherwise},
                    \\end{cases}

            where the model parameters :math:`w_\\text{fc}` and :math:`w_\\text{wilt}`
            are the field capacity and wilting point, respectively,
            and the variable :math:`w_g` is the soil water content.

        References:
            Equations 9.28 and 9.31 from the CLASS book.
        """
        f2 = jnp.where(
            wg > self.wwilt,
            (self.wfc - self.wwilt) / (wg - self.wwilt),
            1.0e8,
        )
        return self.rssoilmin * f2

    def compute_temp_soil_tend(
        self, gf: Array, temp_soil: Array, temp2: Array
    ) -> Array:
        """Compute the soil temperature tendency ``temp_soil_tend``.

        Notes:
            The dynamics of heat transport in the soil is given by

            .. math::
                \\frac{\\text{d}T_s}{\\text{d}t}
                =
                C_T G - \\frac{2\\pi}{\\tau} (T_s - T_2),

            :math:`T_2` is the temperature of the second layer in the soil,
            :math:`T_s` is the soil temperature,
            where :math:`\\tau` is the time constant of one day (86400s),
            :math:`G` is the ground the heat flux and
            and :math:`C_T` is the surface soil/vegetation heat capacity, which can be parametrized as

            .. math::
                C_T = C_{T,\\text{sat}} \\left(\\frac{w_{\\text{sat}}}{w_2}\\right)^{\\frac{b}{2\\log(10)}}

            where :math:`C_{T,\\text{sat}}` is the saturated heat capacity,
            :math:`w_{\\text{sat}}` is the saturation water content,
            :math:`w_2` is the water content at the second layer
            and :math:`b` is a parameter from Clapp and Hornberger (1978).
            I have no idea where this log comes from.

        References:
            Equation 9.32 of the CLASS book.
        """
        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * jnp.log(10.0)))
        return cg * gf - 2.0 * jnp.pi / 86400.0 * (temp_soil - temp2)

    def compute_wgtend(self, wg: Array, le_soil: Array) -> Array:
        """Compute the soil moisture tendency ``wgtend``.

        Notes:
            The dynamics of soil moisture in the top soil layer is described by

            .. math::
                \\frac{\\mathrm{d}w_g}{\\mathrm{d}t}
                =
                -\\frac{C_1}{\\rho_w d_1} \\frac{LE_{\\text{soil}}}{L_v}
                - \\frac{C_2}{\\tau} (w_g - w_{eq}),

            where the coefficients :math:`C_1` and :math:`C_2` are calculated as

            .. math::
                C_1 = C_{1,\\text{sat}} \\left(\\frac{w_{sat}}{w_g}\\right)^{b/2 + 1},

            .. math::
                C_2 = C_{2,\\text{ref}} \\left(\\frac{w_2}{w_{sat} - w_2}\\right),

            where :math:`C_{1,\\text{sat}}` and :math:`C_{2,\\text{sat}}` are parameters from Clapp-Hornberger (1978)
            and the equilibrium soil moisture is given by

            .. math::
                w_{eq} = w_2 - w_{sat} a \\left(\\left(\\frac{w_2}{w_{sat}}\\right)^p
                \\left[1 - \\left(\\frac{w_2}{w_{sat}}\\right)^{8p}\\right]\\right).

            In these equations, :math:`w_g` is the volumetric soil moisture in the top layer,
            :math:`LE_{\\text{soil}}` is the latent heat flux from the soil,
            :math:`L_v` is the latent heat of vaporization,
            :math:`\\rho_w` is the density of water,
            :math:`d_1` is the depth of the first soil layer,
            :math:`\\tau` is a time constant (here, 86400 s = 1 day),
            :math:`w_2` is the soil moisture in the second layer,
            :math:`w_{sat}` is the saturated soil moisture, and
            :math:`a`, :math:`b` and :math:`p` are parameters from Clapp-Hornberger (1978).

            The first term represents the loss of soil moisture due to evaporation,
            and the second term represents the relaxation of soil moisture toward equilibrium with the lower layer.

        References:
            - (9.34)–(9.37) in the CLASS book.
            - Clapp, R. B., & Hornberger, G. M. (1978). Empirical equations for some soil hydraulic properties. Water resources research, 14(4), 601-604.

        """
        c1 = self.c1sat * (self.wsat / wg) ** (self.b / 2.0 + 1.0)
        c2 = self.c2ref * (self.w2 / (self.wsat - self.w2))
        wgeq = self.w2 - self.wsat * self.a * (
            (self.w2 / self.wsat) ** self.p
            * (1.0 - (self.w2 / self.wsat) ** (8.0 * self.p))
        )
        evap_loss = -c1 / (cst.rhow * self.d1) * le_soil / cst.lv
        deep_grad = c2 / 86400.0 * (wg - wgeq)
        return evap_loss + deep_grad

    def run_tends(self, state: StandardSoilState, surf_state) -> StandardSoilState:
        """Compute soil tendencies that depend on surface fluxes."""
        wgtend = self.compute_wgtend(state.wg, surf_state.le_soil)
        temp_soil_tend = self.compute_temp_soil_tend(
            surf_state.gf, state.temp_soil, state.temp2
        )
        return replace(state, wgtend=wgtend, temp_soil_tend=temp_soil_tend)

    def integrate(self, state: StandardSoilState, dt: float) -> StandardSoilState:
        """Integrate soil moisture and temperature forward in time."""
        wg = state.wg + dt * state.wgtend
        temp_soil = state.temp_soil + dt * state.temp_soil_tend
        return replace(state, wg=wg, temp_soil=temp_soil)
