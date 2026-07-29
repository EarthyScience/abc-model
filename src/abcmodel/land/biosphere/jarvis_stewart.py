from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from ...abstracts import AbstractCoupledState
from ..abstracts import AbstractBiosphereModel, AbstractBiosphereState


@dataclass
class JarvisStewartState(AbstractBiosphereState):
    """Jarvis-Stewart biosphere state."""

    rs: Array = field(
        metadata={
            "label": r"$r_s$",
            "unit": "s m^{-1}",
            "description": "Surface resistance",
        },
    )
    """Surface resistance [s m-1]."""
    wl: Array = field(
        metadata={
            "label": r"$w_l$",
            "unit": "m",
            "description": "Canopy water content",
        },
    )
    """Canopy water content [m]."""
    cliq: Array = field(
        metadata={
            "label": r"$dw_l$",
            "unit": "-",
            "description": "Wet fraction of canopy",
        },
    )
    """Wet fraction of canopy [-]."""
    wCO2: Array = field(
        metadata={
            "label": r"$w_{CO_2}$",
            "unit": "mol m-2 s-1",
            "description": "Kinematic CO2 flux",
        },
    )
    """Kinematic CO2 flux [mol m-2 s-1]."""
    cveg: Array = field(
        metadata={
            "label": r"$c_{veg}$",
            "unit": "-",
            "description": "Vegetation fraction",
        },
    )
    """Vegetation fraction [-]."""
    wltend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$dw_l$",
            "unit": "m",
            "description": "Canopy water content tendency",
        },
    )
    """Canopy water content tendency [m]."""


class JarvisStewartModel(AbstractBiosphereModel[JarvisStewartState]):
    """Jarvis-Stewart biosphere model with empirical surface resistance.

    Args:
        rsmin: minimum stomatal resistance [s m-1]. Default is 110.0.
        lai: leaf area index [m2 m-2]. Default is 2.0.
        gD: canopy rad extinction coefficient [-]. Default is 0.0.
        wmax: maximum water storage capacity of the canopy [m]. Default is 0.0002.
        wwilt: soil moisture content at wilting point [m3 m-3]. Default is 0.171.
        wfc: soil moisture content at field capacity [m3 m-3]. Default is 0.323.
        w2: soil moisture content at the second layer [m3 m-3]. Default is 0.21.
    """

    def __init__(
        self,
        rsmin: float = 110.0,
        lai: float = 2.0,
        gD: float = 0.0,
        cveg: float = 0.85,
        wmax: float = 0.0002,
        wwilt: float = 0.171,
        wfc: float = 0.323,
        w2: float = 0.21,
    ):
        self.rsmin = rsmin
        self.lai = lai
        self.gD = gD
        self.wmax = wmax
        self.wwilt = wwilt
        self.wfc = wfc
        self.w2 = w2
        self.cveg = cveg

    def init_state(
        self,
        rs: float = 1.0e6,
        wl: float = 0.0,
        cliq: float = 0.0,
        wCO2: float = 0.0,
    ) -> JarvisStewartState:
        """Initialize the biosphere state.

        Args:
            rs: Surface resistance [s m-1]. Default is 1.0e6.
            wl: Canopy water content [m]. Default is 0.0.
            cliq: Wet fraction of canopy [-]. Default is 0.0.
            wCO2: Kinematic CO2 flux [mol m-2 s-1]. Default is 0.0.
            cveg: vegetation fraction [-]. Default is 0.85.

        Returns:
            The initialized JarvisStewartState.
        """
        return JarvisStewartState(
            rs=jnp.array(rs),
            wl=jnp.array(wl),
            cliq=jnp.array(cliq),
            wCO2=jnp.array(wCO2),
            cveg=jnp.array(self.cveg),  # this is a dirty move...
        )

    def run(self, state: AbstractCoupledState) -> JarvisStewartState:
        """Compute biosphere surface resistance and canopy wet fraction."""
        f1 = self.compute_f1(state.in_srad)
        f2 = self.compute_f2(state.land.wg)
        f3 = self.compute_f3(state.land.esat, state.land.e)
        f4 = self.compute_f4(state.atmos.theta)

        rs = self.rsmin / self.lai * f1 * f2 * f3 * f4
        cliq = self.compute_cliq(state.land.wl)

        return replace(
            state.land.biosphere,
            rs=rs,
            cliq=cliq,
            wCO2=jnp.array(0.0),
        )

    def compute_f1(self, in_srad: Array) -> Array:
        """Compute the radiation stress factor ``f1``.

        Notes:
            The radiation factor follows the Jarvis (1976) formulation:

            .. math::
                f_1 = \\frac{1}{\\min\\!\\left(1,\\,
                    \\dfrac{0.004\\,R_s + 0.05}{0.81\\,(0.004\\,R_s + 1)}\\right)}

            where :math:`R_s` is the incoming solar radiation.
        """
        ratio = (0.004 * in_srad + 0.05) / (0.81 * (0.004 * in_srad + 1.0))
        f1 = 1.0 / jnp.minimum(1.0, ratio)
        return f1

    def compute_f2(self, wg: Array) -> Array:
        """Compute the soil moisture stress factor ``f2``.

        Notes:
            The soil moisture factor follows the Jarvis (1976) formulation:

            .. math::
                f_2 = \\max\\!\\left(1,\\,
                    \\frac{w_{\\text{fc}} - w_{\\text{wilt}}}
                         {w_g - w_{\\text{wilt}}}\\right)

            where :math:`w_g` is the surface soil moisture,
            :math:`w_{\\text{fc}}` is the field capacity and
            :math:`w_{\\text{wilt}}` is the wilting point.
            When the second-layer soil moisture :math:`w_2` drops below the
            wilting point, the factor is set to a large value (effectively
            closing the stomata).
        """
        f2 = jnp.where(
            self.w2 > self.wwilt,
            (self.wfc - self.wwilt) / (wg - self.wwilt),
            1.0e8,
        )
        f2 = jnp.maximum(f2, 1.0)
        return f2

    def compute_f3(self, esat: Array, e: Array) -> Array:
        """Compute the vapour pressure deficit stress factor ``f3``.

        Notes:
            The VPD factor follows the Jarvis (1976) formulation:

            .. math::
                f_3 = \\exp\\!\\left(\\frac{g_D \\, D}{100}\\right)

            where :math:`D = e_{\\text{sat}} - e` is the vapour pressure
            deficit, :math:`e_{\\text{sat}}` is the saturation vapour
            pressure, :math:`e` is the actual vapour pressure, and
            :math:`g_D` is the canopy radiation extinction coefficient.
        """
        vpd = esat - e
        f3 = 1.0 / jnp.exp(-self.gD * vpd / 100.0)
        return f3

    def compute_f4(self, theta: Array) -> Array:
        """Compute the temperature stress factor ``f4``.

        Notes:
            The temperature factor follows the Jarvis (1976) formulation:

            .. math::
                f_4 = \\frac{1}{1 - 0.0016\\,(298 - \\theta)^2}

            where :math:`\\theta` is the potential temperature [K].
        """
        f4 = 1.0 / (1.0 - 0.0016 * (298.0 - theta) ** 2.0)
        return f4

    def compute_cliq(self, wl: Array) -> Array:
        """Compute the wet fraction ``cliq``.

        Notes:
            The wet fraction is defined as

            .. math::
                c_{\\text{liq}} = \\frac{W_l}{\\text{LAI}\\cdot W_{\\text{max}}},

            where :math:`W_l` is the water layer depth,
            :math:`\\text{LAI}` is the leaf area index and
            :math:`W_{\\text{max}}` is the thickness of the water layer on wet vegetation.
            In case :math:`W_l > \\text{LAI}\\cdot W_{\\text{max}}`, the wet fraction is set to 1.

        References:
            Equation 9.19 from the CLASS book.
        """
        wlmx = self.lai * self.wmax
        return jnp.minimum(1.0, wl / wlmx)

    def compute_wltend(self, le_liq: Array) -> Array:
        """Compute the water layer depth tendency ``wltend``.

        Notes:
            The water layer depth tendency is the rate at which water is added to or taken from the vegetation,
            described by

            .. math::
                \\frac{\\text{d} w}{\\text{d} t} = -\\frac{LE_{\\text{liq}}}{\\rho_w L_v},

            where :math:`LE_{\\text{liq}}` is dew, :math:`\\rho_w` is water density and :math:`L_v` is the latent heat of vaporization.

        References:
            Equation 9.20 from the CLASS book, with sign convention.
        """
        from ...utils import PhysicalConstants as cst

        return -le_liq / (cst.rhow * cst.lv)

    def run_tends(self, state: JarvisStewartState, surf_state) -> JarvisStewartState:
        """Compute biosphere tendencies that depend on surface fluxes."""
        wltend = self.compute_wltend(surf_state.le_liq)
        return replace(state, wltend=wltend)

    def integrate(self, state: JarvisStewartState, dt: float) -> JarvisStewartState:
        """Integrate canopy water content forward in time."""
        wl = state.wl + dt * state.wltend
        return replace(state, wl=wl)
