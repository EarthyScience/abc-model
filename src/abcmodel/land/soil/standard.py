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
        """Compute the soil resistance rssoil."""
        f2 = jnp.where(
            wg > self.wwilt,
            (self.wfc - self.wwilt) / (wg - self.wwilt),
            1.0e8,
        )
        return self.rssoilmin * f2

    def compute_temp_soil_tend(
        self, gf: Array, temp_soil: Array, temp2: Array
    ) -> Array:
        """Compute the soil temperature tendency."""
        cg = self.cgsat * (self.wsat / self.w2) ** (self.b / (2.0 * jnp.log(10.0)))
        return cg * gf - 2.0 * jnp.pi / 86400.0 * (temp_soil - temp2)

    def compute_wgtend(self, wg: Array, le_soil: Array) -> Array:
        """Compute the soil moisture tendency."""
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
