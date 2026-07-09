from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import exp1

from ...abstracts import AbstractCoupledState
from ...utils import PhysicalConstants as cst
from ...utils import compute_esat
from ..abstracts import AbstractBiosphereModel, AbstractBiosphereState


@dataclass
class AgsState(AbstractBiosphereState):
    """A-gs biosphere state."""

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
            "label": r"$w'CO_2'$",
            "unit": "mol m^{-2} s^{-1}",
            "description": "Total CO2 flux",
        },
    )
    """Total CO2 flux [mol m-2 s-1]."""
    cveg: Array = field(
        metadata={
            "label": r"$c_{veg}$",
            "unit": "-",
            "description": "Vegetation fraction",
        },
    )
    """Vegetation fraction [-]."""
    rsCO2: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$r_{s,CO2}$",
            "unit": "s m^{-1}",
            "description": "Stomatal resistance to CO2",
        },
    )
    """Stomatal resistance to CO2."""
    gcco2: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$g_{c,CO2}$",
            "unit": "s m^{-1}",
            "description": "Conductance to CO2",
        },
    )
    """Conductance to CO2."""
    ci: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$C_i$",
            "unit": "ppm",
            "description": "Intercellular CO2 concentration",
        },
    )
    """Intercellular CO2 concentration."""
    co2abs: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$CO_{2,abs}$",
            "unit": "kg m^{-3}",
            "description": "CO2 assimilation rate (or concentration?)",
        },
    )
    """CO2 assimilation rate / concentration."""
    wCO2A: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$A_n$",
            "unit": "mol m^{-2} s^{-1}",
            "description": "Net assimilation flux",
        },
    )
    """Net assimilation flux [mol m-2 s-1]."""
    wCO2R: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$R_{soil}$",
            "unit": "mol m^{-2} s^{-1}",
            "description": "Respiration flux",
        },
    )
    """Respiration flux [mol m-2 s-1]."""
    wltend: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$dw_l$",
            "unit": "m",
            "description": "Canopy water content tendency",
        },
    )
    """Canopy water content tendency [m]."""


class AgsModel(AbstractBiosphereModel[AgsState]):
    """Ags land surface biosphere model with coupled photosynthesis and stomatal conductance.

    Args:
        c3c4: string indicating whether the model should use C3 or C4 photosynthesis. Default is "c3".
        lai: leaf area index [m2 m-2]. Default is 2.0.
        cveg: vegetation fraction [-]. Default is 0.85.
        wmax: maximum water storage capacity of the canopy [m]. Default is 0.0002.
        wwilt: soil moisture content at wilting point [m3 m-3]. Default is 0.171.
        wfc: soil moisture content at field capacity [m3 m-3]. Default is 0.323.
        w2: soil moisture content at the second layer [m3 m-3]. Default is 0.21.
    """

    def __init__(
        self,
        c3c4: str = "c3",
        lai: float = 2.0,
        cveg: float = 0.85,
        wmax: float = 0.0002,
        wwilt: float = 0.171,
        wfc: float = 0.323,
        w2: float = 0.21,
    ):
        self.lai = lai
        self.cveg = cveg
        self.wmax = wmax
        self.wwilt = wwilt
        self.wfc = wfc
        self.w2 = w2
        self.c_beta = 0.0

        if c3c4 == "c3":
            self.c3c4 = 0
        elif c3c4 == "c4":
            self.c3c4 = 1
        else:
            raise ValueError(f'''Invalid option "{c3c4}" for "c3c4".''')

        self.co2comp298 = 68.5 if c3c4 == "c3" else 4.3
        self.net_rad10CO2 = 1.5 if c3c4 == "c3" else 1.5
        self.gm298 = 7.0 if c3c4 == "c3" else 17.5
        self.ammax298 = 2.2 if c3c4 == "c3" else 1.7
        self.net_rad10gm = 2.0 if c3c4 == "c3" else 2.0
        self.temp1gm = 278.0 if c3c4 == "c3" else 286.0
        self.temp2gm = 301.0 if c3c4 == "c3" else 309.0
        self.net_rad10Am = 2.0 if c3c4 == "c3" else 2.0
        self.temp1Am = 281.0 if c3c4 == "c3" else 286.0
        self.temp2Am = 311.0 if c3c4 == "c3" else 311.0
        self.f0 = 0.89 if c3c4 == "c3" else 0.85
        self.ad = 0.07 if c3c4 == "c3" else 0.15
        self.alpha0 = 0.017 if c3c4 == "c3" else 0.014
        self.kx = 0.7 if c3c4 == "c3" else 0.7
        self.gmin = 0.25e-3 if c3c4 == "c3" else 0.25e-3
        self.nuco2q = 1.6
        self.cw = 0.0016
        self.wmax = 0.55
        self.wmin = 0.005
        self.r10 = 0.23
        self.e0 = 53.3e3

    def init_state(
        self,
        rs: float = 1.0e6,
        wl: float = 0.0,
        cliq: float = 0.0,
        wCO2: float = 0.0,
    ) -> AgsState:
        """Initialize the Ags state.

        Args:
            rs: Surface resistance [s m-1]. Default is 1.0e6.
            wl: Canopy water content [m]. Default is 0.0.
            cliq: Wet fraction of canopy [-]. Default is 0.0.
            wCO2: Total CO2 flux [mol m-2 s-1]. Default is 0.0.

        Returns:
            The initialized AgsState.
        """
        return AgsState(
            rs=jnp.array(rs),
            wl=jnp.array(wl),
            cliq=jnp.array(cliq),
            wCO2=jnp.array(wCO2),
            cveg=jnp.array(self.cveg),  # this is a dirty move...
        )

    def run(self, state: AbstractCoupledState) -> AgsState:
        """Compute stomatal resistance and CO2 fluxes."""
        land = state.land
        atmos = state.atmos
        thetasurf = atmos.thetasurf

        co2comp = self.compute_co2comp(thetasurf)
        gm = self.compute_gm(thetasurf)
        fmin = self.compute_fmin(gm)
        ds = self.compute_ds(thetasurf, land.e)
        d0 = self.compute_d0(fmin)
        ci, co2abs = self.compute_internal_co2(ds, d0, fmin, atmos.co2, co2comp)
        ammax = self.compute_max_gross_primary_production(thetasurf)
        fstr = self.compute_soil_moisture_stress_factor(self.w2)
        am = self.compute_gross_assimilation(ammax, gm, ci, co2comp)
        rdark = self.compute_dark_respiration(am)
        par = self.compute_absorbed_par(state.in_srad)
        alphac = self.compute_light_use_efficiency(co2abs, co2comp)

        gcco2 = self.compute_canopy_co2_conductance(
            alphac,
            par,
            am,
            rdark,
            fstr,
            co2abs,
            co2comp,
            ds,
            d0,
            fmin,
        )
        rs = self.compute_rs(gcco2)

        rsCO2 = self.compute_surface_co2_resistance(gcco2)
        an = self.compute_net_assimilation(co2abs, ci, atmos.ra, rsCO2)
        fw = self.compute_soil_water_fraction(land.wg)
        resp = self.compute_respiration(land.soil.temp_soil, fw)
        wCO2A = self.scale_flux_to_mol(an)
        wCO2R = self.scale_flux_to_mol(resp)
        wCO2 = wCO2A + wCO2R

        cliq = self.compute_cliq(land.wl)

        return replace(
            land.biosphere,
            rs=rs,
            cliq=cliq,
            rsCO2=rsCO2,
            gcco2=gcco2,
            ci=ci,
            co2abs=co2abs,
            wCO2A=wCO2A,
            wCO2R=wCO2R,
            wCO2=wCO2,
        )

    def compute_co2comp(self, thetasurf: Array) -> Array:
        """Compute the CO₂ compensation concentration."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10CO2, temp_diff)
        return self.co2comp298 * cst.rho * exp_term

    def compute_gm(self, thetasurf: Array) -> Array:
        """Compute the mesophyll conductance."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10gm, temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1gm - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2gm))
        gm = self.gm298 * exp_term / (temp_factor1 * temp_factor2)
        return gm / 1000.0

    def compute_fmin(self, gm: Array) -> Array:
        """Compute minimum stomatal conductance factor."""
        fmin0 = self.gmin / self.nuco2q - 1.0 / 9.0 * gm
        fmin_sq_term = jnp.power(fmin0, 2.0) + 4 * self.gmin / self.nuco2q * gm
        fmin = -fmin0 + jnp.power(fmin_sq_term, 0.5) / (2.0 * gm)
        return fmin

    def compute_ds(self, surf_temp: Array, e: Array) -> Array:
        """Compute vapor pressure deficit (ds) in kPa."""
        ds = (compute_esat(surf_temp) - e) / 1000.0  # kPa
        return ds

    def compute_d0(self, fmin: Array) -> Array:
        """Compute reference vapor pressure deficit."""
        d0 = (self.f0 - fmin) / self.ad
        return d0

    def compute_internal_co2(
        self,
        ds: Array,
        d0: Array,
        fmin: Array,
        co2: Array,
        co2comp: Array,
    ) -> tuple[Array, Array]:
        """Compute cfrac, co2abs, and ci (internal CO2 concentration)."""
        cfrac = self.f0 * (1.0 - (ds / d0)) + fmin * (ds / d0)
        co2abs = co2 * (cst.mco2 / cst.mair) * cst.rho
        ci = cfrac * (co2abs - co2comp) + co2comp
        return ci, co2abs

    def compute_max_gross_primary_production(self, thetasurf: Array) -> Array:
        """Compute maximal gross primary production."""
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10Am, temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1Am - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2Am))
        ammax = self.ammax298 * exp_term / (temp_factor1 * temp_factor2)
        return ammax

    def compute_soil_moisture_stress_factor(self, w2: float) -> Array:
        """Compute effect of soil moisture stress."""
        soil_moisture_ratio = (w2 - self.wwilt) / (self.wfc - self.wwilt)
        betaw = jnp.clip(soil_moisture_ratio, 1e-3, 1.0)

        def case_zero(_):
            return betaw

        def case_low(_):
            p = 6.4 * self.c_beta
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        def case_medium(_):
            p = 7.6 * self.c_beta - 0.3
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        def case_high(_):
            p = 2.0 ** (3.66 * self.c_beta + 0.34) - 1.0
            numerator = 1.0 - jnp.exp(-p * betaw)
            denominator = 1.0 - jnp.exp(-p)
            return numerator / denominator

        branch_index = jnp.where(
            self.c_beta == 0,
            0,
            jnp.where(self.c_beta < 0.25, 1, jnp.where(self.c_beta < 0.50, 2, 3)),
        )

        result = jax.lax.switch(
            branch_index, [case_zero, case_low, case_medium, case_high], None
        )
        return result

    def compute_gross_assimilation(
        self,
        ammax: Array,
        gm: Array,
        ci: Array,
        co2comp: Array,
    ) -> Array:
        """Compute gross assimilation rate."""
        assimilation_factor = -(gm * (ci - co2comp) / ammax)
        am = ammax * (1.0 - jnp.exp(assimilation_factor))
        return am

    def compute_dark_respiration(self, am: Array) -> Array:
        """Compute dark respiration."""
        rdark = (1.0 / 9.0) * am
        return rdark

    def compute_absorbed_par(self, in_srad: Array) -> Array:
        """Compute absorbed photosynthetically active rad."""
        par = 0.5 * jnp.maximum(1e-1, in_srad * self.cveg)
        return par

    def compute_canopy_co2_conductance(
        self,
        alphac: Array,
        par: Array,
        am: Array,
        rdark: Array,
        fstr: Array,
        co2abs: Array,
        co2comp: Array,
        ds: Array,
        d0: Array,
        fmin: Array,
    ) -> Array:
        """Compute CO2 conductance at canopy level."""
        y = alphac * self.kx * par / (am + rdark)
        exp1_arg1 = jnp.array([y * jnp.exp(-self.kx * self.lai)])
        exp1_arg2 = jnp.array([y])
        exp1_term = exp1(exp1_arg1) - exp1(exp1_arg2)
        exp1_term = jnp.squeeze(exp1_term)
        an = (am + rdark) * (1.0 - (1.0 / (self.kx * self.lai)) * exp1_term)
        a1 = 1.0 / (1.0 - self.f0)
        dstar = d0 / (a1 * (self.f0 - fmin))
        conductance_factor = a1 * fstr * an / ((co2abs - co2comp) * (1.0 + ds / dstar))
        gcco2 = self.lai * (self.gmin / self.nuco2q + conductance_factor)
        return gcco2

    def compute_rs(self, gcco2: Array) -> Array:
        """Compute surface resistance."""
        return 1.0 / (1.6 * gcco2)

    def compute_light_use_efficiency(
        self,
        co2abs: Array,
        co2comp: Array,
    ) -> Array:
        """Compute light use efficiency."""
        co2_ratio = (co2abs - co2comp) / (co2abs + 2.0 * co2comp)
        alphac = self.alpha0 * co2_ratio
        return alphac

    def compute_surface_co2_resistance(self, gcco2: Array) -> Array:
        """Compute surface resistance to CO₂."""
        return 1.0 / gcco2

    def compute_net_assimilation(
        self, co2abs: Array, ci: Array, ra: Array, rsCO2: Array
    ) -> Array:
        """Compute net CO₂ assimilation rate."""
        return -(co2abs - ci) / (ra + rsCO2)

    def compute_soil_water_fraction(self, wg: Array) -> Array:
        """Compute soil water fraction."""
        return self.cw * self.wmax / (wg + self.wmin)

    def compute_respiration(
        self,
        temp_soil: Array,
        fw: Array,
    ) -> Array:
        """Compute soil respiration."""
        temp_ratio = 1.0 - 283.15 / temp_soil
        resp_factor = jnp.exp(self.e0 / (283.15 * 8.314) * temp_ratio)
        resp = self.r10 * (1.0 - fw) * resp_factor
        return resp

    def scale_flux_to_mol(self, flux: Array) -> Array:
        """Scale a flux to mol m⁻² s⁻¹."""
        return flux * (cst.mair / (cst.rho * cst.mco2))

    def compute_cliq(self, wl: Array) -> Array:
        """Compute wet fraction of canopy."""
        wlmx = self.lai * self.wmax
        return jnp.minimum(1.0, wl / wlmx)

    def compute_wltend(self, le_liq: Array) -> Array:
        """Compute canopy water storage tendency."""
        return -le_liq / (cst.rhow * cst.lv)

    def run_tends(self, state: AgsState, surf_state) -> AgsState:
        """Compute biosphere tendencies that depend on surface fluxes."""
        wltend = self.compute_wltend(surf_state.le_liq)
        return replace(state, wltend=wltend)

    def integrate(self, state: AgsState, dt: float) -> AgsState:
        """Integrate canopy water content forward in time."""
        wl = state.wl + dt * state.wltend
        return replace(state, wl=wl)
