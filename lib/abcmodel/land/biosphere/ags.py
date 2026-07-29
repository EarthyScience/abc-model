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
        self.net_rad10CO2 = 1.5
        self.gm298 = 7.0 if c3c4 == "c3" else 17.5
        self.ammax298 = 2.2 if c3c4 == "c3" else 1.7
        self.net_rad10gm = 2.0
        self.temp1gm = 278.0 if c3c4 == "c3" else 286.0
        self.temp2gm = 301.0 if c3c4 == "c3" else 309.0
        self.net_rad10Am = 2.0
        self.temp1Am = 281.0 if c3c4 == "c3" else 286.0
        self.temp2Am = 311.0
        self.f0 = 0.89 if c3c4 == "c3" else 0.85
        self.ad = 0.07 if c3c4 == "c3" else 0.15
        self.alpha0 = 0.017 if c3c4 == "c3" else 0.014
        self.kx = 0.7
        self.gmin = 0.25e-3
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
        """Compute the CO₂ compensation concentration ``co2comp``.

        Notes:
            The CO₂ compensation point :math:`\\Gamma` is the CO₂
            concentration at which net photosynthesis is zero. It follows
            a Q₁₀ temperature response:

            .. math::
                \\Gamma = \\Gamma_{298} \\cdot \\rho \\cdot
                    Q_{10}^{\\,0.1\\,(\\theta_s - 298)}

            where :math:`\\Gamma_{298}` is the compensation point at 298 K,
            :math:`\\rho` is the air density, :math:`Q_{10}` is the relative
            increase per 10 K, and :math:`\\theta_s` is the surface potential
            temperature.
        """
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10CO2, temp_diff)
        return self.co2comp298 * cst.rho * exp_term

    def compute_gm(self, thetasurf: Array) -> Array:
        """Compute the mesophyll conductance ``gm``.

        Notes:
            Mesophyll conductance :math:`g_m` controls the diffusion of
            CO₂ from intercellular spaces to the sites of carboxylation.
            It follows a temperature response with a Q₁₀ factor and high/
            low temperature inhibition:

            .. math::
                g_m = \\frac{g_{m,298} \\cdot
                    Q_{10}^{\\,0.1\\,(\\theta_s - 298)}}
                    {\\bigl(1 + e^{0.3\\,(T_1 - \\theta_s)}\\bigr)
                     \\bigl(1 + e^{0.3\\,(\\theta_s - T_2)}\\bigr)}

            where :math:`g_{m,298}` is the mesophyll conductance at 298 K,
            :math:`\\theta_s` is the surface potential temperature, and
            :math:`T_1, T_2` are temperature thresholds.
        """
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10gm, temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1gm - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2gm))
        gm = self.gm298 * exp_term / (temp_factor1 * temp_factor2)
        return gm / 1000.0

    def compute_fmin(self, gm: Array) -> Array:
        """Compute the minimum stomatal conductance factor ``fmin``.

        Notes:
            The minimum conductance factor :math:`f_{\\min}` is derived
            from the quadratic relation between minimum stomatal
            conductance :math:`g_{\\min}` and mesophyll conductance
            :math:`g_m`:

            .. math::
                f_{\\min} = \\frac{-f_0 + \\sqrt{f_0^2 +
                    \\dfrac{4 g_{\\min} g_m}{\\nu}}}{2 g_m},
                \\qquad
                f_0 = \\frac{g_{\\min}}{\\nu} - \\frac{g_m}{9},

            where :math:`\\nu = 1.6` is the ratio of diffusivity of water
            vapour to CO₂, and :math:`g_{\\min}` is the minimum stomatal
            conductance.
        """
        fmin0 = self.gmin / self.nuco2q - 1.0 / 9.0 * gm
        fmin_sq_term = jnp.power(fmin0, 2.0) + 4 * self.gmin / self.nuco2q * gm
        fmin = -fmin0 + jnp.power(fmin_sq_term, 0.5) / (2.0 * gm)
        return fmin

    def compute_ds(self, surf_temp: Array, e: Array) -> Array:
        """Compute the vapour pressure deficit ``ds``.

        Notes:
            The vapour pressure deficit at the surface is given by

            .. math::
                D_s = \\frac{e_{\\text{sat}}(T_s) - e}{1000},

            where :math:`e_{\\text{sat}}` is the saturation vapour pressure
            at the surface temperature :math:`T_s` and :math:`e` is the
            actual vapour pressure.             The result is in kPa.
        """
        ds = (compute_esat(surf_temp) - e) / 1000.0  # kPa
        return ds

    def compute_d0(self, fmin: Array) -> Array:
        """Compute the reference vapour pressure deficit ``d0``.

        Notes:
            The reference VPD is derived from the minimum conductance
            factor:

            .. math::
                D_0 = \\frac{f_0 - f_{\\min}}{a_d},

            where :math:`f_0` is a shape parameter and :math:`a_d` is
            the sensitivity of the VPD response.
        """
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
        """Compute the intercellular CO₂ concentration ``ci``.

        Notes:
            The intercellular CO₂ concentration :math:`C_i` is computed
            from the CO₂ absorption concentration and the compensation
            point:

            .. math::
                c_f &= f_0 \\left(1 - \\frac{D_s}{D_0}\\right) +
                      f_{\\min} \\frac{D_s}{D_0} \\\\
                \\text{CO}_{2,\\text{abs}} &=
                    \\text{CO}_2 \\frac{M_{\\text{CO}_2}}
                                     {M_{\\text{air}}} \\rho \\\\
                C_i &= c_f (\\text{CO}_{2,\\text{abs}} - \\Gamma) + \\Gamma

            where :math:`c_f` is the fractional reduction factor,
            :math:`\\text{CO}_2` is the atmospheric CO₂ concentration,
            :math:`M_{\\text{CO}_2}` and :math:`M_{\\text{air}}` are the
            molar masses of CO₂ and dry air, and :math:`\\rho` is the
            air density.

        Returns:
            A tuple ``(ci, co2abs)``.
        """
        cfrac = self.f0 * (1.0 - (ds / d0)) + fmin * (ds / d0)
        co2abs = co2 * (cst.mco2 / cst.mair) * cst.rho
        ci = cfrac * (co2abs - co2comp) + co2comp
        return ci, co2abs

    def compute_max_gross_primary_production(self, thetasurf: Array) -> Array:
        """Compute the maximal gross primary production ``ammax``.

        Notes:
            The maximum gross primary production :math:`A_{m,\\max}`
            follows a temperature response identical in structure to
            mesophyll conductance:

            .. math::
                A_{m,\\max} = \\frac{A_{m,298} \\cdot
                    Q_{10}^{\\,0.1\\,(\\theta_s - 298)}}
                    {\\bigl(1 + e^{0.3\\,(T_1 - \\theta_s)}\\bigr)
                     \\bigl(1 + e^{0.3\\,(\\theta_s - T_2)}\\bigr)}

            where :math:`A_{m,298}` is the value at 298 K and
            :math:`\\theta_s` is the surface potential temperature.
        """
        temp_diff = 0.1 * (thetasurf - 298.0)
        exp_term = jnp.power(self.net_rad10Am, temp_diff)
        temp_factor1 = 1.0 + jnp.exp(0.3 * (self.temp1Am - thetasurf))
        temp_factor2 = 1.0 + jnp.exp(0.3 * (thetasurf - self.temp2Am))
        ammax = self.ammax298 * exp_term / (temp_factor1 * temp_factor2)
        return ammax

    def compute_soil_moisture_stress_factor(self, w2: float) -> Array:
        """Compute the soil moisture stress factor ``fstr``.

        Notes:
            The soil moisture stress factor :math:`\\beta_w` is computed
            from the relative soil moisture:

            .. math::
                \\beta_w = \\frac{w_2 - w_{\\text{wilt}}}
                               {w_{\\text{fc}} - w_{\\text{wilt}}}

            clipped to :math:`[\\varepsilon, 1]`. The stress factor is then
            adjusted using a piecewise function depending on the parameter
            :math:`c_{\\beta}`:

            .. math::
                f_{\\text{str}} = \\begin{cases}
                    \\beta_w & c_\\beta = 0 \\\\[4pt]
                    \\dfrac{1 - e^{-p \\beta_w}}{1 - e^{-p}} & c_\\beta > 0
                \\end{cases}

            where the shape parameter :math:`p` increases with
            :math:`c_{\\beta}`.
        """
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
        """Compute the gross assimilation rate ``am``.

        Notes:
            The gross assimilation rate follows an exponential approach
            to saturation (Collatz et al., 1991):

            .. math::
                A_m = A_{m,\\max} \\left(1 - \\exp\\!\\left(
                    -\\frac{g_m\\,(C_i - \\Gamma)}{A_{m,\\max}}
                \\right)\\right)

            where :math:`A_{m,\\max}` is the maximal gross primary
            production, :math:`g_m` is the mesophyll conductance,
            :math:`C_i` is the intercellular CO₂ concentration, and
            :math:`\\Gamma` is the CO₂ compensation concentration.
        """
        assimilation_factor = -(gm * (ci - co2comp) / ammax)
        am = ammax * (1.0 - jnp.exp(assimilation_factor))
        return am

    def compute_dark_respiration(self, am: Array) -> Array:
        """Compute the dark respiration rate ``rdark``.

        Notes:
            Dark respiration is proportional to the gross assimilation
            rate:

            .. math::
                R_{\\text{dark}} = \\frac{A_m}{9}

            where :math:`A_m` is the gross assimilation rate.
        """
        rdark = (1.0 / 9.0) * am
        return rdark

    def compute_absorbed_par(self, in_srad: Array) -> Array:
        """Compute the absorbed photosynthetically active radiation ``par``.

        Notes:
            The absorbed PAR is estimated as 50% of the incoming solar
            radiation, scaled by the vegetation fraction, with a lower
            bound:

            .. math::
                \\text{PAR} = \\max\\!\\left(0.5 \\cdot R_s \\cdot
                    c_{\\text{veg}},\\, 0.1\\right)

            where :math:`R_s` is the incoming solar radiation and
            :math:`c_{\\text{veg}}` is the vegetation fraction.
        """
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
        """Compute the canopy CO₂ conductance ``gcco2``.

        Notes:
            The canopy CO₂ conductance :math:`g_{c,\\text{CO}_2}` is
            computed by scaling leaf-level photosynthesis to the canopy
            using the big-leaf approach with the exponential integral
            :math:`E_1` (Sellers et al., 1996):

            .. math::
                y &= \\frac{\\alpha_c k_x \\text{PAR}}{A_m + R_{\\text{dark}}}
                \\\\
                A_n &= (A_m + R_{\\text{dark}})
                    \\left(1 - \\frac{E_1(y e^{-k_x L}) - E_1(y)}{k_x L}
                    \\right) \\\\
                D_* &= \\frac{D_0}{a_1 (f_0 - f_{\\min})} \\\\
                g_{c,\\text{CO}_2} &=
                    L \\left(\\frac{g_{\\min}}{\\nu} +
                    \\frac{a_1 f_{\\text{str}} A_n}
                         {(\\text{CO}_{2,\\text{abs}} - \\Gamma)
                          (1 + D_s / D_*)}\\right)

            where :math:`L` is the leaf area index, :math:`k_x` is the
            extinction coefficient, :math:`\\nu = 1.6` is the diffusivity
            ratio, and :math:`a_1 = 1 / (1 - f_0)`.
        """
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
        """Compute the surface resistance ``rs``.

        Notes:
            The surface (stomatal) resistance is related to the canopy
            CO₂ conductance by

            .. math::
                r_s = \\frac{1}{1.6 \\, g_{c,\\text{CO}_2}}

            where the factor :math:`1.6` is the ratio of diffusivity of
            water vapour to CO₂, and :math:`g_{c,\\text{CO}_2}` is the
            canopy CO₂ conductance.
        """
        return 1.0 / (1.6 * gcco2)

    def compute_light_use_efficiency(
        self,
        co2abs: Array,
        co2comp: Array,
    ) -> Array:
        """Compute the light use efficiency ``alphac``.

        Notes:
            The light use efficiency depends on the CO₂ concentration:

            .. math::
                \\alpha_c = \\alpha_0 \\,
                    \\frac{\\text{CO}_{2,\\text{abs}} - \\Gamma}
                         {\\text{CO}_{2,\\text{abs}} + 2\\Gamma}

            where :math:`\\alpha_0` is the quantum efficiency and
            :math:`\\Gamma` is the CO₂ compensation concentration.
        """
        co2_ratio = (co2abs - co2comp) / (co2abs + 2.0 * co2comp)
        alphac = self.alpha0 * co2_ratio
        return alphac

    def compute_surface_co2_resistance(self, gcco2: Array) -> Array:
        """Compute the surface resistance to CO₂ ``rsCO2``.

        Notes:
            The surface resistance to CO₂ is the reciprocal of the
            canopy CO₂ conductance:

            .. math::
                r_{s,\\text{CO}_2} = \\frac{1}{g_{c,\\text{CO}_2}}

            where :math:`g_{c,\\text{CO}_2}` is the canopy CO₂ conductance.
        """
        return 1.0 / gcco2

    def compute_net_assimilation(
        self, co2abs: Array, ci: Array, ra: Array, rsCO2: Array
    ) -> Array:
        """Compute the net CO₂ assimilation rate ``an``.

        Notes:
            The net assimilation rate follows Fick's law of diffusion:

            .. math::
                A_n = -\\frac{\\text{CO}_{2,\\text{abs}} - C_i}
                             {r_a + r_{s,\\text{CO}_2}}

            where :math:`\\text{CO}_{2,\\text{abs}}` is the CO₂ absorption
            concentration, :math:`C_i` is the intercellular CO₂
            concentration, :math:`r_a` is the aerodynamic resistance,
            and :math:`r_{s,\\text{CO}_2}` is the surface resistance to
            CO₂.
        """
        return -(co2abs - ci) / (ra + rsCO2)

    def compute_soil_water_fraction(self, wg: Array) -> Array:
        """Compute the soil water fraction ``fw``.

        Notes:
            The soil water fraction used for respiration scaling is

            .. math::
                f_w = \\frac{c_w \\, w_{\\max}}{w_g + w_{\\min}}

            where :math:`w_g` is the surface soil moisture,
            :math:`w_{\\max}` and :math:`w_{\\min}` are empirical
            parameters, and :math:`c_w` is a scaling constant.
        """
        return self.cw * self.wmax / (wg + self.wmin)

    def compute_respiration(
        self,
        temp_soil: Array,
        fw: Array,
    ) -> Array:
        """Compute the soil respiration rate ``resp``.

        Notes:
            Soil respiration follows an Arrhenius-type temperature
            response with a moisture limitation:

            .. math::
                R = R_{10} \\cdot (1 - f_w) \\cdot
                    \\exp\\!\\left(\\frac{E_0}{283.15 \\cdot R}
                        \\left(1 - \\frac{283.15}{T_{\\text{soil}}}
                        \\right)\\right)

            where :math:`R_{10}` is the respiration rate at 283.15 K,
            :math:`f_w` is the soil water fraction, :math:`E_0` is the
            activation energy, and :math:`T_{\\text{soil}}` is the soil
            temperature.
        """
        temp_ratio = 1.0 - 283.15 / temp_soil
        resp_factor = jnp.exp(self.e0 / (283.15 * 8.314) * temp_ratio)
        resp = self.r10 * (1.0 - fw) * resp_factor
        return resp

    def scale_flux_to_mol(self, flux: Array) -> Array:
        """Scale a flux to mol m⁻² s⁻¹."""
        return flux * (cst.mair / (cst.rho * cst.mco2))

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
        return -le_liq / (cst.rhow * cst.lv)

    def run_tends(self, state: AgsState, surf_state) -> AgsState:
        """Compute biosphere tendencies that depend on surface fluxes."""
        wltend = self.compute_wltend(surf_state.le_liq)
        return replace(state, wltend=wltend)

    def integrate(self, state: AgsState, dt: float) -> AgsState:
        """Integrate canopy water content forward in time."""
        wl = state.wl + dt * state.wltend
        return replace(state, wl=wl)
