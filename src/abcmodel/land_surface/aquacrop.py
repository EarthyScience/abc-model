import numpy as np
from scipy.special import exp1

from ..models import (
    AbstractDiagnostics,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants, get_esat
from .standard import (
    AbstractStandardLandSurfaceModel,
    StandardLandSurfaceDiagnostics,
    StandardLandSurfaceInitConds,
    StandardLandSurfaceParams,
)


class AquaCropParams(StandardLandSurfaceParams):
    """Data class for AquaCrop model parameters.

    Arguments
    ---------
    - all arguments from StandardLandSurfaceParams.
    - ``c3c4``: plant type, either "c3" or "c4".

    Extra
    -----
    - ``co2comp298``: CO2 compensation concentration [mg m-3].
    - ``net_rad10CO2``: function parameter to calculate CO2 compensation concentration [-].
    - ``gm298``: mesophyill conductance at 298 K [mm s-1].
    - ``ammax298``: CO2 maximal primary productivity [mg m-2 s-1].
    - ``net_rad10gm``: function parameter to calculate mesophyll conductance [-].
    - ``temp1gm``: reference temperature to calculate mesophyll conductance gm [K].
    - ``temp2gm``: reference temperature to calculate mesophyll conductance gm [K].
    - ``net_rad10Am``: function parameter to calculate maximal primary profuctivity Ammax.
    - ``temp1Am``: reference temperature to calculate maximal primary profuctivity Ammax [K].
    - ``temp2Am``: reference temperature to calculate maximal primary profuctivity Ammax [K].
    - ``f0``: maximum value Cfrac [-].
    - ``ad``: regression coefficient to calculate Cfrac [kPa-1].
    - ``alpha0``: initial low light conditions [mg J-1].
    - ``kx``: extinction coefficient PAR [-].
    - ``gmin``: cuticular (minimum) conductance [mm s-1].
    - ``nuco2q``: ratio molecular viscosity water to carbon dioxide.
    - ``cw``: constant water stress correction (eq. 13 Jacobs et al. 2007) [-].
    - ``wmax``: upper reference value soil water [-].
    - ``wmin``: lower reference value soil water [-].
    - ``r10``: respiration at 10 C [mg CO2 m-2 s-1].
    - ``e0``: activation energy [53.3 kJ kmol-1].
    """

    def __init__(self, c3c4: str, **kwargs):
        super().__init__(**kwargs)
        self.c3c4 = c3c4
        self.co2comp298 = [68.5, 4.3]
        self.net_rad10CO2 = [1.5, 1.5]
        self.gm298 = [7.0, 17.5]
        self.ammax298 = [2.2, 1.7]
        self.net_rad10gm = [2.0, 2.0]
        self.temp1gm = [278.0, 286.0]
        self.temp2gm = [301.0, 309.0]
        self.net_rad10Am = [2.0, 2.0]
        self.temp1Am = [281.0, 286.0]
        self.temp2Am = [311.0, 311.0]
        self.f0 = [0.89, 0.85]
        self.ad = [0.07, 0.15]
        self.alpha0 = [0.017, 0.014]
        self.kx = [0.7, 0.7]
        self.gmin = [0.25e-3, 0.25e-3]
        self.nuco2q = 1.6
        self.cw = 0.0016
        self.wmax = 0.55
        self.wmin = 0.005
        self.r10 = 0.23
        self.e0 = 53.3e3


class AquaCropInitConds(StandardLandSurfaceInitConds):
    """Data class for AquaCrop model initial conditions.

    Arguments
    ---------
    - all arguments from StandardLandSurfaceInitConds.
    """

    pass


class AquaCropDiagnostics(StandardLandSurfaceDiagnostics["AquaCropModel"]):
    """Class for AquaCrop model diagnostics.

    Variables
    ---------
    - all arguments from StandardLandSurfaceDiagnostics.
    - ``rsCO2``: stomatal resistance to CO2.
    - ``gcco2``: conductance to CO2.
    - ``ci``: intercellular CO2 concentration.
    - ``co2abs``: CO2 assimilation rate.
    """

    def post_init(self, tsteps: int):
        super().post_init(tsteps)
        self.rsCO2 = np.zeros(tsteps)
        self.gcco2 = np.zeros(tsteps)
        self.ci = np.zeros(tsteps)
        self.co2abs = np.zeros(tsteps)

    def store(self, t: int, model: "AquaCropModel"):
        super().store(t, model)
        self.rsCO2[t] = model.rsCO2
        self.gcco2[t] = model.gcco2
        self.ci[t] = model.ci
        self.co2abs[t] = model.co2abs


class AquaCropModel(AbstractStandardLandSurfaceModel):
    """AquaCrop land surface model with coupled photosynthesis and stomatal conductance.

    A bit more advanced land surface model implementing the AquaCrop approach with coupled
    photosynthesis-stomatal conductance calculations. Includes detailed biochemical
    processes for both C3 and C4 vegetation types, soil moisture stress effects,
    and explicit CO2 flux calculations.

    Processes
    ---------
    1. Inherit all standard land surface processes from parent class.
    2. Calculate CO2 compensation concentration based on temperature.
    3. Compute mesophyll conductance with temperature response functions.
    4. Determine internal CO2 concentration using stomatal optimization.
    5. Calculate gross primary productivity with light and moisture limitations.
    6. Scale from leaf-level to canopy-level fluxes using extinction functions.
    7. Compute surface resistance from canopy conductance.
    8. Calculate net CO2 fluxes including plant assimilation and soil respiration.

    Updates
    --------
    - ``rs``: surface resistance for transpiration [s m-1].
    - ``rsCO2``: surface resistance for CO2 [s m-1].
    - ``gcco2``: canopy conductance for CO2 [m s-1].
    - ``ci``: internal CO2 concentration [mg m-3].
    - ``co2abs``: absolute CO2 concentration [mg m-3].
    - ``mixed_layer.wCO2A``: CO2 flux from plant assimilation [kg kg-1 m s-1].
    - ``mixed_layer.wCO2R``: CO2 flux from soil respiration [kg kg-1 m s-1].
    - ``mixed_layer.wCO2``: net CO2 flux [kg kg-1 m s-1].
    - all updates from ``AbstractStandardLandSurfaceModel``.
    """

    rsCO2: float
    gcco2: float
    ci: float

    def __init__(
        self,
        params: AquaCropParams,
        init_conds: AquaCropInitConds,
        diagnostics: AbstractDiagnostics = AquaCropDiagnostics(),
    ):
        super().__init__(params, init_conds, diagnostics)

        if params.c3c4 == "c3":
            self.c3c4 = 0
        elif params.c3c4 == "c4":
            self.c3c4 = 1
        else:
            raise ValueError(f'''Invalid option "{params.c3c4}" for "c3c4".''')

        self.co2comp298 = params.co2comp298
        self.net_rad10CO2 = params.net_rad10CO2
        self.gm298 = params.gm298
        self.ammax298 = params.ammax298
        self.net_rad10gm = params.net_rad10gm
        self.temp1gm = params.temp1gm
        self.temp2gm = params.temp2gm
        self.net_rad10Am = params.net_rad10Am
        self.temp1Am = params.temp1Am
        self.temp2Am = params.temp2Am
        self.f0 = params.f0
        self.ad = params.ad
        self.alpha0 = params.alpha0
        self.kx = params.kx
        self.gmin = params.gmin
        self.nuco2q = params.nuco2q
        self.cw = params.cw
        self.wmax = params.wmax
        self.wmin = params.wmin
        self.r10 = params.r10
        self.e0 = params.e0

    def calculate_co2_compensation_concentration(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
    ) -> float:
        """Calculate CO2 compensation concentration."""
        temp_diff = 0.1 * (surface_layer.thetasurf - 298.0)
        exp_term = pow(self.net_rad10CO2[self.c3c4], temp_diff)
        return self.co2comp298[self.c3c4] * const.rho * exp_term

    def calculate_mesophyll_conductance(
        self,
        surface_layer: AbstractSurfaceLayerModel,
    ) -> float:
        """Calculate mesophyll conductance."""
        temp_diff = 0.1 * (surface_layer.thetasurf - 298.0)
        exp_term = pow(self.net_rad10gm[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + np.exp(
            0.3 * (self.temp1gm[self.c3c4] - surface_layer.thetasurf)
        )
        temp_factor2 = 1.0 + np.exp(
            0.3 * (surface_layer.thetasurf - self.temp2gm[self.c3c4])
        )
        gm = self.gm298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return gm / 1000.0

    def calculate_internal_co2(
        self,
        const: PhysicalConstants,
        mixed_layer: AbstractMixedLayerModel,
        co2comp: float,
        gm: float,
    ) -> tuple[float, float, float, float, float]:
        """Calculate internal CO2 concentration and related parameters."""
        fmin0 = self.gmin[self.c3c4] / self.nuco2q - 1.0 / 9.0 * gm
        fmin_sq_term = pow(fmin0, 2.0) + 4 * self.gmin[self.c3c4] / self.nuco2q * gm
        fmin = -fmin0 + pow(fmin_sq_term, 0.5) / (2.0 * gm)

        ds = (get_esat(self.surf_temp) - mixed_layer.e) / 1000.0  # kPa
        d0 = (self.f0[self.c3c4] - fmin) / self.ad[self.c3c4]

        cfrac = self.f0[self.c3c4] * (1.0 - (ds / d0)) + fmin * (ds / d0)
        co2abs = mixed_layer.co2 * (const.mco2 / const.mair) * const.rho
        ci = cfrac * (co2abs - co2comp) + co2comp
        return ci, co2abs, fmin, ds, d0

    def calculate_max_gross_primary_production(
        self,
        surface_layer: AbstractSurfaceLayerModel,
    ) -> float:
        """Calculate maximal gross primary production in high light conditions (Ag)."""
        temp_diff = 0.1 * (surface_layer.thetasurf - 298.0)
        exp_term = pow(self.net_rad10Am[self.c3c4], temp_diff)
        temp_factor1 = 1.0 + np.exp(
            0.3 * (self.temp1Am[self.c3c4] - surface_layer.thetasurf)
        )
        temp_factor2 = 1.0 + np.exp(
            0.3 * (surface_layer.thetasurf - self.temp2Am[self.c3c4])
        )
        ammax = self.ammax298[self.c3c4] * exp_term / (temp_factor1 * temp_factor2)
        return ammax

    def calculate_soil_moisture_stress_factor(self) -> float:
        """Calculate effect of soil moisture stress on gross assimilation rate."""
        soil_moisture_ratio = (self.w2 - self.wwilt) / (self.wfc - self.wwilt)
        betaw = max(1e-3, min(1.0, soil_moisture_ratio))

        if self.c_beta == 0:
            return betaw

        if self.c_beta < 0.25:
            p = 6.4 * self.c_beta
        elif self.c_beta < 0.50:
            p = 7.6 * self.c_beta - 0.3
        else:
            p = 2 ** (3.66 * self.c_beta + 0.34) - 1
        return (1.0 - np.exp(-p * betaw)) / (1 - np.exp(-p))

    def calculate_gross_assimilation_and_light_use(
        self,
        ammax: float,
        gm: float,
        ci: float,
        co2comp: float,
        co2abs: float,
        radiation: AbstractRadiationModel,
    ) -> tuple[float, float, float, float]:
        """Calculate gross assimilation rate, dark respiration, PAR, and light use efficiency."""
        assimilation_factor = -(gm * (ci - co2comp) / ammax)
        am = ammax * (1.0 - np.exp(assimilation_factor))
        rdark = (1.0 / 9.0) * am
        par = 0.5 * max(1e-1, radiation.in_srad * self.cveg)
        co2_ratio = (co2abs - co2comp) / (co2abs + 2.0 * co2comp)
        alphac = self.alpha0[self.c3c4] * co2_ratio
        return am, rdark, par, alphac

    def calculate_canopy_co2_conductance(
        self,
        alphac: float,
        par: float,
        am: float,
        rdark: float,
        fstr: float,
        co2abs: float,
        co2comp: float,
        ds: float,
        d0: float,
        fmin: float,
    ) -> float:
        """Calculate upscaling from leaf to canopy and CO2 conductance at canopy level."""
        y = alphac * self.kx[self.c3c4] * par / (am + rdark)
        exp1_arg1 = y * np.exp(-self.kx[self.c3c4] * self.lai)
        exp1_arg2 = y
        exp1_term = exp1(exp1_arg1) - exp1(exp1_arg2)
        an = (am + rdark) * (1.0 - (1.0 / (self.kx[self.c3c4] * self.lai)) * exp1_term)

        a1 = 1.0 / (1.0 - self.f0[self.c3c4])
        dstar = d0 / (a1 * (self.f0[self.c3c4] - fmin))
        conductance_factor = a1 * fstr * an / ((co2abs - co2comp) * (1.0 + ds / dstar))
        gcco2 = self.lai * (self.gmin[self.c3c4] / self.nuco2q + conductance_factor)
        return gcco2

    def compute_surface_resistance(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        """
        Compute surface resistance using AquaCrop photosynthesis-conductance model.

        Parameters
        ----------
        - ``const``: physical constants. Uses ``rho``, ``mco2``, ``mair``.
        - ``radiation``: radiation model. Uses ``in_srad``.
        - ``surface_layer``: surface layer model. Uses ``thetasurf`` and ``ra``.
        - ``mixed_layer``: mixed layer model. Uses ``e``, ``co2``.

        Updates
        -------
        Updates ``self.rs`` based on canopy-scale CO2 conductance derived from
        coupled photosynthesis-stomatal conductance calculations. Also updates
        ``self.gcco2``, ``self.ci``, and ``self.co2abs``.
        """
        co2comp = self.calculate_co2_compensation_concentration(const, surface_layer)
        gm = self.calculate_mesophyll_conductance(surface_layer)

        self.ci, self.co2abs, fmin, ds, d0 = self.calculate_internal_co2(
            const, mixed_layer, co2comp, gm
        )

        ammax = self.calculate_max_gross_primary_production(surface_layer)
        fstr = self.calculate_soil_moisture_stress_factor()

        am, rdark, par, alphac = self.calculate_gross_assimilation_and_light_use(
            ammax, gm, self.ci, co2comp, self.co2abs, radiation
        )

        self.gcco2 = self.calculate_canopy_co2_conductance(
            alphac, par, am, rdark, fstr, self.co2abs, co2comp, ds, d0, fmin
        )

        self.rs = 1.0 / (1.6 * self.gcco2)

    def compute_co2_flux(
        self,
        const: PhysicalConstants,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
    ):
        self.rsCO2 = 1.0 / self.gcco2
        an = -(self.co2abs - self.ci) / (self.ra + self.rsCO2)
        fw = self.cw * self.wmax / (self.wg + self.wmin)
        temp_ratio = 1.0 - 283.15 / self.temp_soil
        resp_factor = np.exp(self.e0 / (283.15 * 8.314) * temp_ratio)
        resp = self.r10 * (1.0 - fw) * resp_factor

        mixed_layer.wCO2A = an * (const.mair / (const.rho * const.mco2))
        mixed_layer.wCO2R = resp * (const.mair / (const.rho * const.mco2))
        mixed_layer.wCO2 = mixed_layer.wCO2A + mixed_layer.wCO2R
