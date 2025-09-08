import numpy as np

from ..models import (
    AbstractCloudModel,
    AbstractDiagnostics,
    AbstractInitConds,
    AbstractParams,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from ..utils import PhysicalConstants
from .stats import AbstractStandardStatsModel

# limamau: is there a better way to do this?
const = PhysicalConstants()
# conversion factor mgC m-2 s-1 to ppm m s-1
FAC = const.mair / (const.rho * const.mco2)


class MinimalMixedLayerParams(AbstractParams["MinimalMixedLayerModel"]):
    """Data class for minimal mixed layer model parameters.

    Extra
    -----
    - ``wstar``: convective velocity scale [m s-1]. Defaults to 1e-6.
    - ``wqe``: entrainment moisture flux [kg kg-1 m s-1]. Defaults to 0.0.
    - ``wCO2e``: entrainment CO2 flux [ppm m s-1]. Defaults to 0.0.
    """

    def __init__(self):
        self.wstar = 1e-6
        self.wqe = 0.0
        self.wCO2e = 0.0


class MinimalMixedLayerInitConds(AbstractInitConds["MinimalMixedLayerModel"]):
    """Data class for minimal mixed layer model initial conditions.

    Arguments
    ---------
    - ``abl_height``: initial ABL height [m].
    - ``surf_pressure``: surface pressure [Pa].
    - ``theta``: initial mixed-layer potential temperature [K].
    - ``dtheta``: initial temperature jump at h [K].
    - ``wtheta``: surface kinematic heat flux [K m/s].
    - ``q``: initial mixed-layer specific humidity [kg/kg].
    - ``dq``: initial specific humidity jump at h [kg/kg].
    - ``wq``: surface kinematic moisture flux [kg/kg m/s].
    - ``co2``: initial mixed-layer CO2 [ppm].
    - ``dCO2``: initial CO2 jump at h [ppm].
    - ``wCO2``: surface kinematic CO2 flux [mgC/m²/s].
    - ``u``: initial mixed-layer u-wind speed [m/s].
    - ``v``: initial mixed-layer v-wind speed [m/s].
    - ``dz_h``: transition layer thickness [-].

    Extra
    -----
    - ``wCO2A``: surface assimulation CO2 flux [ppm m s-1]. Defaults to 0.0.
    - ``wCO2R``: surface respiration CO2 flux [ppm m s-1]. Defaults to 0.0.
    - ``wCO2M``: CO2 mass flux [ppm m s-1]. Defaults to 0.0.
    """

    def __init__(
        self,
        abl_height: float,
        surf_pressure: float,
        theta: float,
        dtheta: float,
        wtheta: float,
        q: float,
        dq: float,
        wq: float,
        co2: float,
        dCO2: float,
        wCO2: float,
        u: float,
        v: float,
        dz_h: float,
    ):
        self.abl_height = abl_height
        self.surf_pressure = surf_pressure
        self.theta = theta
        self.dtheta = dtheta
        self.wtheta = wtheta
        self.q = q
        self.dq = dq
        self.wq = wq
        self.co2 = co2
        self.dCO2 = dCO2
        self.wCO2 = wCO2
        self.u = u
        self.v = v
        self.dz_h = dz_h
        self.wCO2A = 0.0
        self.wCO2R = 0.0
        self.wCO2M = 0.0


class MinimalMixedLayerDiagnostics(AbstractDiagnostics["MinimalMixedLayerModel"]):
    """Class for minimal mixed layer model diagnostics."""

    def post_init(self, tsteps: int):
        self.abl_height = np.zeros(tsteps)
        self.theta = np.zeros(tsteps)
        self.thetav = np.zeros(tsteps)
        self.dtheta = np.zeros(tsteps)
        self.wtheta = np.zeros(tsteps)
        self.wthetav = np.zeros(tsteps)
        self.q = np.zeros(tsteps)
        self.dq = np.zeros(tsteps)
        self.wq = np.zeros(tsteps)
        self.wqe = np.zeros(tsteps)
        self.qsat = np.zeros(tsteps)
        self.e = np.zeros(tsteps)
        self.esat = np.zeros(tsteps)
        self.co2 = np.zeros(tsteps)
        self.dCO2 = np.zeros(tsteps)
        self.wCO2 = np.zeros(tsteps)
        self.wCO2e = np.zeros(tsteps)
        self.wCO2R = np.zeros(tsteps)
        self.wCO2A = np.zeros(tsteps)
        self.wCO2M = np.zeros(tsteps)
        self.u = np.zeros(tsteps)
        self.v = np.zeros(tsteps)
        self.dz_h = np.zeros(tsteps)

    def store(self, t: int, model: "MinimalMixedLayerModel"):
        self.abl_height[t] = model.abl_height
        self.theta[t] = model.theta
        self.thetav[t] = model.thetav
        self.dtheta[t] = model.dtheta
        self.wtheta[t] = model.wtheta
        self.wthetav[t] = model.wthetav
        self.q[t] = model.q
        self.dq[t] = model.dq
        self.wq[t] = model.wq
        self.wqe[t] = model.wqe
        self.qsat[t] = model.qsat
        self.e[t] = model.e
        self.esat[t] = model.esat
        self.co2[t] = model.co2
        self.dCO2[t] = model.dCO2
        self.wCO2[t] = model.wCO2 / FAC
        self.wCO2e[t] = model.wCO2e / FAC
        self.wCO2R[t] = model.wCO2R / FAC
        self.wCO2A[t] = model.wCO2A / FAC
        self.wCO2M[t] = model.wCO2M / FAC
        self.u[t] = model.u
        self.v[t] = model.v
        self.dz_h[t] = model.dz_h


class MinimalMixedLayerModel(AbstractStandardStatsModel):
    """Minimal mixed layer model with constant properties."""

    def __init__(
        self,
        params: MinimalMixedLayerParams,
        init_conds: MinimalMixedLayerInitConds,
        diagnostics: AbstractDiagnostics = MinimalMixedLayerDiagnostics(),
    ):
        self.abl_height = init_conds.abl_height
        self.surf_pressure = init_conds.surf_pressure
        self.theta = init_conds.theta
        self.dtheta = init_conds.dtheta
        self.wtheta = init_conds.wtheta
        self.wstar = params.wstar
        self.q = init_conds.q
        self.dq = init_conds.dq
        self.wq = init_conds.wq
        self.dz_h = init_conds.dz_h
        self.co2 = init_conds.co2
        self.dCO2 = init_conds.dCO2
        self.wCO2 = init_conds.wCO2 * FAC
        self.wCO2A = init_conds.wCO2A
        self.wCO2R = init_conds.wCO2R
        self.wCO2M = init_conds.wCO2M
        self.u = init_conds.u
        self.v = init_conds.v
        self.wqe = params.wqe
        self.wCO2e = params.wCO2e
        self.diagnostics = diagnostics

    def run(
        self,
        const: PhysicalConstants,
        radiation: AbstractRadiationModel,
        surface_layer: AbstractSurfaceLayerModel,
        clouds: AbstractCloudModel,
    ):
        """No calculations."""
        pass

    def integrate(self, dt: float):
        """No integration."""
        pass
