from abcmodel.clouds import StandardCumulusParams
from abcmodel.mixed_layer import BulkMixedLayerParams
from abcmodel.radiation import StandardRadiationParams
from abcmodel.surface_layer import StandardSurfaceLayerParams

radiation = StandardRadiationParams(
    # latitude [deg]
    lat=51.97,
    # longitude [deg]
    lon=-4.93,
    # day of the year [-]
    doy=268.0,
    # time of the day [h UTC]
    tstart=6.8,
    # cloud cover fraction [-]
    cc=0.0,
    # cloud top radiative divergence [W m-2]
    dFz=0.0,
)

surface_layer = StandardSurfaceLayerParams()

clouds = StandardCumulusParams()

mixed_layer = BulkMixedLayerParams(
    sw_ml=True,
    sw_shearwe=True,
    sw_fixft=True,
    sw_wind=True,
    surf_pressure=101300.0,
    divU=0.0,
    coriolis_param=1.0e-4,
    gammatheta=0.006,
    advtheta=0.0,
    beta=0.2,
    gammaq=0.0,
    advq=0.0,
    gammaCO2=0.0,
    advCO2=0.0,
    gammau=0.0,
    advu=0.0,
    gammav=0.0,
    advv=0.0,
)
