from abcmodel.clouds import StandardCumulusInitConds
from abcmodel.mixed_layer import BulkMixedLayerInitConds
from abcmodel.radiation import StandardRadiationInitConds
from abcmodel.surface_layer import StandardSurfaceLayerInitConds

THETA = 288.0

radiation = StandardRadiationInitConds(
    # net surface radiation [W/m²]
    net_rad=400,
)

surface_layer = StandardSurfaceLayerInitConds(
    # surface friction velocity [m s-1]
    ustar=0.3,
    # roughness length for momentum [m]
    z0m=0.02,
    # roughness length for scalars [m]
    z0h=0.002,
    # initial mixed-layer potential temperature [K]
    theta=THETA,
)

clouds = StandardCumulusInitConds()

mixed_layer = BulkMixedLayerInitConds(
    abl_height=200.0,
    theta=THETA,  # THETA is 288.0
    dtheta=1.0,
    wtheta=0.1,
    q=0.008,
    dq=-0.001,
    wq=1e-4,
    co2=422.0,
    dCO2=-44.0,
    wCO2=0.0,
    u=6.0,
    du=4.0,
    v=-4.0,
    dv=4.0,
    dz_h=150.0,
)
