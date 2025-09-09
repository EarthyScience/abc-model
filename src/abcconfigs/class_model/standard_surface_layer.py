from abcmodel.surface_layer import (
    StandardSurfaceLayerInitConds,
    StandardSurfaceLayerParams,
)

THETA = 288.0

# params
params = StandardSurfaceLayerParams()

# init conds
init_conds = StandardSurfaceLayerInitConds(
    ustar=0.3,
    z0m=0.02,
    z0h=0.002,
    theta=THETA,
)
