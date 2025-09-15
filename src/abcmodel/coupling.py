from .models import (
    AbstractCloudModel,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from .utils import PhysicalConstants


class ABCoupler:
    def __init__(
        self,
        mixed_layer: AbstractMixedLayerModel,
        surface_layer: AbstractSurfaceLayerModel,
        radiation: AbstractRadiationModel,
        land_surface: AbstractLandSurfaceModel,
        clouds: AbstractCloudModel,
    ):
        # constants
        self.const = PhysicalConstants()

        # models and diagnostics
        self.radiation = radiation
        self.land_surface = land_surface
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.clouds = clouds

    # store model output
    def store(self, t):
        self.radiation.store(t)
        self.land_surface.store(t)
        self.surface_layer.store(t)
        self.mixed_layer.store(t)
        self.clouds.store(t)
