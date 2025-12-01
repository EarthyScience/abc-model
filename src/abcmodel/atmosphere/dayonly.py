from jaxtyping import PyTree

from ..abstracts import AbstractAtmosphereModel
from ..utils import PhysicalConstants
from .abstracts import (
    AbstractCloudModel,
    AbstractMixedLayerModel,
    AbstractSurfaceLayerModel,
)


class DayOnlyAtmosphereModel(AbstractAtmosphereModel):
    """Atmosphere model aggregating surface layer, mixed layer, and clouds during the day-time."""

    def __init__(
        self,
        surface_layer: AbstractSurfaceLayerModel,
        mixed_layer: AbstractMixedLayerModel,
        clouds: AbstractCloudModel,
    ):
        self.surface_layer = surface_layer
        self.mixed_layer = mixed_layer
        self.clouds = clouds

    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        state = self.surface_layer.run(state, const)
        state = self.clouds.run(state, const)
        state = self.mixed_layer.run(state, const)
        return state

    def statistics(self, state: PyTree, t: int, const: PhysicalConstants) -> PyTree:
        state = self.mixed_layer.statistics(state, t, const)
        return state

    def integrate(self, state: PyTree, dt: float) -> PyTree:
        # Only mixed layer has prognostic variables to integrate
        state = self.mixed_layer.integrate(state, dt)
        return state
