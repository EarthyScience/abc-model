from dataclasses import dataclass, replace
from typing import Generic

from jax import Array

from ..abstracts import (
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
)
from .abstracts import (
    AbstractBiosphereModel,
    AbstractSoilModel,
    AbstractSurfaceModel,
    BiosphereT,
    SoilT,
    SurfaceT,
)


@dataclass
class StandardLandState(AbstractLandState, Generic[BiosphereT, SoilT, SurfaceT]):
    """Standard land surface model state aggregating biosphere, soil, and surface."""

    biosphere: BiosphereT
    soil: SoilT
    surface: SurfaceT

    @property
    def alpha(self) -> Array:
        return self.surface.alpha

    @property
    def surf_temp(self) -> Array:
        return self.surface.surf_temp

    @property
    def rs(self) -> Array:
        return self.biosphere.rs

    @property
    def wg(self) -> Array:
        return self.soil.wg

    @property
    def wl(self) -> Array:
        return self.biosphere.wl

    @property
    def esat(self) -> Array:
        return self.surface.esat

    @property
    def qsat(self) -> Array:
        return self.surface.qsat

    @property
    def dqsatdT(self) -> Array:
        return self.surface.dqsatdT

    @property
    def e(self) -> Array:
        return self.surface.e

    @property
    def qsatsurf(self) -> Array:
        return self.surface.qsatsurf

    @property
    def wtheta(self) -> Array:
        return self.surface.wtheta

    @property
    def wq(self) -> Array:
        return self.surface.wq

    @property
    def wCO2(self) -> Array:
        return self.biosphere.wCO2


class StandardLandModel(AbstractLandModel[StandardLandState]):
    """Standard land model coordinating biosphere, soil, and surface models."""

    def __init__(
        self,
        biosphere: AbstractBiosphereModel,
        soil: AbstractSoilModel,
        surface: AbstractSurfaceModel,
    ):
        self.biosphere = biosphere
        self.soil = soil
        self.surface = surface

    def init_state(
        self,
        biosphere_state: BiosphereT,
        soil_state: SoilT,
        surface_state: SurfaceT,
    ) -> StandardLandState[BiosphereT, SoilT, SurfaceT]:
        """Initialize standard land state."""
        return StandardLandState(
            biosphere=biosphere_state,
            soil=soil_state,
            surface=surface_state,
        )

    def run(self, state: AbstractCoupledState) -> StandardLandState:
        """Run standard land components sequentially.

        Execution order:
          1. biosphere diagnostics
          2. soil diagnostics
          3. surface fluxes and surf_temp
          4. biosphere tendencies
          5. soil tendencies
        """
        # biosphere
        bio_state = self.biosphere.run(state)
        land_state = replace(state.land, biosphere=bio_state)
        state = state.replace(land=land_state)

        # soil
        soil_state = self.soil.run(state)
        land_state = replace(land_state, soil=soil_state)
        state = state.replace(land=land_state)

        # surface
        surf_state = self.surface.run(state)
        land_state = replace(land_state, surface=surf_state)
        state = state.replace(land=land_state)

        # tendencies (require surface fluxes computed above)
        bio_state = self.biosphere.run_tends(bio_state, surf_state)
        soil_state = self.soil.run_tends(soil_state, surf_state)

        return StandardLandState(
            biosphere=bio_state,
            soil=soil_state,
            surface=surf_state,
        )

    def integrate(self, state: StandardLandState, dt: float) -> StandardLandState:
        """Integrate biosphere, soil, and surface states forward in time."""
        biosphere = self.biosphere.integrate(state.biosphere, dt)
        soil = self.soil.integrate(state.soil, dt)
        surface = self.surface.integrate(state.surface, dt)
        return replace(
            state,
            biosphere=biosphere,
            soil=soil,
            surface=surface,
        )
