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
    """Standard land surface model state aggregating biosphere, soil, and surface.

    Args:
        biosphere: The biosphere component state.
        soil: The soil component state.
        surface: The surface component state.
    """

    biosphere: BiosphereT
    soil: SoilT
    surface: SurfaceT

    @property
    def alpha(self) -> Array:
        """Surface albedo [-], range 0 to 1."""
        return self.surface.alpha

    @property
    def surf_temp(self) -> Array:
        """Surface temperature [K]."""
        return self.surface.surf_temp

    @property
    def rs(self) -> Array:
        """Surface resistance [s m-1]."""
        return self.biosphere.rs

    @property
    def wg(self) -> Array:
        """Ground water storage [m]."""
        return self.soil.wg

    @property
    def wl(self) -> Array:
        """Land water storage [m]."""
        return self.biosphere.wl

    @property
    def esat(self) -> Array:
        """Saturation vapor pressure [Pa]."""
        return self.surface.esat

    @property
    def qsat(self) -> Array:
        """Saturation specific humidity [kg kg-1]."""
        return self.surface.qsat

    @property
    def dqsatdT(self) -> Array:
        """Derivative of saturation specific humidity with respect to temperature [kg kg-1 K-1]."""
        return self.surface.dqsatdT

    @property
    def e(self) -> Array:
        """Vapor pressure [Pa]."""
        return self.surface.e

    @property
    def qsatsurf(self) -> Array:
        """Saturation specific humidity at the surface [kg kg-1]."""
        return self.surface.qsatsurf

    @property
    def wtheta(self) -> Array:
        """Kinematic heat flux [K m/s]."""
        return self.surface.wtheta

    @property
    def wq(self) -> Array:
        """Kinematic moisture flux [kg kg-1 m/s]."""
        return self.surface.wq

    @property
    def wCO2(self) -> Array:
        """Kinematic CO2 flux [kg m/s]."""
        return self.biosphere.wCO2


class StandardLandModel(AbstractLandModel[StandardLandState]):
    """Standard land model coordinating biosphere, soil, and surface models.

    Args:
        biosphere: Biosphere model instance (e.g., :class:`~abcmodel.land.biosphere.ags.AgsModel` or
            :class:`~abcmodel.land.biosphere.jarvis_stewart.JarvisStewartModel`).
        soil: Soil model instance (e.g., :class:`~abcmodel.land.soil.standard.StandardSoilModel`).
        surface: Surface model instance (e.g., :class:`~abcmodel.land.surface.standard.StandardSurfaceModel`).
    """

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
        """Initialize standard land state.

        Args:
            biosphere_state: Initialized biosphere state.
            soil_state: Initialized soil state.
            surface_state: Initialized surface state.

        Returns:
            The aggregated :class:`StandardLandState`.
        """
        return StandardLandState(
            biosphere=biosphere_state,
            soil=soil_state,
            surface=surface_state,
        )

    def run(self, state: AbstractCoupledState) -> StandardLandState:
        """Run the full land surface model for one time step.

        Execution order:
          1. biosphere diagnostics (stomatal resistance, CO2 fluxes, wet fraction)
          2. soil diagnostics (soil resistance)
          3. surface fluxes and skin temperature (energy balance, latent/sensible/ground heat)
          4. biosphere tendencies (canopy water)
          5. soil tendencies (moisture and temperature)

        Args:
            state: CoupledState carrying atmos, land, and radiation state.

        Returns:
            The updated land state object with all computed fluxes and diagnostics.
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
        """Integrate biosphere, soil, and surface states forward in time.

        Args:
            state: the land state object carrying all variables.
            dt: the time step [s].

        Returns:
            The updated land state object.
        """
        biosphere = self.biosphere.integrate(state.biosphere, dt)
        soil = self.soil.integrate(state.soil, dt)
        surface = self.surface.integrate(state.surface, dt)
        return replace(
            state,
            biosphere=biosphere,
            soil=soil,
            surface=surface,
        )
