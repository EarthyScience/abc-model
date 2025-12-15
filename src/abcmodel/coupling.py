from dataclasses import dataclass
from typing import Generic

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractRadiationModel,
    AtmosT,
    LandT,
    RadT,
)
from .utils import PhysicalConstants


@dataclass
class CoupledState(
    AbstractCoupledState[RadT, LandT, AtmosT], Generic[RadT, LandT, AtmosT]
):
    """Hierarchical coupled state, generic over component types."""

    rad: RadT
    land: LandT
    atmos: AtmosT


class ABCoupler:
    """Coupling class to bound all the components."""

    def __init__(
        self,
        rad: AbstractRadiationModel,
        land: AbstractLandModel,
        atmos: AbstractAtmosphereModel,
    ):
        self.rad = rad
        self.land = land
        self.atmos = atmos
        self.const = PhysicalConstants()

    @staticmethod
    def init_state(
        rad_state: RadT,
        land_state: LandT,
        atmos_state: AtmosT,
    ) -> CoupledState[RadT, LandT, AtmosT]:
        return CoupledState(
            rad=rad_state,
            land=land_state,
            atmos=atmos_state,
        )

    def compute_diagnostics(self, state: AbstractCoupledState) -> AbstractCoupledState:
        """Compute diagnostic variables for total water budget."""
        return state
