from dataclasses import dataclass

import jax

from .abstracts import (
    AbstractAtmosphereModel,
    AbstractAtmosphereState,
    AbstractCoupledState,
    AbstractLandModel,
    AbstractLandState,
    AbstractRadiationModel,
    AbstractRadiationState,
)
from .utils import PhysicalConstants


@jax.tree_util.register_pytree_node_class
@dataclass
class DiagnosticsState:
    """Diagnostic variables for the coupled system."""

    total_water_mass: float = 0.0
    total_energy: float = 0.0

    def tree_flatten(self):
        return (self.total_water_mass, self.total_energy), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class CoupledState(AbstractCoupledState):
    """Hierarchical coupled state."""

    atmosphere: AbstractAtmosphereState
    land: AbstractLandState
    radiation: AbstractRadiationState
    diagnostics: DiagnosticsState = DiagnosticsState()

    def tree_flatten(self):
        children = (self.atmosphere, self.land, self.radiation, self.diagnostics)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class ABCoupler:
    """Coupling class to bound all the components."""

    def __init__(
        self,
        radiation: AbstractRadiationModel,
        land: AbstractLandModel,
        atmosphere: AbstractAtmosphereModel,
    ):
        self.radiation = radiation
        self.land = land
        self.atmosphere = atmosphere
        self.const = PhysicalConstants()

    @staticmethod
    def init_state(
        radiation_state: AbstractRadiationState,
        land_state: AbstractLandState,
        atmosphere_state: AbstractAtmosphereState,
    ) -> CoupledState:
        return CoupledState(
            radiation=radiation_state,
            land=land_state,
            atmosphere=atmosphere_state,
        )

    def compute_diagnostics(self, state: CoupledState) -> CoupledState:
        """Compute diagnostic variables for total water budget."""
        return state
