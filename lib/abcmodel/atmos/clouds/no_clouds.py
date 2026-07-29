from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from ...coupling import AbstractCoupledState
from ..abstracts import AbstractCloudModel, AbstractCloudState


@dataclass
class NoCloudState(AbstractCloudState):
    """No cloud initial state."""

    cc_frac: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$a_{cc}$",
            "unit": "-",
            "description": "Cloud core fraction",
        },
    )
    """Cloud core fraction [-], range 0 to 1."""
    cc_mf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$M_{cc}$",
            "unit": "s^{-1}",
            "description": "Cloud core mass flux",
        },
    )
    """Cloud core mass flux [kg/kg/s]."""
    cc_qf: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'q')_{cc}$",
            "unit": "kg kg^{-1} s^{-1}",
            "description": "Cloud core moisture flux",
        },
    )
    """Cloud core moisture flux [kg/kg/s]."""
    cl_trans: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\tau_{cl}$",
            "unit": "-",
            "description": "Cloud layer transmittance",
        },
    )
    """Cloud layer transmittance [-], range 0 to 1."""
    q2_h: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\sigma^2_{q,h}$",
            "unit": "kg^2 kg^{-2}",
            "description": "Humidity variance at mixed-layer top",
        },
    )
    """Humidity variance at mixed-layer top [kg²/kg²]."""
    top_CO22: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$\sigma^2_{CO2,h}$",
            "unit": "ppm^2",
            "description": "CO2 variance at mixed-layer top",
        },
    )
    """CO2 variance at mixed-layer top [ppm²]."""
    wCO2M: Array = field(
        default_factory=lambda: jnp.array(0.0),
        metadata={
            "label": r"$(w'CO_2')_M$",
            "unit": "mgC m^{-2} s^{-1}",
            "description": "CO2 mass flux",
        },
    )
    """CO2 mass flux [mgC/m²/s]."""


class NoCloudModel(AbstractCloudModel[NoCloudState]):
    """No cloud is formed using this model."""

    def __init__(self):
        pass

    def init_state(self) -> NoCloudState:
        """Initialize the model state.

        Returns:
            The initial cloud state.
        """
        return NoCloudState()

    def run(self, state: AbstractCoupledState) -> NoCloudState:
        """No calculations."""
        return state.atmos.clouds
