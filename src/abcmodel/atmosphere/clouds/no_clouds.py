from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, PyTree
from simple_pytree import Pytree

from ...utils import PhysicalConstants
from ..abstracts import AbstractCloudModel


@dataclass
class NoCloudInitConds(Pytree):
    """No cloud initial state."""

    cc_frac: Array = 0.0
    """Cloud core fraction [-], range 0 to 1."""
    cc_mf: Array = 0.0
    """Cloud core mass flux [kg/kg/s]."""
    cc_qf: Array = 0.0
    """Cloud core moisture flux [kg/kg/s]."""
    cl_trans: Array = 1.0
    """Cloud layer transmittance [-], range 0 to 1."""
    q2_h: Array = 0.0
    """Humidity variance at mixed-layer top [kg²/kg²]."""
    top_CO22: Array = 0.0
    """CO2 variance at mixed-layer top [ppm²]."""
    wCO2M: Array = 0.0
    """CO2 mass flux [mgC/m²/s]."""


class NoCloudModel(AbstractCloudModel):
    """No cloud is formed using this model."""

    def __init__(self):
        pass

    def run(self, state: PyTree, const: PhysicalConstants):
        """No calculations."""
        return state
