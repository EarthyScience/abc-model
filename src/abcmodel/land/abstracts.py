"""Abstract classes for land sub-modules."""

from abc import abstractmethod
from typing import Generic, TypeVar

from jax import Array

from ..abstracts import AbstractCoupledState, AbstractModel, AbstractState


class AbstractBiosphereState(AbstractState):
    """Abstract biosphere state."""

    rs: Array
    """Surface resistance [s m-1]."""
    wl: Array
    """Canopy water content [m]."""
    wltend: Array
    """Canopy water storage tendency [m s-1]."""
    cliq: Array
    """Wet fraction of the canopy [-]."""
    wCO2: Array
    """Kinematic CO2 flux [kg/kg m/s] or [mol m-2 s-1]."""
    cveg: Array
    """Vegetation fraction [-]."""


class AbstractSoilState(AbstractState):
    """Abstract soil state."""

    wg: Array
    """Soil moisture content in the root zone [m3 m-3]."""
    temp_soil: Array
    """Soil temperature [K]."""
    temp2: Array
    """Deep soil temperature [K]."""
    rssoil: Array
    """Soil resistance [s m-1]."""
    wgtend: Array
    """Soil moisture tendency [m3 m-3 s-1]."""
    temp_soil_tend: Array
    """Soil temperature tendency [K s-1]."""


class AbstractSurfaceState(AbstractState):
    """Abstract surface state."""

    alpha: Array
    """Surface albedo [-]."""
    surf_temp: Array
    """Surface temperature [K]."""
    qsatsurf: Array
    """Saturation specific humidity at surface temperature [kg/kg]."""
    le_veg: Array
    """Latent heat flux from vegetation [W m-2]."""
    le_liq: Array
    """Latent heat flux from liquid water [W m-2]."""
    le_soil: Array
    """Latent heat flux from soil [W m-2]."""
    le: Array
    """Total latent heat flux [W m-2]."""
    hf: Array
    """Sensible heat flux [W m-2]."""
    gf: Array
    """Ground heat flux [W m-2]."""
    le_pot: Array
    """Potential latent heat flux [W m-2]."""
    le_ref: Array
    """Reference latent heat flux [W m-2]."""
    vpd: Array
    """Vapor pressure deficit [Pa]."""
    wtheta: Array
    """Kinematic heat flux [K m/s]."""
    wq: Array
    """Kinematic moisture flux [kg/kg m/s]."""
    esat: Array
    """Saturation vapor pressure [Pa]."""
    qsat: Array
    """Saturation specific humidity [kg/kg]."""
    dqsatdT: Array
    """Derivative of saturation specific humidity with respect to temperature [kg/kg/K]."""
    e: Array
    """Vapor pressure [Pa]."""


BiosphereT = TypeVar("BiosphereT", bound=AbstractBiosphereState)
SoilT = TypeVar("SoilT", bound=AbstractSoilState)
SurfaceT = TypeVar("SurfaceT", bound=AbstractSurfaceState)


class AbstractBiosphereModel(AbstractModel, Generic[BiosphereT]):
    """Abstract biosphere model class."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> BiosphereT:
        raise NotImplementedError

    @abstractmethod
    def run_tends(
        self, state: BiosphereT, surf_state: "AbstractSurfaceState"
    ) -> BiosphereT:
        """Compute and store biosphere tendencies that depend on surface fluxes."""
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: BiosphereT, dt: float) -> BiosphereT:
        raise NotImplementedError


class AbstractSoilModel(AbstractModel, Generic[SoilT]):
    """Abstract soil model class."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> SoilT:
        raise NotImplementedError

    @abstractmethod
    def run_tends(self, state: SoilT, surf_state: "AbstractSurfaceState") -> SoilT:
        """Compute and store soil tendencies that depend on surface fluxes."""
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: SoilT, dt: float) -> SoilT:
        raise NotImplementedError


class AbstractSurfaceModel(AbstractModel, Generic[SurfaceT]):
    """Abstract surface model class."""

    @abstractmethod
    def run(self, state: AbstractCoupledState) -> SurfaceT:
        raise NotImplementedError

    def integrate(self, state: SurfaceT, dt: float) -> SurfaceT:
        return state
