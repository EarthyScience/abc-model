"""Abstract classes for atmosphere sub-modules."""

from abc import abstractmethod

from jaxtyping import Array, PyTree

from ..abstracts import AbstractModel
from ..utils import PhysicalConstants


class AbstractSurfaceLayerModel(AbstractModel):
    """Abstract surface layer model class to define the interface for all surface layer models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_ra(state: PyTree) -> Array:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel):
    """Abstract mixed layer model class to define the interface for all mixed layer models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, state: PyTree, t: int, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: PyTree, dt: float) -> PyTree:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel):
    """Abstract cloud model class to define the interface for all cloud models."""

    @abstractmethod
    def run(self, state: PyTree, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError
