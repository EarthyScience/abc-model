import numpy as np

from .clouds import NoCloudModel
from .land_surface import MinimalLandSurfaceModel
from .models import (
    AbstractCloudModel,
    AbstractLandSurfaceModel,
    AbstractMixedLayerModel,
    AbstractRadiationModel,
    AbstractSurfaceLayerModel,
)
from .utils import PhysicalConstants


class ABCModel:
    def __init__(
        self,
        dt: float,
        runtime: float,
        mixed_layer: AbstractMixedLayerModel,
        surface_layer: AbstractSurfaceLayerModel,
        radiation: AbstractRadiationModel,
        land_surface: AbstractLandSurfaceModel,
        clouds: AbstractCloudModel,
    ):
        # constants
        self.const = PhysicalConstants()

        # running configuration
        self.dt = dt
        self.runtime = runtime
        self.tsteps = int(np.floor(self.runtime / self.dt))
        self.t = 0

        # models and diagnostics
        self.radiation = radiation
        self.radiation.diagnostics.post_init(self.tsteps)
        self.land_surface = land_surface
        # self.land_surface.diagnostics.post_init(self.tsteps) - TBD
        self.surface_layer = surface_layer
        self.surface_layer.diagnostics.post_init(self.tsteps)
        self.mixed_layer = mixed_layer
        self.mixed_layer.diagnostics.post_init(self.tsteps)
        self.clouds = clouds
        self.clouds.diagnostics.post_init(self.tsteps)

        # initialize output
        self.out = ABCOutput(self.tsteps)

    def run(self):
        self.warmup()
        for self.t in range(self.tsteps):
            self.timestep()

    def warmup(self):
        self.mixed_layer.statistics(self.t, self.const)

        # calculate initial diagnostic variables
        self.radiation.run(
            self.t,
            self.dt,
            self.const,
            self.land_surface,
            self.mixed_layer,
        )

        for _ in range(10):
            self.surface_layer.run(self.const, self.land_surface, self.mixed_layer)

        self.land_surface.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.mixed_layer,
        )

        if not isinstance(self.clouds, NoCloudModel):
            self.mixed_layer.run(
                self.const,
                self.radiation,
                self.surface_layer,
                self.clouds,
            )
            self.clouds.run(self.mixed_layer)

        self.mixed_layer.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.clouds,
        )

    def timestep(self):
        self.mixed_layer.statistics(self.t, self.const)

        # run radiation model
        self.radiation.run(
            self.t,
            self.dt,
            self.const,
            self.land_surface,
            self.mixed_layer,
        )

        # run surface layer model
        self.surface_layer.run(self.const, self.land_surface, self.mixed_layer)

        # run land surface model
        self.land_surface.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.mixed_layer,
        )

        # run cumulus parameterization
        self.clouds.run(self.mixed_layer)

        # run mixed-layer model
        self.mixed_layer.run(
            self.const,
            self.radiation,
            self.surface_layer,
            self.clouds,
        )

        # store output before time integration
        self.store()

        # time integrate land surface model
        self.land_surface.integrate(self.dt)

        # time integrate mixed-layer model
        self.mixed_layer.integrate(self.dt)

    # store model output
    def store(self):
        t = self.t
        # limamau: IMO tstart should be taken out of radiation
        self.out.t[t] = t * self.dt / 3600.0 + self.radiation.tstart
        self.radiation.store(t)
        self.surface_layer.store(t)
        self.clouds.store(t)

        if not isinstance(self.land_surface, MinimalLandSurfaceModel):
            self.out.rs[t] = self.land_surface.rs
            self.out.hf[t] = self.land_surface.hf
            self.out.le[t] = self.land_surface.le
            self.out.le_liq[t] = self.land_surface.le_liq
            self.out.le_veg[t] = self.land_surface.le_veg
            self.out.le_soil[t] = self.land_surface.le_soil
            self.out.le_pot[t] = self.land_surface.le_pot
            self.out.le_ref[t] = self.land_surface.le_ref
            self.out.gf[t] = self.land_surface.gf

        self.mixed_layer.store(t)


class ABCOutput:
    def __init__(self, tsteps):
        # time [s]
        self.t = np.zeros(tsteps)

        # land surface variables
        # aerodynamic resistance [s m-1]
        self.ra = np.zeros(tsteps)
        # surface resistance [s m-1]
        self.rs = np.zeros(tsteps)
        # sensible heat flux [W m-2]
        self.hf = np.zeros(tsteps)
        # evapotranspiration [W m-2]
        self.le = np.zeros(tsteps)
        # open water evaporation [W m-2]
        self.le_liq = np.zeros(tsteps)
        # transpiration [W m-2]
        self.le_veg = np.zeros(tsteps)
        # soil evaporation [W m-2]
        self.le_soil = np.zeros(tsteps)
        # potential evaporation [W m-2]
        self.le_pot = np.zeros(tsteps)
        # reference evaporation at rs = rsmin / LAI [W m-2]
        self.le_ref = np.zeros(tsteps)
        # ground heat flux [W m-2]
        self.gf = np.zeros(tsteps)
