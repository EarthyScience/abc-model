import numpy as np

from .clouds import NoCloudModel
from .coupling import ABCoupler


def warmup(coupler: ABCoupler, t: int, dt: float):
    coupler.mixed_layer.statistics(t, coupler.const)

    # calculate initial diagnostic variables
    coupler.radiation.run(
        t,
        dt,
        coupler.const,
        coupler.land_surface,
        coupler.mixed_layer,
    )

    for _ in range(10):
        coupler.surface_layer.run(
            coupler.const, coupler.land_surface, coupler.mixed_layer
        )

    coupler.land_surface.run(
        coupler.const,
        coupler.radiation,
        coupler.surface_layer,
        coupler.mixed_layer,
    )

    if not isinstance(coupler.clouds, NoCloudModel):
        coupler.mixed_layer.run(
            coupler.const,
            coupler.radiation,
            coupler.surface_layer,
            coupler.clouds,
        )
        coupler.clouds.run(coupler.mixed_layer)

    coupler.mixed_layer.run(
        coupler.const,
        coupler.radiation,
        coupler.surface_layer,
        coupler.clouds,
    )


def timestep(coupler: ABCoupler, t: int, dt: float):
    coupler.mixed_layer.statistics(t, coupler.const)

    # run radiation model
    coupler.radiation.run(
        t,
        dt,
        coupler.const,
        coupler.land_surface,
        coupler.mixed_layer,
    )

    # run surface layer model
    coupler.surface_layer.run(coupler.const, coupler.land_surface, coupler.mixed_layer)

    # run land surface model
    coupler.land_surface.run(
        coupler.const,
        coupler.radiation,
        coupler.surface_layer,
        coupler.mixed_layer,
    )

    # run cumulus parameterization
    coupler.clouds.run(coupler.mixed_layer)

    # run mixed-layer model
    coupler.mixed_layer.run(
        coupler.const,
        coupler.radiation,
        coupler.surface_layer,
        coupler.clouds,
    )

    # store output before time integration
    coupler.store(t)

    # time integrate land surface model
    coupler.land_surface.integrate(dt)

    # time integrate mixed-layer model
    coupler.mixed_layer.integrate(dt)


def integrate(
    coupler: ABCoupler,
    dt: float,
    runtime: float,
    initial_state=None,
):
    """Integrate the coupler forward in time."""
    tsteps = int(np.floor(runtime / dt))

    # initialize diagnostics
    coupler.radiation.diagnostics.post_init(tsteps)
    coupler.land_surface.diagnostics.post_init(tsteps)
    coupler.surface_layer.diagnostics.post_init(tsteps)
    coupler.mixed_layer.diagnostics.post_init(tsteps)
    coupler.clouds.diagnostics.post_init(tsteps)

    # warmup
    warmup(coupler, 0, dt)

    # let's go
    for t in range(tsteps):
        timestep(coupler, t, dt)

    times = np.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart

    return times
