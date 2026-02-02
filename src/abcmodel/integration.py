from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .abstracts import AbstractCoupledState, AtmosT, LandT, RadT
from .coupling import ABCoupler


def warmup(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
    tstart: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmos.warmup(coupler.rad, coupler.land, state, t, dt, tstart)
    return state


def timestep(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    t: int,
    dt: float,
    tstart: float,
) -> AbstractCoupledState[RadT, LandT, AtmosT]:
    """Run a single timestep of the model."""
    atmos = coupler.atmos.statistics(state, t)
    state = state.replace(atmos=atmos)
    rad = coupler.rad.run(state, t, dt, tstart)
    state = state.replace(rad=rad)
    land = coupler.land.run(state)
    state = state.replace(land=land)
    atmos = coupler.atmos.run(state)
    state = state.replace(atmos=atmos)
    land = coupler.land.integrate(state.land, dt)
    state = state.replace(land=land)
    atmos = coupler.atmos.integrate(state.atmos, dt)
    state = state.replace(atmos=atmos)
    state = coupler.compute_diagnostics(state)
    return state


def inner_step(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    t: int,
    coupler: ABCoupler,
    inner_dt: float,
    tstart: float,
):
    """Single physics timestep."""
    state = timestep(state, coupler, t, inner_dt, tstart)
    return state, state


def outter_step(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    t: int,
    coupler: ABCoupler,
    inner_dt: float,
    inner_tsteps: int,
    tstart: float,
):
    """A block of inner steps averaging the result."""
    timesteps = t + jnp.arange(inner_tsteps)
    step_fn_configured = partial(
        inner_step, coupler=coupler, inner_dt=inner_dt, tstart=tstart
    )
    state, inner_traj = jax.lax.scan(
        step_fn_configured, state, timesteps, length=inner_tsteps
    )
    avg_traj = jax.tree.map(lambda x: jnp.mean(x, axis=0), inner_traj)
    return state, avg_traj


def integrate(
    state: AbstractCoupledState[RadT, LandT, AtmosT],
    coupler: ABCoupler,
    inner_dt: float,
    outter_dt: float,
    runtime: float,
    tstart: float,
):
    """Integrate the coupler forward in time."""

    inner_tsteps = int(np.floor(outter_dt / inner_dt))
    outter_tsteps = int(np.floor(runtime / outter_dt))

    # warmup and initial diagnostics
    state = warmup(state, coupler, 0, inner_dt, tstart)
    state = coupler.compute_diagnostics(state)

    # configure outter step function
    scan_fn = partial(
        outter_step,
        coupler=coupler,
        inner_dt=inner_dt,
        inner_tsteps=inner_tsteps,
        tstart=tstart,
    )

    # create the array of start times for each outer block
    # these are integer step indices, not physical times
    timesteps = jnp.arange(outter_tsteps) * inner_tsteps

    # this is effectively the integration
    state, trajectory = jax.lax.scan(scan_fn, state, timesteps, length=outter_tsteps)

    times = jnp.arange(outter_tsteps) * outter_dt / 3600.0 + tstart

    return times, trajectory
