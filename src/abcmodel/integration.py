import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .coupling import ABCoupler


# limamau: this (or something similar) could be a check to run after
# warmup since the static type checking for the state is now bad
def print_nan_variables(state: PyTree):
    """Print all variables in a CoupledState that have NaN values"""
    nan_vars = []

    for name, value in state.__dict__.items():
        try:
            is_nan = False

            # check JAX arrays
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                if jnp.issubdtype(value.dtype, jnp.floating):
                    if jnp.any(jnp.isnan(value)):
                        is_nan = True
            # check numpy arrays
            elif hasattr(value, "dtype") and np.issubdtype(value.dtype, np.floating):
                if np.any(np.isnan(value)):
                    is_nan = True
            # check regular float values
            elif isinstance(value, float) and math.isnan(value):
                is_nan = True

            if is_nan:
                nan_vars.append((name, value))
                print(f"Variable '{name}' contains NaN: {value}")

        except (TypeError, AttributeError, Exception):
            # skip variables that can't be checked for NaN
            continue

    return nan_vars


def warmup(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    """Warmup the model by running it for a few timesteps."""
    state = coupler.atmosphere.statistics(state, t, coupler.const)

    # calculate initial diagnostic variables
    state = coupler.radiation.run(state, t, dt, coupler.const)

    # warmup atmosphere and land
    # limamau: would it be possible to warmup land then atmosphere?
    state = coupler.atmosphere.warmup(state, coupler.const, coupler.land)

    return state


def timestep(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    """Run a single timestep of the model."""
    state = coupler.atmosphere.statistics(state, t, coupler.const)
    state = coupler.radiation.run(state, t, dt, coupler.const)
    state = coupler.land.run(state, coupler.const)
    state = coupler.atmosphere.run(state, coupler.const)
    state = coupler.land.integrate(state, dt)
    state = coupler.atmosphere.integrate(state, dt)
    state = coupler.compute_diagnostics(state)
    return state


def integrate(state: PyTree, coupler: ABCoupler, dt: float, runtime: float):
    """Integrate the coupler forward in time.

    Args:
        state: Initial coupled state.
        coupler: ABCoupler instance.
        dt: Time step [s].
        runtime: Total runtime [s].

    Returns:
        times: Array of time values [h].
        trajectory: PyTree containing the full state trajectory.
    """
    tsteps = int(np.floor(runtime / dt))

    # warmup
    state = warmup(state, coupler, 0, dt)
    state = coupler.compute_diagnostics(state)

    def iter_fn(state, t):
        state = timestep(state, coupler, t, dt)
        return state, state

    timesteps = jnp.arange(tsteps)
    state, trajectory = jax.lax.scan(iter_fn, state, timesteps, length=tsteps)

    times = jnp.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart

    return times, trajectory
