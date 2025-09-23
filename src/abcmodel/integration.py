import math
from types import SimpleNamespace
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from .clouds import NoCloudModel
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
    state = coupler.mixed_layer.statistics(state, t, coupler.const)

    # calculate initial diagnostic variables
    state = coupler.radiation.run(state, t, dt, coupler.const)

    for _ in range(10):
        state = coupler.surface_layer.run(state, coupler.const)

    state = coupler.land_surface.run(state, coupler.const, coupler.surface_layer)

    if not isinstance(coupler.clouds, NoCloudModel):
        state = coupler.mixed_layer.run(state, coupler.const)
        state = coupler.clouds.run(state, coupler.const)

    state = coupler.mixed_layer.run(state, coupler.const)

    print_nan_variables(state)

    return state


def timestep(state: PyTree, coupler: ABCoupler, t: int, dt: float) -> PyTree:
    state = coupler.mixed_layer.statistics(state, t, coupler.const)

    # run radiation model
    state = coupler.radiation.run(state, t, dt, coupler.const)

    # run surface layer model
    state = coupler.surface_layer.run(state, coupler.const)

    # run land surface model
    state = coupler.land_surface.run(state, coupler.const, coupler.surface_layer)

    # run cumulus parameterization
    state = coupler.clouds.run(state, coupler.const)

    # run mixed-layer model
    state = coupler.mixed_layer.run(state, coupler.const)

    # time integrate land surface model
    state = coupler.land_surface.integrate(state, dt)

    # time integrate mixed-layer model
    state = coupler.mixed_layer.integrate(state, dt)

    return state


class TrajectoryCollector:
    """Collects state variables over time into separate lists."""

    def __init__(self, initial_state=None):
        self.data: Dict[str, List[Any]] = {}
        self.initialized = False

        if initial_state is not None:
            self._initialize_from_state(initial_state)

    def _initialize_from_state(self, state):
        """Initialize the collector with variable names from the first state."""
        self.data = {name: [] for name in state.__dict__.keys()}
        self.initialized = True

    def append(self, state):
        """Append current state values to the trajectory."""
        if not self.initialized:
            self._initialize_from_state(state)

        # Add any new variables that might have appeared
        for name in state.__dict__.keys():
            if name not in self.data:
                # Backfill with None for missing timesteps
                self.data[name] = [None] * len(next(iter(self.data.values())))

        # Append current values (making copies to avoid reference issues)
        for name, value in state.__dict__.items():
            try:
                # Make a copy of the value to avoid reference issues
                if hasattr(value, "copy"):
                    copied_value = value.copy()
                elif hasattr(value, "__array__"):
                    copied_value = jnp.array(value)  # JAX arrays
                else:
                    copied_value = value  # Scalars, immutable types

                self.data[name].append(copied_value)
            except Exception:
                # Fallback: just store the value as-is
                self.data[name].append(value)

    def to_arrays(self, use_jax=True):
        """Convert lists to arrays and return as SimpleNamespace."""
        array_data = {}

        for name, values in self.data.items():
            try:
                if use_jax:
                    # Try JAX stack first
                    if all(hasattr(v, "shape") for v in values if v is not None):
                        array_data[name] = jnp.stack(
                            [v for v in values if v is not None]
                        )
                    else:
                        array_data[name] = jnp.array(values)
                else:
                    # Use numpy
                    array_data[name] = np.array(values)
            except Exception as e:
                print(f"Warning: Could not convert '{name}' to array: {e}")
                # Keep as list if conversion fails
                array_data[name] = values

        return SimpleNamespace(**array_data)

    def __len__(self):
        """Return number of timesteps collected."""
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))

    def get_variable(self, name: str):
        """Get the trajectory for a specific variable."""
        return self.data.get(name, [])


def integrate(
    state: PyTree,
    coupler: ABCoupler,
    dt: float,
    runtime: float,
):
    """Integrate the coupler forward in time."""
    tsteps = int(np.floor(runtime / dt))

    # warmup
    state = warmup(state, coupler, 0, dt)

    # # ---------------------------------------------
    # # limamau: this is the old way without scan
    # trajectory = TrajectoryCollector()
    # for t in range(tsteps):
    #     state = timestep(state, coupler, t, dt)
    #     trajectory.append(state)
    # trajectory = trajectory.to_arrays(use_jax=True)
    # # ---------------------------------------------

    # ------------------------------------------------------------------------
    # limamau: wip

    def iter_fn(state, t):
        state = timestep(state, coupler, t, dt)
        return state, state

    timesteps = jnp.arange(tsteps)
    state, trajectory = jax.lax.scan(iter_fn, state, timesteps, length=tsteps)
    # ------------------------------------------------------------------------

    times = np.arange(tsteps) * dt / 3600.0 + coupler.radiation.tstart

    return times, trajectory
