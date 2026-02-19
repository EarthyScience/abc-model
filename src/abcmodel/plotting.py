from operator import attrgetter
from typing import Any

import matplotlib.pyplot as plt
from jax import Array

from .abstracts import AbstractCoupledState


def simple(
    time: Array,
    trajectory: AbstractCoupledState,
    left_top_path: str = "atmos.mixed.h_abl",
    mid_top_path: str = "atmos.mixed.theta",
    right_top_path: str = "atmos.mixed.q",
    left_bottom_path: str = "atmos.clouds.cc_frac",
    mid_bottom_path: str = "land.le",
    right_bottom_path: str = "land.wCO2",
    axes: Any = None,
    **kwargs,
):
    """
    Plot trajectories of variables against time.

    Args:
        time: time array.
        trajectory: coupled state trajectory.
        left_top_path: path to the variable within the trajectory object
        to be plotted on the left top subplot. Default is `"atmos.mixed.h_abl"`.
        mid_top_path: path to the variable within the trajectory object
        to be plotted on the mid top subplot. Default is `"atmos.mixed.theta"`.
        right_top_path: path to the variable within the trajectory object
        to be plotted on the right top subplot. Default is `"atmos.mixed.q"`.
        left_bottom_path: path to the variable within the trajectory object
        to be plotted on the left bottom subplot. Default is `"atmos.clouds.cc_frac"`.
        mid_bottom_path: path to the variable within the trajectory object
        to be plotted on the mid bottom subplot. Default is `"land.le"`.
        right_bottom_path: path to the variable within the trajectory object
        to be plotted on the right bottom subplot. Default is `"land.wCO2"`.
        axes: optional matplotlib axes to plot on.
        **kwargs: additional keyword arguments to pass to matplotlib's plot function.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    else:
        fig = None
    axes = axes.flatten()

    var_paths = [
        left_top_path,
        mid_top_path,
        right_top_path,
        left_bottom_path,
        mid_bottom_path,
        right_bottom_path,
    ]

    for i, path in enumerate(var_paths):
        ax = axes[i]
        try:
            getter = attrgetter(path)
            data = getter(trajectory)
        except AttributeError:
            raise ValueError(f"Could not access path '{path}' in trajectory.")
        label = path
        parts = path.split(".")
        current_obj = trajectory

        try:
            # we traverse the objects to find the leaf's parent
            for part in parts[:-1]:
                current_obj = getattr(current_obj, part)
            # now current_obj is the *instance* holding the field
            # we need its class to inspect fields.
            cls = type(current_obj)
            field_name = parts[-1]

            # find the field in the class fields using the
            # __dataclass_fields__ dictionary which is pretty hacky :P
            field_obj = getattr(cls, "__dataclass_fields__", {}).get(field_name)
            label = get_label_from_metadata(field_obj.metadata, label)  # type: ignore
        except Exception:
            raise ValueError(f"Data not found: {path}")

        ax.plot(time, data, **kwargs)
        ax.set_title(label)
        if i > 2:
            ax.set_xlabel("time [h]")

    return fig, axes


def get_label_from_metadata(meta: dict, default_label: str) -> str:
    """Extract label from field metadata."""
    if "label" in meta:
        label = meta["label"]
        if "unit" in meta:
            unit = meta["unit"]
            # if unit contains math chars and isn't wrapped in $, wrap it
            if any(c in unit for c in "^\\_") and not unit.startswith("$"):
                unit = f"${unit}$"
            label = f"{label} [{unit}]"
        return label
    elif "description" in meta:
        return meta["description"]
    return default_label
