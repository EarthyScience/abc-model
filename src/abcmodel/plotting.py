from dataclasses import fields
from operator import attrgetter

import matplotlib.pyplot as plt


def show(time, trajectory, *var_paths: str):
    """
    Plot trajectories of variables against time.

    Args:
        time: Array of time points.
        trajectory: The state trajectory object (nested dataclass/PyTree).
        *var_paths: Strings representing the path to the variable within the trajectory object,
                    e.g., "atmos.mixed.h_abl". Exactly 6 paths should be provided.
    """
    if len(var_paths) != 6:
        raise ValueError(
            f"Expected exactly 6 variable paths, but got {len(var_paths)}: {var_paths}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, path in enumerate(var_paths):
        ax = axes[i]

        # 1. Retrieve data
        try:
            getter = attrgetter(path)
            data = getter(trajectory)
        except AttributeError:
            print(f"Warning: Could not access path '{path}' in trajectory.")
            data = None

        # 2. Retrieve label from metadata
        # We need to traverse the CLASS structure, not the instance, to get field metadata reliably?
        # Actually, `trajectory` is likely a PyTree of arrays. The metadata is on the class fields.
        # We need to traverse the *type* of trajectory to find the field definition.

        label = path  # Default label is the path itself

        parts = path.split(".")
        current_obj = trajectory

        try:
            # We traverse the objects to find the leaf's parent
            for part in parts[:-1]:
                current_obj = getattr(current_obj, part)

            # Now current_obj is the *instance* holding the field.
            # We need its class to inspect fields.
            cls = type(current_obj)
            field_name = parts[-1]

            # Find the field in the class fields
            for f in fields(cls):
                if f.name == field_name:
                    if "description" in f.metadata:
                        desc = f.metadata["description"]
                        # heuristics to split unit?
                        # The descriptions are "Label [Unit]"
                        label = desc
                    break
        except Exception as e:
            print(f"Warning: Could not extract metadata for '{path}': {e}")

        # 3. Plot
        if data is not None:
            ax.plot(time, data)
            ax.set_ylabel(label)
            ax.set_xlabel("time [h]")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f"Data not found: {path}", ha="center", va="center")

    plt.show()
