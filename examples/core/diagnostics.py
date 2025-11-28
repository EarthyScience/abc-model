import matplotlib.pyplot as plt

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.utils import PhysicalConstants


def main():
    # time step [s]
    dt = 60.0
    # total run time [s]
    runtime = 12 * 3600.0

    # radiation
    radiation_init_conds = abcmodel.radiation.StandardRadiationInitConds(
        **cm.standard_radiation.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs,
    )

    # land surface
    land_surface_init_conds = abcmodel.land_surface.JarvisStewartInitConds(
        **cm.jarvis_stewart.init_conds_kwargs,
    )
    land_surface_model = abcmodel.land_surface.JarvisStewartModel(
        **cm.jarvis_stewart.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = abcmodel.surface_layer.StandardSurfaceLayerInitConds(
        **cm.standard_surface_layer.init_conds_kwargs
    )
    surface_layer_model = abcmodel.surface_layer.StandardSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.clouds.StandardCumulusInitConds()
    cloud_model = abcmodel.clouds.StandardCumulusModel()

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        mixed_layer=mixed_layer_model,
        surface_layer=surface_layer_model,
        radiation=radiation_model,
        land_surface=land_surface_model,
        clouds=cloud_model,
    )
    state = abcoupler.init_state(
        radiation_init_conds,
        land_surface_init_conds,
        surface_layer_init_conds,
        mixed_layer_init_conds,
        cloud_init_conds,
    )

    # run model with diagnostics enabled
    time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)
    const = PhysicalConstants()

    # plot diagnostic evolution
    _, axes = plt.subplots(1, 3, figsize=(12, 4))

    # row 1: water budget
    axes[0].plot(time, trajectory.total_water_mass)
    axes[0].set_xlabel("time [h]")
    axes[0].set_ylabel("Total water mass [kg m-2]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, trajectory.q * trajectory.abl_height * const.rho, label="Vapor")
    axes[1].plot(time, trajectory.wg * const.rhow * 0.1, label="Soil layer 1")
    axes[1].plot(time, trajectory.wl * const.rhow, label="Canopy")
    axes[1].set_xlabel("time [h]")
    axes[1].set_ylabel("Water mass [kg m-2]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, trajectory.le_veg, label="LE vegetation")
    axes[2].plot(time, trajectory.le_soil, label="LE soil")
    axes[2].plot(time, trajectory.le_liq, label="LE liq")
    axes[2].plot(time, trajectory.le, label="LE total")
    axes[2].set_xlabel("time [h]")
    axes[2].set_ylabel("Latent heat flux [W m-2]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
