import jax.numpy as jnp
import numpy as np
import abcconfigs.class_model as cm
import abcmodel

def main():
    # time step [s]
    dt = 15.0
    # total run time [s]
    runtime = 12 * 3600.0

    # radiation with clouds
    radiation_init_conds = abcmodel.radiation.StandardRadiationwCloudsInitConds(
        **cm.standard_radiation_w_clouds.init_conds_kwargs
    )
    radiation_model = abcmodel.radiation.StandardRadiationwCloudsModel(
        **cm.standard_radiation_w_clouds.model_kwargs,
    )

    # land surface
    land_surface_init_conds = abcmodel.land.AquaCropInitConds(
        **cm.aquacrop.init_conds_kwargs,
    )
    land_surface_model = abcmodel.land.AquaCropModel(
        **cm.aquacrop.model_kwargs,
    )

    # surface layer
    surface_layer_init_conds = (
        abcmodel.atmosphere.surface_layer.StandardSurfaceLayerInitConds(
            **cm.standard_surface_layer.init_conds_kwargs
        )
    )
    surface_layer_model = abcmodel.atmosphere.surface_layer.StandardSurfaceLayerModel()

    # mixed layer
    mixed_layer_init_conds = abcmodel.atmosphere.mixed_layer.BulkMixedLayerInitConds(
        **cm.bulk_mixed_layer.init_conds_kwargs,
    )
    mixed_layer_model = abcmodel.atmosphere.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs,
    )

    # clouds
    cloud_init_conds = abcmodel.atmosphere.clouds.StandardCumulusInitConds()
    cloud_model = abcmodel.atmosphere.clouds.StandardCumulusModel()

    # define atmosphere model
    atmosphere_model = abcmodel.atmosphere.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    # define coupler and coupled state
    abcoupler = abcmodel.ABCoupler(
        radiation=radiation_model,
        land=land_surface_model,
        atmosphere=atmosphere_model,
    )
    state = abcoupler.init_state(
        radiation_init_conds,
        land_surface_init_conds,
        surface_layer_init_conds,
        mixed_layer_init_conds,
        cloud_init_conds,
    )

    # run
    time, trajectory = abcmodel.integrate(state, abcoupler, dt=dt, runtime=runtime)

    # Analyze stability
    # Check for NaNs
    if jnp.any(jnp.isnan(trajectory.theta)):
        print("FAIL: NaNs detected in theta")
        return

    # Check for oscillations in thetasurf
    # Calculate first backward difference
    diff_theta = jnp.diff(trajectory.thetasurf)
    # Calculate second backward difference (acceleration/oscillation)
    diff2_theta = jnp.diff(diff_theta)
    
    # Check if there are extreme oscillations
    # A simple smooth curve should have small 2nd derivatives. 
    # High frequency oscillations will have high 2nd derivatives.
    max_oscillation = jnp.max(jnp.abs(diff2_theta))
    mean_oscillation = jnp.mean(jnp.abs(diff2_theta))

    print(f"Max 2nd derivative of thetasurf: {max_oscillation}")
    print(f"Mean 2nd derivative of thetasurf: {mean_oscillation}")

    # Heuristic check: for a smooth 15s timestep simulation, 2nd derivative shouldn't be huge
    # purely empirical threshold, but "varying too much" usually means erratic jumps
    if max_oscillation > 1.0: # 1 K per step change in rate is huge for 15s dt
         print("WARNING: High oscillation detected in thetasurf")
    else:
         print("SUCCESS: trajectory appears smooth")

    # Also check cloud transmittance and net radiation
    print(f"Final cl_trans: {trajectory.cl_trans[-1]}")
    print(f"Final net_rad: {trajectory.net_rad[-1]}")

if __name__ == "__main__":
    main()
