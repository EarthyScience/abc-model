import matplotlib.pyplot as plt

import abcmodel


def main():
    inner_dt = 60.0 * 5
    outter_dt = 60.0 * 30
    runtime = 5 * 86400.0
    tstart = 7.0

    rad_model = abcmodel.rad.StandardRadiationModel()
    rad_state = rad_model.init_state()

    bio_model = abcmodel.land.biosphere.AgsModel()
    bio_state = bio_model.init_state()
    soil_model = abcmodel.land.soil.StandardSoilModel()
    soil_state = soil_model.init_state()
    surface_model = abcmodel.land.surface.StandardSurfaceModel()
    surface_state = surface_model.init_state()
    land_model = abcmodel.land.StandardLandModel(
        biosphere=bio_model,
        soil=soil_model,
        surface=surface_model,
    )
    land_state = land_model.init_state(
        biosphere_state=bio_state,
        soil_state=soil_state,
        surface_state=surface_state,
    )

    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state()
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel()
    mixed_layer_state = mixed_layer_model.init_state()
    sbl_model = abcmodel.atmos.stable_layer.ZilitinkevichModel()
    sbl_state = sbl_model.init_state()
    residual_model = abcmodel.atmos.residual_layer.FrozenResidualModel()
    residual_state = residual_model.init_state()
    cloud_model = abcmodel.atmos.clouds.NoCloudModel()
    cloud_state = cloud_model.init_state()
    atmos_model = abcmodel.atmos.DayAndNightAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        sbl_layer=sbl_model,
        residual_layer=residual_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_state,
        mixed=mixed_layer_state,
        sbl=sbl_state,
        residual=residual_state,
        clouds=cloud_state,
    )

    abcoupler = abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    time, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )

    abcmodel.plotting.simple(
        time,
        trajectory,
        left_top_path="atmos.h_abl",
        mid_top_path="atmos.theta",
        right_top_path="atmos.q",
        left_bottom_path="atmos.active_bl.is_night",
        mid_bottom_path="land.surface.le",
        right_bottom_path="land.wCO2",
    )
    plt.show()


if __name__ == "__main__":
    main()
