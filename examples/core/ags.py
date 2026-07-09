import matplotlib.pyplot as plt

import abcmodel


def main():
    # integration parameters
    inner_dt = 60.0 * 5
    outter_dt = 60.0 * 15
    runtime = 12 * 3600.0
    tstart = 6.5

    # rad with clouds
    rad_model = abcmodel.rad.CloudyRadiationModel()
    rad_state = rad_model.init_state()

    # land surface
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

    # atmos
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state()
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel()
    mixed_layer_state = mixed_layer_model.init_state()
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_state,
        mixed=mixed_layer_state,
        clouds=cloud_state,
    )

    # coupler and coupled state
    abcoupler = abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    # run run run
    time, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )
    abcmodel.plotting.simple(time, trajectory)
    plt.show()


if __name__ == "__main__":
    main()
