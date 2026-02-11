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
    land_model = abcmodel.land.AgsModel()
    land_state = land_model.init_state()

    # surface layer
    surface_layer_model = abcmodel.atmos.surface_layer.ObukhovModel()
    surface_layer_state = surface_layer_model.init_state()

    # mixed layer
    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkModel()
    mixed_layer_state = mixed_layer_model.init_state()

    # clouds
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    # atmos
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
    abcmodel.plotting.show(
        time,
        trajectory,
        "atmos.mixed.h_abl",
        "atmos.mixed.theta",
        "atmos.mixed.q",
        "atmos.clouds.cc_frac",
        "land.le",
        "land.wCO2",
    )


if __name__ == "__main__":
    main()
