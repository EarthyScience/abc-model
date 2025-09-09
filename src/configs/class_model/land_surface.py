from abcmodel.land_surface import (
    AquaCropInitConds,
    AquaCropParams,
    JarvisStewartInitConds,
    JarvisStewartParams,
)

# params for
# 1. Jarvis Stewart
jarvis_stewart_params = JarvisStewartParams(
    a=0.219,
    b=4.90,
    p=4.0,
    cgsat=3.56e-6,
    wsat=0.472,
    wfc=0.323,
    wwilt=0.171,
    c1sat=0.132,
    c2ref=1.8,
    lai=2.0,
    gD=0.0,
    rsmin=110.0,
    rssoilmin=50.0,
    alpha=0.25,
    cveg=0.85,
    wmax=0.0002,
    lam=5.9,
)

# 2. AquaCrop
aquacrop_params = AquaCropParams(
    a=0.219,
    b=4.90,
    p=4.0,
    cgsat=3.56e-6,
    wsat=0.472,
    wfc=0.323,
    wwilt=0.171,
    c1sat=0.132,
    c2ref=1.8,
    lai=2.0,
    gD=0.0,
    rsmin=110.0,
    rssoilmin=50.0,
    alpha=0.25,
    cveg=0.85,
    wmax=0.0002,
    lam=5.9,
    c3c4="c3",
)

# init conds for
# 1. Jarvis Stewart
jarvis_stewart_init_conds = JarvisStewartInitConds(
    wg=0.21,
    w2=0.21,
    temp_soil=285.0,
    temp2=286.0,
    surf_temp=290.0,
    wl=0.0000,
)

# 2. AquaCrop
aquacrop_init_conds = AquaCropInitConds(
    wg=0.21,
    w2=0.21,
    temp_soil=285.0,
    temp2=286.0,
    surf_temp=290.0,
    wl=0.0000,
)
