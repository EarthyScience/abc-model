import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from utils import (
    HybridObukhovModel,
    NeuralNetwork,
)

import abcconfigs.class_model as cm
import abcmodel
from abcmodel.atmos.surface_layer.obukhov import (
    ObukhovSurfaceLayerModel,
    # compute_psih,
    # compute_psim,
)


def train_emulator(target_fn, key, label):
    print(f"training {label}...")
    zeta_train = jnp.linspace(-5.0, 2.0, 1000).reshape(-1, 1)
    targets = target_fn(zeta_train)

    model = NeuralNetwork(rngs=nnx.Rngs(key))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    def loss_fn(model, x, y):
        pred = model(x)
        return jnp.mean((pred - y) ** 2)

    @nnx.jit
    def update(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(grads)
        return loss

    for step in range(2000):
        loss = update(model, optimizer, zeta_train, targets)
        if step % 500 == 0:
            print(f"  step {step}, loss: {loss:.6f}")

    return model


def run_simulation(surface_layer_model):
    # time settings
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    runtime = 12 * 3600.0
    tstart = 6.5

    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )
    rad_state = rad_model.init_state(**cm.standard_radiation.state_kwargs)

    land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)
    land_state = land_model.init_state(**cm.jarvis_stewart.state_kwargs)

    surface_layer_state = surface_layer_model.init_state(
        **cm.obukhov_surface_layer.state_kwargs
    )

    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )
    mixed_layer_state = mixed_layer_model.init_state(**cm.bulk_mixed_layer.state_kwargs)

    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    cloud_state = cloud_model.init_state()

    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=surface_layer_model,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )
    atmos_state = atmos_model.init_state(
        surface=surface_layer_state, mixed=mixed_layer_state, clouds=cloud_state
    )

    abcoupler = abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)
    state = abcoupler.init_state(rad_state, land_state, atmos_state)

    times, trajectory = abcmodel.integrate(
        state, abcoupler, inner_dt, outter_dt, runtime, tstart
    )
    return times, trajectory


def main():
    root_key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(root_key)

    psim_emulator = train_emulator(ObukhovSurfaceLayerModel.compute_psim, k1, "psim")
    psih_emulator = train_emulator(ObukhovSurfaceLayerModel.compute_psih, k2, "psih")
    hybrid_model = HybridObukhovModel(
        psim_emulator=psim_emulator, psih_emulator=psih_emulator
    )

    # visual verification
    zeta_test = jnp.linspace(-6.0, 3.0, 200).reshape(-1, 1)

    psim_orig = ObukhovSurfaceLayerModel.compute_psim(zeta_test)
    psim_ml = hybrid_model.compute_psim(zeta_test)

    psih_orig = ObukhovSurfaceLayerModel.compute_psih(zeta_test)
    psih_ml = hybrid_model.compute_psih(zeta_test)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(zeta_test, psim_orig, "C0-", label="standard")
    plt.plot(zeta_test, psim_ml, "C1--", label="hybrid")
    plt.xlabel(r"$\zeta = z/L$")
    plt.ylabel(r"$\Psi_m$")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(zeta_test, psih_orig, "C0-", label="standard")
    plt.plot(zeta_test, psih_ml, "C1--", label="hybrid")
    plt.xlabel(r"$\zeta = z/L$")
    plt.ylabel(r"$\Psi_h$")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("running standard simulations...")
    standard_model = ObukhovSurfaceLayerModel()
    times, traj_std = run_simulation(standard_model)

    print("running hybrid simulation...")
    times, traj_hybrid = run_simulation(hybrid_model)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(times, traj_std.atmos.mixed.h_abl, "C0-", label="standard")
    plt.plot(times, traj_hybrid.atmos.mixed.h_abl, "C1--", label="hybrid")
    plt.xlabel("time [h]")
    plt.ylabel("PBL height [m]")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(times, traj_std.atmos.mixed.theta, "C0-", label="standard")
    plt.plot(times, traj_hybrid.atmos.mixed.theta, "C1--", label="hybrid")
    plt.xlabel("time [h]")
    plt.ylabel("potential temperature [K]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
