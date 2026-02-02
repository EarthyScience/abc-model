# use u and v to code the emulator and make a loss function based on LE in the next time step?
# probably more natural to use xarray instead of h5py in order (at least) to read the data...
# should be nice to extract inner + outter step functions into one so that it can be used here
# and then there is maybe going to be some forcing
# in the online step one would call the entire run_simulation function...

import os

import h5py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from utils import HybridObukhovModel, NeuralNetwork

import abcconfigs.class_model as cm
import abcmodel


# todo: x needs to be the entire state, t needs to be returned, and y needs to be normalized
def load_data(key: Array, ratio: float = 0.8) -> tuple[Array, ...]:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read data
    with h5py.File(os.path.join(script_dir, "../../data/dataset.h5"), "r") as f:
        u = jnp.expand_dims(jnp.array(f["u"]), axis=1)
        v = jnp.expand_dims(jnp.array(f["v"]), axis=1)
        lhf = jnp.array(f["le"])

    # adjust dims so that we have (u_t, v_t) |-> LE_(t+1)
    x = jnp.concatenate((u, v), axis=1)
    x = x[..., :-1]
    y = lhf[:, 1:]

    # split into train and test sets
    num_ensembles = y.shape[0]
    perm_idxs = jax.random.permutation(key, num_ensembles)
    train_idxs = perm_idxs[: int(ratio * num_ensembles)]
    test_idxs = perm_idxs[int(ratio * num_ensembles) :]
    x_train, x_test = x[train_idxs], x[test_idxs]
    y_train, y_test = y[train_idxs], y[test_idxs]

    return x_train, x_test, y_train, y_test


def load_model(key: Array):
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )
    land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)

    # definition od the hybrid model
    mkey, hkey = jax.random.split(key)
    mnet = NeuralNetwork(rngs=nnx.Rngs(mkey))
    hnet = NeuralNetwork(rngs=nnx.Rngs(hkey))
    hybrid_surface = HybridObukhovModel(mnet, hnet)

    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=hybrid_surface,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    return abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)


def train(model, x: Array, y: Array):
    # time settings: should be the same as the
    # ones that were used to generate the dataset
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    tstart = 6.5
    inner_tsteps = int(outter_dt / inner_dt)

    print("training...")
    optimizer = nnx.Optimizer(model.atmos.surface_layer.psim_emulator, optax.adam(1e-3))
    # optimizer2 = nnx.Optimizer(model.atmos.mixed_layer.psih_emulator, optax.adam(1e-3))

    def loss_fn(model, x, y):
        pred = abcmodel.integration.outter_step(
            x,
            t,  # this should come from the loading of the dataset
            coupler=model,
            inner_dt=inner_dt,
            inner_tsteps=inner_tsteps,
            tstart=tstart,
        )
        # this should be normalized
        pred_le = pred.land.le  # type: ignore
        return jnp.mean((pred_le - y) ** 2)

    @jax.jit
    def update(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(grads)
        return loss

    for step in range(2000):
        loss = update(model, optimizer, x, y)
        if step % 500 == 0:
            print(f"  step {step}, loss: {loss:.6f}")

    return model


def main():
    key = jax.random.PRNGKey(42)
    data_key, model_key = jax.random.split(key)
    x_train, x_test, y_train, y_test = load_data(data_key)
    model = load_model(model_key)
    model = train(model, x_train, y_train)


if __name__ == "__main__":
    main()
