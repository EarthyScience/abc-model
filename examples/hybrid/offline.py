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

from abcmodel.integration import outter_step_fn


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


def train(x: Array, y: Array, key: Array):
    print("training...")
    mkey, hkey = jax.random.split(key)
    mnet = NeuralNetwork(rngs=nnx.Rngs(mkey))
    hnet = NeuralNetwork(rngs=nnx.Rngs(hkey))
    net = HybridObukhovModel(mnet, hnet)
    optimizer = nnx.Optimizer(net, optax.adam(1e-3))

    def loss_fn(model, x, y):
        pred = model(x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def update(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(grads)
        return loss

    for step in range(2000):
        loss = update(net, optimizer, x, y)
        if step % 500 == 0:
            print(f"  step {step}, loss: {loss:.6f}")

    return net


def plot_results(model, x: Array, y: Array):
    pass


def main():
    key = jax.random.PRNGKey(42)
    data_key, train_key = jax.random.split(key)
    x_train, x_test, y_train, y_test = load_data(data_key)
    model = train(x_train, y_train, train_key)
    plot_results(model, x_test, y_test)


if __name__ == "__main__":
    main()
