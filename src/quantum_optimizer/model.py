from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import pennylane as qml

n_wires = 5

jax.config.update("jax_platform_name", "cpu")

def build_dataset() -> tuple[jnp.ndarray, jnp.ndarray]:
    data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
    targets = jnp.array([-0.2, 0.4, 0.35, 0.2])
    return data, targets


def make_device() -> qml.Device:
    return qml.device("default.qubit", wires=n_wires)


def circuit(data: jnp.ndarray, weights: jnp.ndarray) -> qml.measurements.ExpectationMP:
    dev = make_device()

    @qml.qnode(dev)
    def _circuit(data, weights):
        for i in range(n_wires):
            qml.RY(data[i], wires=i)

        for i in range(n_wires):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RX(weights[i, 2], wires=i)
            qml.CNOT(wires=[i, (i + 1) % n_wires])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    return _circuit(data, weights)


def build_model() -> qml.QNode:
    dev = make_device()

    @qml.qnode(dev)
    def _circuit(data, weights):
        for i in range(n_wires):
            qml.RY(data[i], wires=i)

        for i in range(n_wires):
            qml.RX(weights[i, 0], wires=i)
            qml.RY(weights[i, 1], wires=i)
            qml.RX(weights[i, 2], wires=i)
            qml.CNOT(wires=[i, (i + 1) % n_wires])

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    return _circuit


def model(data: jnp.ndarray, weights: jnp.ndarray, bias: float) -> jnp.ndarray:
    return circuit(data, weights) + bias


def loss_fn(params: dict[str, Any], data: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    predictions = model(data, params["weights"], params["bias"])
    return jnp.sum((targets - predictions) ** 2 / len(data))


def loss_and_grad(params: dict[str, Any], data: jnp.ndarray, targets: jnp.ndarray, print_training: bool, i: int) -> tuple[jnp.ndarray, dict[str, Any]]:
    loss_val, grad_val = jax.value_and_grad(loss_fn)(params, data, targets)

    def print_fn() -> None:
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

    jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)
    return loss_val, grad_val


def optimize(
    params: dict[str, Any],
    data: jnp.ndarray,
    targets: jnp.ndarray,
    steps: int = 100,
    learning_rate: float = 0.3,
    print_training: bool = False,
) -> dict[str, Any]:
    opt = jaxopt.GradientDescent(
        lambda p, d, t, prt, idx: loss_and_grad(p, d, t, prt, idx),
        stepsize=learning_rate,
        value_and_grad=True,
    )
    opt_state = opt.init_state(params)

    for i in range(steps):
        params, opt_state = opt.update(params, opt_state, data, targets, print_training, i)

    return params


def optimize_jit(
    params: dict[str, Any],
    data: jnp.ndarray,
    targets: jnp.ndarray,
    steps: int = 100,
    learning_rate: float = 0.3,
    print_training: bool = False,
) -> dict[str, Any]:
    @jax.jit
    def _optimization_jit(current_params, current_data, current_targets):
        opt = jaxopt.GradientDescent(
            lambda p, d, t, prt, idx: loss_and_grad(p, d, t, prt, idx),
            stepsize=learning_rate,
            value_and_grad=True,
        )
        opt_state = opt.init_state(current_params)

        def update(i, args):
            p, s, d, t, prt = args
            p, s = opt.update(p, s, d, t, prt, i)
            return (p, s, d, t, prt)

        args = (current_params, opt_state, current_data, current_targets, print_training)
        params_out, _, _, _, _ = jax.lax.fori_loop(0, steps, update, args)
        return params_out

    return _optimization_jit(params, data, targets)
