import jax.numpy as jnp

from quantum_optimizer.model import build_dataset, loss_fn, optimize


def test_build_dataset_shapes():
    data, targets = build_dataset()
    assert data.shape == (5, 4)
    assert targets.shape == (4,)


def test_loss_fn_returns_scalar():
    data, targets = build_dataset()
    params = {"weights": jnp.ones((5, 3)), "bias": jnp.array(0.0)}
    loss = loss_fn(params, data, targets)
    assert loss.shape == ()


def test_optimize_runs_without_error():
    data, targets = build_dataset()
    params = {"weights": jnp.ones((5, 3)), "bias": jnp.array(0.0)}
    final_params = optimize(params, data, targets, steps=3, learning_rate=0.1)
    assert final_params["weights"].shape == (5, 3)
    assert final_params["bias"].shape == ()
