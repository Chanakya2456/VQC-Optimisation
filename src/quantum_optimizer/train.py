from __future__ import annotations

import argparse

import jax.numpy as jnp

from .model import build_dataset, optimize


def run_training(steps: int = 100, learning_rate: float = 0.3, print_training: bool = False) -> dict:
    data, targets = build_dataset()
    weights = jnp.ones([5, 3])
    bias = jnp.array(0.0)
    params = {"weights": weights, "bias": bias}

    final_params = optimize(
        params=params,
        data=data,
        targets=targets,
        steps=steps,
        learning_rate=learning_rate,
        print_training=print_training,
    )

    prediction = final_params["weights"].sum(axis=1) + final_params["bias"]
    loss = jnp.sum((targets - prediction[: len(targets)]) ** 2 / len(targets))
    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "loss": float(loss),
        "final_bias": float(final_params["bias"]),
        "final_weights": final_params["weights"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the quantum variational optimizer")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimization steps")
    parser.add_argument("--learning-rate", type=float, default=0.3, help="Gradient descent learning rate")
    parser.add_argument("--print-training", action="store_true", help="Print loss every 5 steps")
    args = parser.parse_args()

    result = run_training(steps=args.steps, learning_rate=args.learning_rate, print_training=args.print_training)
    print(f"Training complete: steps={result['steps']}, learning_rate={result['learning_rate']}, final_bias={result['final_bias']}")
    print("Final weights:")
    print(result["final_weights"])


if __name__ == "__main__":
    main()
