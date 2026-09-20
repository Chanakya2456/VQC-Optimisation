# VQC-Optimisation

This repository packages the variational quantum optimization experiment from the notebook into a reusable Python project.

## Overview

The project trains a small PennyLane circuit with JAX and JAXopt to fit a simple target function over a synthetic dataset. It includes:

- a reusable PennyLane model
- JAX-based loss and gradient evaluation
- a gradient descent optimizer loop
- a CLI entry point for running the training job
- a basic automated test suite

## Project structure

```text
quantum-optimizer-repo/
+-- README.md
+-- pyproject.toml
+-- requirements.txt
+-- .gitignore
+-- src/
¦   +-- quantum_optimizer/
¦       +-- __init__.py
¦       +-- __main__.py
¦       +-- model.py
¦       +-- train.py
+-- tests/
¦   +-- test_model.py
+-- notebooks/
    +-- Quantum.ipynb
```

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m quantum_optimizer --steps 100
```

## Example output

The training script prints a loss summary at each interval and then returns the final parameters.

## Notes

- This project uses JAX on the CPU by default.
- PennyLane and JAXopt are required for the optimizer loop.
- The notebook in the `notebooks` directory is kept as a reference for the original interactive experiment.
