# Triton Property-Based Testing (PBT)

This project uses **Hypothesis** to automatically discover edge cases in Triton kernels by randomly generating shapes, data types, and hardware configurations.

## Prerequisite

- Nvidia GPU (Ampere or newer architecture)
- Slurm Workload Manager (e.g., University of Utah CHPC)

## Quickstart

If you have a Nvidia GPU locally, you can skip the following process and install the required Python packages listed in `./scripts/init.sh`.

```sh
pip install torch triton hypothesis pytest numpy
```

First, login to a dedicated CHPC environment.

Activate the environment:
```sh
# 1. Create the environment (only run this the first time)
# It's gonna take a while to install required packages.
# Do not run this after SSH into a GPU node. You will get OOM killed when installing packages.
source ./scripts/init.sh

# 2. Allocate a GPU node
sbatch ./scripts/tunnel.slr

# 3. Check which node was allocated
squeue -u $USER

# 4. SSH into the allocated node (e.g., notch329)
ssh <node_name>

# 5. Activate the environment in the allocated node.
source ./scripts/setup.sh
```

Run all tests with `dev` profile (less examples, faster).
```sh
pytest -v tests/ --hypothesis-profile dev
```

To run a thorough coverage, use `stress` profile.
```sh
pytest -v tests/ --hypothesis-profile stress
```

Run a single test manually to observe register pressure.

```sh
pytest tests/test_linalg.py -k "test_matmul_manual" --M 1015 --N 803 --K 704 --warps 8 --stages 2 -s
```
