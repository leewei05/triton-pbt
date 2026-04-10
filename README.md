# Triton Property-Based Testing (PBT)

This project uses **Hypothesis** to automatically discover edge cases in Triton kernels by randomly generating shapes, data types, and hardware configurations.

## Prerequisite

- Nvidia GPU (Ampere or newer recommended for `tl.dot` tests)
- Slurm Workload Manager (e.g., University of Utah CHPC)

## Quickstart

Activate the environment:
```sh
# 1. Allocate a GPU node
sbatch ./scripts/tunnel.slr

# 2. Check which node was allocated
squeue -u $USER

# 3. SSH into the allocated node (e.g., notch329)
ssh <node_name>

# 4. Create the environment (only run this the first time)
source ./scripts/init.sh

# 5. Load the environment
source ./scripts/setup.sh
```

Run all tests with `dev` profile (less examples, faster).
```sh
pytest - v tests/ --hypothesis-profile dev
```

To run a thorough coverage, use `stress` profile.
```sh
pytest - v tests/ --hypothesis-profile stress
```

Run a single test manually to observe register pressure.

```sh
pytest tests/test_linalg.py -k "test_matmul_manual" --M 1015 --N 803 --K 704 --warps 8 --stages 2 -s
```