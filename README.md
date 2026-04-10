# Triton Property-Based Testing (PBT)

This project uses **Hypothesis** to automatically discover edge cases in Triton kernels by randomly generating shapes, data types, and hardware configurations.

## Quickstart

Allocate a GPU node and activate the environment:
```sh
sbatch ./scripts/tunnel.slr
# run this the first time
source ./scripts/init.sh
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