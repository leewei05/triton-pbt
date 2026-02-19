## Quickstart

```sh
# allocate GPU
sbatch ./scripts/tunnel.slr

# login to the allocated node
# activate conda env
source ./scripts/setup.sh

# run test
pytest tests

# print hypothesis test cases on a single test file
pytest tests/test_core.py --hypothesis-verbosity=verbose -s
```