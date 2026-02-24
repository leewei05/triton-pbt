## Quickstart

```sh
# allocate GPU
sbatch ./scripts/tunnel.slr

# login to the allocated node
# activate conda env
source ./scripts/setup.sh

# run to check GPU status
python tests/test_core.py
GPU: NVIDIA A40
Triton version: 3.6.0
Starting Hypothesis tests...
All tests passed!

# run only test
pytest tests

# print hypothesis test cases on a single test file
pytest tests/test_core.py --hypothesis-verbosity=verbose -s

# test only one op
pytest tests/test_core.py -k "cos"
```