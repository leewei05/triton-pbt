## Quickstart

```sh
# allocate GPU
sbatch ./scripts/tunnel.slr

# login to the allocated node
# activate conda env
source ./scripts/setup.sh

# run to check GPU status
python tests/test_math.py
GPU: NVIDIA A40
Triton version: 3.6.0

# run only test
pytest tests

# print hypothesis test cases on a single test file
pytest tests/test_math.py --hypothesis-verbosity=verbose -s

# test only one op
pytest tests/test_math.py -k "cos"

# test only one test function
pytest tests/test_math.py -k "test_binary"

# captures print to stdout
pytest tests/test_math.py -k "test_binary" -s

# rerun last failed test case
pytest tests/test_math.py -k "test_binary" -s --lf

# manually run matrix multiplication test case
pytest tests/test_linalg.py -k "test_matmul_manual" --M 1015 --N 803 --K 704 --warps 8 --stages 2 -s
```