## Triton-PBT

This is a Triton Property-Based Testing project. We utilize Hypothesis to test Triton randomly.

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

# run tests
# pytest -v --hypothesis-profile stress tests/ for a more extensive test
pytest -v --hypothesis-profile dev tests/
tests/test_linalg.py::test_matmul_basic PASSED                                                                                                                                [  3%]
tests/test_linalg.py::test_matmul_manual SKIPPED (Manual parameters (M, N, K) not provided.)                                                                                  [  6%]
tests/test_math.py::test_unary[abs] PASSED                                                                                                                                    [  9%]
tests/test_math.py::test_unary[ceil] PASSED                                                                                                                                   [ 12%]
tests/test_math.py::test_unary[cos] PASSED                                                                                                                                    [ 15%]
tests/test_math.py::test_unary[erf] PASSED                                                                                                                                    [ 18%]
tests/test_math.py::test_unary[exp] PASSED                                                                                                                                    [ 21%]
tests/test_math.py::test_unary[exp2] PASSED                                                                                                                                   [ 24%]
tests/test_math.py::test_unary[floor] PASSED                                                                                                                                  [ 27%]
tests/test_math.py::test_unary[log] PASSED                                                                                                                                    [ 30%]
tests/test_math.py::test_unary[log2] 
PASSED                                                                                                                                [ 36%]
tests/test_math.py::test_unary[sigmoid] PASSED                                                                                                                                [ 39%]
tests/test_math.py::test_unary[sin] PASSED                                                                                                                                    [ 42%]
tests/test_math.py::test_unary[sqrt] PASSED                                                                                                                                   [ 45%]
tests/test_math.py::test_unary[sqrt_rn] PASSED                                                                                                                                [ 48%]
tests/test_math.py::test_binary[+] PASSED                                                                                                                                     [ 51%]
tests/test_math.py::test_binary[-] PASSED                                                                                                                                     [ 54%]
tests/test_math.py::test_binary[*] PASSED                                                                                                                                     [ 57%]
tests/test_math.py::test_binary[/] PASSED                                                                                                                                     [ 60%]
tests/test_math.py::test_binary[%] PASSED                                                                                                                                     [ 63%]
tests/test_math.py::test_binary_broadcast[+] PASSED                                                                                                                           [ 66%]
tests/test_math.py::test_binary_broadcast[-] PASSED                                                                                                                           [ 69%]
tests/test_math.py::test_binary_broadcast[*] PASSED                                                                                                                           [ 72%]
tests/test_math.py::test_binary_broadcast[/] PASSED                                                                                                                           [ 75%]
tests/test_math.py::test_binary_broadcast[%] PASSED                                                                                                                           [ 78%]
tests/test_negative.py::test_arange_non_power_of_two PASSED                                                                                                                   [ 81%]
tests/test_negative.py::test_arange_negative_bounds PASSED                                                                                                                    [ 84%]
tests/test_negative.py::test_arange_oversized PASSED                                                                                                                          [ 87%]
tests/test_negative.py::test_dot_inner_dim_mismatch PASSED                                                                                                                    [ 90%]
tests/test_negative.py::test_dot_too_small PASSED                                                                                                                             [ 93%]
tests/test_negative.py::test_reshape_mismatch PASSED                                                                                                                          [ 96%]
tests/test_shape.py::test_general_shape_ops PASSED                                                                                                                            [100%]

===================================================================== 32 passed, 1 skipped in 478.09s (0:07:58) ==

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