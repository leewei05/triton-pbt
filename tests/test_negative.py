from hypothesis import given, strategies as st, assume
import pytest
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

#####################
# Shape Constraints #
#####################
@triton.jit
def arange_kernel(START: tl.constexpr, END: tl.constexpr):
    x = tl.arange(START, END)

@given(
    start=st.integers(min_value=0, max_value=1024),
    end=st.integers(min_value=0, max_value=1024),
)
def test_arange_non_power_of_two(start, end):
    diff = end - start

    assume(diff > 0)
    assume((diff & (diff - 1)) != 0)

    with pytest.raises(CompilationError, match="power of 2"):
        arange_kernel[(1,)](START=start, END=end)

@given(
    val=st.integers(min_value=-1024, max_value=-1)
)
def test_arange_negative_bounds(val):
    # Testing that negative start or end triggers the 'int32' error
    with pytest.raises(CompilationError, match="arange must fit in int32"):
        arange_kernel[(1,)](START=val, END=val + 8)

# end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = 1048576
@given(
    size=st.integers(min_value=1048577, max_value=2000000)
)
def test_arange_oversized(size):
    # We need to make sure 'size' is a power of 2 to avoid
    # triggering the 'power of 2' error first.
    # Find the next power of 2 above the limit.
    oversized_pow2 = 1 << size.bit_length()

    with pytest.raises(CompilationError, match="exceeds triton maximum tens"):
        arange_kernel[(1,)](START=0, END=oversized_pow2)

@triton.jit
def dot_kernel(M: tl.constexpr, N: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr):
    # mismatch inner size
    a = tl.zeros((M, K1), dtype=tl.float32)
    b = tl.zeros((K2, N), dtype=tl.float32)
    c = tl.dot(a, b)

@given(
    m=st.sampled_from([16, 32, 64]),
    n=st.sampled_from([16, 32, 64]),
    k1=st.integers(min_value=16, max_value=64),
    k2=st.integers(min_value=16, max_value=64)
)
def test_dot_inner_dim_mismatch(m, n, k1, k2):
    assume(k1 != k2)
    with pytest.raises(CompilationError, match="mismatch inner size"):
        dot_kernel[(1,)](M=m, N=n, K1=k1, K2=k2)

@given(
    k=st.sampled_from([2, 4, 8]),
)
def test_dot_too_small(k):
    with pytest.raises(CompilationError, match="K >= 16"):
        dot_kernel[(1,)](M=1, N=1, K1=k, K2=k)

@triton.jit
def reshape_kernel(BLOCK_SIZE: tl.constexpr, TARGET_SHAPE: tl.constexpr):
    x = tl.arange(0, BLOCK_SIZE)
    y = tl.reshape(x, TARGET_SHAPE)

@given(
    block_size=st.sampled_from([16, 32, 64, 128]),
    target_shape=st.lists(st.sampled_from([2, 4, 8, 16]), min_size=2, max_size=4).map(tuple)
)
def test_reshape_mismatch(block_size, target_shape):
    target_total = 1
    for dim in target_shape:
        target_total *= dim

    # mismatch
    assume(target_total != block_size)

    with pytest.raises(CompilationError, match="number of elements"):
        reshape_kernel[(1,)](BLOCK_SIZE=block_size, TARGET_SHAPE=target_shape)
