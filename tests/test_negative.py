from hypothesis import given, strategies as st, assume
import pytest
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

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