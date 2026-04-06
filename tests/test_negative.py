import pytest
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

def test_arange_invalid_range():
    @triton.jit
    def arange_kernel(START: tl.constexpr, END: tl.constexpr):
        x = tl.arange(START, END)

    with pytest.raises(CompilationError, match="arange's range must be a power of 2"):
        arange_kernel[(1,)](START=0, END=7)

    with pytest.raises(CompilationError, match="arange's range must be a power of 2"):
        arange_kernel[(1,)](START=0, END=10)

    with pytest.raises(CompilationError, match="end argument must be greater"):
        arange_kernel[(1,)](START=10, END=2)