import torch
import triton
import triton.language as tl
from hypothesis import given, strategies as st, settings
import pytest

@triton.jit
def max_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute element-wise maximum
    output = tl.maximum(x, y)

    # Store result
    tl.store(out_ptr + offsets, output, mask=mask)

# 2. Hypothesis Test Suite
@settings(max_examples=50, deadline=None)
@given(
    # Generate random tensor sizes, including non-powers-of-two
    n=st.integers(min_value=1, max_value=4096),
    # Randomly select a block size to test tiling logic
    block_size=st.sampled_from([128, 256, 512, 1024])
)
def test_triton_maximum(n, block_size):
    # Setup random data
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.randn(n, device='cuda', dtype=torch.float32)
    output_triton = torch.empty_like(x)

    # Launch Grid
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    max_kernel[grid](x, y, output_triton, n, BLOCK_SIZE=block_size)

    # Reference Oracle (PyTorch)
    expected_output = torch.maximum(x, y)

    # Assert numerical correctness
    torch.testing.assert_close(output_triton, expected_output, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    print("🚀 Starting Hypothesis tests for tl.maximum...")
    try:
        test_triton_maximum()
        print("✅ All randomized tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
