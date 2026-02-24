import torch
import triton
import triton.language as tl
from hypothesis import given, strategies as st, settings

# ----------------
# data types
# ----------------
int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
uint_dtypes = [torch.uint8]  # Note: PyTorch currently only supports uint8
integral_dtypes = int_dtypes + uint_dtypes

float_dtypes = [torch.float16, torch.float32, torch.float64]
float_dtypes_without_fp16 = [torch.float32, torch.float64]
float_dtypes_with_bfloat16 = float_dtypes + [torch.bfloat16]

dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + [torch.bfloat16]

# A GPU executes instructions in groups of 32 threads called a Warp
# 4 * 32, 8 * 32 ...
BLOCK_SIZES = [128, 256, 512, 1024]
NUM_WARPS = [4, 8, 16]

# ----------------
# test math ops
# ----------------
@triton.jit
def ceil_kernel(x_ptr, z_ptr, n_elements, size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    z = tl.ceil(x)
    tl.store(z_ptr + offsets, z, mask=mask)

@settings(max_examples=50, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=4096),
    block_size=st.sampled_from(BLOCK_SIZES),
    num_warps=st.sampled_from(NUM_WARPS),
    # tl.ceil() doesn't support fp16
    dtype=st.sampled_from(float_dtypes_without_fp16)
)
def test_ceil(n, block_size, num_warps, dtype):
    device = 'cuda'
    
    # Generate random floats (including negatives to test rounding behavior)
    x_torch = torch.randn(n, device=device, dtype=dtype) * 10
    z_torch = torch.empty_like(x_torch)

    # Reference result
    z_ref = torch.ceil(x_torch)

    grid = (triton.cdiv(n, block_size),)
    ceil_kernel[grid](
        x_ptr=x_torch, 
        z_ptr=z_torch, 
        n_elements=n,
        size=block_size,
        num_warps=num_warps
    )

    # Use a higher tolerance for smaller float types
    tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
    torch.testing.assert_close(z_torch, z_ref, rtol=tol, atol=tol)
    

@triton.jit
def abs_kernel(x_ptr, z_ptr, n_elements, size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    # a mask to prevent out-of-bounds access
    # For instance, we have 100 elements and size is 128.
    # Index 100 to 127 would be out-of-bounds access.
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    z = tl.abs(x)
    tl.store(z_ptr + offsets, z, mask=mask)

@settings(max_examples=50, deadline=None)
@given(
    # random tensor sizes
    n=st.integers(min_value=1, max_value=4096),
    block_size=st.sampled_from(BLOCK_SIZES),
    num_warps=st.sampled_from(NUM_WARPS),
    # all data types
    dtype=st.sampled_from(dtypes)
)
def test_abs(n, block_size, num_warps, dtype):
    device = 'cuda'
    
    # inputs
    if dtype in float_dtypes_with_bfloat16:
        x_torch = torch.randn(n, device=device, dtype=dtype)
    elif dtype == torch.uint8:
        # Unsigned cannot be negative; 0 to 255 is the valid range
        x_torch = torch.randint(0, 255, (n,), device=device, dtype=dtype)
    else:
        x_torch = torch.randint(-100, 100, (n,), device=device, dtype=dtype)

    z_torch = torch.empty_like(x_torch)

    # reference result 
    z_ref = torch.abs(x_torch)

    # We need enough blocks to cover 'n' elements
    # If we have 130 elements, Triton will schedule 2 blocks.
    # The first block handles 0 - 127 elements, the rest in the second block.
    grid = (triton.cdiv(n, block_size),)
    abs_kernel[grid](
        x_ptr=x_torch, 
        z_ptr=z_torch, 
        n_elements=n,
        size=block_size,
        num_warps=num_warps
    )

    tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
    torch.testing.assert_close(z_torch, z_ref, rtol=tol, atol=tol)

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")
    print("Starting Hypothesis tests...")
    test_abs()
    test_ceil()
    print("All tests passed!")