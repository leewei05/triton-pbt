import torch
import triton
import triton.language as tl
from hypothesis import given, strategies as st, settings
import numpy as np

# ----------------
# data types
# ----------------
int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
integral_dtypes = int_dtypes + uint_dtypes
float_dtypes = ['float16', 'float32', 'float64']
float_dtypes_with_bfloat16 = float_dtypes + ['bfloat16']
dtypes = integral_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']

# ----------------
# test math ops
# ----------------
@triton.jit
def abs_kernel(x_ptr, z_ptr, size: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    x = tl.load(x_ptr + offsets)
    z = tl.abs(x)
    tl.store(z_ptr + offsets, z)

def test_abs(dtype=torch.float32, device='cuda'):
    SIZE = 128
    
    # inputs
    x_np = np.random.uniform(-10, 10, size=(SIZE,)).astype(np.float32)
    x_torch = torch.from_numpy(x_np).to(device)
    z_torch = torch.empty_like(x_torch)

    # reference result 
    z_ref = np.abs(x_np)

    abs_kernel[(1,)](
        x_ptr=x_torch, 
        z_ptr=z_torch, 
        size=SIZE, 
        num_warps=4
    )

    z_tri_np = z_torch.cpu().numpy()
    np.testing.assert_allclose(z_ref, z_tri_np, rtol=1e-5)
    print(f"Test Passed for {dtype}!")

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")
    print("Starting Hypothesis tests...")
    test_abs()