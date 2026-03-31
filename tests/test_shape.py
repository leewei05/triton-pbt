import torch
import triton
import triton.language as tl
from hypothesis import given, strategies as st, settings
import pytest
from common import print_kernel_stats

@triton.jit
def shape_op_kernel(
    in_ptr, out_ptr, 
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # This kernel assumes we are processing one M x N tile for simplicity
    # In a real kernel, you'd use program_id to tile a larger matrix
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    
    # Physical address: in_ptr + row * N + col
    in_ptrs = in_ptr + (rm[:, None] * N + rn[None, :])
    x = tl.load(in_ptrs) # x is shape (BLOCK_M, BLOCK_N)

    # no memory is moved, only the logical shape is changed.
    x_t = tl.trans(x) # x_t is now shape (BLOCK_N, BLOCK_M)

    # store a (BLOCK_N x BLOCK_M) matrix back to a 1D array.
    out_offsets = rn[:, None] * M + rm[None, :]
    out_ptrs = out_ptr + out_offsets
    
    tl.store(out_ptrs, x_t)

@given(
    # TODO: various shapes for testing alignment
    M=st.sampled_from([16, 32, 64]),
    N=st.sampled_from([16, 32, 64]),
    num_warps=st.sampled_from([4, 8])
)
def test_triton_transpose(M, N, num_warps):
    device = 'cuda'
    # Input is a flat array that represents an M x N matrix
    x = torch.randn(M * N, device=device, dtype=torch.float32)
    z_triton = torch.empty_like(x)
    
    # Reference: Reshape to 2D, transpose, then flatten back
    x_2d = x.view(M, N)
    z_ref = x_2d.t().contiguous().view(-1)

    # We launch exactly one program to handle this M x N block
    grid = (1,)
    
    shape_op_kernel[grid](
        x, z_triton, 
        M=M, N=N,
        BLOCK_M=M, BLOCK_N=N,
        num_warps=num_warps
    )

    torch.testing.assert_close(z_triton, z_ref)
    # print_kernel_stats(shape_op_kernel)