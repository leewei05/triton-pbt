import torch
import triton
import triton.language as tl
from hypothesis import given, assume, strategies as st, settings
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

@triton.jit
def general_shape_kernel(
    in_ptr, out_ptr,
    SHAPE: tl.constexpr,
    PERMUTE: tl.constexpr
):
    # start with 1D
    total_elements = 1
    for dim in SHAPE:
        total_elements *= dim

    offsets = tl.arange(0, total_elements)
    x = tl.load(in_ptr + offsets)

    # logical SHAPE view
    x_nd = tl.view(x, SHAPE)
    x_manipulated = tl.permute(x_nd, PERMUTE)

    # back to 1D
    final_x = tl.view(x_manipulated, (total_elements,))
    tl.store(out_ptr + offsets, final_x)

@st.composite
def shape_and_permute_strategy(draw):
    # n dimensions
    rank = draw(st.integers(min_value=2, max_value=6))

    # TODO: use st.integers for better test coverage
    shape = tuple(draw(st.lists(
        st.sampled_from([2, 4, 8, 16]),
        min_size=rank,
        max_size=rank
    )))

    total_elements = 1
    for dim in shape:
        total_elements *= dim
    assume(64 <= total_elements <= 4096)

    # Generate a random permutation of the indices [0, 1, ..., rank-1]
    indices = list(range(rank))
    permute = tuple(draw(st.permutations(indices)))

    return shape, permute, total_elements

@given(data=st.data(), num_warps=st.sampled_from([4, 8]))
def test_general_shape_ops(data, num_warps):
    shape, permute, total_elements = data.draw(shape_and_permute_strategy())

    device = 'cuda'
    x = torch.randn(total_elements, device=device, dtype=torch.float32)
    z_triton = torch.empty_like(x)

    # --- PyTorch Reference ---
    # We must use .contiguous() after permute because Triton's
    # tl.store(offsets, x_flat) effectively expects a contiguous layout.
    z_ref = x.view(shape).permute(permute).contiguous().view(-1)

    grid = (1,)
    general_shape_kernel[grid](
        x, z_triton,
        SHAPE=shape,
        PERMUTE=permute,
        num_warps=num_warps
    )

    # Assert correctness
    torch.testing.assert_close(z_triton, z_ref)