import torch
import triton
import triton.language as tl
from hypothesis import given, assume, strategies as st, settings
import pytest
from common import *

@triton.jit
def general_shape_kernel(
    in_ptr, out_ptr,
    SHAPE: tl.constexpr,
    TOTAL_ELEMENTS: tl.constexpr,
):
    # 1. Create a 1D grid of indices [0, 1, 2, ..., 63]
    idx = tl.arange(0, TOTAL_ELEMENTS)

    # 2. Map those indices to N-D coordinates
    idx_nd = tl.reshape(idx, SHAPE)

    # 3. Permute the INDICES (The "Logic Shuffle")
    # This creates a map of "where the data SHOULD come from"
    idx_permuted = PERMUTE_OP # Injected: tl.permute(idx_nd, (0, 2, 1))

    # 4. Flatten the shuffled indices
    # If idx_permuted[1] is 16, then we will load the 16th element
    # and put it in the 1st register.
    idx_flat = tl.reshape(idx_permuted, (TOTAL_ELEMENTS,))

    # 5. Load using the shuffled indices
    # This is a "Gather" operation
    x = tl.load(in_ptr + idx_flat)

    # 6. Store linearly
    # This writes the shuffled data into a contiguous 1D array
    tl.store(out_ptr + idx, x)

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

    permute_string = f"tl.permute(idx_nd, {permute})"
    patched_kernel = patch_kernel(
        general_shape_kernel,
        {"PERMUTE_OP": permute_string}
    )

    grid = (1,)
    patched_kernel[grid](
        x, z_triton,
        SHAPE=shape,
        TOTAL_ELEMENTS=total_elements,
        num_warps=num_warps
    )

    # Assert correctness
    torch.testing.assert_close(z_triton, z_ref)