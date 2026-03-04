import torch
import triton
import triton.language as tl
from hypothesis import given, assume, strategies as st, settings
import pytest
from common import *

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # map program ID to a tile in the C matrix
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # create 2D matrix of memory addresses
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # using default Nvidia tf32 precision
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
        
        # pointer advancement along the K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty))
    
@settings(max_examples=50, deadline=None)
@given(
    # standard shapes that are multiples of 16 for Tensor Cores since we don't have masking yet
    M=st.integers(min_value=1, max_value=128).map(lambda x: x * 16),
    N=st.integers(min_value=1, max_value=128).map(lambda x: x * 16),
    K=st.integers(min_value=1, max_value=32).map(lambda x: x * 16),
    data=st.data()
)
def test_matmul_basic(M, N, K, data):
    device = 'cuda'
    # Use float16 for the inputs to ensure Tensor Core usage
    dtype = torch.float16
    # Dynamically draw block sizes that are valid for the chosen M, N, K
    bm = data.draw(st.sampled_from([16, 32, 64]))
    bn = data.draw(st.sampled_from([16, 32, 64]))
    bk = data.draw(st.sampled_from([16, 32]))

    # block is larger than the matrix (for now)
    assume(bm <= M and bn <= N and bk <= K)
    # TODO: add masking, right now we test standard shapes
    assume(M % bm == 0)
    assume(N % bn == 0)
    assume(K % bk == 0)
    
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c_triton = torch.empty((M, N), device=device, dtype=dtype)
    c_ref = torch.matmul(a, b)

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_kernel[grid](
        a, b, c_triton,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_triton.stride(0), c_triton.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=4
    )

    # Matmul results can drift slightly due to floating point accumulation order
    torch.testing.assert_close(c_triton, c_ref, atol=1e-2, rtol=1e-2)