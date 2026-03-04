import torch
import triton
import triton.language as tl

# ----------------
# data types
# ----------------
int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
# TODO: add uint16, uint32, uint64
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

def patch_kernel(template_fn, to_replace):
    src = template_fn.src
    for key, value in to_replace.items():
        src = src.replace(key, value)

    # create a new JITFunction object to replace
    new_kernel = triton.runtime.jit.JITFunction(template_fn.fn)
    new_kernel.hash = None
    new_kernel._src = src

    return new_kernel

def debug_mismatch(op_name, n, block_size, z_torch, z_ref, tol, dtypes, *inputs):
    """
    Unified diagnostic report for Triton kernel failures.
    Works for both Unary and Binary ops.
    """
    print(f"\n{'!'*20} TRITON DEBUG REPORT {'!'*20}")
    print(f"Op: {op_name} | N: {n} | Block: {block_size}")
    print(f"Dtypes: {', '.join([str(d) for d in dtypes])} | Out: {z_ref.dtype}")

    # 1. Locate the exact indices where bits differed
    # We use a strict check to find exactly where it failed the tolerance
    mask = ~torch.isclose(z_torch, z_ref, rtol=tol, atol=tol, equal_nan=True)
    indices = torch.nonzero(mask).flatten()
    num_errors = len(indices)

    print(f"Mismatched Elements: {num_errors} / {n} ({(num_errors/n)*100:.2f}%)")

    # 2. Only show the first 5 errors to keep it clean
    peek = indices[:5]

    print(f"\nSample of failing indices: {peek.tolist()}")

    # 3. Dynamically print inputs (handles X or X, Y)
    for i, inp in enumerate(inputs):
        label = "X" if len(inputs) == 1 else ("X" if i == 0 else "Y")
        print(f"Input {label} at failure:  {inp[peek].tolist()}")

    print(f"Triton Result:       {z_torch[peek].tolist()}")
    print(f"Expected Reference:  {z_ref[peek].tolist()}")

    # 4. Bit-level inspection for the first failure (Compiler Engineer's favorite)
    if num_errors > 0:
        idx = peek[0]
        print(f"\n[Bit Inspection @ idx {idx}]")
        print(f"Triton Hex: {hex(z_torch[idx].view(torch.int32).item()) if z_torch.element_size() == 4 else 'N/A'}")
        print(f"Ref Hex:    {hex(z_ref[idx].view(torch.int32).item()) if z_ref.element_size() == 4 else 'N/A'}")

    print(f"{'!'*60}\n")
