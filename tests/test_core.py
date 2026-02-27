import torch
import triton
import triton.language as tl
from hypothesis import given, assume, strategies as st, settings
import pytest
import copy

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

# ----------------
# test math ops
# ----------------
@triton.jit
def unary_kernel(x_ptr, z_ptr, n_elements, size: tl.constexpr, op_name: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    z = UNARY_EXPR

    tl.store(z_ptr + offsets, z, mask=mask)

OP_CONFIGS = {
    "abs": (torch.abs, "tl.abs(x)", dtypes_with_bfloat16),
    "ceil": (torch.ceil, "tl.ceil(x)", float_dtypes_without_fp16),
    "cos": (torch.cos, "tl.cos(x)", float_dtypes_without_fp16),
    "erf": (torch.erf, "tl.erf(x)", float_dtypes_without_fp16),
    "exp": (torch.exp, "tl.exp(x)", float_dtypes_without_fp16),
    "exp2": (torch.exp2, "tl.exp2(x)", float_dtypes_without_fp16),
    "floor": (torch.floor, "tl.floor(x)", float_dtypes_without_fp16),
    "log": (torch.log, "tl.log(x)", float_dtypes_without_fp16),
    "log2": (torch.log2, "tl.log2(x)", float_dtypes_without_fp16),
    "rsqrt": (torch.rsqrt, "tl.rsqrt(x)", float_dtypes_without_fp16),
    "sigmoid": (torch.sigmoid, "tl.sigmoid(x)", float_dtypes_without_fp16),
    "sin": (torch.sin, "tl.sin(x)", float_dtypes_without_fp16),
    "sqrt": (torch.sqrt, "tl.sqrt(x)", float_dtypes_without_fp16),
    "sqrt_rn": (torch.sqrt, "tl.math.sqrt_rn(x.to(tl.float32)).to(x.dtype)", [torch.float32]),
}

# Guarantee that every op has 100 test examples
@pytest.mark.parametrize("op_name", list(OP_CONFIGS.keys()))
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=4096),
    block_size=st.sampled_from(BLOCK_SIZES),
    num_warps=st.sampled_from(NUM_WARPS),
    data=st.data()
)
def test_unary(n, block_size, num_warps, op_name, data):
    # ref_func is a reference from PyTorch or NumPy
    ref_func, triton_expr, allowed_dtypes = OP_CONFIGS[op_name]
    device = 'cuda'

    # Sample a dtype from the allowed list for this specific op
    dtype = data.draw(st.sampled_from(allowed_dtypes))

    if dtype in float_dtypes_with_bfloat16:
        if op_name in ["log", "log2", "sqrt", "rsqrt", "sqrt_rn"]:
            # Ensure inputs are in range [0.1, 10.1] to avoid domain errors
            # log(negative) -> nan
            # log(0) -> inf
            x_torch = torch.rand(n, device=device, dtype=dtype) * 10 + 0.1
        else:
            x_torch = torch.randn(n, device=device, dtype=dtype) * 10
    elif dtype == torch.uint8:
        x_torch = torch.randint(0, 255, (n,), device=device, dtype=dtype)
    else:
        x_torch = torch.randint(-100, 100, (n,), device=device, dtype=dtype)

    z_torch = torch.empty_like(x_torch)
    z_ref = ref_func(x_torch)

    grid = (triton.cdiv(n, block_size),)
    patched_kernel = patch_kernel(
        unary_kernel,
        {"UNARY_EXPR": triton_expr}
    )

    patched_kernel[grid](
        x_ptr=x_torch,
        z_ptr=z_torch,
        n_elements=n,
        size=block_size,
        op_name=op_name,
        num_warps=num_warps
    )

    tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
    try:
        torch.testing.assert_close(z_torch, z_ref, rtol=tol, atol=tol)
    except AssertionError as e:
        debug_mismatch(
            op_name, n, block_size, z_torch, z_ref, tol, 
            [dtype], x_torch
        )
        raise e

@triton.jit
def binary_kernel(x_ptr, y_ptr, z_ptr, n_elements, size: tl.constexpr, op_name: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    z = BINARY_EXPR

    tl.store(z_ptr + offsets, z, mask=mask)

def get_triton_reference(ref_func, x, y, op_name, dtype_x, dtype_y):
    """
    Implements Triton's internal type promotion rules to generate a bit-accurate reference.
    """
    # Triton promotes 16-bit / and % to float32 to match hardware behavior
    is_fp16 = lambda d: d in [torch.float16, torch.bfloat16]
    if op_name in ('/', '%') and (is_fp16(dtype_x) or is_fp16(dtype_y)):
        x, y = x.to(torch.float32), y.to(torch.float32)

    # If mixed signed/unsigned and unsigned is at least as wide as signed,
    # Triton favors the unsigned type (effectively zero-extending the signed int).
    width_x = x.element_size() * 8
    width_y = y.element_size() * 8
    
    # Check if we need to force unsigned semantics to match Triton's LLVM backend
    if (dtype_x == torch.uint8 and dtype_y in int_dtypes and width_x >= width_y):
        y = y.to(torch.uint8) # Interpret int8 bits as uint8
    elif (dtype_y == torch.uint8 and dtype_x in int_dtypes and width_y >= width_x):
        x = x.to(torch.uint8)

    return ref_func(x, y)

BINARY_OP_CONFIGS = {
    "+": (torch.add, "x + y", dtypes_with_bfloat16),
    "-": (torch.sub, "x - y", dtypes_with_bfloat16),
    "*": (torch.mul, "x * y", dtypes_with_bfloat16),
    "/": (torch.div, "x / y", dtypes_with_bfloat16),
    "%": (torch.fmod, "x % y", dtypes_with_bfloat16),
}

@pytest.mark.parametrize("op_name", list(BINARY_OP_CONFIGS.keys()))
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=4096),
    block_size=st.sampled_from(BLOCK_SIZES),
    num_warps=st.sampled_from(NUM_WARPS),
    data=st.data()
)
def test_binary(n, block_size, num_warps, op_name, data):
    ref_func, triton_expr, allowed_dtypes = BINARY_OP_CONFIGS[op_name]
    device = 'cuda'

    dtype_x = data.draw(st.sampled_from(allowed_dtypes))
    dtype_y = data.draw(st.sampled_from(allowed_dtypes))

    is_u_x = (dtype_x == torch.uint8)
    is_u_y = (dtype_y == torch.uint8)
    is_int_x = dtype_x in int_dtypes
    is_int_y = dtype_y in int_dtypes

    mixed_signedness = (is_u_x and is_int_y) or (is_int_x and is_u_y)
    should_fail = (op_name in ["/", "%"]) and mixed_signedness

    def gen_data(dtype):
        if dtype in float_dtypes_with_bfloat16:
            return torch.randn(n, device=device, dtype=dtype)
        elif dtype == torch.uint8:
            return torch.randint(0, 255, (n,), device=device, dtype=dtype)
        else:
            return torch.randint(-100, 100, (n,), device=device, dtype=dtype)

    x_torch = gen_data(dtype_x)
    y_torch = gen_data(dtype_y)

    if op_name in ["/", "%"]:
        # We cannot assume based on the whole tensor easily, 
        # so we ensure y doesn't contain zeros.
        assume(not torch.any(y_torch == 0))

    if op_name == "%" and dtype_y in float_dtypes_with_bfloat16:
        # Ensure the divisor isn't so small that precision loss is guaranteed
        # floor(x/y) should not exceed a reasonable precision limit
        assume(torch.all(torch.abs(x_torch / y_torch) < 1e5))

    z_ref = get_triton_reference(ref_func, x_torch, y_torch, op_name, dtype_x, dtype_y)
    z_torch = torch.empty_like(z_ref)

    grid = (triton.cdiv(n, block_size),)
    # update op placeholder
    patched_kernel = patch_kernel(
        binary_kernel,
        {"BINARY_EXPR": triton_expr}
    )

    if should_fail:
       with pytest.raises(triton.TritonError, match="signedness"):
            # we don't care about the output since it failed
            z_placeholder = torch.empty_like(x_torch)
            patched_kernel[grid](
                x_ptr=x_torch, y_ptr=y_torch, z_ptr=z_placeholder,
                n_elements=n, size=block_size, op_name=op_name, num_warps=num_warps
            )
    else:
        patched_kernel[grid](
            x_ptr=x_torch, y_ptr=y_torch, z_ptr=z_torch,
            n_elements=n, size=block_size, op_name=op_name, num_warps=num_warps
        )

        # Find the "weakest link" in the precision chain
        all_dtypes = [dtype_x, dtype_y, z_ref.dtype]
        low_prec_types = [torch.float16, torch.bfloat16]
    
        if any(d in low_prec_types for d in all_dtypes):
            tol = 1e-2 # 16-bit precision
        else:
            tol = 1e-5 # 32-bit or integer precision
        try:
            torch.testing.assert_close(
                z_torch,
                z_ref,
                rtol=tol,
                atol=tol,
                # Treats NaN == NaN as True. We're testing the output of Triton and PyTorch
                # Not the semantics of NaN == NaN.
                equal_nan=True
            )
        except AssertionError as e:
            debug_mismatch(
                op_name, n, block_size, z_torch, z_ref, tol, 
                [dtype_x, dtype_y], x_torch, y_torch
            )
            raise e

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")