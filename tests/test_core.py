import torch
import triton
import triton.language as tl
from hypothesis import given, assume, strategies as st, settings
import pytest

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
def unary_kernel(x_ptr, z_ptr, n_elements, size: tl.constexpr, op_name: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if op_name == "abs":
        z = tl.abs(x)
    elif op_name == "ceil":
        z = tl.ceil(x)
    elif op_name == "cos":
        z = tl.cos(x)
    elif op_name == "erf":
        z = tl.erf(x)
    elif op_name == "exp":
        z = tl.exp(x)
    elif op_name == "exp2":
        z = tl.exp2(x)
    elif op_name == "floor":
        z = tl.floor(x)
    elif op_name == "log":
        z = tl.log(x)
    elif op_name == "log2":
        z = tl.log2(x)
    elif op_name == "rsqrt":
        z = tl.rsqrt(x)
    elif op_name == "sigmoid":
        z = tl.sigmoid(x)
    elif op_name == "sin":
        z = tl.sin(x)
    elif op_name == "sqrt":
        z = tl.sqrt(x)
    elif op_name == "sqrt_rn":
        # hardware intrinsic mapping
        z = tl.math.sqrt_rn(x.to(tl.float32)).to(x.dtype)

    tl.store(z_ptr + offsets, z, mask=mask)

OP_CONFIGS = {
    "abs": (torch.abs, dtypes_with_bfloat16),
    "ceil": (torch.ceil, float_dtypes_without_fp16),
    "cos": (torch.cos, float_dtypes_without_fp16),
    "erf": (torch.erf, float_dtypes_without_fp16),
    "exp": (torch.exp, float_dtypes_without_fp16),
    "exp2": (torch.exp2, float_dtypes_without_fp16),
    "floor": (torch.floor, float_dtypes_without_fp16),
    "log": (torch.log, float_dtypes_without_fp16),
    "log2": (torch.log2, float_dtypes_without_fp16),
    "rsqrt": (torch.rsqrt, float_dtypes_without_fp16),
    "sigmoid": (torch.sigmoid, float_dtypes_without_fp16),
    "sin": (torch.sin, float_dtypes_without_fp16),
    "sqrt": (torch.sqrt, float_dtypes_without_fp16),
    "sqrt_rn": (torch.sqrt, [torch.float32]),
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
    ref_func, allowed_dtypes = OP_CONFIGS[op_name]
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
    unary_kernel[grid](
        x_ptr=x_torch,
        z_ptr=z_torch,
        n_elements=n,
        size=block_size,
        op_name=op_name,
        num_warps=num_warps
    )

    tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-5
    torch.testing.assert_close(z_torch, z_ref, rtol=tol, atol=tol)

@triton.jit
def binary_kernel(x_ptr, y_ptr, z_ptr, n_elements, size: tl.constexpr, op_name: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * size + tl.arange(0, size)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Cast to f32 for ops like 'pow' or 'div' if needed
    x_f = x.to(tl.float32)
    y_f = y.to(tl.float32)

    if op_name == "+":
        z = x + y
    elif op_name == "-":
        z = x - y
    elif op_name == "*":
        z = x * y
    elif op_name == "/":
        # Triton promotes 16-bit floating-point / and % to 32-bit because there
        # are no native div or FRem operations on float16. Since we have to
        # convert anyway, we may as well take the accuracy bump.
        if x.dtype == tl.float16 or x.dtype == tl.bfloat16:
            z = x.to(tl.float32) / y.to(tl.float32)
        else:
            z = x / y

    tl.store(z_ptr + offsets, z, mask=mask)

BINARY_OP_CONFIGS = {
    "+": (torch.add, dtypes_with_bfloat16),
    "-": (torch.sub, dtypes_with_bfloat16),
    "*": (torch.mul, dtypes_with_bfloat16),
    # TODO: test floating point numbers
    "/": (torch.div, integral_dtypes),
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
    ref_func, allowed_dtypes = BINARY_OP_CONFIGS[op_name]
    device = 'cuda'

    dtype_x = data.draw(st.sampled_from(allowed_dtypes))
    dtype_y = data.draw(st.sampled_from(allowed_dtypes))

    is_u_x = (dtype_x == torch.uint8)
    is_u_y = (dtype_y == torch.uint8)
    is_int_x = dtype_x in int_dtypes
    is_int_y = dtype_y in int_dtypes

    mixed_signedness = (is_u_x and is_int_y) or (is_int_x and is_u_y)
    should_fail = op_name == "/" and mixed_signedness

    def gen_data(dtype):
        if dtype in float_dtypes_with_bfloat16:
            return torch.randn(n, device=device, dtype=dtype)
        elif dtype == torch.uint8:
            return torch.randint(0, 255, (n,), device=device, dtype=dtype)
        else:
            return torch.randint(-100, 100, (n,), device=device, dtype=dtype)

    x_torch = gen_data(dtype_x)
    y_torch = gen_data(dtype_y)

    z_ref = ref_func(x_torch, y_torch)
    z_torch = torch.empty_like(z_ref)

    grid = (triton.cdiv(n, block_size),)

    if should_fail:
       with pytest.raises(triton.TritonError, match="signedness"):
            # we don't care about the output since it failed
            z_placeholder = torch.empty_like(x_torch)
            binary_kernel[grid](
                x_ptr=x_torch, y_ptr=y_torch, z_ptr=z_placeholder,
                n_elements=n, size=block_size, op_name=op_name, num_warps=num_warps
            )
    else:
        binary_kernel[grid](
            x_ptr=x_torch, y_ptr=y_torch, z_ptr=z_torch,
            n_elements=n, size=block_size, op_name=op_name, num_warps=num_warps
        )

        out_dtype = z_ref.dtype
        tol = 1e-2 if out_dtype in [torch.float16, torch.bfloat16] else 1e-5
        torch.testing.assert_close(
            z_torch,
            z_ref,
            rtol=tol,
            atol=tol,
            # Treats NaN == NaN as True. We're testing the output of Triton and PyTorch
            # Not the semantics of NaN == NaN.
            equal_nan=True
        )

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton version: {triton.__version__}")