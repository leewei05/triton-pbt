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

def print_kernel_stats(kernel):
    """
    Extracts and formats metadata from the Triton GPU compiler backend.
    Uses n_regs and n_spills directly from the CompiledKernel instance.
    """
    print("\n" + "="*95)
    print(f"{'PROFILING TRITON KERNEL: ' + kernel.__name__:^95}")
    print("="*95)
    print(f"{'Device':<10} | {'Regs':<5} | {'Spills':<10} | {'SRAM (KB)':<10} | {'Metadata'}")
    print("-" * 95)

    for device, cache_info in kernel.device_caches.items():
        kernel_cache = cache_info[0]

        for key, compiled_bin in kernel_cache.items():
            # 1. Trigger the lazy initialization to populate n_regs and n_spills
            compiled_bin._init_handles()

            # 2. Access the stats directly from the object (not metadata)
            regs = compiled_bin.n_regs
            spills = compiled_bin.n_spills

            # 3. Access hardware/scheduling info from the metadata namedtuple
            shared_bytes = compiled_bin.metadata.shared
            warps = compiled_bin.metadata.num_warps
            # num_stages is a constexpr, so it's in the metadata
            stages = getattr(compiled_bin.metadata, 'num_stages', 1)

            # Formatting
            sram_kb = shared_bytes / 1024
            spill_str = f"{spills} B" if spills > 0 else "None"
            reg_warn = "⚠️" if regs > 224 else " "
            spill_warn = "🔥" if spills > 0 else " "

            metadata_str = f"Warps: {warps}, Stages: {stages}"
            print(f"CUDA:{device:<5} | {regs:>4}{reg_warn} | {spill_str:>10}{spill_warn} | {sram_kb:>9.2f} | {metadata_str}")

    print("="*95 + "\n")