import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt  # ग्राफ बनाने के लिए इम्पोर्ट किया

# =====================================================================
# 1. HARDWARE AUTOTUNING CONFIGURATION (The Professional Touch)
# =====================================================================
# This decorator tests different block sizes and warps at run-time.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_STAGES': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_STAGES': 3}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_block_ptr_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Production-Grade 2D Tiled GEMM using Modern Block Pointers.
    Optimized for Hardware-Assisted Asynchronous Memory Transfers.
    """
    # Identify this specific thread block within the Grid.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # -----------------------------------------------------------------
    # MODERN BLOCK POINTERS SETUP (No manual 2D arithmetic & masking)
    # -----------------------------------------------------------------
    # Block pointer for Matrix A: Row-Major (order=(1, 0))
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0)
    )

    # Block pointer for Matrix B: Column-Major Aligned (order=(1, 0))
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0)
    )

    # Initialize the accumulator register matrix in FP32 precision.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # -----------------------------------------------------------------
    # THE INNER K-REDUCTION LOOP
    # -----------------------------------------------------------------
    # Loop to cover the entire K-dimension.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # HBM से सीधे GPU SRAM (Shared Memory) में टाइल लोड करें
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))

        # Perform matrix multiplication using Tensor Cores.
        accumulator = tl.dot(a_tile, b_tile, accumulator)

        # Advance the pointers in the horizontal (A) and vertical (B) directions (K-step jump).
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # -----------------------------------------------------------------
    # EPILOGUE: WRITE BACK TO HBM
    # -----------------------------------------------------------------
    # Cast the output precision from FP32 to FP16 (Matrix C layout).
    c = accumulator.to(tl.float16)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# =====================================================================
# 2. CPU LAUNCHER & PYTORCH INTERFACE
# =====================================================================
def triton_gemm(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous(), "Matrices must be contiguous in memory!"
    assert a.is_cuda and b.is_cuda, "Tensors must be allocated on Nvidia GPU VRAM!"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )

    # Launch the Kernel
    gemm_block_ptr_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c


# =====================================================================
# 3. TRITON PERF REPORT ENGINE FOR GRAPH GENERATION
# =====================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['S'],  # Square matrix size (S x S)
        x_vals=[128, 256, 512, 1024, 2048, 4096],  # विभिन्न मैट्रिक्स साइज
        line_arg='provider',
        line_vals=['triton', 'pytorch'],
        line_names=['Custom Triton GEMM', 'PyTorch cuBLAS'],
        styles=[('blue', '-'), ('orange', '--')],
        ylabel='TFLOPS',  # GPU की कंप्यूट क्षमता मापने का पैमाना
        plot_name='Triton vs PyTorch: 2D Block-Tiled GEMM Performance',
        args={},
    )
)
def benchmark(S, provider):
    a = torch.randn((S, S), device='cuda', dtype=torch.float16)
    b = torch.randn((S, S), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_gemm(a, b), quantiles=quantiles)
    
    # GEMM TFLOPS गणना: (2 * M * N * K) / (time_in_ms * 1e9)
    tflops = lambda ms: 2 * (S ** 3) / (ms * 1e9)
    return tflops(ms)


# =====================================================================
# 4. EXECUTION AND VERIFICATION BLOCK
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print(" Allocating 2048 x 2048 Tensor Structures on Silicon...")
    a = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
    b = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)

    print(" Executing Custom Triton Block-Pointer Kernel...")
    triton_output = triton_gemm(a, b)

    print("⚡ Executing Industry Golden Baseline (PyTorch cuBLAS)...")
    pytorch_output = torch.matmul(a, b)

    if torch.allclose(triton_output, pytorch_output, atol=1e-2, rtol=1e-2):
        print("\n MATCH VALIDATED: Your Triton custom layout perfectly matches native silicon results!")
        triton_ms = triton.testing.do_bench(lambda: triton_gemm(a, b))
        pytorch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

        print(f"\n --- HARDWARE PERFORMANCE DATA ---")
        print(f" PyTorch cuBLAS Time : {pytorch_ms:.4f} ms")
        print(f" Your Triton GEMM Time: {triton_ms:.4f} ms")
        print(f" Relative Speedup : {pytorch_ms / triton_ms:.2f}x")
        
        # -------------------------------------------------------------
        
        # -------------------------------------------------------------
        print("\n Generating Triton vs PyTorch Performance Graph (128 to 4096 scale)...")
        benchmark.run(show_plots=False, save_path='.')  # वर्तमान फोल्डर में सेव करेगा
        print(" Success! Graph saved as 'Triton vs PyTorch: 2D Block-Tiled GEMM Performance.png'")
        
    else:
        print("\n MATHEMATICAL MISMATCH: Check layout alignment or accumulator casts.")
