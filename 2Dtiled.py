import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt 

# =====================================================================
# 1. BASELINE HARDWARE AUTOTUNING CONFIGURATION
# =====================================================================
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
    [YOUR ORIGINAL KERNEL - FULLY PRESERVED]
    Production-Grade 2D Tiled GEMM using Modern Block Pointers.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(tl.float16)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# =====================================================================
# 2. END-LEVEL SWIZZLED KERNEL FOR L2 CACHE OPTIMIZATION
# =====================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'NUM_STAGES': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'NUM_STAGES': 3}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'NUM_STAGES': 3}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_swizzled_block_ptr_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Advanced 2D Block-Tiled GEMM with Macro-Tile Swizzling Engine.
    Saturates L2 Cache lines to prevent compute pipeline starvation.
    """
    # 1D Program ID Mapping to 2D Swizzled Grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Modern Block Pointers
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = accumulator.to(tl.float16)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# =====================================================================
# 3. CPU LAUNCHERS & INTERFACES
# =====================================================================
def triton_gemm(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    gemm_block_ptr_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)
    )
    return c

def triton_gemm_swizzled(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Swizzled कर्नल के लिए 1D ग्रिड लॉन्च
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    gemm_swizzled_block_ptr_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1)
    )
    return c


# =====================================================================
# 4. TRITON PERF REPORT ENGINE FOR GRAPH GENERATION
# =====================================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['S'],  
        x_vals=[128, 256, 512, 1024, 2048, 4096],  
        line_arg='provider',
        line_vals=['triton_baseline', 'triton_swizzled', 'pytorch'],
        line_names=['Custom Triton GEMM (Baseline)', 'Swizzled Triton GEMM (L2 Opt)', 'PyTorch cuBLAS'],
        styles=[('blue', '-'), ('green', '-.'), ('orange', '--')],
        ylabel='TFLOPS',  
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
    if provider == 'triton_baseline':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_gemm(a, b), quantiles=quantiles)
    if provider == 'triton_swizzled':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_gemm_swizzled(a, b), quantiles=quantiles)
    
    tflops = lambda ms: 2 * (S ** 3) / (ms * 1e9)
    return tflops(ms)


# =====================================================================
# 5. EXECUTION AND VERIFICATION BLOCK
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    print(" Allocating 2048 x 2048 Tensor Structures on Silicon...")
    a = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
    b = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)

    print(" Executing Custom Triton Baseline Kernel...")
    baseline_output = triton_gemm(a, b)

    print(" Executing Optimized Swizzled Triton Kernel...")
    swizzled_output = triton_gemm_swizzled(a, b)

    print("⚡ Executing Industry Golden Baseline (PyTorch cuBLAS)...")
    pytorch_output = torch.matmul(a, b)

    # दोनों कर्नल्स की शुद्धता की जांच
    baseline_match = torch.allclose(baseline_output, pytorch_output, atol=1e-2, rtol=1e-2)
    swizzled_match = torch.allclose(swizzled_output, pytorch_output, atol=1e-2, rtol=1e-2)

    if baseline_match and swizzled_match:
        print("\n MATCH VALIDATED: All Triton custom layouts perfectly match native silicon results!")
        
        pytorch_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
        baseline_ms = triton.testing.do_bench(lambda: triton_gemm(a, b))
        swizzled_ms = triton.testing.do_bench(lambda: triton_gemm_swizzled(a, b))

        print(f"\n --- HARDWARE PERFORMANCE DATA (2048 x 2048) ---")
        print(f" PyTorch cuBLAS Time      : {pytorch_ms:.4f} ms")
        print(f" Triton Baseline GEMM Time: {baseline_ms:.4f} ms")
        print(f" Triton Swizzled GEMM Time: {swizzled_ms:.4f} ms")
        print(f"------------------------------------------------")
        print(f" Baseline Speedup Over cuBLAS: {pytorch_ms / baseline_ms:.2f}x")
        print(f" Swizzled Speedup Over cuBLAS: {pytorch_ms / swizzled_ms:.2f}x")
        
        print("\n Generating 3-Line Performance Graph (128 to 4096 scale)...")
