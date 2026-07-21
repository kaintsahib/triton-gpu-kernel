import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt


# --- 1. CORE TRITON KERNEL ---
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# --- 2. BENCHMARK & PLOT ENGINE ---
if __name__ == '__main__':
    # 2^12 (4096) se lekar 2^24 (~16 Million) tak ke sizes test karenge
    sizes = [2**i for i in range(12, 25)]
    triton_gbps = []
    pytorch_gbps = []

    print("🚀 Benchmarking chal raha hai, thoda sabar rakho...")

    for size in sizes:
        x = torch.rand(size, device='cuda', dtype=torch.float32)
        y = torch.rand(size, device='cuda', dtype=torch.float32)

        # FIXED: do_bench ab single float value return karega bina unpack error ke
        ms_py = triton.testing.do_bench(lambda: x + y)
        ms_tri = triton.testing.do_bench(lambda: triton_add(x, y))

        # Formula: (3 * size * 4 bytes) / (time_in_ms * 1e6) -> GB/s
        py_speed = (3 * size * 4) / (ms_py * 1e6)
        tri_speed = (3 * size * 4) / (ms_tri * 1e6)

        pytorch_gbps.append(py_speed)
        triton_gbps.append(tri_speed)
        print(f"Size: {size:10d} | PyTorch: {py_speed:6.1f} GB/s | Triton: {tri_speed:6.1f} GB/s")

    # --- 3. PLOTTING LAYER ---
    plt.ioff()  # Turn off interactive mode to ensure safe rendering
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, triton_gbps, marker='o', color='#1f77b4', linewidth=2, label='Custom Triton Kernel')
    ax.plot(sizes, pytorch_gbps, marker='s', color='#ff7f0e', linestyle='--', linewidth=2, label='PyTorch Native')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Vector Size (Log Scale base 2)', fontsize=12)
    ax.set_ylabel('Memory Bandwidth Speed (GB/s)', fontsize=12)
    ax.set_title('Triton vs PyTorch: Hardware Memory Layer Benchmark', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="-", color='0.85')
    ax.legend(fontsize=11)

    # 4. SAVE BACKUP AND SHOW
    # WSL2 GUI conflicts se bachne ke liye image hamesha folder mein save ho jayegi
    plt.savefig('hardware_benchmark.png')
    print("\n✅ GRAPH SAVED: Apne folder mein 'hardware_benchmark.png' dekho!")
    
    try:
        print("🖥️ Screen par popup window display karne ki koshish ho rahi hai...")
        plt.show()
    except Exception as e:
        print("ℹ️ Note: WSL2 environment ke karan directly popup window block hui, par png file ready hai.")
