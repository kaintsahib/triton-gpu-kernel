


# Triton GPU Kernel Labs & Hardware Profiling

This repository tracks my hands-on implementation of low-level, hardware-aware GPU kernels using the OpenAI Triton language compiler ecosystem.

## 🖥️ Hardware & Systems Environment
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **Host OS**: Ubuntu (WSL2 Architecture)
- **Compiler Layer**: CUDA Toolkit 12.0 (`nvcc`) / Nvidia Graphic Driver (CUDA 13.2 compatible)
- **Environment Management**: Poetry (`py3.12`)

## 📊 Phase 1 Labs Accomplished

### Lab 1: Vector Addition Memory Squeeze
- Developed a high-performance continuous memory elementwise vector addition kernel using OpenAI Triton.
- Bypassed standard Python interpretation loops to interface directly with GPU streaming multiprocessors (SMs).
- **Result**: Achieved maximum hardware memory bandwidth saturation at **146.1 GB/s** on the GTX 1650 physical silicon layer, matching PyTorch's native C++ framework performance.
<img width="993" height="645" alt="Screenshot from 2026-07-27 13-03-16" src="https://github.com/user-attachments/assets/4c952620-7afc-4965-8c5a-9cbbea394019" />

### Lab 2: Nsight Compute Hardware Profiling (ncu)
- Conducted a full 31-pass instrumentation profiling using NVIDIA Nsight Compute CLI (`ncu`).
- Tracked deep hardware execution counters, memory roofline limits, and register configurations directly from the physical GPU.
- Generated and uploaded binary profiling report (`vector_add_profile.ncu-rep`).
- <img width="1919" height="971" alt="Screenshot from 2026-07-27 14-04-13" src="https://github.com/user-attachments/assets/79a36799-ebfa-4b7c-bfac-e8af072fe975" />

### Lab 3: Nsight Systems Telemetry Tracing (nsys)
- Captured live application behaviors, thread execution, and framework latencies using NVIDIA Nsight Systems (`nsys`).
- Uploaded 10 distinct runtime metric capture logs (`report1.qdstrm` to `report10.qdstrm`) tracking timeline events and context switches.

## 📊 Phase 2: Advanced Block-Tiled GEMM & Hardware Autotuning

### Lab 4: Production-Grade 2D Block-Tiled GEMM
* **Architecture:** Implemented an optimized 2D matrix multiplication ($C = A \times B$) kernel using Triton's modern **Block Pointer API** (`tl.make_block_ptr` and `tl.advance`). This bypasses legacy manual 2D index arithmetic, enabling cleaner hardware-assisted asynchronous memory transfers.
<img width="839" height="528" alt="Screenshot from 2026-07-28 09-34-58" src="https://github.com/user-attachments/assets/3499a502-6ecf-4349-b816-122aa826af5d" />

  
* **Autotuning Engine:** Integrated `@triton.autotune` to dynamically sweep across execution space configurations (`BLOCK_SIZE_M/N/K`, `num_warps`, and `NUM_STAGES`) at runtime to extract peak hardware efficiency.
* **Result:** Achieved a massive **1.40x Relative Speedup** over PyTorch's native highly-optimized cuBLAS baseline on $2048 \times 2048$ matrices!
  * **PyTorch cuBLAS Baseline:** 1005.8693 ms
  * **Custom Triton GEMM Kernel:** 719.6263 ms

### Lab 5: Deep-Dive Memory Roofline Profiling (Nsight Compute)
Conducted a deep multi-pass hardware instrumentation suite using **NVIDIA Nsight Compute GUI** to isolate performance bottlenecks on consumer silicon. Generated and uploaded the full `2Dtiled_new.ncu-rep` profiling report.
<img width="1916" height="972" alt="Screenshot from 2026-07-28 09-28-29" src="https://github.com/user-attachments/assets/080737a1-6900-4b62-9675-9892a6639fea" />


* **Architectural Diagnosis (Memory Bound Regime):**
  * **Memory Throughput (Speed of Light):** **76.17%** — Successfully saturated the physical DRAM bandwidth limit of the GTX 1650.
  * **Compute (SM) Throughput:** **10.90%**
  * **Analysis:** The profiler explicitly flags the execution as memory-bound. On bandwidth-constrained consumer hardware, the Tensor Core compute pipelines experience instruction starvation while waiting for global tile offsets to be preloaded into GPU SRAM (Shared Memory).
  * **L2 Cache Efficiency:** Hit **45.36% L2 Cache Throughput**, identifying a concrete vector for future cache-line swizzling optimizations.


