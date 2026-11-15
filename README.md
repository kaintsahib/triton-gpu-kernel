# Triton GPU Kernel Labs & Hardware Profiling

This repository tracks my hands-on implementation of low-level, hardware-aware GPU kernels using the OpenAI Triton language compiler ecosystem.

## Hardware & Systems Environment
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM, Turing TU117 Architecture)
- **Host OS**: Ubuntu (WSL2 Architecture)
- **Compiler Layer**: CUDA Toolkit 12.0 (`nvcc`) / NVIDIA Graphics Driver (CUDA 13.2 compatible)
- **Environment Management**: Poetry (`py3.14`)

---

## Phase 1 Labs Accomplished

### Lab 1: Vector Addition Memory Squeeze
- Developed a high-performance continuous memory elementwise vector addition kernel using OpenAI Triton.
- Bypassed standard Python interpretation loops to interface directly with GPU streaming multiprocessors (SMs).
- **Result**: Achieved maximum hardware memory bandwidth saturation at **146.1 GB/s** on the GTX 1650 physical silicon layer, matching PyTorch's native C++ framework performance.

<img width="993" height="645" alt="Screenshot from 2026-07-27 13-03-16" src="https://github.com/user-attachments/assets/4c952620-7afc-4965-8c5a-9cbbea394019" />

### Lab 2: Nsight Compute Hardware Profiling (`ncu`)
- Conducted a full 31-pass instrumentation profiling using NVIDIA Nsight Compute CLI (`ncu`).
- Tracked deep hardware execution counters, memory roofline limits, and register configurations directly from the physical GPU.
- Generated and uploaded binary profiling report (`vector_add_profile.ncu-rep`).

<img width="1919" height="971" alt="Screenshot from 2026-07-27 14-04-13" src="https://github.com/user-attachments/assets/79a36799-ebfa-4b7c-bfac-e8af072fe975" />

### Lab 3: Nsight Systems Telemetry Tracing (`nsys`)
- Captured live application behaviors, thread execution, and framework latencies using NVIDIA Nsight Systems (`nsys`).
- Uploaded 10 distinct runtime metric capture logs (`report1.qdstrm` to `report10.qdstrm`) tracking timeline events and context switches.

---

## 📊 Phase 2: Advanced Block-Tiled GEMM, L2 Swizzling & Autotuning

### Lab 4: Production-Grade 2D Block-Tiled & L2-Swizzled GEMM
- **Architecture:** Implemented an optimized 2D matrix multiplication ($C = A \times B$) kernel using 2D pointer arithmetic (`offs_am`, `offs_bn`, `offs_k`) with boundary masking for clean, hardware-assisted asynchronous memory transfers across Triton 3.7+.
- **L2 Cache Swizzling:** Implemented 1D grid super-grouping (`GROUP_SIZE_M`) to maximize L2 cache locality and reduce redundant global DRAM fetches across thread blocks.
- **Hardware Saturation:** Reached **96.54% DRAM Memory Throughput** (Speed of Light) efficiency.
- **Autotuning Engine:** Integrated `@triton.autotune` to dynamically sweep across execution space configurations (`BLOCK_SIZE_M/N/K`, `num_warps`, and `num_stages`) at runtime to extract peak hardware efficiency.

<img width="640" height="480" alt="Triton vs PyTorch: 2D Block-Tiled GEMM Performance" src="https://github.com/user-attachments/assets/18fce9bf-5d33-4b9b-8c40-eb477691387d" />

- **Benchmark Results (`2048 x 2048` FP16 Matrices via `triton.testing.do_bench`):**
  - **PyTorch cuBLAS Baseline:** `54.3520 ms` (`0.32 TFLOPS`)
  - **Custom Triton Baseline GEMM:** `28.8986 ms` (`0.59 TFLOPS`) — **1.88x Speedup over cuBLAS**
  - **Custom Triton L2-Swizzled GEMM:** `31.7813 ms` (`0.54 TFLOPS`) — **1.71x Speedup over cuBLAS**
  - *Hardware Takeaway:* On consumer Turing silicon (GTX 1650) with a compact 1MB L2 cache, the unswizzled 2D baseline slightly edges out the 1D swizzled grid due to integer division/modulo indexing overhead.

### Lab 5: Deep-Dive Memory Roofline Profiling (Nsight Compute)
Conducted a deep multi-pass hardware instrumentation suite using **NVIDIA Nsight Compute GUI** to isolate performance bottlenecks on consumer silicon. Generated and uploaded the full `2Dtiled_new.ncu-rep` profiling report.

<img width="1916" height="972" alt="Screenshot from 2026-07-28 09-28-29" src="https://github.com/user-attachments/assets/080737a1-6900-4b62-9675-9892a6639fea" />

- **Architectural Diagnosis (Memory-Bound Regime):**
  - **Memory Throughput (Speed of Light):** **76.17%** — Successfully saturated the physical DRAM bandwidth limit of the GTX 1650.
  - **Compute (SM) Throughput:** **10.90%**
  - **Analysis:** The profiler explicitly flags the execution as memory-bound. On bandwidth-constrained consumer hardware, the **FP16/FP32 CUDA Core** compute pipelines experience instruction starvation while waiting for global tile offsets to be preloaded into GPU SRAM (Shared Memory).
  - **L2 Cache Efficiency:** Hit **45.36% L2 Cache Throughput** on the unswizzled baseline, motivating the **L2 Cache Swizzling** optimization implemented in `2Dtiled.py`.
