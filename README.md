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

### Lab 2: Nsight Compute Hardware Profiling (ncu)
- Conducted a full 31-pass instrumentation profiling using NVIDIA Nsight Compute CLI (`ncu`).
- Tracked deep hardware execution counters, memory roofline limits, and register configurations directly from the physical GPU.
- Generated and uploaded binary profiling report (`vector_add_profile.ncu-rep`).

### Lab 3: Nsight Systems Telemetry Tracing (nsys)
- Captured live application behaviors, thread execution, and framework latencies using NVIDIA Nsight Systems (`nsys`).
- Uploaded 10 distinct runtime metric capture logs (`report1.qdstrm` to `report10.qdstrm`) tracking timeline events and context switches.

### Lab 4: Tiled GEMM (Matrix Multiplication)
- Implemented a custom 2D block-tiled matrix multiplication kernel using strict 32x32 hardware tiles to leverage GPU Shared Memory (SRAM) and prevent VRAM bandwidth degradation.
