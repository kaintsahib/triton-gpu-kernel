cat << 'EOF' > README.md
# Triton GPU Kernel Labs & Hardware Profiling

This repository tracks my hands-on implementation of low-level, hardware-aware GPU kernels using the OpenAI Triton language compiler ecosystem.

##  Hardware & Systems Environment
- **GPU**: NVIDIA GeForce GTX 1650 (4GB VRAM)
- **Host OS**: Ubuntu (WSL2 Architecture)
- **Compiler Layer**: CUDA Toolkit 12.0 (`nvcc`) / Nvidia Graphic Driver (CUDA 13.2 compatible)
- **Environment Management**: Poetry (`py3.12`)

##  Phase 1 Labs Accomplished

### Lab 1: Vector Addition Memory Squeeze
- Developed a high-performance continuous memory elementwise vector addition kernel using OpenAI Triton.
- Bypassed standard Python interpretation loops to interface directly with GPU streaming multiprocessors (SMs).
- **Result**: Achieved maximum hardware memory bandwidth saturation at **146.1 GB/s** on the GTX 1650 physical silicon layer, matching PyTorch's native C++ framework performance.

### Lab 2: Low-Level Profiling (Nsight Compute Interception)
- Conducted a full 31-pass instrumentation analysis of the custom kernel.
- Captured deep hardware counters, memory pipelines, and execution grids using NVIDIA Nsight Compute CLI (`ncu`).
- Tracked hardware metrics and compiled binary reports (`vector_add_profile.ncu-rep`).

### Lab 3: Tiled GEMM (Matrix Multiplication)
- Implemented a custom 2D block-tiled matrix multiplication kernel using strict 32x32 hardware tiles to leverage GPU Shared Memory (SRAM) and prevent VRAM bandwidth degradation.
EOF
