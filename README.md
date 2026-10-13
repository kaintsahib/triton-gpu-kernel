# Triton GPU Kernel Labs & Hardware Profiling

This repository tracks hands-on implementation of low-level, hardware-aware GPU kernels using the OpenAI Triton language compiler ecosystem on an NVIDIA GeForce GTX 1650 (WSL2 Ubuntu).

## Phase 1 & Phase 2 Overview
- **Phase 1 Labs:** Covered Vector Addition Memory Squeeze reaching **146.1 GB/s** bandwidth saturation, alongside full 31-pass Nsight Compute (ncu) and Nsight Systems (nsys) hardware telemetry tracing.
- **Phase 2 Labs:** Features an optimized 2D Block-Tiled GEMM utilizing Triton's Block Pointer API and `@triton.autotune`.
- **Updated Metrics:** Achieved a remarkable **1.73x Relative Speedup** over PyTorch's cuBLAS baseline and **96.54% DRAM Memory Throughput** (Speed of Light) saturation.

