# Parity Lab Runners

This directory will house two tiny harnesses:

1. `wgpu/` — a Rust binary that uploads the shared case buffers, dispatches the WGSL versions of each kernel, and copies results back through map/readback for the CPU assertions.
2. `metal/` — a SwiftPM target (reusing `MetalRenderer` plumbing) that loads the same case data and runs the MSL kernels outside of the main Gaussian Slicer UI.

The runners intentionally disable any optimizations that might reorder floating-point operations. Their only job is to report residuals and dump JSON snapshots for the CPU reference checks in `parity_lab/tests/`.
