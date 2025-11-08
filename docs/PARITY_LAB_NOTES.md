# Parity Lab Context Notes

These notes condense how the current Swift/Metal build and the Rust/wgpu port are structured so the new `parity_lab/` workspace can mirror the same math kernels.

## Swift / Metal reference

- `Sources/GaussianSlicer/MetalRenderer.swift` drives the three compute kernels (K1 precalc, K2 update, K3 evaluation) and the fullscreen visualization pass. It owns the Gaussian buffers, sets the uniforms (`Config`, `VisualizationConfig`), and schedules the dispatch order that we need to reproduce.
- `Sources/GaussianSlicer/Shaders.metal` contains the exact Metal implementations of K1–K3 plus the fragment shader for rendering the `r32float` density texture. The kernels operate on `Gaussian3D`, `PrecalculatedParams`, and `DynamicParams`, matching the Swift structs sent over `MTLBuffer`.
- SwiftUI scaffolding (`ContentView.swift`, `GaussianSlicerApp.swift`) configures renderer settings, exposes the runtime controls, and routes CLI overrides (via `RuntimeConfig`), but the important parity hooks live inside the renderer and shader files mentioned above.

## Rust / WGSL port

- `crates/slicer_core/` mirrors the Swift data model: `config.rs` builds the uniform payload, `gmm.rs` generates deterministic Gaussian mixtures, and `gpu.rs` defines the POD structs shared with WGSL.
- `crates/slicer_shaders/` stores the WGSL translations of the three kernels and the visualization shader. Each kernel lives in `src/kernels/*.wgsl`, and the crate already validates them with `naga` in unit tests.
- `crates/slicer_gfx/` owns the `wgpu` device setup, buffer allocation, pipeline creation, and bind-group layout. The `RendererResources` type exposes `run_precalc()` (K1) and `run_update_and_evaluate()` (K2/K3), mirroring the Metal dispatch order.
- `crates/slicer_app/` wires everything into a winit + egui loop, exposes the CLI, and contains the debug dump utilities that we can reuse inside the parity lab runner.

## Implications for the parity lab

1. **Math sources of truth:** The Metal kernels are already trusted; their WGSL twins live under `crates/slicer_shaders`. The parity lab should link directly against both copies to prevent silent drift.
2. **Host structs:** `slicer_core::gpu::*` and `slicer_core::config::KernelConfig` already encode the layouts needed for GPU dispatch. The parity lab CPU references can reuse these types so that no translation layer diverges.
3. **Reference inputs:** Both the Swift and Rust apps load Gaussian PLY files, so the parity tests can ingest the same datasets (or JSON snapshots) and then feed them into the CPU + GPU harnesses.
4. **Kernel ordering:** For deterministic comparisons we must execute the kernels in the exact K1 → K2 → K3 order with identical loop structures (no tiling or parallel reductions yet). The existing renderer implementations show how to wire that up for both Metal and `wgpu`.

These notes will evolve as the dedicated `parity_lab/` workspace takes shape, but they capture the shared architecture we must preserve while building the scalar/vector/matrix/SPD parity tests from `AGENT-PARITYLAB.md`.

## Current parity runners

- `cargo run -p parity_lab --bin wgpu_scalars -- --count=128` now generates **labeled case sets** for scalar/vector, mat3 operations, and SPD/Cholesky solves. It dispatches the corresponding WGSL kernels and compares against CPU references (fail-fast if any residual exceeds `1e-6`). Pass `--export-json=cases.json` to capture the exact inputs, `--cases-json=cases.json` to replay a saved set, or `--skip-dispatch` if you only need to generate/export data without touching the GPU (useful on machines without Metal/Vulkan access).
- `cd parity_lab/runner/metal && SWIFT_MODULE_CACHE_PATH=.swift-module-cache CLANG_MODULE_CACHE_PATH=.clang-module-cache swift run parity-lab-metal --cases=cases.json` loads the same JSON, runs the scalar/vector, mat3, and SPD Metal kernels, and prints per-set residuals. Add `--print-layout` to simply dump struct sizes/alignments. The explicit module cache paths avoid sandboxed filesystem restrictions.
- For a one-command end-to-end check use `scripts/run_parity.sh` (honors `COUNT`, `SEED`, and `EPSILON` env vars). It exports the JSON via the wgpu runner (`--skip-dispatch`) and immediately feeds it to the Metal runner.

## Recent parity-lab deltas

- **Scalar/vector parity tightened:** CPU references, WGSL, and Metal all now use fused multiply-adds for the axpy path, eliminating the ~2e-6 drift we saw when large case batches mixed very small and very large magnitudes.
- **SPD validation hardened:** The Rust generator drops matrices whose CPU solutions exceed a residual of `5e-7`, and both runners measure `‖A·x - b‖∞` plus Cholesky deltas instead of comparing solutions directly. This keeps tolerances meaningful even when different platforms pick slightly different `L` factorizations.
- **Metal runner resilience:** When SwiftPM cannot embed a precompiled metallib (e.g., sandboxed builds), the runner falls back to compiling `Shaders.metal` straight from the bundle source and allocates buffers using `withUnsafeBytes` to avoid ARC warnings.
- **Automation:** `scripts/run_parity.sh` now uses the updated CLI signature (`swift run parity-lab-metal --cases=…`) and redirects module caches locally, so `COUNT/SEED/EPSILON` sweeps can run in a single command.
