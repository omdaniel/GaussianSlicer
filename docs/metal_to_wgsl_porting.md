# Metal → WGSL Porting Guide

A short checklist distilled from the Swift/Metal → Rust/wgpu/WGSL migration for Gaussian Slicer.

## 1. Align the Data Model First
- **Mirror host structs verbatim.** Give every struct a single source of truth (Rust `#[repr(C)]` + WGSL `struct`). Pad `vec3`/`float3` members to 16 B and split `mat3`/`float3x3` into three `vec4` columns so std430 and Metal agree.
- **Validate at build time.** Run `cargo test -p slicer_shaders layout_snapshot` (naga layouter) plus any Swift-side `MemoryLayout` asserts to catch regressions before they hit the GPU.
- **Normalize strides between languages.** When exporting buffers (RAW/JSON), stick to little-endian `f32` blocks so Python scripts and Swift/Rust agree on record lengths.

## 2. Translate Kernels Incrementally (K1/K2/K3)
- **Keep pipeline boundaries identical.** Each Metal `MTLComputePipelineState` maps cleanly to a `wgpu::ComputePipeline`. Dispatch counts stay the same (64-wide for K1/K2, 8×8 tiles for K3) so validation scripts can reuse references.
- **Instrument every stage.** Add WGSL storage buffers (`CovarianceDebug`, dynamic/density dumps) and CLI flags (`--dump-…-raw`) so you can diff CPU ↔ GPU at each kernel: K1 matrices, K2 dynamic params, K3 density textures.
- **Prefer explicit matrix math.** Instead of relying on implicit transposes, call out `world_to_slice`/`slice_to_world` multiplies exactly as Metal does; we fixed the major parity bug by matching the multiply order and column packing.

## 3. Tooling & Validation Loop
- **Python or bust.** Scripts such as `scripts/compare_cov_debug.py`, `compare_dynamic.py`, `compare_density.py`, and `compare_volume_exports.py` rebuild the CPU reference math and compare against GPU dumps (full grid or sampled). Keep tolerances tight (≤1e-5) to surface issues quickly.
- **Automate the happy path.** `Tools/red_green_cycle.sh` now dumps the triplet scene and runs the compare scripts; `Tools/run_parity_suite.sh` adds 50 k procedural scenes (256²/512²) with deterministic sampling. Make these gated checks in CI.
- **Capture end-to-end exports.** Both Swift and Rust apps expose `--export-volume` (plus `--export-log-normalized`). Headless runs sweep all slices, produce RAW/MHD stacks, and feed into `scripts/compare_volume_exports.py` for final parity sign-off.

## 4. Debugging WGSL vs. MSL
- **Use wgpu + naga diagnostics.** `wgpu` validation + `naga` parser errors highlight binding/typing issues; the custom layout test ensures WGSL structs respect std430 alignment.
- **Read back everything.** Temporary staging buffers (density, dynamic, kernel config) and Swift dumps make it easy to stare at raw floats when shader output looks wrong.
- **Compare intermediate math, not just frames.** We caught the worst bugs (anisotropic covariances collapsing, zeroed dynamic factors) by diffing intermediate matrices and per-Gaussian scalars, not just the final texture.

## 5. Operational Tips
- **Shared CLI surface.** Match the host flags (`--num-distributions`, `--grid-resolution`, `--seed`, `--gaussian-ply`, `--export-volume`, etc.) so scenarios can be replayed verbatim across Swift/Rust.
- **Deterministic datasets.** Keep tiny fixtures (triplet PLY) for fast loops and large procedural cases (50 k gaussians) for stress tests. Deterministic seeds make regressions reproducible.
- **Quick visual parity.** Optional PNG captures (`--capture-frame` on Swift, `--capture-frame` flag on Rust) help confirm GUI parity once numeric checks are green.

Following this playbook keeps the port grounded: align structs, instrument kernels, validate numerically, then gate everything through automation before touching UI polish.
