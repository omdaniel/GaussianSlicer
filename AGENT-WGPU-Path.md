# AGENT‑WGPU‑Path.md: Gaussian Slicer Port (Rust/wgpu/egui) — **Consolidated Plan (No RT)**

> **Constraint:** This plan is a straight, feature‑parity port of the Swift/Metal app to Rust + wgpu + egui. **No hardware‑accelerated ray tracing and no ray queries** anywhere in the architecture, milestones, or code paths.

---

## Progress Snapshot (In-Repo Status)

- **Workspace & Core (M0–M2):** Cargo workspace, core configs, and WGSL shader translations are live with Naga-based validation in `cargo test`.
- **GPU Runtime (M3):** `slicer_gfx` provisions K1 → K3 compute pipelines plus the fullscreen visualization pass, managing buffers/bind groups for the translated kernels.
- **UI Shell (M4 kick-off):** `slicer_app` now runs a winit + egui loop, dispatches the compute passes every frame, and displays the density texture with interactive visualization controls.
- **Red/Green Loop:** `tools/red_green_cycle.sh` automates `cargo check`, `cargo test`, and a short `slicer_app` smoke-run (`--exit-after-ms`) so agents can iterate quickly and capture runtime validation errors.
- **Outstanding Milestones:** Export/readback flows, full UI parity, and CI packaging (M4–M6) remain open; next focus is adding staging-buffer exports and broader settings coverage.

## 0) TL;DR
- Port to **Rust + wgpu + egui** with the same features and visual output as Swift/Metal.
- Structure as a **Cargo workspace**: `slicer_core`, `slicer_gfx`, `slicer_app`, `slicer_shaders`.
- Keep compute split as **K1 (Precalc) → K2 (Update) → K3 (Evaluation)**; visualize a single `r32float` density texture.
- Use **WGSL** for all shaders; mirror host structs (`#[repr(C)]`, `bytemuck::{Pod, Zeroable}`).
- Rigorous **validation**: golden images, numeric checksums, cross‑backend sanity; **no RT / ray queries**.
- CI builds + headless screenshot diffs across macOS/Metal, Linux/Vulkan, Windows/D3D12.

---

## 1) Scope, Constraints, Ethos
- **Goal:** Feature‑parity port of Gaussian Slicer from Swift/Metal to Rust/wgpu/egui.
- **Targets:** Windows (DX12/Vulkan), macOS (Metal), Linux (Vulkan) via wgpu backends.
- **Out of scope:** Any hardware RT (DXR/VKRT/MetalRT) and **ray queries**.
- **Ethos:** Deterministic, reproducible builds; zero‑UB Rust; plain WGSL; agent‑friendly logs and validation.

---

## 2) Technology Stack
- **Language:** Rust (stable)
- **GPU:** `wgpu` (compute + render), **WGSL** shaders
- **UI & Windowing:** `egui` (+ `egui-wgpu`, `egui-winit`) with `winit` event loop
- **Math:** `glam`
- **Layout & Bytes:** `bytemuck` (`Pod`, `Zeroable`) + `#[repr(C)]`
- **RNG & CLI:** `rand`, optional `clap` (to mirror `RuntimeConfig.swift` args)
- **Logging:** `tracing`, `tracing-subscriber`; enable `RUST_LOG=wgpu_core=trace,wgpu_hal=info` during bring‑up

Optional (dev ergonomics): `just` for task runs; Nix dev‑shell; GitHub Actions matrix CI with cache.

---

## 3) Workspace Layout
```
repo/
  Cargo.toml               # workspace
  crates/
    slicer_app/            # winit loop + egui + app state wiring
    slicer_gfx/            # wgpu device/surface, pipelines, passes
    slicer_core/           # domain logic (no GPU/UI), config + GMM
    slicer_shaders/        # WGSL sources + build helpers (optional)
  assets/                  # textures, LUTs, demo configs
  tests/golden_scenes/     # scenes + reference PNGs
  ci/workflow.yml          # headless build + image diff
```
**Feature flags (suggested):** `export-mhd`, `export-vdb` (gate C++ tool), `dev` (validation, debug labels).

---

## 4) Swift/Metal → Rust/wgpu Mapping
| Swift/Metal | Rust/wgpu/egui | Notes |
|---|---|---|
| `MTLDevice`, `MTLCommandQueue` | `wgpu::Device`, `wgpu::Queue` | init in `slicer_gfx` |
| `MTLCommandBuffer` | (implicit in submit) | `Queue::submit` |
| `MTLCommandEncoder` | `wgpu::CommandEncoder` | record passes |
| `MTLComputePipelineState` | `wgpu::ComputePipeline` | K1/K2/K3 |
| `MTLRenderPipelineState` | `wgpu::RenderPipeline` | fullscreen visualize |
| `MTLBuffer` | `wgpu::Buffer` | bytemuck on host |
| `MTLTexture` | `wgpu::Texture` + `TextureView` | `r32float` density |
| `MTLSamplerState` | `wgpu::Sampler` | LUT/colormap sampling |
| `MTKView`/drawable | `winit` + `wgpu::Surface` | swapchain |
| SwiftUI/AppKit | `egui` | panels, sliders |
| MSL | WGSL | strict parity |
| Argument buffers | `BindGroupLayout` + `BindGroup` | document indices |
| Function constants | (avoid) or push constants (backend‑guarded) | prefer uniforms |

**Clip/NDC:** Normalize Y flip + depth range in `slicer_gfx` so visuals match Metal.

---

## 5) Milestones

### M0 — Bootstrap (Days 0‑2)
- Cargo workspace; window + surface; `egui` wired (or `eframe` to start).
- **Exit:** Clear color and an egui demo panel render.

### M1 — Core Parity (Days 2‑5)
- Port config & settings; define host structs (`#[repr(C)]`, `Pod`, `Zeroable`) using `glam`.
- Port **GMMGenerator** (seeded via `rand`), SPD covariance creation.
- **Exit:** Identical inputs/config state as Metal; deterministic seeds visible in UI/log.

### M2 — Shaders (MSL → WGSL) (Days 4‑10)
- Inventory shaders; translate to WGSL with 1:1 structs/bindings.
- Replace Metal threadgroups with WGSL workgroups (`@workgroup_size`).
- **Exit:** All WGSL validates under `naga`; unit smoke tests pass.

### M3 — Pipelines & Passes (Days 8‑14)
- Create **K1/K2/K3 compute pipelines** and the **fullscreen render** pipeline.
- Define **BindGroupLayouts** and label all resources for GPU debuggers.
- **Exit:** Canonical scene matches Metal visually (first pixels).

### M4 — UI Port (Days 12‑18)
- Rebuild panels (Data, Grid, Visualization) in `egui`.
- **Apply Settings**: structural changes trigger **K1**; per‑frame changes run **K2 → K3**.
- Add **orientation gizmo**, FPS, timings.
- **Exit:** Feature‑complete UI; behaviors mirror Metal.

### M5 — Validation & Fidelity (Days 16‑24)
- **Golden images:** fixed configs → PNGs; thresholded diffs.
- **Numeric checks:** CPU checksums/metrics after K1/K2/K3 (float tolerance).
- **Perf baselines:** Metal/Vulkan/D3D12 with **RT disabled**.
- **Exit:** Parity within tolerances across backends.

### M6 — Packaging & CI (Days 20‑28)
- Headless mode (offscreen target) for CI image tests.
- Release artifacts for macOS/Linux/Windows.
- Structured logs + crash capture.
- **Exit:** Matrix CI green; 1‑click build flows.

---

## 6) Technical Notes

### 6.1 Data Layout & Buffers
- Host structs: `#[repr(C)]` + `bytemuck::{Pod, Zeroable}`; avoid implicit padding surprises.
- WGSL structs mirror host layout. Validate with `naga` warnings and small layout docs.
- Prefer storage buffers for bulk arrays; uniforms for small immutable config.

### 6.2 Kernels & Responsibilities
- **K1 – Precalc (1D):** orientation/covariance prep, SPD checks, inverses/partitions. Run on load or structural changes.
- **K2 – Update (1D):** slice‑offset dependent params (conditional mean/scale). Per‑frame when offset changes.
- **K3 – Evaluation (2D):** sum densities into **`r32float`** texture. Separate read/write passes; explicit barriers via pass boundaries.

### 6.3 Visualization
- Fullscreen quad (vertex+fragment) samples `r32float` density.
- Normalization modes: linear/log; **colormap LUTs** (Viridis/Plasma, etc.).
- Contour edging via derivatives (`dpdx`/`dpdy`/`fwidth`) with a quality toggle.
- Keep color‑space handling explicit (linear vs sRGB) and apply gamma once.

### 6.4 egui Integration
- Run egui first to mutate `AppState`, then encode GPU passes using that state.
- Keep UI separate from GPU code; no `egui` imports in `slicer_gfx`.
- Add **Safe Mode** (fixed timestep, deterministic seeds, single‑threaded) for bisecting.

### 6.5 Coordinates & Limits
- Centralize projection conventions to solve Y‑flip/depth differences.
- Query and respect `DeviceLimits` for workgroup sizes, bindings, and texture formats.

### 6.6 Performance & Diagnostics
- Start workgroup sizes from Metal equivalents (8×8/16×16), then profile per backend.
- Tag resources/passes with debug labels; enable `RUST_LOG` categories for wgpu.
- Prefer buffers over images for large reductions; avoid undefined WGSL ops.

---

## 7) Export & Tooling
- Headless export loop over Z slices: **K2 → K3 → staging readback (`map_async`)**.
- Optional CPU‑side normalization before write.
- Write **RAW (f32)** + **MHD**; record dims, spacing, endianness.
- Optional **OpenVDB** conversion via external `vdb_writer` (`std::process::Command`), gated behind `export-vdb` feature.

---

## 8) Testing & Validation
- **Unit (core):** math, configs, GMM with fixed seeds.
- **Numeric parity:** checksums/metrics after K1/K2/K3 vs Metal (ε tolerance).
- **Golden images:** offscreen render; PNG diffs (optionally SSIM/PSNR).
- **Cross‑backend smoke:** run the same scene on Metal/Vulkan/D3D12.
- Store references and diffs under `tests/golden_scenes/` in CI artifacts.

---

## 9) Agent‑Friendly Iteration
- Single command flows: `cargo run -F dev`, `just run`, `just headless`.
- Deterministic seeds printed each run; all configs serializable (TOML/JSON).
- Failure surfaces are **actionable**: wgpu validation messages + structured logs + labeled resources.
- Keep changes granular: commit per stage (K1/K2/K3/UI) to enable stepwise agent reasoning.

---

## 10) CI & Packaging
- **Matrix:** macOS‑latest (Metal), ubuntu‑latest (Vulkan), windows‑latest (DX12).
- Cache Cargo + shader build outputs; run **headless golden‑image** job.
- Release build artifacts for all platforms; optional notarization/signing steps later.

---

## 11) Success Criteria
1) No `wgpu` validation warnings on all targets.  
2) Visual parity with Swift/Metal under canonical seeds/configs.  
3) Comparable interactive frame times for representative workloads.  
4) RAW/MHD and OpenVDB exports match baseline within numeric tolerances.  
5) RT remains **disabled** and no ray queries are used.

---

## Reference from AGENTS.md
Add a bullet pointing here, for example:

> **Rust + wgpu + egui Port:** See `AGENT‑WGPU‑Path.md` for the consolidated milestone plan and architecture mapping (**no hardware RT or ray queries**).
