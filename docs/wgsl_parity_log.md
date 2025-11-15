# WGSL ⇄ Metal Density Parity Log

> Tracking the ongoing effort to make the Rust/wgpu Gaussian slicer produce the same density field as the Swift/Metal implementation. Use this as a hand-off document for future iterations.

---

## Context
- Goal: run the same Gaussian PLY data through both stacks and obtain pixel-parity captures of the slice visualization.
- Strategy: mirror the compute pipeline (precalc → update → evaluate) and verify every intermediate buffer before worrying about fragment output.
- Tooling: the Rust shell exposes `--capture-frame`, `--dump-{precalc,dynamic,density}-raw`, and (newly) `--dump-precalc-debug-raw` to snapshot GPU state for offline inspection.

Example invocation:

```bash
cargo run -p slicer_app -- \
  --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
  --dump-precalc-raw=tmp/precalc.raw \
  --dump-precalc-debug-raw=tmp/precalc_debug.raw \
  --dump-dynamic-raw=tmp/dynamic.raw \
  --dump-density-raw=tmp/density.raw \
  --exit-after-ms=2000
```

All dumps are little-endian `f32` blobs (see `tmp/*.raw`).

---

## Attempt Log

### 1. Buffer Plumbing & CLI Flags (✅)
- Added staging buffers + CLI switches for every interesting GPU buffer.
- `precalc_debug` storage buffer (binding 3) now records the three columns of `cov_prime` (rotation of each Gaussian covariance into slice space) before the Schur complement.
- Confirmed `tmp/precalc_debug.raw` updates each frame and matches buffer sizes (`3 * sizeof(vec4<f32>) * num_gaussians`).

### 2. Rotation Matrix Handling (partial)

| Attempt | Notes | Outcome |
|---------|-------|---------|
| Use host-provided `rotation_matrix` uniform directly in WGSL | Treated uniform columns as slice axes in world space (`slice_to_world = mat3x3(col0,col1,col2)`). | Produced obviously wrong covariances (anisotropic gaussians collapsed). |
| Rebuild basis in WGSL via cross products (`build_slice_basis`) | Mirrors Swift `getRotationMatrix` logic; avoids trusting the uniform. | Still zero σₙₙ for Gaussians 1/2, but easier to reason about. |
| Explicit change-of-basis (compute `cov_prime = Rᵀ Σ R` manually) | Implemented via `covariance * column` and `row ⋅ column` dot-products to catch row/col ordering issues. | Covariance columns still wrong for anisotropic cases (see below). |

### 3. Covariance Debug vs CPU Reference

Using the PLY sample (`assets/examples/gaussian_triplet.ply`):

| Gaussian | CPU `cov_prime` (expected) | WGSL `cov_prime` (dump) | Status |
|----------|---------------------------|-------------------------|--------|
| 0 (isotropic) | Diagonal ≈ 0.25 | Matches (within ~1e-9) | ✅ |
| 1 (anisotropic, diag `[0.09, 0.25, 0.49]`) | `[[0.3339, 0.0710, 0.1818], [0.0710, 0.2495, -0.0013], [0.1818, -0.0013, 0.2466]]` | `[[0.2450, -0.3074, 0.2450], [-0.1768, 0.0, 0.1768], [-0.2450, -0.3074, -0.2450]]` | ❌ third column aligns with in-plane axes instead of normal; σₙₙ collapses |
| 2 (off-diagonal covariance) | Non-zero `cov_prime[2,*]` (see CPU calc) | Same issue as Gaussian 1 (third column zero apart from mirrored XY) | ❌ |

Observations:
- The GPU’s “column 2” is still derived from the in-plane columns, suggesting that we are using row vectors instead of column vectors when projecting onto the normal.
- `sigma_n_n` = `cov_prime[2].z` ends up negative/zero, so the Schur complement short-circuits and the dynamic buffer stays zero → density texture constant.

### 4. Additional Experiments
- Tried transposing the uniform before packing (`mat3_to_padded_columns` storing rows instead of columns). Result: isotropic case still worked, but anisotropic cases unchanged (third column still ~zero). Reverted to original packing and fell back to the WGSL `build_slice_basis`.
- Verified raw Gaussian storage buffer (`tmp/gaussians.raw`) matches Swift’s data: columns are std430-padded (each column is `[x,y,z,0]`). So the error isn’t in uploading covariance data.
- Density range dump (`tmp/density.raw`) remains constant (min=max), confirming the compute path still bails out before writing meaningful values.

### 5. Column Packing Verification (Nov 8, 2025)
- Added `scripts/compare_cov_debug.py` to automate CPU↔WGSL covariance diffs. Example:
  ```bash
  cargo run -p slicer_app -- --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
    --dump-gaussians-raw=tmp/gaussians.raw \
    --dump-precalc-debug-raw=tmp/precalc_debug.raw \
    --dump-kernel-config-raw=tmp/kernel.raw \
    --exit-after-ms=2000

  python3 scripts/compare_cov_debug.py \
    --gaussians tmp/gaussians.raw \
    --precalc-debug tmp/precalc_debug.raw \
    --kernel-config tmp/kernel.raw \
    --tolerance 1e-6
  ```
  Output (current build) reports `max_abs_diff ≈ 5.26e-01` with all three Gaussians failing because `cov_prime[2]` is zeroed on the GPU.
- Host now writes column-aligned matrices via `KernelConfig::rotation_matrix_cols`, and WGSL consumes a matching field.
- Instrumented `precalc_debug` to stash `world_to_slice[col][2]` in the `.w` component of each column. Those values are `0` for every column, proving the **entire third row of W2S is zero** inside the shader even though the CPU rotation matrix has non-zero normals. In other words, the normal axis never survives the uniform upload, so σₙₙ collapses to 0 and K2 short-circuits.
- Suspected cause: the mat4/array layout mismatch when mapping the uniform. Reverting the WGSL field type back to `mat4x4<f32>` did **not** fix the zero row, so the issue likely sits elsewhere (possibly in how we rehydrate the matrix on the WGSL side).

➤ **Next debugging steps**
1. Dump the raw uniform buffer (`tmp/kernel.raw`) and interpret it inside a tiny shader (copy the three columns into `precalc_debug`) to confirm what WGSL actually sees. Right now the evidence suggests column 2 arrives as zero even though the CPU dump shows `[0.7274, 0.3637, 0.5819]`.
2. If the column truly becomes zero only after the `slice_to_world_matrix()` helper, rewrite that helper without `mat3x3` constructors (build rows manually and skip `transpose`) to rule out Naga/`mat3` semantics.
3. Once column 2 survives, rerun `compare_cov_debug.py`; success criteria: `max_abs_diff` ≤ 1e-6 so K2 can finally compute σₙₙ > 0.

### 6. Expanded Debug Buffer + Explicit Transpose (Nov 8, 2025 — afternoon)
- `CovarianceDebug` now stores both `cov_prime` columns and the raw slice basis columns (see `crates/slicer_shaders/src/kernels/precalculate.wgsl:13-24`). The host buffer stride doubled (96 B) via `COV_DEBUG_STRIDE` in `crates/slicer_gfx/src/lib.rs:44`.
- `scripts/compare_cov_debug.py` understands the new layout and can show the captured basis via `--show-basis`. Current output:
  ```
  GPU basis cols: ( 2.83999e-01, -9.31518e-01,  2.27199e-01),
                  ( 7.27393e-01,  3.63696e-01,  5.81914e-01),
                  ( 0.00000e+00,  0.00000e+00,  0.00000e+00)
  ```
  i.e. column 0 = host column 1, column 1 = host column 2, column 2 = zero. This proves the uniform read is dropping the first column entirely before the transpose even happens.
- Added an explicit transpose helper in both WGSL kernels (no reliance on `transpose()`), but the basis columns still show the same pattern, so the issue is unequivocally happening when WGSL reads the uniform.
- **Action item:** investigate why column 0 vanishes. Hypotheses include:
  1. The uniform struct still maps to std140 differently than expected (the `mat4x4` might need to be declared as `array<vec4<f32>,4>` to prevent driver-side swizzling).
  2. The CPU buffer write uses `bytes_of(&kernel_config)` where the struct still contains legacy field names; verify that no other code rewrites the uniform with row-packed data (search for `rotation_matrix_cols` assignments).
  3. Create a micro shader that writes `config.rotation_matrix_cols[n]` straight into an SSBO to see the data before any math touches it.

Until the uniform columns match, `scripts/compare_cov_debug.py` will keep failing with `max_abs_diff ≈ 5.26e-01`.

### 7. std140 Header Alignment Fix (Nov 8, 2025 — evening)
- Host/WGSL structs now share an explicit std140 header: the first field is a `vec4<u32>` (`KernelConfig::num_distributions` on Rust is `[u32; 4]`, GPU reads it via `config.num_distributions.x`). This removes the hidden padding that previously shifted the rotation matrix columns.
- `precalc_debug` now mirrors the CPU slice basis exactly:
  ```
  GPU basis cols: (-6.24695e-01,  0.00000e+00,  7.80869e-01),
                  ( 2.83999e-01, -9.31518e-01,  2.27199e-01),
                  ( 7.27393e-01,  3.63696e-01,  5.81914e-01)
  ```
- With the alignment fixed, `sigma_n_n` becomes positive and the density texture finally shows a range (`min=0.0, max≈1.1e-1`). `scripts/compare_cov_debug.py` still reports `max_abs_diff ≈ 6.4e-01`, but the discrepancy now reflects real math differences (change-of-basis math) rather than missing normals.
- **Next up:** focus purely on K1 math—iterate on the change-of-basis multiply, compare against the CPU oracle after every tweak, and drive the diff below 1e-6 before touching K2/K3 again.

### 8. Full K1 Instrumentation (Nov 8, 2025 — late evening)
- `CovarianceDebug` now stores, per Gaussian:
  - `cov_col*`: Σ′ columns in slice space.
  - `rot_col*`: uniform slice-to-world columns.
  - `w2s_col*`: world-to-slice columns (transpose of slice-to-world).
  - `rotw_col*`: the intermediate `covariance * slice_to_world` columns (current implementation).
  - `dot_col*`: `world_to_slice[row] · rotw_col` (the final Σ′ entries as computed on-GPU).
- `scripts/compare_cov_debug.py` gained `--show-intermediates` and `--check-symmetry`. It now reports symmetry error (currently ~6.6e-1) and the mismatch between Σ′ and the reconstructed dot rows (currently ~3e-8, meaning the per-row dot accumulation agrees with Σ′).
- Findings:
  - The basis columns and world-to-slice rows match the CPU reference exactly.
  - `rotw_col*` is wrong for anisotropic Gaussians (e.g. expected `[-0.056, 0, 0.382]` but GPU shows `[0.339, -0.156, 0]`). This proves the first multiply is using the wrong operands/order.
  - Because dot reconstruction matches the stored Σ′, the rest of the kernel is consistent; fixing `covariance * slice_to_world` should automatically restore symmetry and parity.
### 6. K1 Parity Closure (Nov 14, 2025)
- `KernelConfig::rotation_matrix_cols` is now packed as **world→slice rows**, mirroring the Swift shader’s `float3x3` layout. WGSL loads the matrix directly without additional transposes, while the CPU oracle multiplies `R_ws * Σ * R_wsᵀ`.
- `Gaussian3D` on the GPU side now matches the Swift stride exactly (five `vec4<f32>` blocks: mean, three covariance columns, weight/pad). This eliminated the silent aliasing where every Gaussian reused the previous covariance data.
- `scripts/compare_cov_debug.py` reads the world→slice basis from the kernel dump, rebuilds the CPU change-of-basis, and supports `--check-symmetry`/`--show-intermediates` for deeper inspection.
- Canonical validation loop:
  ```bash
  cargo run -p slicer_app -- \
    --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
    --dump-gaussians-raw=tmp/gaussians.raw \
    --dump-precalc-debug-raw=tmp/precalc_debug.raw \
    --dump-kernel-config-raw=tmp/kernel.raw \
    --exit-after-ms=2000

  python3 scripts/compare_cov_debug.py \
    --gaussians tmp/gaussians.raw \
    --precalc-debug tmp/precalc_debug.raw \
    --kernel-config tmp/kernel.raw \
    --tolerance 1e-6 \
    --check-symmetry
  ```
  Output: `max_abs_diff=2.24e-08` (all entries within tolerance, symmetry error ≈ 1e-7). This confirms K1 parity between Swift/Metal and Rust/wgpu for the triplet PLY.

### 7. K2/K3 Tooling Drop (Nov 14, 2025)
- Added `scripts/parity_common.py`, a shared helper used by all parity scripts to load buffers and re-run the Swift math on CPU.
- **Dynamic params (K2):** `scripts/compare_dynamic.py` loads the Gaussian dump + kernel config, recomputes conditional means/combined factors, and compares them against `--dump-dynamic-raw`. Example:
  ```bash
  python3 scripts/compare_dynamic.py \
    --gaussians tmp/gaussians.raw \
    --kernel-config tmp/kernel.raw \
    --dynamic tmp/dynamic.raw \
    --tolerance 1e-5 \
    --top 5
  ```
  This script reports per-Gaussian diffs for both `mean2d` and `combined_factor` (currently failing until K2 is fixed).
- **Density texture (K3):** `scripts/compare_density.py` mirrors the evaluation kernel on CPU and compares the result to `--dump-density-raw`. Supports `--resolution` overrides and `--limit-gaussians` for heavy scenes:
  ```bash
  python3 scripts/compare_density.py \
    --gaussians tmp/gaussians.raw \
    --kernel-config tmp/kernel.raw \
    --density tmp/density.raw \
    --resolution 256 \
    --tolerance 1e-5
  ```
  Outputs max/mean absolute error plus the coordinates of the worst texels so we can zero in on K3 regressions once K2 parity lands.

### 8. Latest Tooling Run (Nov 14, 2025 @ 23:58)
- `python3 scripts/compare_dynamic.py --gaussians tmp/gaussians.raw --kernel-config tmp/kernel.raw --dynamic tmp/dynamic.raw --tolerance 1e-5 --top 5` now reports `mean diff=1.824e-07` and `combined diff=7.549e-09` (all entries within tolerance).
- `python3 scripts/compare_density.py --gaussians tmp/gaussians.raw --kernel-config tmp/kernel.raw --density tmp/density.raw --resolution 256 --tolerance 1e-5 --top 5` yields `max_abs_diff=9.558e-08` and `mean_abs_diff=8.393e-10` (all texels within tolerance).
- Keep these dumps (triplet PLY, 256 grid) around as the new parity baseline; re-run the two scripts after every shader/layout edit to make sure K2/K3 stay green.

### 9. Dynamic Buffer Alignment Fix (Nov 14, 2025 @ 23:55)
- `DynamicParams` now matches WGSL’s std430 layout explicitly: `(vec2 mean2d)` @ 0 bytes, `combined_factor` @ 8 bytes, `_padding0` f32 @ 12 bytes, and an unused `vec4` pad @ 16 bytes (total 32 bytes per record). This keeps the shader’s implicit 16-byte alignment from trampling the next Gaussian’s data.
- Host structs (`crates/slicer_core/src/gpu.rs`) gained `_padding0` and `_pad` to mirror the WGSL definition, and the parity tooling (`scripts/parity_common.py`) now treats dynamic dumps as 8 floats per record.
- Added `cargo test -p slicer_shaders layout_snapshot`, which parses `update_params.wgsl` with `naga`, asserts the struct member offsets (`mean2d=0`, `combined_factor=8`, `_padding0=12`, `_pad=16`), and verifies the total sizes (`Gaussian3D=80`, `PrecalculatedParams=40`, `DynamicParams=32`). Re-run this whenever the WGSL structs change to catch regressions before they hit the GPU.

### 10. Large Scene Validation (Nov 15, 2025 @ 00:15)
- `slicer_app` now accepts `--num-distributions`, `--grid-resolution`, and `--seed` CLI overrides, so we can deterministically stress-test larger Gaussian sets without editing source.
- 50 000 procedurally generated gaussians (seed = 1234) @ **256² grid**: `compare_dynamic.py` reports `mean diff=2.185e-06`, `combined diff=3.606e-11`; `compare_density.py --sample-count 4096` hits `max_abs_diff=1.826e-08`, `mean_abs_diff=1.168e-09`.
- Same gaussians @ **512² grid**: `compare_dynamic.py` unchanged (struct-only), `compare_density.py --sample-count 8192` shows `max_abs_diff=2.214e-08`, `mean_abs_diff=1.105e-09`.
- Full-grid density diffs for 50 k gaussians are infeasible (O(N·pixels) ≈ 3.3e9 ops), so `compare_density.py` now supports deterministic texel sampling; the sample counts above keep runtime <2 min while still covering thousands of pixels.

### 11. Red/Green Automation (Nov 15, 2025 @ 00:25)
- `Tools/red_green_cycle.sh` now runs the parity loop automatically: the triplet PLY smoke-run emits all dumps, then `compare_dynamic.py` and `compare_density.py` execute (full-grid) before the script reports success. Any shader/layout regression now fails the red/green cycle immediately.
- `scripts/compare_volume_exports.py` is staged for the upcoming RAW/MHD export parity work—point it at the Swift + Rust exports (with optional voxel sampling) to sanity-check headless volume stacks once the Rust exporter lands.

### 12. Parity Suite Script (Nov 15, 2025 @ 00:35)
- Added `Tools/run_parity_suite.sh`, which chains three scenarios: (1) triplet PLY (full-grid diff), (2) procedural 50 k @ 256² with 4 096 sampled texels, and (3) procedural 50 k @ 512² with 8 192 sampled texels. Set `PARITY_SKIP_HEAVY=1` (or override `PARITY_SAMPLE_{256,512}`) when a lighter loop is desired, but the defaults match the large-scene diffs logged above.
- CI TODO: invoke this script (possibly nightly) so sampled parity failures show up without humans running the Python combos manually.

### 13. CLI Volume Export (Nov 15, 2025 @ 00:45)
- `slicer_app` gained `--export-volume=/path/to/output.{mhd,raw}` plus `--export-log-normalized`. When specified, the app runs headless, sweeps K2→K3 across all `grid_resolution` slices, and writes a RAW/MHD pair using the same normalization logic as the Swift `VolumeExporter`.
- Use the new `scripts/compare_volume_exports.py --reference-{mhd,raw}=SWIFT --candidate-{mhd,raw}=RUST` helper (optionally `--sample-count=N`) to diff the exported stacks once the Rust exporter lands in the GUI path. This unlocks numeric parity for the export/readback milestone (M5) before UI polish.

### 14. Swift↔Rust Export Parity (Nov 15, 2025 @ 01:20)
- The Swift app now honors the same CLI flags as Rust for headless exports. Example (triplet PLY, 64³ grid):
  ```bash
  swift run GaussianSlicer --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
    --grid-resolution=64 --export-volume=$(pwd)/tmp/swift_volume.mhd

  cargo run -p slicer_app -- --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
    --grid-resolution=64 --export-volume=$(pwd)/tmp/rust_volume.mhd

  python3 scripts/compare_volume_exports.py \
    --reference-mhd tmp/swift_volume.mhd --reference-raw tmp/swift_volume.raw \
    --candidate-mhd tmp/rust_volume.mhd --candidate-raw tmp/rust_volume.raw \
    --tolerance 1e-5
  ```
  Result: `max_abs_diff=4.172e-07`, `mean_abs_diff=3.638e-10` (all 262 144 voxels within tolerance). The dumps under `tmp/swift_volume.*` / `tmp/rust_volume.*` are the new RAW/MHD baseline for CI.

### 15. Bilinear ↔ Nearest Visualization Toggle (Nov 15, 2025 @ 14:00)
- Added a `filter_mode` flag to both visualization configs plus dual samplers (bilinear + nearest). WGSL now writes densities to `rgba16float`, samples via the selected filter, and switches outlines accordingly (gradient-based for bilinear, discrete neighbor-band detection for nearest). Metal mirrors the same behavior.
- CLI / UI wiring: `slicer_app --filter-mode={bilinear|nearest}` toggles Rust headlessly, the egui sidebar exposes a combo, and the Swift UI gained a “Nearest Neighbor Filter” switch (persisted through `RendererSettings`). Default remains bilinear to match the historic Metal look; nearest restores the old blocky style.
 
---

## Persistent Issues
1. **Need broader datasets.** Coverage now includes a deterministic 50 k‑Gaussian procedural scene at 256²/512², but we still need to ingest production PLYs and extreme grid sizes (>768²) to watch for precision issues (hook `Tools/run_parity_suite.sh` up to those inputs).
2. **Automation/CI still limited.** The red/green cycle now catches triplet regressions locally, but CI still ignores the sampled parity runs—wire `Tools/run_parity_suite.sh` (or an equivalent GitHub Action) into CI so failures show up without manual intervention.
3. **Export/readback parity automation.** Headless exports now match numerically; next up is wiring the Swift+Rust commands plus `scripts/compare_volume_exports.py` into CI so RAW/MHD regressions are caught automatically.
4. **Filter-mode snapshots.** Capture a couple of reference frames (bilinear vs nearest, both stacks) so we can sanity-check the toggle in future regressions.

---

## Suggested Next Paths
1. **Scale the parity set.** Feed any production PLYs through `Tools/run_parity_suite.sh` (or extend it with additional scenes) and record the sampled diffs here so we spot issues beyond the procedural generator.
2. **Automate the loop.** Wire `Tools/run_parity_suite.sh` into CI (triplet + sampled heavy scenes) so parity failures block merges; keep `Tools/red_green_cycle.sh` for the quick developer loop.
3. **Plan CI/export coverage.** Hook the new `--export-volume` flag + `scripts/compare_volume_exports.py` into CI so both exporters stay in lockstep, then resume the egui feature work + packaging (M4–M6).

---

## File Map / Tooling Links
- `crates/slicer_shaders/src/kernels/precalculate.wgsl` — current WGSL logic, includes `CovarianceDebug`.
- `crates/slicer_app/src/main.rs` — CLI parsing + dump plumbing.
- `tmp/precalc_debug.raw` — latest GPU dump (little-endian f32, 12 floats per Gaussian).
- `tmp/precalc.raw`, `tmp/dynamic.raw`, `tmp/density.raw` — for completeness.

---

## Handoff Snapshot — 2025‑11‑14 23:58 EST

### What’s currently in-flight
- Host ↔ WGSL layouts are locked in (`DynamicParams` stride is 32 B, matrices stay three `vec4`s) and enforced by `cargo test -p slicer_shaders layout_snapshot`.
- Parity scripts are green on the triplet PLY @ 256²: `compare_cov_debug.py` (`max_abs_diff=2.24e-08` from earlier run), `compare_dynamic.py` (`mean diff=1.824e-07`, `combined diff=7.549e-09`), and `compare_density.py` (`max_abs_diff=9.558e-08`, `mean_abs_diff=8.393e-10`). Fresh dumps live under `tmp/`.
- Need to scale this validation beyond the tiny scene and hook the Python checks into automation so layout/shader regressions surface fast.

### Immediate next steps for the next agent
1. Re-run the dump + compare loop on a larger Gaussian set (≥50 k) and multiple grid resolutions; capture the new diffs here so we know the stride fix holds under load.
2. Fold `compare_dynamic.py`/`compare_density.py` into `Tools/red_green_cycle.sh` or a CI job (the scripts exit non-zero when tolerances blow up), keeping artifacts for inspection.
3. Resume the export/readback/CI packaging milestones (M4–M6) now that K2/K3 are trustworthy—reuse the same numeric diff approach to validate headless exports before wiring UI polish.

### Known blockers to mention if reassigned
- The parity loop still requires spinning up `slicer_app` interactively to produce dumps; a headless/deterministic export path would make CI wiring easier.
- CI is unaware of these checks; until automation lands we rely on agents to run the loop manually.
- Export/readback parity still lacks tooling, so downstream milestones remain speculative until we replicate today’s diff strategy there.

Keep this section updated as soon as a major checkpoint is reached so the next agent can resume without rediscovering the same alignment issues.

---

_Last updated: 2025‑11‑15 after RAW/MHD export parity verification._
