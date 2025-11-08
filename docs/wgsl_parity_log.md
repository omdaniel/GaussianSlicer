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
- **Action item:** rewrite the intermediate multiply to explicitly compute `covariance * slice_to_world[column]` (column-major multiply) rather than treating the columns as rows. Re-run the dump + oracle loop until `GPU cov*S cols` match the CPU (`cov @ S`). Only then resume the Schur complement work.

---

## Persistent Issues
1. **Wrong basis used for covariance projection.** Even after explicitly computing `cov_prime = Rᵀ Σ R`, the third column equals the in-plane columns, so the normal variance collapses. Hypothesis: the “rows” we dot against are still the in-plane axes because WGSL matrices index as column-major and we’re reusing `slice_to_world` both as column and row vectors.
2. **`sigma_n_n < 0`** for Gaussians 1 & 2. The Schur complement logic (`cov_n_n_val >= EPSILON`) therefore never runs, leaving `inv_cov2d` zero and `combined_factor` zero.
3. **No density variation** in `tmp/density.raw` (min= max) → render pass samples a uniform texture and ends up with a single color PNG.

---

## Suggested Next Paths
1. **Derive proper row vectors explicitly.** Instead of reusing `slice_to_world[0].x` etc., compute `world_to_slice = transpose(slice_to_world)` once and use its rows when projecting both means and covariances. The current manual dot-product probably mixes up rows/columns.
2. **Cross-validate with CPU script.** Since `tmp/precalc_debug.raw` and `tmp/gaussians.raw` are available, write a quick script (`scripts/compare_cov_debug.py`) that:
   - Rebuilds the CPU `cov_prime` using the same basis.
   - Prints the per-element delta vs `precalc_debug`.
   - This will pinpoint whether only the third column is wrong or the whole matrix is rotated.
3. **Unit-test the basis math.** Add a Rust test in `slicer_core` that compares `rotation_matrix_for_normal` against the WGSL `build_slice_basis` by exporting its output (e.g., via pushing the basis into a debug buffer). Ensures both stacks agree on column ordering.

Once `cov_prime` matches, rerun the dumps to confirm `sigma_n_n` > 0 for every Gaussian, then resume the PNG capture comparison.

---

## File Map / Tooling Links
- `crates/slicer_shaders/src/kernels/precalculate.wgsl` — current WGSL logic, includes `CovarianceDebug`.
- `crates/slicer_app/src/main.rs` — CLI parsing + dump plumbing.
- `tmp/precalc_debug.raw` — latest GPU dump (little-endian f32, 12 floats per Gaussian).
- `tmp/precalc.raw`, `tmp/dynamic.raw`, `tmp/density.raw` — for completeness.

---

## Handoff Snapshot — 2025‑11‑08 08:35 EST

### What’s currently in-flight
- Host-side structs already enforce an 80 B `Gaussian3D` stride (see `crates/slicer_core/src/gpu.rs`), but the accompanying WGSL rewrite from `gs_wgsl_parity_patch.diff` has **not** been applied yet. Both `precalculate.wgsl` and `update_params.wgsl` still rely on row-assembled rotation matrices.
- `Config.rotation_matrix` is being populated with *column* data, while the shader interprets it as rows (`slice_to_world_matrix` transposes manually). This mismatch is the primary suspect for the incorrect `cov_prime` third column.
- Latest dumps live under `tmp/` (see `git status`); keep them for reference before rerunning tests.

### Immediate next steps for the next agent
1. **Review and apply `gs_wgsl_parity_patch.diff`.**  
   - The patch modernizes `Gaussian3D`, rewrites K1/K2 to use explicit matrix multiplications (`W2S * Σ * W2Sᵀ`), and adds the Schur complement guard rails described above.  
   - Apply the patch incrementally (e.g., `patch -p1 < gs_wgsl_parity_patch.diff` or cherry‑pick the relevant hunks) and run `cargo fmt`.
2. **Regenerate GPU dumps after each atomic edit.**  
   ```
   cargo run -p slicer_app -- --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
     --dump-precalc-debug-raw=tmp/precalc_debug.raw \
     --dump-precalc-raw=tmp/precalc.raw \
     --dump-dynamic-raw=tmp/dynamic.raw \
     --dump-density-raw=tmp/density.raw \
     --exit-after-ms=2000
   ```
   - Keep old dumps for comparison (rename before rerunning).
3. **CPU vs GPU comparison script.**  
   - Implement `scripts/compare_cov_debug.py` that rebuilds `cov_prime` on the CPU using the same config and reports `max_abs_diff` per Gaussian vs `tmp/precalc_debug.raw`.  
   - Target tolerance: ≤ 1e‑6 after the WGSL fix. Fail the script if exceeded.
4. **Document each test loop.**  
   - After every successful pass (patch apply → cargo run → comparison script), append findings to this log so future agents see the progression.

### Known blockers to mention if reassigned
- `sigma_n_n` still collapses to zero for anisotropic Gaussians until the rotation uniform semantics are corrected.
- Density PNGs are still single-color because K2/K3 short-circuit when `cov_prime[2].z ≤ 0`.
- CI/tests do not yet cover any of this; manual loops (`cargo run` + Python diff) are required for now.

Keep this section updated as soon as a major checkpoint is reached so the next agent can resume without rediscovering the same alignment issues.

---

_Last updated: 2025‑11‑08 by WGSL parity investigation._
