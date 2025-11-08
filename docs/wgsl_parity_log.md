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
