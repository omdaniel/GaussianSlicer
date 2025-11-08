# Parity Lab: scope & goals

**Goal:** Establish a small battery of **known-good Metal shaders** and **WGSL twins** that solve trivial problems exactly (or within tight tolerances). From these we extract repeatable **translation patterns** you’ll apply to the real kernels.

## Current parity lab status (updated)

* Shared JSON schema now emits labeled case sets for scalar/vector, mat3, and SPD fixtures; legacy scalar arrays still parse for older snapshots.
* Both GPU runners consume those case sets. The `wgpu` CLI (`cargo run -p parity_lab --bin wgpu_scalars`) dispatches all three WGSL kernels, while the SwiftPM runner (`parity_lab/runner/metal`) mirrors the Metal shaders and validates residuals per set.
* Scalar/vector paths use fused multiply-adds everywhere (Rust refs, WGSL, Metal) so large case batches stay within the 1e-6 tolerance without double-rounding drift.
* SPD harnesses check `|A·x - b|` residuals (rather than direct CPU vector deltas) and the case generator filters ill-conditioned matrices up front, keeping tolerance enforcement consistent across platforms.
* `scripts/run_parity.sh` provides the end-to-end JSON export + Metal replay flow, honoring `COUNT`, `SEED`, and `EPSILON` env vars and wiring the Swift module caches into the repo for reproducible builds.

## Minimal set of parity problems (build up)

1. **Scalars & vectors**

   * dot, axpy (`y = a·x + y`), length², normalize (guard ε).
2. **2×2 & 3×3 matrix ops**

   * mat·vec, mat·mat, transpose (verify row/col conventions).
3. **SPD solves (no explicit inverse)**

   * 2×2 and 3×3 **Cholesky** (LLᵀ) + forward/back substitution.
4. **Schur complement (stable form)**

   * `S = A − B·solve(C, Bᵀ)` (never `C⁻¹`).
5. **Change-of-basis / rotations**

   * Orthonormal R (det=+1): verify `R·Rᵀ = I`, and `R·v` parity.
6. **Order-sensitivity**

   * Accumulation with *fixed loop order* vs tile-optimized order; measure drift and set policy.

Each case runs against **identical CPU references** (Rust `glam`) and identical test vectors, on both pipelines.

# Harness & structure (tiny and testable)

* **Workspace**: add `parity_lab/` (binary + tests)

  * `metal/` MSL kernels (the already-correct ones)
  * `wgsl/` WGSL twins
  * `cpu/` Rust refs (glam)
  * `runner/` wgpu + Metal (Swift) simple runners
* **Data path**: always use **storage buffers** (no textures) for math tests.
* **One issue per test**: each shader does one thing; host verifies residuals.

### Acceptance thresholds

* Scalars/vectors/matmul: **bit-exact** or `≤ 1e-7` abs err
* Cholesky: `‖LLᵀ − A‖∞ ≤ 1e-5` (A scaled to ~O(1))
* Solve: `‖A·x − b‖∞ ≤ 1e-5`
* Schur: `‖S_ref − S_gpu‖∞ ≤ 1e-5`

# WGSL ↔ Metal translation patterns (lock these in)

* **Matrix & vector convention**

  * WGSL is **column-major**; treat vectors as **columns**; multiply as `M * v`.
  * Encode this in a short **“Matrix Discipline”** doc + unit tests (render axis basis).
* **Thread indexing**

  * Metal `thread_position_in_grid` ↔ WGSL `@builtin(global_invocation_id)`.
* **Shared memory**

  * Metal `threadgroup` ↔ WGSL `var<workgroup>`.
  * Barrier: `threadgroup_barrier(mem_flags)` ↔ `workgroupBarrier()`.
* **Bindings**

  * Metal `[[buffer(i)]]` ↔ WGSL `@group(g) @binding(b)` (document a fixed layout).
* **No explicit inverses**

  * Always **Cholesky + solve** (or CPU precompute) for SPD.
* **Order of ops**

  * If parity matters, match **loop order** first; tile/optimize later behind a flag.
* **Precision**

  * Keep core math **f32**; only use `f16` in visualization paths.

# Tiny WGSL snippets to seed the lab

### 3×3 SPD Cholesky + solve (core of K1/K2)

```wgsl
// wgsl/cholesky3.wgsl
struct Mat3 { c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32> }; // column-major
fn m3_get(m: Mat3, r: u32, c: u32) -> f32 {
  return (c == 0u ? m.c0[r] : (c == 1u ? m.c1[r] : m.c2[r]));
}
fn m3_set(inout m: Mat3, r: u32, c: u32, v: f32) {
  if (c == 0u) { m.c0[r] = v; } else if (c == 1u) { m.c1[r] = v; } else { m.c2[r] = v; }
}

// Cholesky factorization A = L·L^T, A SPD (3x3). Returns L; adds epsilon to diag if needed.
fn cholesky3(a: Mat3) -> Mat3 {
  var L = Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  // i = 0..2
  for (var i: u32 = 0u; i < 3u; i++) {
    for (var j: u32 = 0u; j <= i; j++) {
      var sum = m3_get(a, i, j);
      for (var k: u32 = 0u; k < j; k++) {
        sum -= m3_get(L, i, k) * m3_get(L, j, k);
      }
      if (i == j) {
        let d = max(sum, 1e-12); // jitter guard
        m3_set(L, i, j, sqrt(d));
      } else {
        m3_set(L, i, j, sum / m3_get(L, j, j));
      }
    }
  }
  return L;
}

fn forward_substitute(L: Mat3, b: vec3<f32>) -> vec3<f32> {
  var y = vec3<f32>(0.0);
  y.x = b.x / m3_get(L, 0u, 0u);
  y.y = (b.y - m3_get(L,1u,0u)*y.x) / m3_get(L,1u,1u);
  y.z = (b.z - m3_get(L,2u,0u)*y.x - m3_get(L,2u,1u)*y.y) / m3_get(L,2u,2u);
  return y;
}
fn back_substitute(L: Mat3, y: vec3<f32>) -> vec3<f32> {
  var x = vec3<f32>(0.0);
  x.z = y.z / m3_get(L,2u,2u);
  x.y = (y.y - m3_get(L,2u,1u)*x.z) / m3_get(L,1u,1u);
  x.x = (y.x - m3_get(L,1u,0u)*x.y - m3_get(L,2u,0u)*x.z) / m3_get(L,0u,0u);
  return x;
}
fn solve_spd3(A: Mat3, b: vec3<f32>) -> vec3<f32> {
  let L = cholesky3(A);
  let y = forward_substitute(L, b);
  return back_substitute(L, y);
}
```

### Stable Schur complement (K2 core pattern)

```wgsl
// S = A - B * inv(C) * B^T  →  S = A - B * solve(C, B^T)
fn schur3(A: Mat3, B: mat3x3<f32>, C: Mat3) -> Mat3 {
  let LC = cholesky3(C);
  // Solve C·X = B^T  → X has 3 columns (solve each column)
  var X0 = solve_spd3(C, vec3<f32>(B[0].x, B[1].x, B[2].x));
  var X1 = solve_spd3(C, vec3<f32>(B[0].y, B[1].y, B[2].y));
  var X2 = solve_spd3(C, vec3<f32>(B[0].z, B[1].z, B[2].z));
  let X = mat3x3<f32>(X0, X1, X2); // columns
  // S = A - B * X
  let BX = mat3x3<f32>(
    B[0].x*X0 + B[0].y*X1 + B[0].z*X2,
    B[1].x*X0 + B[1].y*X1 + B[1].z*X2,
    B[2].x*X0 + B[2].y*X1 + B[2].z*X2
  );
  return Mat3(BX[0], BX[1], BX[2]); // then subtract from A per-element in your kernel
}
```

> Keep these kernels extremely small; they’re for **correctness parity**, not performance.

# Test vectors (simple & revealing)

* Identity / diagonal SPD (easy ground truth).
* Random SPD via `M·Mᵀ + λI` with λ in {1e-6, 1e-4, 1e-2}.
* Well-conditioned rotation matrices (det +1).
* Near-singular cases to validate guards (expect larger residuals but bounded).

# Process to reach parity fast

1. **Lock inputs**: same seeds, same constants/LUTs, dumped to JSON once and reused.
2. **Run each test on both pipelines** (Metal app & wgpu parity runner); copy buffers back; compute residuals on CPU.
3. **Fix the first failing stage** before moving up the ladder (don’t chase downstream symptoms).
4. **Document the pattern** that made WGSL match Metal (e.g., column-vector convention, loop order, ε-jitter, SoA layout).
5. **Promote the pattern** into K1/K2/K3, guarded behind a `--reference-mode` switch for easy A/B.

# Practical tips that usually break the deadlock

* **SoA over AoS** for Gaussian params in compute passes; fewer indexing bugs and better coalescing.
* **No per-frame pipeline/bind-group creation** (only buffer writes).
* **Fixed loop order** in “reference” versions; optimize later behind a flag.
* **One source of truth** for layout: write a tiny script that prints `sizeof/offsetof` for Rust structs at startup and verify in a WGSL “layout check” kernel with static asserts (sizes/strides baked as `override` constants).

---

If you want, I can package this into a small `parity_lab/` crate (Rust + WGSL) with:

* ready-to-run tests for dot/matmul/Cholesky/solve/Schur,
* a CLI that runs both **reference** and **optimized** variants,
* and a markdown “Translation Patterns” page you can link from AGENTS.md.
