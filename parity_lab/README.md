# Gaussian Slicer Parity Lab

This crate hosts the cross-backend math kernels described in `AGENT-PARITYLAB.md`. The goal is to keep the canonical scalar/vector/matrix/SPD implementations in one place, feed identical test vectors through Metal + WGSL kernels, and make regressions obvious via CPU references.

## Layout

```
parity_lab/
  cpu/        # Rust references that can double as CPU fallbacks
  metal/      # Known-good MSL snippets that track the Swift app
  wgsl/       # One-to-one WGSL twins that feed the wgpu runner
  runner/     # Host-side harnesses (wgpu + Swift Metal) for tiny tests
  tests/      # Rust integration tests that validate the references
```

The initial drop focuses on the “Scalars & Vectors” milestone. `cpu/scalars.rs` provides dot/axpy/length²/normalize helpers and deterministic generators (`generate_scalar_vector_cases`) that will be shared with both GPU runners. The integration test `tests/scalars.rs` ensures these stay deterministic and documents the tolerance limits we expect from the GPU backends (currently 1e-6 abs err for basic ops).

### Latest status

- JSON export/import now emits labeled case sets that bundle scalar/vector, mat3, and SPD fixtures in one file (legacy scalar-only arrays are still accepted).
- Scalar/vector axpy paths in Rust, WGSL, and Metal use fused multiply-adds to stay under the 1e-6 tolerance even with large mixed-magnitude batches.
- SPD references gate inputs by residual (`‖A·x-b‖∞ ≤ 5e-7`) and both GPU runners validate via residuals plus Cholesky deltas, which proved more stable than raw vector diffs.
- The Swift Metal runner can rebuild the shader library from bundled source when necessary and the `scripts/run_parity.sh` helper now drives the entire export→Metal replay loop with local module caches.

Upcoming work will:

1. Add tiny Metal + WGSL kernels under `metal/` and `wgsl/` that consume the shared case buffers.
2. Build the `runner/` binaries (`wgpu` + SwiftPM) that drive those kernels and copy results back. The first step of this is the `cargo run -p parity_lab --bin wgpu_scalars` helper, which dispatches the scalar/vector WGSL kernel and reports residuals against the CPU references (failing if they exceed 1e-6). Use `--export-json=cases.json` to dump the deterministic inputs, `--cases-json=cases.json` to replay a previously exported batch (handy for the Metal runner), and `--skip-dispatch` if you only need the data without touching the GPU. The JSON now stores labeled case sets (e.g., `basic_random`, `stress_edges`) so both GPU runners cover multiple parity fixtures. A matching SwiftPM runner lives under `runner/metal/`; run it via `SWIFT_MODULE_CACHE_PATH=.swift-module-cache CLANG_MODULE_CACHE_PATH=.clang-module-cache swift run parity-lab-metal --cases=/path/to/cases.json` to execute the Metal kernel on the exact same data (add `--print-layout` to inspect struct sizes/alignment). For a one-command exchange, run `scripts/run_parity.sh` (respects `COUNT`, `SEED`, `EPSILON`).
3. Extend the CPU module hierarchy to cover the matrix, SPD, Schur, and rotation tests listed in `AGENT-PARITYLAB.md`.

Run `cargo test -p parity_lab` to execute the reference suite. The binary (`cargo run -p parity_lab`) currently emits guidance and will evolve into a CLI for selecting parity problem sets and dumping JSON snapshots consumed by the GPU harnesses.
