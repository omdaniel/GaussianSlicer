# Parity Lab Metal Runner

This Swift Package runs the Metal reference kernels for scalar/vector math, 3×3 matrix ops, and SPD/Cholesky solves using the exact JSON case sets emitted by the wgpu runner. It mirrors the buffer layouts used on the Rust side so both toolchains ingest identical data.

Run the parity check:

```bash
cd parity_lab/runner/metal
SWIFT_MODULE_CACHE_PATH=.swift-module-cache CLANG_MODULE_CACHE_PATH=.clang-module-cache swift run parity-lab-metal --cases=/path/to/cases.json
```

The extra environment variables redirect the compiler module caches into the repository (some sandboxed environments block writes to the default locations). To inspect buffer layout only, add `--print-layout`. To run the parity comparison, export a JSON dataset from the wgpu runner (`cargo run -p parity_lab --bin wgpu_scalars -- --export-json=cases.json --skip-dispatch`) and feed it to the command above. The Metal tool runs all three compute kernels and prints residuals versus the CPU references (failing if any metric exceeds `1e-6`).

**Shortcut:** `scripts/run_parity.sh` performs the entire exchange (export via wgpu, import via Metal) and respects the `COUNT`, `SEED`, and `EPSILON` environment variables.
