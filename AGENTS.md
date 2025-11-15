# Agent Validation: Gaussian Slicer (Metal/SwiftPM)

This workflow validates that the generated Swift Package Manager (SwiftPM) project for the Gaussian Slicer compiles successfully on macOS. This confirms that both the Swift code and the Metal shaders are valid and correctly configured for the target environment.

## Prerequisites

* **Environment:** macOS 13.0+ (as specified in `Package.swift`).
* **Tools:** Xcode Command Line Tools (includes the Swift toolchain and Metal compiler).

## Validation Steps

1.  **Execute Bootstrap Script**

    Ensure the bootstrap script runs successfully and creates the project.
    ```python
    agwa.shell.run("./bootstrap_gaussianslicer.sh")
    ```

2.  **Navigate to Project Directory**

    Change the current working directory to the root of the generated project.
    ```python
    agwa.shell.run("cd GaussianSlicer")
    ```

3.  **Clean Previous Builds**

    Ensure a fresh build environment.
    ```python
    agwa.shell.run("swift package clean")
    ```

4.  **Build the Project (Debug Configuration)**

    Execute the Swift build command. SwiftPM automatically invokes the Metal compiler (`metal`) to compile `Shaders.metal` into a `default.metallib` and links it with the Swift code. A failure in either compilation stage will cause this step to fail.
    ```python
    # Set the expectation for the command to succeed
    agwa.shell.run("swift build -c debug")
    ```

5.  **Build the Project (Release Configuration)**

    Verify that the project builds with release optimizations enabled. This is the configuration optimized for performance on the M1 Pro.
    ```python
    # Set the expectation for the command to succeed
    agwa.shell.run("swift build -c release")
    ```

6.  **Verify Executable Creation**

    Check that the final executable binary was created in the release build directory.
    ```python
    # The exact path depends on the architecture (e.g., arm64-apple-macosx).
    # We use a wildcard search to find the executable robustly.
    agwa.shell.run("find .build -type f -name 'GaussianSlicer' -path '*/release/*' | grep .")
    ```

7.  **Cleanup**

    Navigate back to the original directory.
    ```python
    agwa.shell.run("cd ..")
    ```

## Success Criteria

The validation is successful if all steps complete without error (exit code 0), confirming successful compilation of the Swift and Metal code.

## Command-Line Runtime & Export Flags

Both the Swift and Rust frontends share the same CLI surface so you can drive deterministic workloads without touching UI state:

| Flag | Description |
|------|-------------|
| `--num-distributions=INT` | Override the procedural point count (default `50_000`). |
| `--grid-resolution=INT`   | Set the square slice resolution (default `256`). |
| `--seed=UINT64`           | Fix the RNG seed used by the procedural generator. |
| `--gaussian-ply=PATH`     | Load a Gaussian Splat PLY (e.g., `assets/examples/gaussian_triplet.ply`). |
| `--export-volume=PATH`    | Run headless, sweep all slices (K2→K3), and write a RAW/MHD pair. Use `.mhd` or `.raw` for the path extension. |
| `--export-log-normalized` | (Optional) Apply the log-space normalization that mirrors the in-app visualization before writing the volume. |
| `--filter-mode=MODE`      | Choose `bilinear` (default) or `nearest` sampling for the visualization/export path. Matches the UI toggle on both apps. |

Any of the visualization parameters (colormap, density range, etc.) can also be specified; see `Sources/GaussianSlicer/RuntimeConfig.swift` for the full list.

## Optional Runtime & Export Parity Checks

1. **Interactive parity (triplet PLY)**

   After a successful build you can launch the macOS app against the shared-test Gaussian splat file that both the Swift and Rust ports understand. This is useful for visual investigations:

```bash
swift run GaussianSlicer --gaussian-ply="$(pwd)/assets/examples/gaussian_triplet.ply"
# Compare against Rust/wgpu
cargo run -p slicer_app -- --gaussian-ply=assets/examples/gaussian_triplet.ply
```

2. **Headless RAW/MHD export parity**

   Use the shared CLI to generate matching volumes, then diff them numerically:

```bash
# Swift export (headless)
swift run GaussianSlicer \
  --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
  --grid-resolution=64 \
  --export-volume=$(pwd)/tmp/swift_volume.mhd

# Rust export (headless)
cargo run -p slicer_app -- \
  --gaussian-ply=$(pwd)/assets/examples/gaussian_triplet.ply \
  --grid-resolution=64 \
  --export-volume=$(pwd)/tmp/rust_volume.mhd

# Numeric diff (full volume or use --sample-count=N for huge grids)
python3 scripts/compare_volume_exports.py \
  --reference-mhd tmp/swift_volume.mhd --reference-raw tmp/swift_volume.raw \
  --candidate-mhd tmp/rust_volume.mhd --candidate-raw tmp/rust_volume.raw \
  --tolerance 1e-5
```

3. **Full parity suite (optional)**

   To exercise the triplet scene plus 50 k‑Gaussian procedural workloads, run:

```bash
Tools/run_parity_suite.sh                # defaults: triplet + 50k@256²/512²
PARITY_SKIP_HEAVY=1 Tools/run_parity_suite.sh   # only the triplet baseline
```

This script dumps the necessary buffers, runs `scripts/compare_dynamic.py`, `scripts/compare_density.py`, and records sampled diffs for the heavy scenes.
