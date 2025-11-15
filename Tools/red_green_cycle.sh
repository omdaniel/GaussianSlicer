#!/usr/bin/env bash

set -euo pipefail

echo "[red/green] cargo check"
cargo check

echo "[red/green] cargo test"
cargo test

OUT_DIR=tmp/red_green
mkdir -p "${OUT_DIR}"
LOG_FILE=/tmp/gaussian_slicer_smoke.log

echo "[red/green] smoke-run slicer_app (triplet ply + dumps)"
SMOKE_ARGS=(
    --gaussian-ply="$(pwd)/assets/examples/gaussian_triplet.ply"
    --exit-after-ms=2000
    --dump-gaussians-raw="${OUT_DIR}/gaussians.raw"
    --dump-precalc-raw="${OUT_DIR}/precalc.raw"
    --dump-precalc-debug-raw="${OUT_DIR}/precalc_debug.raw"
    --dump-dynamic-raw="${OUT_DIR}/dynamic.raw"
    --dump-density-raw="${OUT_DIR}/density.raw"
    --dump-kernel-config-raw="${OUT_DIR}/kernel.raw"
)
cargo run -p slicer_app -- "${SMOKE_ARGS[@]}" >"${LOG_FILE}" 2>&1 || {
    cat "${LOG_FILE}"
    exit 1
}

echo "[red/green] compare_dynamic (triplet)"
python3 scripts/compare_dynamic.py \
    --gaussians "${OUT_DIR}/gaussians.raw" \
    --kernel-config "${OUT_DIR}/kernel.raw" \
    --dynamic "${OUT_DIR}/dynamic.raw" \
    --tolerance 1e-5 \
    --top 3

echo "[red/green] compare_density (triplet)"
python3 scripts/compare_density.py \
    --gaussians "${OUT_DIR}/gaussians.raw" \
    --kernel-config "${OUT_DIR}/kernel.raw" \
    --density "${OUT_DIR}/density.raw" \
    --resolution 256 \
    --tolerance 1e-5 \
    --top 3

echo "[red/green] success"
