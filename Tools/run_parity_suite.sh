#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="tmp/parity_suite"
mkdir -p "${OUT_DIR}"
LOG_FILE=/tmp/gaussian_slicer_parity.log

run_triplet() {
  local prefix="triplet"
  local base="${OUT_DIR}/${prefix}"
  echo "[parity] Triplet baseline"
  cargo run -p slicer_app -- \
    --gaussian-ply="$(pwd)/assets/examples/gaussian_triplet.ply" \
    --exit-after-ms=2000 \
    --dump-gaussians-raw="${base}_gaussians.raw" \
    --dump-precalc-raw="${base}_precalc.raw" \
    --dump-dynamic-raw="${base}_dynamic.raw" \
    --dump-density-raw="${base}_density.raw" \
    --dump-kernel-config-raw="${base}_kernel.raw" \
    --dump-precalc-debug-raw="${base}_precalc_debug.raw" >"${LOG_FILE}" 2>&1 || {
      cat "${LOG_FILE}"
      exit 1
    }

  python3 scripts/compare_dynamic.py \
    --gaussians "${base}_gaussians.raw" \
    --kernel-config "${base}_kernel.raw" \
    --dynamic "${base}_dynamic.raw" \
    --tolerance 1e-5 \
    --top 3

  python3 scripts/compare_density.py \
    --gaussians "${base}_gaussians.raw" \
    --kernel-config "${base}_kernel.raw" \
    --density "${base}_density.raw" \
    --resolution 256 \
    --tolerance 1e-5 \
    --top 3
}

run_procedural() {
  local resolution="$1"
  local sample_count="$2"
  local prefix="proc_${resolution}"
  local base="${OUT_DIR}/${prefix}"
  echo "[parity] Procedural 50k @ ${resolution}^2 (sample ${sample_count})"
  cargo run -p slicer_app -- \
    --num-distributions=50000 \
    --seed=1234 \
    --grid-resolution="${resolution}" \
    --exit-after-ms=2000 \
    --dump-gaussians-raw="${base}_gaussians.raw" \
    --dump-precalc-raw="${base}_precalc.raw" \
    --dump-dynamic-raw="${base}_dynamic.raw" \
    --dump-density-raw="${base}_density.raw" \
    --dump-kernel-config-raw="${base}_kernel.raw" >"${LOG_FILE}" 2>&1 || {
      cat "${LOG_FILE}"
      exit 1
    }

  python3 scripts/compare_dynamic.py \
    --gaussians "${base}_gaussians.raw" \
    --kernel-config "${base}_kernel.raw" \
    --dynamic "${base}_dynamic.raw" \
    --tolerance 1e-5 \
    --top 5

  python3 scripts/compare_density.py \
    --gaussians "${base}_gaussians.raw" \
    --kernel-config "${base}_kernel.raw" \
    --density "${base}_density.raw" \
    --resolution "${resolution}" \
    --tolerance 1e-5 \
    --top 5 \
    --sample-count "${sample_count}"
}

run_triplet

if [[ "${PARITY_SKIP_HEAVY:-0}" == "1" ]]; then
  echo "[parity] Skipping heavy scenes (PARITY_SKIP_HEAVY=1)"
else
  PROC_SAMPLE_256="${PARITY_SAMPLE_256:-4096}"
  PROC_SAMPLE_512="${PARITY_SAMPLE_512:-8192}"
  run_procedural 256 "${PROC_SAMPLE_256}"
  run_procedural 512 "${PROC_SAMPLE_512}"
fi

echo "[parity] suite complete"
