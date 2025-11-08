#!/usr/bin/env bash
set -euo pipefail

COUNT=${COUNT:-128}
SEED=${SEED:-0xA11CE}
EPSILON=${EPSILON:-1e-6}
TMP_CASES=$(mktemp /tmp/gaussian_parity_cases.XXXXXX.json)
cleanup() {
  rm -f "$TMP_CASES"
}
trap cleanup EXIT

echo "[1/3] Exporting parity case sets via wgpu runner (count=$COUNT, seed=$SEED)..."
cargo run -p parity_lab --bin wgpu_scalars -- --count="$COUNT" --seed="$SEED" \
  --epsilon="$EPSILON" --export-json="$TMP_CASES" --skip-dispatch

echo "[2/3] Running Metal parity runner on exported cases..."
(
  cd parity_lab/runner/metal
  SWIFT_MODULE_CACHE_PATH=.swift-module-cache \
  CLANG_MODULE_CACHE_PATH=.clang-module-cache \
  swift run parity-lab-metal --cases="$TMP_CASES" --epsilon="$EPSILON"
)

echo "[3/3] Done."
