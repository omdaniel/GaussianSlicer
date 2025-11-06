#!/usr/bin/env bash

set -euo pipefail

echo "[red/green] cargo check"
cargo check

echo "[red/green] cargo test"
cargo test

echo "[red/green] smoke-run slicer_app (--exit-after-ms=2000)"
cargo run -p slicer_app -- --exit-after-ms=2000 >/tmp/gaussian_slicer_smoke.log 2>&1 || {
    cat /tmp/gaussian_slicer_smoke.log
    exit 1
}

echo "[red/green] success"
