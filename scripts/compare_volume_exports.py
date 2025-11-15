#!/usr/bin/env python3
"""
Compare two exported Gaussian Slicer volumes (RAW + MHD metadata).

Intended for headless parity between the Swift/Metal exporter and the Rust/wgpu
port once the latter gains RAW/MHD export support. Supports full-volume diffs
or deterministic sampling for large grids.
"""

from __future__ import annotations

import argparse
import random
import sys
from array import array
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_TOLERANCE = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two exported density volumes.")
    parser.add_argument("--reference-mhd", type=Path, required=True, help="Baseline (Swift) .mhd header")
    parser.add_argument("--reference-raw", type=Path, required=True, help="Baseline (Swift) .raw data")
    parser.add_argument("--candidate-mhd", type=Path, required=True, help="Rust/wgpu .mhd header")
    parser.add_argument("--candidate-raw", type=Path, required=True, help="Rust/wgpu .raw data")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Max allowed abs diff per voxel")
    parser.add_argument("--sample-count", type=int, default=None, help="Number of voxels to sample (full volume if omitted)")
    parser.add_argument("--top", type=int, default=5, help="Print the worst-N diffs when tolerance is exceeded")
    args = parser.parse_args()

    ref_meta = parse_mhd(args.reference_mhd)
    cand_meta = parse_mhd(args.candidate_mhd)

    dims = parse_dims(ref_meta, description="reference")
    cand_dims = parse_dims(cand_meta, description="candidate")
    if dims != cand_dims:
        sys.exit(f"dimension mismatch: reference {dims} vs candidate {cand_dims}")

    reference = load_raw(args.reference_raw, dims)
    candidate = load_raw(args.candidate_raw, dims)

    total_voxels = len(reference)
    if len(candidate) != total_voxels:
        sys.exit(
            f"candidate raw has {len(candidate)} floats but reference has {total_voxels}"
        )

    sample_count = args.sample_count
    if sample_count is None or sample_count >= total_voxels:
        diffs = [
            (abs(r - c), idx, r, c) for idx, (r, c) in enumerate(zip(reference, candidate))
        ]
        mode = f"full volume ({total_voxels} voxels)"
    else:
        sample_count = max(1, sample_count)
        rng = random.Random(0xDEADBEEF)
        indices = sorted(rng.sample(range(total_voxels), sample_count))
        diffs = [
            (abs(reference[idx] - candidate[idx]), idx, reference[idx], candidate[idx])
            for idx in indices
        ]
        mode = f"{sample_count} sampled voxels"

    diffs.sort(key=lambda entry: entry[0], reverse=True)
    max_diff = diffs[0][0]
    mean_diff = sum(d[0] for d in diffs) / len(diffs)

    print(
        f"Compared {mode} in volume {dims[0]}x{dims[1]}x{dims[2]} "
        f"(max_abs_diff={max_diff:.3e}, mean_abs_diff={mean_diff:.3e}, tolerance={args.tolerance:.1e})."
    )

    if max_diff <= args.tolerance:
        print("Volume within tolerance.")
        return

    print("Top voxel differences (diff, idx, coord, ref, cand):")
    for diff, idx, ref_val, cand_val in diffs[: args.top]:
        coord = flatten_to_coord(idx, dims)
        print(
            f"  Δ={diff:.3e} at {coord}: ref={ref_val:.5e}, cand={cand_val:.5e}"
        )
    sys.exit(1)


def parse_mhd(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    data = path.read_text().splitlines()
    for line in data:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def parse_dims(meta: Dict[str, str], description: str) -> Tuple[int, int, int]:
    raw = meta.get("DimSize")
    if raw is None:
        sys.exit(f"{description} header missing DimSize")
    parts = raw.split()
    if len(parts) != 3:
        sys.exit(f"{description} DimSize must have 3 entries, got {raw}")
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(f"{description} DimSize contains non-integer: {raw}") from exc
    return dims  # type: ignore[return-value]


def load_raw(path: Path, dims: Tuple[int, int, int]) -> List[float]:
    expected_floats = dims[0] * dims[1] * dims[2]
    data = path.read_bytes()
    floats = array("f")
    floats.frombytes(data)
    if floats.itemsize != 4:
        sys.exit("volume RAW must contain 32-bit floats")
    if len(floats) != expected_floats:
        sys.exit(
            f"{path} contains {len(floats)} floats but expected {expected_floats} from DimSize"
        )
    if sys.byteorder != "little":
        floats.byteswap()
    return list(floats)


def flatten_to_coord(idx: int, dims: Tuple[int, int, int]) -> Tuple[int, int, int]:
    width, height, depth = dims
    slice_size = width * height
    z = idx // slice_size
    rem = idx % slice_size
    y = rem // width
    x = rem % width
    return (x, y, z)


if __name__ == "__main__":
    main()
