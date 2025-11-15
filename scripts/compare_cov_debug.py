#!/usr/bin/env python3
"""
Compare the WGSL precalc covariance debug buffer against a CPU oracle.

The script reconstructs Σ' = W2S * Σ * W2Sᵀ for every Gaussian and
reports the maximum absolute difference versus the buffer dumped from
the GPU (`--dump-precalc-debug-raw`). Exit code is non-zero if any
Gaussian exceeds the requested tolerance.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from parity_common import (
    Vec3,
    compute_precalc_for_gaussian,
    load_float_array,
    load_gaussians,
    load_kernel_config,
)

COV_DEBUG_STRIDE_BYTES = 240
COV_DEBUG_FLOATS = COV_DEBUG_STRIDE_BYTES // 4  # 60
DEFAULT_TOLERANCE = 1e-6


@dataclass
class DebugEntry:
    cov_cols: Tuple[Vec3, Vec3, Vec3]
    basis_cols: Tuple[Vec3, Vec3, Vec3]
    world_to_slice_cols: Tuple[Vec3, Vec3, Vec3]
    cov_times_rot_cols: Tuple[Vec3, Vec3, Vec3]
    dot_rows: Tuple[Vec3, Vec3, Vec3]


def load_precalc_debug(path: Path, expected: int) -> List[DebugEntry]:
    floats = load_float_array(path)
    if len(floats) % COV_DEBUG_FLOATS != 0:
        raise ValueError(
            f"{path} length {len(floats)} floats is not a whole number of CovarianceDebug records"
        )
    count = len(floats) // COV_DEBUG_FLOATS
    if count < expected:
        raise ValueError(
            f"{path} only contains {count} records but {expected} gaussians were provided"
        )
    debug_entries: List[DebugEntry] = []
    for idx in range(expected):
        start = idx * COV_DEBUG_FLOATS
        cov_cols: List[Vec3] = []
        basis_cols: List[Vec3] = []
        w2s_cols: List[Vec3] = []
        rotw_cols: List[Vec3] = []
        for c in range(3):
            offset = start + c * 4
            cov_cols.append(tuple(floats[offset : offset + 3]))  # type: ignore[assignment]
        for c in range(3):
            offset = start + 12 + c * 4
            basis_cols.append(tuple(floats[offset : offset + 3]))  # type: ignore[assignment]
        for c in range(3):
            offset = start + 24 + c * 4
            w2s_cols.append(tuple(floats[offset : offset + 3]))  # type: ignore[assignment]
        for c in range(3):
            offset = start + 36 + c * 4
            rotw_cols.append(tuple(floats[offset : offset + 3]))  # type: ignore[assignment]
        dot_rows: List[Vec3] = []
        for c in range(3):
            offset = start + 48 + c * 4
            dot_rows.append(tuple(floats[offset : offset + 3]))  # type: ignore[assignment]
        debug_entries.append(
            DebugEntry(
                cov_cols=tuple(cov_cols),
                basis_cols=tuple(basis_cols),
                world_to_slice_cols=tuple(w2s_cols),
                cov_times_rot_cols=tuple(rotw_cols),
                dot_rows=tuple(dot_rows),
            )
        )
    return debug_entries


def matrix_from_cols(cols: Sequence[Vec3]) -> List[List[float]]:
    return [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ]


def max_abs_diff(a: Sequence[Vec3], b: Sequence[Vec3]) -> float:
    return max(abs(a[c][r] - b[c][r]) for c in range(3) for r in range(3))


def max_symmetry_error(cols: Sequence[Vec3]) -> float:
    mat = matrix_from_cols(cols)
    err = 0.0
    for i in range(3):
        for j in range(3):
            err = max(err, abs(mat[i][j] - mat[j][i]))
    return err


def max_row_diff(cols: Sequence[Vec3], dot_rows: Sequence[Vec3]) -> float:
    mat = matrix_from_cols(cols)
    err = 0.0
    for row in range(3):
        for col in range(3):
            err = max(err, abs(mat[row][col] - dot_rows[row][col]))
    return err


def format_vec3(v: Vec3) -> str:
    return f"({v[0]: .5e}, {v[1]: .5e}, {v[2]: .5e})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WGSL precalc debug buffer with CPU reference.")
    parser.add_argument("--gaussians", type=Path, required=True, help="Path to --dump-gaussians-raw output")
    parser.add_argument("--precalc-debug", type=Path, required=True, help="Path to --dump-precalc-debug-raw output")
    parser.add_argument("--kernel-config", type=Path, required=True, help="Path to --dump-kernel-config-raw output")
    parser.add_argument("--limit", type=int, default=None, help="Only compare the first N gaussians")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Maximum allowed absolute diff")
    parser.add_argument("--top", type=int, default=5, help="Print details for the N worst offenders")
    parser.add_argument(
        "--show-basis",
        action="store_true",
        help="Dump the slice-to-world basis columns captured in the debug buffer.",
    )
    parser.add_argument(
        "--show-intermediates",
        action="store_true",
        help="Dump world-to-slice columns and covariance*rotation intermediates.",
    )
    parser.add_argument(
        "--check-symmetry",
        action="store_true",
        help="Report symmetry error and dot-product reconstruction error for each offending gaussian.",
    )
    args = parser.parse_args()

    gaussians = load_gaussians(args.gaussians)
    kernel = load_kernel_config(args.kernel_config)
    debug_entries = load_precalc_debug(args.precalc_debug, len(gaussians))

    if kernel.num_distributions != len(gaussians):
        print(
            f"warning: kernel config expects {kernel.num_distributions} gaussians but dump has {len(gaussians)}",
            file=sys.stderr,
        )

    total = len(gaussians)
    sample_count = args.limit if args.limit is not None else total
    sample_count = min(sample_count, total)

    failures: List[Tuple[int, float, Tuple[Vec3, Vec3, Vec3], Tuple[Vec3, Vec3, Vec3]]] = []
    worst = 0.0
    for idx in range(sample_count):
        precalc = compute_precalc_for_gaussian(gaussians[idx], kernel)
        cpu_cols = precalc.cov_cols
        gpu_cols = debug_entries[idx].cov_cols
        diff = max_abs_diff(cpu_cols, gpu_cols)
        worst = max(worst, diff)
        if diff > args.tolerance:
            failures.append((idx, diff, cpu_cols, gpu_cols))

    mismatch_count = len(failures)
    print(
        f"Compared {sample_count}/{total} gaussians "
        f"(max_abs_diff={worst:.3e}, tolerance={args.tolerance:.1e})."
    )
    if mismatch_count == 0:
        print("All entries are within tolerance.")
        return

    print(f"{mismatch_count} gaussians exceed tolerance:")
    failures.sort(key=lambda item: item[1], reverse=True)
    for idx, (gauss_idx, diff, cpu_cols, gpu_cols) in enumerate(failures[: args.top]):
        print(
            f"  #{idx + 1}: gaussian {gauss_idx} diff={diff:.3e}\n"
            f"    CPU cols: {', '.join(format_vec3(col) for col in cpu_cols)}\n"
            f"    GPU cols: {', '.join(format_vec3(col) for col in gpu_cols)}"
        )
        if args.show_basis:
            basis = debug_entries[gauss_idx].basis_cols
            print(
                f"    GPU basis cols: {', '.join(format_vec3(col) for col in basis)}"
            )
        if args.show_intermediates:
            w2s = debug_entries[gauss_idx].world_to_slice_cols
            rotw = debug_entries[gauss_idx].cov_times_rot_cols
            print(
                f"    GPU W2S cols: {', '.join(format_vec3(col) for col in w2s)}\n"
                f"    GPU cov*S cols: {', '.join(format_vec3(col) for col in rotw)}"
            )
        if args.check_symmetry:
            sym_err = max_symmetry_error(gpu_cols)
            row_err = max_row_diff(gpu_cols, debug_entries[gauss_idx].dot_rows)
            print(
                f"    Symmetry max error: {sym_err:.3e}; dot reconstruction error: {row_err:.3e}"
            )
    sys.exit(1)


if __name__ == "__main__":
    main()
