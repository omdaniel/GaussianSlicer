#!/usr/bin/env python3
"""
Compare the WGSL dynamic parameter buffer (K2) against a CPU reference.

The script recomputes the conditional mean and combined factor for every
Gaussian, matching the Metal/Swift logic, and compares it to the data
dumped via `--dump-dynamic-raw`. Use this after K1 parity is green to
validate that K2 outputs stay aligned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from parity_common import (
    DynamicResult,
    compute_all_dynamic,
    compute_all_precalc,
    load_dynamic_buffer,
    load_gaussians,
    load_kernel_config,
    max_abs_diff_vec2,
)

DEFAULT_TOLERANCE = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WGSL dynamic buffer with CPU reference.")
    parser.add_argument("--gaussians", type=Path, required=True, help="Path to --dump-gaussians-raw output")
    parser.add_argument("--kernel-config", type=Path, required=True, help="Path to --dump-kernel-config-raw output")
    parser.add_argument("--dynamic", type=Path, required=True, help="Path to --dump-dynamic-raw output")
    parser.add_argument("--limit", type=int, default=None, help="Only compare the first N gaussians")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Max allowed abs diff")
    parser.add_argument("--top", type=int, default=5, help="Print details for the N worst offenders")
    args = parser.parse_args()

    gaussians = load_gaussians(args.gaussians)
    kernel = load_kernel_config(args.kernel_config)
    limit = args.limit if args.limit is not None else len(gaussians)
    limit = min(limit, len(gaussians))
    if kernel.num_distributions != len(gaussians):
        print(
            f"warning: kernel config expects {kernel.num_distributions} gaussians but dump has {len(gaussians)}",
            file=sys.stderr,
        )

    precalc_results = compute_all_precalc(gaussians, kernel)
    cpu_dynamic = compute_all_dynamic(gaussians, precalc_results, kernel)
    gpu_dynamic = load_dynamic_buffer(args.dynamic, len(gaussians))

    failures: List[Tuple[int, float, float, DynamicResult, DynamicResult]] = []
    worst_mean = 0.0
    worst_combined = 0.0
    for idx in range(limit):
        cpu_entry = cpu_dynamic[idx]
        gpu_entry = gpu_dynamic[idx]
        mean_diff = max_abs_diff_vec2(cpu_entry.mean2d, gpu_entry.mean2d)
        combined_diff = abs(cpu_entry.combined_factor - gpu_entry.combined_factor)
        worst_mean = max(worst_mean, mean_diff)
        worst_combined = max(worst_combined, combined_diff)
        if mean_diff > args.tolerance or combined_diff > args.tolerance:
            failures.append((idx, mean_diff, combined_diff, cpu_entry, gpu_entry))

    mismatch_count = len(failures)
    print(
        f"Compared {limit}/{len(gaussians)} gaussians "
        f"(mean diff={worst_mean:.3e}, combined diff={worst_combined:.3e}, tolerance={args.tolerance:.1e})."
    )
    if mismatch_count == 0:
        print("All entries are within tolerance.")
        return

    print(f"{mismatch_count} gaussians exceed tolerance:")
    failures.sort(key=lambda item: max(item[1], item[2]), reverse=True)
    for rank, (idx, mean_diff, combined_diff, cpu_entry, gpu_entry) in enumerate(failures[: args.top], start=1):
        print(
            f"  #{rank}: gaussian {idx} mean_diff={mean_diff:.3e} combined_diff={combined_diff:.3e}\n"
            f"    CPU mean2d=({cpu_entry.mean2d[0]: .5e}, {cpu_entry.mean2d[1]: .5e}) "
            f"combined={cpu_entry.combined_factor: .5e}\n"
            f"    GPU mean2d=({gpu_entry.mean2d[0]: .5e}, {gpu_entry.mean2d[1]: .5e}) "
            f"combined={gpu_entry.combined_factor: .5e}"
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
