#!/usr/bin/env python3
"""
Compare the WGSL density texture (K3) against a CPU reference.

The script recomputes the evaluation kernel on the CPU, using the same
Gaussian inputs/config, and checks the dumped density texture
(`--dump-density-raw`). This complements the K1/K2 parity scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple
import random

from parity_common import (
    compute_all_dynamic,
    compute_all_precalc,
    compute_density_grid,
    compute_density_samples,
    load_density,
    load_gaussians,
    load_kernel_config,
)

DEFAULT_TOLERANCE = 1e-5


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WGSL density texture with CPU reference.")
    parser.add_argument("--gaussians", type=Path, required=True, help="Path to --dump-gaussians-raw output")
    parser.add_argument("--kernel-config", type=Path, required=True, help="Path to --dump-kernel-config-raw output")
    parser.add_argument("--density", type=Path, required=True, help="Path to --dump-density-raw output")
    parser.add_argument("--resolution", type=int, default=None, help="Override grid resolution (auto-detected otherwise)")
    parser.add_argument("--limit-gaussians", type=int, default=None, help="Only use the first N gaussians for CPU reference")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Max allowed absolute diff per texel")
    parser.add_argument("--top", type=int, default=3, help="Print details for the N worst texel deltas")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="If set, randomly sample this many texels instead of diffing the entire grid (speeds up dense scenes).",
    )
    args = parser.parse_args()

    gaussians = load_gaussians(args.gaussians)
    kernel = load_kernel_config(args.kernel_config)
    if kernel.num_distributions != len(gaussians):
        print(
            f"warning: kernel config expects {kernel.num_distributions} gaussians but dump has {len(gaussians)}",
            file=sys.stderr,
        )

    gauss_limit = args.limit_gaussians if args.limit_gaussians is not None else len(gaussians)
    gauss_limit = min(gauss_limit, len(gaussians))
    if gauss_limit < len(gaussians):
        print(f"Using first {gauss_limit} gaussians for CPU reference.", file=sys.stderr)
    gauss_slice = gaussians[:gauss_limit]

    precalc_results = compute_all_precalc(gauss_slice, kernel)
    dynamic_results = compute_all_dynamic(gauss_slice, precalc_results, kernel)

    gpu_resolution, gpu_density = load_density(args.density, args.resolution)
    total_texels = len(gpu_density)
    sample_count = args.sample_count
    if sample_count is None or sample_count >= total_texels:
        cpu_density = compute_density_grid(precalc_results, dynamic_results, kernel, gpu_resolution)
        if len(cpu_density) != len(gpu_density):
            raise ValueError(
                f"CPU density has {len(cpu_density)} samples but GPU dump has {len(gpu_density)}"
            )
        report_full_grid(cpu_density, gpu_density, gpu_resolution, args)
    else:
        sample_count = max(1, sample_count)
        rng = random.Random(0xC0FFEE)
        sample_indices = sorted(rng.sample(range(total_texels), sample_count))
        cpu_samples = compute_density_samples(
            precalc_results, dynamic_results, kernel, gpu_resolution, sample_indices
        )
        report_samples(cpu_samples, gpu_density, gpu_resolution, args)


def report_full_grid(cpu_density, gpu_density, gpu_resolution, args):
    max_diff, max_idx = max_diff_with_index(cpu_density, gpu_density)
    mean_diff = sum(abs(c - g) for c, g in zip(cpu_density, gpu_density)) / len(cpu_density)
    print(
        f"Compared density grid {gpu_resolution}x{gpu_resolution} "
        f"(max_abs_diff={max_diff:.3e}, mean_abs_diff={mean_diff:.3e}, tolerance={args.tolerance:.1e})."
    )
    if max_diff <= args.tolerance:
        print("All texels within tolerance.")
        return
    print("Top texel differences (cpu, gpu, diff, x, y):")
    worst = find_top_diffs(cpu_density, gpu_density, gpu_resolution, args.top)
    for diff, idx, cpu_val, gpu_val in worst:
        x = idx % gpu_resolution
        y = idx // gpu_resolution
        print(f"  Δ={diff:.3e} at ({x}, {y}): CPU={cpu_val:.5e}, GPU={gpu_val:.5e}")
    sys.exit(1)


def report_samples(cpu_samples, gpu_density, gpu_resolution, args):
    diffs = []
    for idx, cpu_val in cpu_samples:
        gpu_val = gpu_density[idx]
        diffs.append((abs(cpu_val - gpu_val), idx, cpu_val, gpu_val))
    if not diffs:
        raise ValueError("no samples were evaluated")
    diffs.sort(key=lambda entry: entry[0], reverse=True)

    max_diff = diffs[0][0]
    mean_diff = sum(d[0] for d in diffs) / len(diffs)
    print(
        f"Compared {len(cpu_samples)} sampled texels out of grid {gpu_resolution}x{gpu_resolution} "
        f"(max_abs_diff={max_diff:.3e}, mean_abs_diff={mean_diff:.3e}, tolerance={args.tolerance:.1e})."
    )
    if max_diff <= args.tolerance:
        print("Sampled texels within tolerance.")
        return
    print("Top sampled differences (cpu, gpu, diff, x, y):")
    for diff, idx, cpu_val, gpu_val in diffs[: args.top]:
        x = idx % gpu_resolution
        y = idx // gpu_resolution
        print(f"  Δ={diff:.3e} at ({x}, {y}): CPU={cpu_val:.5e}, GPU={gpu_val:.5e}")
    sys.exit(1)


def max_diff_with_index(cpu: list[float], gpu: list[float]) -> Tuple[float, int]:
    max_diff = 0.0
    max_idx = -1
    for idx, (c, g) in enumerate(zip(cpu, gpu)):
        diff = abs(c - g)
        if diff > max_diff:
            max_diff = diff
            max_idx = idx
    return max_diff, max_idx


def find_top_diffs(
    cpu: list[float],
    gpu: list[float],
    resolution: int,
    top_n: int,
) -> list[Tuple[float, int, float, float]]:
    diffs = [
        (abs(c - g), idx, c, g)
        for idx, (c, g) in enumerate(zip(cpu, gpu))
    ]
    diffs.sort(key=lambda entry: entry[0], reverse=True)
    return diffs[:top_n]


if __name__ == "__main__":
    main()
