#!/usr/bin/env python3
"""
Shared helpers for Swift↔WGSL parity scripts.

Provides struct loaders, CPU reference implementations for K1/K2/K3, and
utility math helpers so individual comparison scripts can stay focused on
reporting.
"""

from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

GAUSSIAN_STRIDE_BYTES = 80
GAUSSIAN_FLOATS = GAUSSIAN_STRIDE_BYTES // 4  # 20
PRECALC_STRIDE_BYTES = 40
PRECALC_FLOATS = PRECALC_STRIDE_BYTES // 4  # 10
DYNAMIC_STRIDE_BYTES = 32
DYNAMIC_FLOATS = DYNAMIC_STRIDE_BYTES // 4  # 8
KERNEL_CONFIG_BYTES = 160

EPSILON = 1e-6
TWO_PI = 6.28318530717958647692
SQRT_2_PI = 2.50662827463100050242

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Matrix3 = List[List[float]]


@dataclass
class KernelConfigData:
    num_distributions: int
    world_to_slice_columns: Tuple[Vec3, Vec3, Vec3]
    plane_normal: Vec3
    plane_offset: float
    grid_min: float
    grid_max: float

    def world_to_slice_matrix(self) -> Matrix3:
        return mat_from_columns(self.world_to_slice_columns)

    def slice_to_world_matrix(self) -> Matrix3:
        return mat_transpose(self.world_to_slice_matrix())


@dataclass
class GaussianRecord:
    mean: Vec3
    covariance_cols: Tuple[Vec3, Vec3, Vec3]
    weight: float


@dataclass
class PrecalcResult:
    cov_cols: Tuple[Vec3, Vec3, Vec3]
    inv_cov2d_col0: Vec2
    inv_cov2d_col1: Vec2
    norm_const2d: float
    sigma_n_n: float
    mean_adj_factor: Vec2


@dataclass
class DynamicResult:
    mean2d: Vec2
    combined_factor: float


def load_float_array(path: Path) -> array:
    data = path.read_bytes()
    floats = array("f")
    floats.frombytes(data)
    if floats.itemsize != 4:
        raise ValueError("unexpected float size in array()")
    if struct.pack("=f", 1.0) != struct.pack("<f", 1.0):
        floats.byteswap()
    return floats


def load_gaussians(path: Path) -> List[GaussianRecord]:
    floats = load_float_array(path)
    if len(floats) % GAUSSIAN_FLOATS != 0:
        raise ValueError(
            f"{path} length {len(floats)} floats is not a whole number of Gaussian records"
        )
    count = len(floats) // GAUSSIAN_FLOATS
    records: List[GaussianRecord] = []
    for idx in range(count):
        base = idx * GAUSSIAN_FLOATS
        mean = tuple(floats[base : base + 3])  # type: ignore[assignment]
        cursor = base + 4  # skip mean pad
        cols: List[Vec3] = []
        for _ in range(3):
            col = tuple(floats[cursor : cursor + 3])  # type: ignore[assignment]
            cols.append(col)
            cursor += 4  # skip padded w component
        weight = floats[base + 16]
        records.append(GaussianRecord(mean=mean, covariance_cols=(cols[0], cols[1], cols[2]), weight=weight))
    return records


def load_kernel_config(path: Path) -> KernelConfigData:
    data = path.read_bytes()
    if len(data) < KERNEL_CONFIG_BYTES:
        raise ValueError(
            f"{path} is only {len(data)} bytes (need at least {KERNEL_CONFIG_BYTES})"
        )
    num_distributions, _, _, _ = struct.unpack_from("<IIII", data, 0)
    rotation = struct.unpack_from("<16f", data, 16)
    columns = tuple(
        (
            rotation[i + 0],
            rotation[i + 1],
            rotation[i + 2],
        )
        for i in range(0, 12, 4)
    )
    plane_normal = struct.unpack_from("<3f", data, 16 + 16 * 4)
    grid_params = struct.unpack_from("<4f", data, 16 + 16 * 4 + 16)
    return KernelConfigData(
        num_distributions=num_distributions,
        world_to_slice_columns=columns,  # type: ignore[arg-type]
        plane_normal=plane_normal,  # type: ignore[arg-type]
        plane_offset=grid_params[0],
        grid_min=grid_params[1],
        grid_max=grid_params[2],
    )


def mat_from_columns(cols: Sequence[Vec3]) -> Matrix3:
    return [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ]


def mat_transpose(m: Matrix3) -> Matrix3:
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def mat_cols(m: Matrix3) -> Tuple[Vec3, Vec3, Vec3]:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def mat_mul(a: Matrix3, b: Matrix3) -> Matrix3:
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def mat_vec_mul(m: Matrix3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def determinant_2x2(col0: Vec2, col1: Vec2) -> float:
    return col0[0] * col1[1] - col1[0] * col0[1]


def compute_precalc_for_gaussian(
    gaussian: GaussianRecord, kernel: KernelConfigData
) -> PrecalcResult:
    world_to_slice = kernel.world_to_slice_matrix()
    slice_to_world = mat_transpose(world_to_slice)
    cov_world = mat_from_columns(gaussian.covariance_cols)
    cov_prime = mat_mul(mat_mul(world_to_slice, cov_world), slice_to_world)
    cov_cols = mat_cols(cov_prime)

    cov_uv_col0 = (cov_prime[0][0], cov_prime[0][1])
    cov_uv_col1 = (cov_prime[1][0], cov_prime[1][1])
    c02 = cov_prime[2][0]
    c12 = cov_prime[2][1]
    cov_n_n = cov_prime[2][2]

    if cov_n_n >= EPSILON:
        cov_n_inv = 1.0 / cov_n_n
        adj_00 = (c02 * c02) * cov_n_inv
        adj_01 = (c02 * c12) * cov_n_inv
        adj_11 = (c12 * c12) * cov_n_inv
        cov_2d_col0 = (cov_uv_col0[0] - adj_00, cov_uv_col0[1] - adj_01)
        cov_2d_col1 = (cov_uv_col1[0] - adj_01, cov_uv_col1[1] - adj_11)
        mean_adj = (c02 * cov_n_inv, c12 * cov_n_inv)
    else:
        cov_2d_col0 = cov_uv_col0
        cov_2d_col1 = cov_uv_col1
        mean_adj = (0.0, 0.0)

    det_cov = determinant_2x2(cov_2d_col0, cov_2d_col1)
    if det_cov > EPSILON:
        inv_det = 1.0 / det_cov
        inv_cov_col0 = (cov_2d_col1[1] * inv_det, -cov_2d_col1[0] * inv_det)
        inv_cov_col1 = (-cov_2d_col0[1] * inv_det, cov_2d_col0[0] * inv_det)
        norm_const = 1.0 / (TWO_PI * math.sqrt(det_cov))
    else:
        inv_cov_col0 = (0.0, 0.0)
        inv_cov_col1 = (0.0, 0.0)
        norm_const = 0.0

    return PrecalcResult(
        cov_cols=cov_cols,
        inv_cov2d_col0=inv_cov_col0,
        inv_cov2d_col1=inv_cov_col1,
        norm_const2d=norm_const,
        sigma_n_n=cov_n_n,
        mean_adj_factor=mean_adj,
    )


def compute_all_precalc(
    gaussians: Sequence[GaussianRecord], kernel: KernelConfigData
) -> List[PrecalcResult]:
    return [compute_precalc_for_gaussian(g, kernel) for g in gaussians]


def compute_dynamic_for_gaussian(
    gaussian: GaussianRecord, precalc: PrecalcResult, kernel: KernelConfigData
) -> DynamicResult:
    world_to_slice = kernel.world_to_slice_matrix()
    plane_point = tuple(
        kernel.plane_normal[i] * kernel.plane_offset for i in range(3)
    )
    mean_shifted = tuple(gaussian.mean[i] - plane_point[i] for i in range(3))  # type: ignore[misc]
    mean_prime = mat_vec_mul(world_to_slice, mean_shifted)  # type: ignore[arg-type]
    mu_uv = (mean_prime[0], mean_prime[1])
    mu_n = mean_prime[2]
    mean_adj = precalc.mean_adj_factor
    conditional_mean = (
        mu_uv[0] - mean_adj[0] * mu_n,
        mu_uv[1] - mean_adj[1] * mu_n,
    )

    scaling = 0.0
    if precalc.sigma_n_n >= EPSILON:
        norm_const_1d = 1.0 / (SQRT_2_PI * math.sqrt(precalc.sigma_n_n))
        exponent = -0.5 * (mu_n * mu_n) / precalc.sigma_n_n
        scaling = norm_const_1d * math.exp(exponent)

    combined = gaussian.weight * scaling * precalc.norm_const2d
    return DynamicResult(mean2d=conditional_mean, combined_factor=combined)


def compute_all_dynamic(
    gaussians: Sequence[GaussianRecord],
    precalc_results: Sequence[PrecalcResult],
    kernel: KernelConfigData,
) -> List[DynamicResult]:
    return [
        compute_dynamic_for_gaussian(g, p, kernel)
        for g, p in zip(gaussians, precalc_results)
    ]


def compute_density_grid(
    precalc_results: Sequence[PrecalcResult],
    dynamic_results: Sequence[DynamicResult],
    kernel: KernelConfigData,
    resolution: int,
) -> List[float]:
    width = max(resolution, 1)
    height = width
    width_minus_one = max(width, 1) - 1
    height_minus_one = max(height, 1) - 1
    grid_range = kernel.grid_max - kernel.grid_min

    densities: List[float] = []
    for y in range(height):
        if height_minus_one > 0:
            v_norm_raw = y / height_minus_one
        else:
            v_norm_raw = 0.0
        v_norm = 1.0 - v_norm_raw
        v_coord = kernel.grid_min + v_norm * grid_range

        for x in range(width):
            if width_minus_one > 0:
                u_norm = x / width_minus_one
            else:
                u_norm = 0.0
            u_coord = kernel.grid_min + u_norm * grid_range

            densities.append(
                evaluate_density_at_point(precalc_results, dynamic_results, u_coord, v_coord)
            )
    return densities


def compute_density_samples(
    precalc_results: Sequence[PrecalcResult],
    dynamic_results: Sequence[DynamicResult],
    kernel: KernelConfigData,
    resolution: int,
    sample_indices: Sequence[int],
) -> List[Tuple[int, float]]:
    width = max(resolution, 1)
    height = width
    width_minus_one = max(width, 1) - 1
    height_minus_one = max(height, 1) - 1
    grid_range = kernel.grid_max - kernel.grid_min
    total_texels = width * height

    results: List[Tuple[int, float]] = []
    for idx in sample_indices:
        if idx < 0 or idx >= total_texels:
            raise ValueError(f"sample index {idx} outside grid (size {total_texels})")
        x = idx % width
        y = idx // width

        if height_minus_one > 0:
            v_norm_raw = y / height_minus_one
        else:
            v_norm_raw = 0.0
        v_norm = 1.0 - v_norm_raw

        if width_minus_one > 0:
            u_norm = x / width_minus_one
        else:
            u_norm = 0.0

        u_coord = kernel.grid_min + u_norm * grid_range
        v_coord = kernel.grid_min + v_norm * grid_range
        density = evaluate_density_at_point(precalc_results, dynamic_results, u_coord, v_coord)
        results.append((idx, density))
    return results


def evaluate_density_at_point(
    precalc_results: Sequence[PrecalcResult],
    dynamic_results: Sequence[DynamicResult],
    u_coord: float,
    v_coord: float,
) -> float:
    total_density = 0.0
    for pre, dyn in zip(precalc_results, dynamic_results):
        if dyn.combined_factor <= EPSILON:
            continue
        x_mu0 = u_coord - dyn.mean2d[0]
        x_mu1 = v_coord - dyn.mean2d[1]
        inv_col0 = pre.inv_cov2d_col0
        inv_col1 = pre.inv_cov2d_col1
        transformed0 = inv_col0[0] * x_mu0 + inv_col1[0] * x_mu1
        transformed1 = inv_col0[1] * x_mu0 + inv_col1[1] * x_mu1
        mahalanobis_sq = x_mu0 * transformed0 + x_mu1 * transformed1
        total_density += dyn.combined_factor * math.exp(-0.5 * mahalanobis_sq)
    return total_density


def load_dynamic_buffer(path: Path, expected: int) -> List[DynamicResult]:
    floats = load_float_array(path)
    if len(floats) % DYNAMIC_FLOATS != 0:
        raise ValueError(
            f"{path} length {len(floats)} floats is not a whole number of DynamicParams records"
        )
    count = len(floats) // DYNAMIC_FLOATS
    if count < expected:
        raise ValueError(
            f"{path} only contains {count} records but {expected} gaussians were provided"
        )
    results: List[DynamicResult] = []
    for idx in range(expected):
        offset = idx * DYNAMIC_FLOATS
        mean2d = (
            floats[offset + 0],
            floats[offset + 1],
        )
        combined = floats[offset + 2]
        # Remaining floats in the stride are padding/alignment.
        results.append(DynamicResult(mean2d=mean2d, combined_factor=combined))
    return results


def load_density(path: Path, resolution_hint: int | None = None) -> Tuple[int, List[float]]:
    floats = load_float_array(path)
    if resolution_hint is not None:
        if resolution_hint <= 0:
            raise ValueError("resolution must be positive")
        expected = resolution_hint * resolution_hint
        if expected != len(floats):
            raise ValueError(
                f"{path} contains {len(floats)} floats but resolution {resolution_hint} expects {expected}"
            )
        return resolution_hint, list(floats)

    resolution = int(math.isqrt(len(floats)))
    if resolution * resolution != len(floats):
        raise ValueError(
            f"{path} contains {len(floats)} floats which is not a perfect square—specify --resolution explicitly"
        )
    return resolution, list(floats)


def max_abs_diff_vec2(a: Vec2, b: Vec2) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def max_abs_diff_list(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))
