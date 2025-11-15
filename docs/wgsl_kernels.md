# WGSL Kernels (Metal → WGSL)

## crates/slicer_shaders/src/kernels/precalculate.wgsl

```
struct Config {
    num_distributions: vec4<u32>,
    rotation_matrix_cols: array<vec4<f32>, 4>,
    plane_normal: vec4<f32>,
    grid_params: vec4<f32>,
    _reserved0: vec4<f32>,
    _reserved1: vec4<f32>,
    _reserved2: vec4<f32>,
};

struct Gaussian3D {
    mean: vec4<f32>,
    covariance_col0: vec4<f32>,
    covariance_col1: vec4<f32>,
    covariance_col2: vec4<f32>,
    weight_pad: vec4<f32>,
};

struct PrecalculatedParams {
    inv_cov2d_col0: vec2<f32>,
    inv_cov2d_col1: vec2<f32>,
    norm_const2d: f32,
    sigma_n_n: f32,
    mean_adj_factor: vec2<f32>,
    _pad: vec2<f32>,
};

struct CovarianceDebug {
    cov_col0: vec4<f32>,
    cov_col1: vec4<f32>,
    cov_col2: vec4<f32>,
    rot_col0: vec4<f32>,
    rot_col1: vec4<f32>,
    rot_col2: vec4<f32>,
    w2s_col0: vec4<f32>,
    w2s_col1: vec4<f32>,
    w2s_col2: vec4<f32>,
    rotw_col0: vec4<f32>,
    rotw_col1: vec4<f32>,
    rotw_col2: vec4<f32>,
    dot_col0: vec4<f32>,
    dot_col1: vec4<f32>,
    dot_col2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read_write> precalc: array<PrecalculatedParams>;
@group(0) @binding(3) var<storage, read_write> precalc_debug: array<CovarianceDebug>;

const EPSILON: f32 = 1e-6;
const TWO_PI: f32 = 6.28318530717958647692;

fn determinant_2x2(col0: vec2<f32>, col1: vec2<f32>) -> f32 {
    return col0.x * col1.y - col1.x * col0.y;
}

fn world_to_slice_matrix() -> mat3x3<f32> {
    let col0 = config.rotation_matrix_cols[0].xyz;
    let col1 = config.rotation_matrix_cols[1].xyz;
    let col2 = config.rotation_matrix_cols[2].xyz;
    return mat3x3<f32>(col0, col1, col2);
}

@compute @workgroup_size(64)
fn precalculate_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= config.num_distributions.x) {
        return;
    }

    let gaussian = gaussians[index];
    let world_to_slice = world_to_slice_matrix();
    let slice_to_world = transpose(world_to_slice);

    let covariance = mat3x3<f32>(
        gaussian.covariance_col0.xyz,
        gaussian.covariance_col1.xyz,
        gaussian.covariance_col2.xyz,
    );

    let cov_prime = world_to_slice * covariance * slice_to_world;

    precalc_debug[index].cov_col0 = vec4<f32>(cov_prime[0], 0.0);
    precalc_debug[index].cov_col1 = vec4<f32>(cov_prime[1], 0.0);
    precalc_debug[index].cov_col2 = vec4<f32>(cov_prime[2], 0.0);
    precalc_debug[index].rot_col0 = vec4<f32>(slice_to_world[0], 0.0);
    precalc_debug[index].rot_col1 = vec4<f32>(slice_to_world[1], 0.0);
    precalc_debug[index].rot_col2 = vec4<f32>(slice_to_world[2], 0.0);
    precalc_debug[index].w2s_col0 = vec4<f32>(world_to_slice[0], 0.0);
    precalc_debug[index].w2s_col1 = vec4<f32>(world_to_slice[1], 0.0);
    precalc_debug[index].w2s_col2 = vec4<f32>(world_to_slice[2], 0.0);

    let rot_times_cov_col0 = covariance * slice_to_world[0];
    let rot_times_cov_col1 = covariance * slice_to_world[1];
    let rot_times_cov_col2 = covariance * slice_to_world[2];
    precalc_debug[index].rotw_col0 = vec4<f32>(rot_times_cov_col0, 0.0);
    precalc_debug[index].rotw_col1 = vec4<f32>(rot_times_cov_col1, 0.0);
    precalc_debug[index].rotw_col2 = vec4<f32>(rot_times_cov_col2, 0.0);

    let w2s_row0 = vec3<f32>(world_to_slice[0][0], world_to_slice[1][0], world_to_slice[2][0]);
    let w2s_row1 = vec3<f32>(world_to_slice[0][1], world_to_slice[1][1], world_to_slice[2][1]);
    let w2s_row2 = vec3<f32>(world_to_slice[0][2], world_to_slice[1][2], world_to_slice[2][2]);

    let dot_row0 = vec3<f32>(
        dot(w2s_row0, rot_times_cov_col0),
        dot(w2s_row0, rot_times_cov_col1),
        dot(w2s_row0, rot_times_cov_col2)
    );

    let dot_row1 = vec3<f32>(
        dot(w2s_row1, rot_times_cov_col0),
        dot(w2s_row1, rot_times_cov_col1),
        dot(w2s_row1, rot_times_cov_col2)
    );

    let dot_row2 = vec3<f32>(
        dot(w2s_row2, rot_times_cov_col0),
        dot(w2s_row2, rot_times_cov_col1),
        dot(w2s_row2, rot_times_cov_col2)
    );

    precalc_debug[index].dot_col0 = vec4<f32>(dot_row0, 0.0);
    precalc_debug[index].dot_col1 = vec4<f32>(dot_row1, 0.0);
    precalc_debug[index].dot_col2 = vec4<f32>(dot_row2, 0.0);

    let cov_uv_col0 = vec2<f32>(cov_prime[0][0], cov_prime[1][0]);
    let cov_uv_col1 = vec2<f32>(cov_prime[0][1], cov_prime[1][1]);
    let c02 = cov_prime[0][2];
    let c12 = cov_prime[1][2];
    let cov_n_n = cov_prime[2][2];

    var inv_cov2d_col0 = vec2<f32>(0.0, 0.0);
    var inv_cov2d_col1 = vec2<f32>(0.0, 0.0);
    var mean_adj = vec2<f32>(0.0, 0.0);
    var cov_2d_col0 = cov_uv_col0;
    var cov_2d_col1 = cov_uv_col1;

    if (cov_n_n >= EPSILON) {
        let inv_cov_n = 1.0 / cov_n_n;
        let adj_00 = c02 * c02 * inv_cov_n;
        let adj_01 = c02 * c12 * inv_cov_n;
        let adj_11 = c12 * c12 * inv_cov_n;
        cov_2d_col0 = vec2<f32>(cov_uv_col0.x - adj_00, cov_uv_col0.y - adj_01);
        cov_2d_col1 = vec2<f32>(cov_uv_col1.x - adj_01, cov_uv_col1.y - adj_11);
        mean_adj = vec2<f32>(c02 * inv_cov_n, c12 * inv_cov_n);
    }

    let det_cov = determinant_2x2(cov_2d_col0, cov_2d_col1);
    var norm_const = 0.0;
    if (det_cov > EPSILON) {
        let inv_det = 1.0 / det_cov;
        inv_cov2d_col0 = vec2<f32>(cov_2d_col1.y * inv_det, -cov_2d_col1.x * inv_det);
        inv_cov2d_col1 = vec2<f32>(-cov_2d_col0.y * inv_det, cov_2d_col0.x * inv_det);
        norm_const = 1.0 / (TWO_PI * sqrt(det_cov));
    }

    precalc[index].inv_cov2d_col0 = inv_cov2d_col0;
    precalc[index].inv_cov2d_col1 = inv_cov2d_col1;
    precalc[index].sigma_n_n = cov_n_n;
    precalc[index].norm_const2d = norm_const;
    precalc[index].mean_adj_factor = mean_adj;
}
```

## crates/slicer_shaders/src/kernels/update_params.wgsl

```
struct Config {
    num_distributions: vec4<u32>,
    rotation_matrix_cols: array<vec4<f32>, 4>,
    plane_normal: vec4<f32>,
    grid_params: vec4<f32>,
    _reserved0: vec4<f32>,
    _reserved1: vec4<f32>,
    _reserved2: vec4<f32>,
};

struct Gaussian3D {
    mean: vec4<f32>,
    covariance_col0: vec4<f32>,
    covariance_col1: vec4<f32>,
    covariance_col2: vec4<f32>,
    weight_pad: vec4<f32>,
};

struct PrecalculatedParams {
    inv_cov2d_col0: vec2<f32>,
    inv_cov2d_col1: vec2<f32>,
    norm_const2d: f32,
    sigma_n_n: f32,
    mean_adj_factor: vec2<f32>,
    _pad: vec2<f32>,
};

struct DynamicParams {
    mean2d: vec2<f32>,
    combined_factor: f32,
    _padding0: f32,
    _pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read> precalc: array<PrecalculatedParams>;
@group(0) @binding(3) var<storage, read_write> dynamic_params: array<DynamicParams>;

const EPSILON: f32 = 1e-6;
const SQRT_2_PI: f32 = 2.50662827463100050242;

fn world_to_slice_matrix() -> mat3x3<f32> {
    let col0 = config.rotation_matrix_cols[0].xyz;
    let col1 = config.rotation_matrix_cols[1].xyz;
    let col2 = config.rotation_matrix_cols[2].xyz;
    return mat3x3<f32>(col0, col1, col2);
}

@compute @workgroup_size(64)
fn update_params_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= config.num_distributions.x) {
        return;
    }

    let gaussian = gaussians[index];
    let pre = precalc[index];

    if (pre.norm_const2d == 0.0) {
        dynamic_params[index].combined_factor = 0.0;
        return;
    }

    let plane_normal = config.plane_normal.xyz;
    let plane_point = plane_normal * config.grid_params.x;
    let world_to_slice = world_to_slice_matrix();
    let mean_shifted = gaussian.mean.xyz - plane_point;
    let mean_prime = world_to_slice * mean_shifted;

    let mu_uv = mean_prime.xy;
    let mu_n = mean_prime.z;
    let mean_adj_factor = pre.mean_adj_factor;

    let conditional_mean = mu_uv - mean_adj_factor * mu_n;
    dynamic_params[index].mean2d = conditional_mean;

    var scaling = 0.0;
    let sigma_n_n = pre.sigma_n_n;
    if (sigma_n_n >= EPSILON) {
        let norm_const_1d = 1.0 / (SQRT_2_PI * sqrt(sigma_n_n));
        let exponent = -0.5 * (mu_n * mu_n) / sigma_n_n;
        scaling = norm_const_1d * exp(exponent);
    }

    dynamic_params[index].combined_factor = gaussian.weight_pad.x * scaling * pre.norm_const2d;
}
```

## crates/slicer_shaders/src/kernels/evaluation.wgsl

```
struct Config {
    num_distributions: vec4<u32>,
    rotation_matrix_cols: array<vec4<f32>, 4>,
    plane_normal: vec4<f32>,
    grid_params: vec4<f32>,
    _reserved0: vec4<f32>,
    _reserved1: vec4<f32>,
    _reserved2: vec4<f32>,
};

struct PrecalculatedParams {
    inv_cov2d_col0: vec2<f32>,
    inv_cov2d_col1: vec2<f32>,
    norm_const2d: f32,
    sigma_n_n: f32,
    mean_adj_factor: vec2<f32>,
    _pad: vec2<f32>,
};

struct DynamicParams {
    mean2d: vec2<f32>,
    combined_factor: f32,
    _padding0: f32,
    _pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(2) var<storage, read> precalc: array<PrecalculatedParams>;
@group(0) @binding(3) var<storage, read> dynamic_params: array<DynamicParams>;
@group(0) @binding(4) var output_texture: texture_storage_2d<r32float, write>;

const EPSILON: f32 = 1e-6;

@compute @workgroup_size(8, 8, 1)
fn evaluation_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    let width = dims.x;
    let height = dims.y;

    if (global_id.x >= width || global_id.y >= height) {
        return;
    }

    let width_minus_one = max(width, 1u) - 1u;
    let height_minus_one = max(height, 1u) - 1u;

    var u_norm: f32;
    if (width_minus_one > 0u) {
        u_norm = f32(global_id.x) / f32(width_minus_one);
    } else {
        u_norm = 0.0;
    }

    var v_norm_raw: f32;
    if (height_minus_one > 0u) {
        v_norm_raw = f32(global_id.y) / f32(height_minus_one);
    } else {
        v_norm_raw = 0.0;
    }

    // Metal uses an inverted vertical axis when copying to the drawable, so we mirror here.
    let v_norm = 1.0 - v_norm_raw;

    let grid_min = config.grid_params.y;
    let grid_max = config.grid_params.z;
    let grid_range = grid_max - grid_min;

    let u_coord = grid_min + u_norm * grid_range;
    let v_coord = grid_min + v_norm * grid_range;

    var total_density = 0.0;
    for (var i: u32 = 0u; i < config.num_distributions.x; i = i + 1u) {
        let dyn = dynamic_params[i];
        if (dyn.combined_factor <= EPSILON) {
            continue;
        }

        let pre = precalc[i];
        let x_mu0 = u_coord - dyn.mean2d.x;
        let x_mu1 = v_coord - dyn.mean2d.y;

        let transformed0 = pre.inv_cov2d_col0.x * x_mu0 + pre.inv_cov2d_col1.x * x_mu1;
        let transformed1 = pre.inv_cov2d_col0.y * x_mu0 + pre.inv_cov2d_col1.y * x_mu1;
        let mahalanobis_sq = x_mu0 * transformed0 + x_mu1 * transformed1;
        total_density = total_density + dyn.combined_factor * exp(-0.5 * mahalanobis_sq);
    }

    textureStore(output_texture, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4<f32>(total_density, 0.0, 0.0, 0.0));
}
```

## crates/slicer_shaders/src/kernels/visualize.wgsl

```
// Visualization macros omitted for brevity; this file contains the fullscreen quad vertex shader,
// the fragment shader that samples the r32float density texture, applies log/linear normalization,
// selects the colormap, and outputs RGBA to the swapchain. See the source for full details.
```
