struct Config {
    num_distributions: u32,
    _pad0: vec3<u32>,
    rotation_matrix: mat4x4<f32>,
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
    _pad: vec3<f32>,
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

    let v_norm = 1.0 - v_norm_raw;

    let grid_min = config.grid_params.y;
    let grid_max = config.grid_params.z;
    let grid_range = grid_max - grid_min;
    let u = grid_min + u_norm * grid_range;
    let v = grid_min + v_norm * grid_range;
    let point = vec2<f32>(u, v);

    var total_density = 0.0;

    for (var i: u32 = 0u; i < config.num_distributions; i = i + 1u) {
        let dynamic_entry = dynamic_params[i];
        if (dynamic_entry.combined_factor <= EPSILON) {
            continue;
        }

        let pre = precalc[i];
        let x_mu = point - dynamic_entry.mean2d;

        let inv_col0 = pre.inv_cov2d_col0;
        let inv_col1 = pre.inv_cov2d_col1;
        let transformed = vec2<f32>(
            inv_col0.x * x_mu.x + inv_col1.x * x_mu.y,
            inv_col0.y * x_mu.x + inv_col1.y * x_mu.y
        );

        let mahalanobis_sq = dot(x_mu, transformed);
        total_density = total_density + dynamic_entry.combined_factor * exp(-0.5 * mahalanobis_sq);
    }

    textureStore(
        output_texture,
        vec2<i32>(i32(global_id.x), i32(global_id.y)),
        vec4<f32>(total_density, 0.0, 0.0, 1.0)
    );
}
