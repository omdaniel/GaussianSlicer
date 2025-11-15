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
