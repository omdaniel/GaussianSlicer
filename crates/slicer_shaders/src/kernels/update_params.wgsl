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

struct Gaussian3D {
    mean: vec3<f32>,
    _mean_pad: f32,
    covariance_col0: vec4<f32>,
    covariance_col1: vec4<f32>,
    covariance_col2: vec4<f32>,
    weight: f32,
    _pad: vec3<f32>,
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
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read> precalc: array<PrecalculatedParams>;
@group(0) @binding(3) var<storage, read_write> dynamic_params: array<DynamicParams>;

const EPSILON: f32 = 1e-6;
const SQRT_2_PI: f32 = 2.50662827463100050242;

fn build_slice_basis(normal_in: vec3<f32>) -> mat3x3<f32> {
    var normal = normal_in;
    let len_sq = dot(normal, normal);
    if (len_sq <= EPSILON) {
        normal = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        normal = normalize(normal);
    }

    let abs_n = abs(normal);
    var min_axis: u32 = 0u;
    if (abs_n.y < abs_n.x && abs_n.y <= abs_n.z) {
        min_axis = 1u;
    } else if (abs_n.z < abs_n.x && abs_n.z < abs_n.y) {
        min_axis = 2u;
    }

    var arbitrary = vec3<f32>(1.0, 0.0, 0.0);
    if (min_axis == 1u) {
        arbitrary = vec3<f32>(0.0, 1.0, 0.0);
    } else if (min_axis == 2u) {
        arbitrary = vec3<f32>(0.0, 0.0, 1.0);
    }

    var u = cross(normal, arbitrary);
    if (dot(u, u) < EPSILON) {
        if (min_axis == 0u) {
            arbitrary = vec3<f32>(0.0, 1.0, 0.0);
        } else if (min_axis == 1u) {
            arbitrary = vec3<f32>(0.0, 0.0, 1.0);
        } else {
            arbitrary = vec3<f32>(1.0, 0.0, 0.0);
        }
        u = cross(normal, arbitrary);
    }
    u = normalize(u);
    let v = normalize(cross(normal, u));
    return mat3x3<f32>(u, v, normal);
}

@compute @workgroup_size(64)
fn update_params_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= config.num_distributions) {
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
    let slice_to_world = build_slice_basis(plane_normal);
    let row0 = slice_to_world[0];
    let row1 = slice_to_world[1];
    let row2 = slice_to_world[2];
    let mean_shifted = gaussian.mean - plane_point;
    let mean_prime = vec3<f32>(
        dot(row0, mean_shifted),
        dot(row1, mean_shifted),
        dot(row2, mean_shifted),
    );

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

    dynamic_params[index].combined_factor = gaussian.weight * scaling * pre.norm_const2d;
}
