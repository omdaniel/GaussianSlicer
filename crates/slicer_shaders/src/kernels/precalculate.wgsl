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
    covariance: mat3x3<f32>,
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

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian3D>;
@group(0) @binding(2) var<storage, read_write> precalc: array<PrecalculatedParams>;

const EPSILON: f32 = 1e-6;
const TWO_PI: f32 = 6.28318530717958647692;

fn determinant_2x2(col0: vec2<f32>, col1: vec2<f32>) -> f32 {
    return col0.x * col1.y - col1.x * col0.y;
}

@compute @workgroup_size(64)
fn precalculate_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= config.num_distributions) {
        return;
    }

    let gaussian = gaussians[index];
    let rotation = mat3x3<f32>(
        config.rotation_matrix[0].xyz,
        config.rotation_matrix[1].xyz,
        config.rotation_matrix[2].xyz,
    );
    let cov_prime = rotation * gaussian.covariance * transpose(rotation);

    let cov_uv_col0 = cov_prime[0].xy;
    let cov_uv_col1 = cov_prime[1].xy;
    let c02 = cov_prime[2].x;
    let c12 = cov_prime[2].y;
    let cov_n_n_val = cov_prime[2].z;

    var cov_2d_col0 = cov_uv_col0;
    var cov_2d_col1 = cov_uv_col1;
    var cov_n_n_inv = 0.0;

    if (cov_n_n_val >= EPSILON) {
        cov_n_n_inv = 1.0 / cov_n_n_val;
        let adj_00 = (c02 * c02) * cov_n_n_inv;
        let adj_01 = (c02 * c12) * cov_n_n_inv;
        let adj_11 = (c12 * c12) * cov_n_n_inv;

        cov_2d_col0 = vec2<f32>(
            cov_uv_col0.x - adj_00,
            cov_uv_col0.y - adj_01
        );
        cov_2d_col1 = vec2<f32>(
            cov_uv_col1.x - adj_01,
            cov_uv_col1.y - adj_11
        );
    } else {
        cov_n_n_inv = 0.0;
    }

    let mean_adj = vec2<f32>(c02, c12) * cov_n_n_inv;
    var norm_const = 0.0;
    var inv_cov_col0 = vec2<f32>(0.0, 0.0);
    var inv_cov_col1 = vec2<f32>(0.0, 0.0);

    let det_cov = determinant_2x2(cov_2d_col0, cov_2d_col1);
    if (det_cov > EPSILON) {
        let inv_det = 1.0 / det_cov;
        inv_cov_col0 = vec2<f32>(
            cov_2d_col1.y * inv_det,
            -cov_2d_col1.x * inv_det
        );
        inv_cov_col1 = vec2<f32>(
            -cov_2d_col0.y * inv_det,
            cov_2d_col0.x * inv_det
        );
        norm_const = 1.0 / (TWO_PI * sqrt(det_cov));
    }

    precalc[index].inv_cov2d_col0 = inv_cov_col0;
    precalc[index].inv_cov2d_col1 = inv_cov_col1;
    precalc[index].norm_const2d = norm_const;
    precalc[index].sigma_n_n = cov_n_n_val;
    precalc[index].mean_adj_factor = mean_adj;
}
