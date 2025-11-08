struct ScalarVectorCase {
    alpha: vec4<f32>,
    x: vec4<f32>,
    y: vec4<f32>,
};

struct ScalarVectorResult {
    dot_len: vec4<f32>,
    normalized: vec4<f32>,
    axpy: vec4<f32>,
    flags: vec4<u32>,
};

struct ScalarVectorConfig {
    count: vec4<u32>,
    epsilon: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<ScalarVectorCase>;
@group(0) @binding(1) var<storage, read_write> results: array<ScalarVectorResult>;
@group(0) @binding(2) var<uniform> config: ScalarVectorConfig;

fn normalize_guarded(v: vec3<f32>, eps: f32) -> vec4<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= eps) {
        return vec4<f32>(v, 0.0);
    }
    let inv_len = inverseSqrt(len_sq);
    return vec4<f32>(v * inv_len, 1.0);
}

@compute @workgroup_size(64)
fn scalar_vector_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= config.count.x) {
        return;
    }

    let input_case = cases[gid.x];
    let x = input_case.x.xyz;
    let y = input_case.y.xyz;
    let dot_val = dot(x, y);
    let length_sq = dot(x, x);
    let axpy = fma(vec3<f32>(input_case.alpha.x), x, y);
    let norm = normalize_guarded(x, config.epsilon.x);
    let normalized_vec = norm.xyz;
    let normalized_flag = select(0u, 1u, norm.w > 0.0);

    results[gid.x].dot_len = vec4<f32>(dot_val, length_sq, 0.0, 0.0);
    results[gid.x].normalized = vec4<f32>(normalized_vec, 0.0);
    results[gid.x].axpy = vec4<f32>(axpy, 0.0);
    results[gid.x].flags = vec4<u32>(normalized_flag, 0u, 0u, 0u);
}
