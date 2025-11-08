struct Mat3Columns {
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
};

struct Mat3Case {
    a: Mat3Columns,
    b: Mat3Columns,
    v: vec4<f32>,
};

struct Mat3Result {
    a_times_v: vec4<f32>,
    ab_col0: vec4<f32>,
    ab_col1: vec4<f32>,
    ab_col2: vec4<f32>,
    at_col0: vec4<f32>,
    at_col1: vec4<f32>,
    at_col2: vec4<f32>,
};

struct Config {
    count: u32,
    _pad0: vec3<u32>,
};

fn mat_from_columns(cols: Mat3Columns) -> mat3x3<f32> {
    return mat3x3<f32>(cols.c0.xyz, cols.c1.xyz, cols.c2.xyz);
}

fn columns_from_mat(m: mat3x3<f32>) -> Mat3Columns {
    return Mat3Columns(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0));
}

@group(0) @binding(0) var<storage, read> cases: array<Mat3Case>;
@group(0) @binding(1) var<storage, read_write> results: array<Mat3Result>;
@group(0) @binding(2) var<uniform> config: Config;

@compute @workgroup_size(64)
fn mat3_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= config.count) {
        return;
    }
    let input = cases[gid.x];
    let a = mat_from_columns(input.a);
    let b = mat_from_columns(input.b);
    let v = input.v.xyz;

    let vec_out = a * v;
    let ab = a * b;
    let at = transpose(a);

    results[gid.x].a_times_v = vec4<f32>(vec_out, 0.0);
    results[gid.x].ab_col0 = vec4<f32>(ab[0], 0.0);
    results[gid.x].ab_col1 = vec4<f32>(ab[1], 0.0);
    results[gid.x].ab_col2 = vec4<f32>(ab[2], 0.0);
    results[gid.x].at_col0 = vec4<f32>(at[0], 0.0);
    results[gid.x].at_col1 = vec4<f32>(at[1], 0.0);
    results[gid.x].at_col2 = vec4<f32>(at[2], 0.0);
}
