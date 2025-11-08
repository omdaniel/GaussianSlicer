struct Mat3Columns {
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
};

struct SpdCase {
    a: Mat3Columns,
    b: vec4<f32>,
};

struct SpdResult {
    chol_col0: vec4<f32>,
    chol_col1: vec4<f32>,
    chol_col2: vec4<f32>,
    solution: vec4<f32>,
};

struct Config {
    count: u32,
    _pad0: u32,
    epsilon: f32,
    _pad1: f32,
};

fn mat_from_columns(cols: Mat3Columns) -> mat3x3<f32> {
    return mat3x3<f32>(cols.c0.xyz, cols.c1.xyz, cols.c2.xyz);
}

fn columns_from_mat(m: mat3x3<f32>) -> Mat3Columns {
    return Mat3Columns(vec4<f32>(m[0], 0.0), vec4<f32>(m[1], 0.0), vec4<f32>(m[2], 0.0));
}

fn cholesky3(a: mat3x3<f32>, eps: f32) -> mat3x3<f32> {
    var l = mat3x3<f32>(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
    );
    for (var i: u32 = 0u; i < 3u; i++) {
        for (var j: u32 = 0u; j <= i; j++) {
            var sum = a[j][i];
            for (var k: u32 = 0u; k < j; k++) {
                sum = sum - l[k][i] * l[k][j];
            }
            if (i == j) {
                l[j][i] = sqrt(max(sum, eps));
            } else {
                l[j][i] = sum / l[j][j];
            }
        }
    }
    return l;
}

fn forward_substitution(l: mat3x3<f32>, b: vec3<f32>) -> vec3<f32> {
    var y = vec3<f32>(0.0);
    let l00 = l[0][0];
    let l10 = l[0][1];
    let l20 = l[0][2];
    let l11 = l[1][1];
    let l21 = l[1][2];
    let l22 = l[2][2];
    y.x = b.x / l00;
    y.y = (b.y - l10 * y.x) / l11;
    y.z = (b.z - l20 * y.x - l21 * y.y) / l22;
    return y;
}

fn backward_substitution(l: mat3x3<f32>, y: vec3<f32>) -> vec3<f32> {
    var x = vec3<f32>(0.0);
    let l00 = l[0][0];
    let l10 = l[0][1];
    let l20 = l[0][2];
    let l11 = l[1][1];
    let l21 = l[1][2];
    let l22 = l[2][2];
    x.z = y.z / l22;
    x.y = (y.y - l21 * x.z) / l11;
    x.x = (y.x - l10 * x.y - l20 * x.z) / l00;
    return x;
}

@group(0) @binding(0) var<storage, read> cases: array<SpdCase>;
@group(0) @binding(1) var<storage, read_write> results: array<SpdResult>;
@group(0) @binding(2) var<uniform> config: Config;

@compute @workgroup_size(64)
fn spd3_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= config.count) {
        return;
    }
    let input = cases[gid.x];
    let a = mat_from_columns(input.a);
    let b = input.b.xyz;
    let chol = cholesky3(a, config.epsilon);
    let y = forward_substitution(chol, b);
    let x = backward_substitution(chol, y);

    results[gid.x].chol_col0 = vec4<f32>(chol[0], 0.0);
    results[gid.x].chol_col1 = vec4<f32>(chol[1], 0.0);
    results[gid.x].chol_col2 = vec4<f32>(chol[2], 0.0);
    results[gid.x].solution = vec4<f32>(x, 0.0);
}
