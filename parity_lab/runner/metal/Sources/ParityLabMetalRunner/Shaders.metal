#include <metal_stdlib>
using namespace metal;

struct ScalarVectorCaseGpu {
    float4 alpha;
    float4 x;
    float4 y;
};

struct ScalarVectorResultGpu {
    float4 dotLen;
    float4 normalized;
    float4 axpy;
    uint4 flags;
};

struct Config {
    uint count;
    uint _pad0;
    float epsilon;
    float _pad1;
};

inline float3 normalize_guarded(float3 v, float eps, thread bool &didNormalize) {
    float len_sq = dot(v, v);
    if (len_sq <= eps) {
        didNormalize = false;
        return v;
    }
    float inv_len = rsqrt(len_sq);
    didNormalize = true;
    return v * inv_len;
}

kernel void scalarVectorKernel(const device ScalarVectorCaseGpu *cases [[buffer(0)]],
                               device ScalarVectorResultGpu *results [[buffer(1)]],
                               constant Config &config [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= config.count) { return; }
    float3 x = float3(cases[gid].x.xyz);
    float3 y = float3(cases[gid].y.xyz);
    float alpha = cases[gid].alpha.x;
    float dotVal = dot(x, y);
    float lengthSq = dot(x, x);
    float3 axpy = fma(alpha, x, y);
    bool didNormalize = false;
    float3 normalized = normalize_guarded(x, config.epsilon, didNormalize);

    results[gid].dotLen = float4(dotVal, lengthSq, 0.0, 0.0);
    results[gid].normalized = float4(normalized, 0.0);
    results[gid].axpy = float4(axpy, 0.0);
    results[gid].flags = uint4(didNormalize ? 1u : 0u, 0u, 0u, 0u);
}

struct Mat3Columns {
    float4 c0;
    float4 c1;
    float4 c2;
};

struct Mat3CaseGpu {
    Mat3Columns a;
    Mat3Columns b;
    float4 v;
};

struct Mat3ResultGpu {
    float4 aTimesV;
    float4 abCol0;
    float4 abCol1;
    float4 abCol2;
    float4 atCol0;
    float4 atCol1;
    float4 atCol2;
};

inline float3x3 mat_from_columns(Mat3Columns cols) {
    return float3x3(cols.c0.xyz, cols.c1.xyz, cols.c2.xyz);
}

kernel void mat3Kernel(const device Mat3CaseGpu *cases [[buffer(0)]],
                       device Mat3ResultGpu *results [[buffer(1)]],
                       constant Config &config [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= config.count) { return; }
    float3x3 a = mat_from_columns(cases[gid].a);
    float3x3 b = mat_from_columns(cases[gid].b);
    float3 v = cases[gid].v.xyz;

    float3 vec_out = a * v;
    float3x3 ab = a * b;
    float3x3 at = transpose(a);

    results[gid].aTimesV = float4(vec_out, 0.0);
    results[gid].abCol0 = float4(ab[0], 0.0);
    results[gid].abCol1 = float4(ab[1], 0.0);
    results[gid].abCol2 = float4(ab[2], 0.0);
    results[gid].atCol0 = float4(at[0], 0.0);
    results[gid].atCol1 = float4(at[1], 0.0);
    results[gid].atCol2 = float4(at[2], 0.0);
}

struct SpdCaseGpu {
    Mat3Columns a;
    float4 b;
};

struct SpdResultGpu {
    float4 cholCol0;
    float4 cholCol1;
    float4 cholCol2;
    float4 solution;
};

inline float3x3 cholesky3(float3x3 a, float eps) {
    float3x3 l = float3x3(float3(0.0), float3(0.0), float3(0.0));
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j <= i; ++j) {
            float sum = a[i][j];
            for (uint k = 0; k < j; ++k) {
                sum -= l[i][k] * l[j][k];
            }
            if (i == j) {
                l[i][j] = sqrt(max(sum, eps));
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    return l;
}

inline float3 forward_sub(float3x3 l, float3 b) {
    float y0 = b.x / l[0][0];
    float y1 = (b.y - l[1][0] * y0) / l[1][1];
    float y2 = (b.z - l[2][0] * y0 - l[2][1] * y1) / l[2][2];
    return float3(y0, y1, y2);
}

inline float3 backward_sub(float3x3 l, float3 y) {
    float x2 = y.z / l[2][2];
    float x1 = (y.y - l[2][1] * x2) / l[1][1];
    float x0 = (y.x - l[1][0] * x1 - l[2][0] * x2) / l[0][0];
    return float3(x0, x1, x2);
}

kernel void spd3Kernel(const device SpdCaseGpu *cases [[buffer(0)]],
                       device SpdResultGpu *results [[buffer(1)]],
                       constant Config &config [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= config.count) { return; }
    float3x3 a = mat_from_columns(cases[gid].a);
    float3 b = cases[gid].b.xyz;
    float3x3 chol = cholesky3(a, config.epsilon);
    float3 y = forward_sub(chol, b);
    float3 x = backward_sub(chol, y);

    results[gid].cholCol0 = float4(chol[0], 0.0);
    results[gid].cholCol1 = float4(chol[1], 0.0);
    results[gid].cholCol2 = float4(chol[2], 0.0);
    results[gid].solution = float4(x, 0.0);
}
