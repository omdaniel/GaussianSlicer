// Reference placeholder for the 3x3 SPD Cholesky kernel.
// The implementation will mirror `wgsl/cholesky3_reference.wgsl` once
// we extract the Swift/Metal version from `Sources/GaussianSlicer/Shaders.metal`.

#include <metal_stdlib>
using namespace metal;

struct Mat3Columns {
    float3 c0;
    float3 c1;
    float3 c2;
};

inline float mat_get(const Mat3Columns m, uint r, uint c) {
    return (c == 0 ? m.c0[r] : (c == 1 ? m.c1[r] : m.c2[r]));
}

inline void mat_set(thread Mat3Columns &m, uint r, uint c, float v) {
    if (c == 0) {
        m.c0[r] = v;
    } else if (c == 1) {
        m.c1[r] = v;
    } else {
        m.c2[r] = v;
    }
}

inline Mat3Columns cholesky3(Mat3Columns A) {
    Mat3Columns L = { float3(0.0), float3(0.0), float3(0.0) };
    for (uint i = 0; i < 3; ++i) {
        for (uint j = 0; j <= i; ++j) {
            float sum = mat_get(A, i, j);
            for (uint k = 0; k < j; ++k) {
                sum -= mat_get(L, i, k) * mat_get(L, j, k);
            }
            if (i == j) {
                float d = max(sum, 1e-12f);
                mat_set(L, i, j, sqrt(d));
            } else {
                mat_set(L, i, j, sum / mat_get(L, j, j));
            }
        }
    }
    return L;
}
