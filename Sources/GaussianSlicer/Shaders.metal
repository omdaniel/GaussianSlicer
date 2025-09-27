// Shaders.metal
#include <metal_stdlib>
using namespace metal;

// Constants
constant float PI = 3.14159265358979323846;
constant float TWO_PI = 6.28318530717958647692;
constant float SQRT_2_PI = 2.50662827463100050242;
// Epsilon for float32 stability checks
constant float EPSILON = 1e-6;

// Perceptually uniform / diverging gradients (sampled from MatPlotLib definitions)
constant uint COLORMAP_LUT_SIZE = 11;
constant float3 PLASMA_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.05098, 0.03137, 0.52941),
    float3(0.25490, 0.01569, 0.61569),
    float3(0.41569, 0.00000, 0.65882),
    float3(0.56078, 0.05098, 0.64314),
    float3(0.69412, 0.16471, 0.56471),
    float3(0.79608, 0.27451, 0.47451),
    float3(0.88235, 0.39216, 0.38431),
    float3(0.94902, 0.51765, 0.29412),
    float3(0.98824, 0.65098, 0.21176),
    float3(0.98824, 0.80784, 0.14510),
    float3(0.94118, 0.97647, 0.12941)
};

constant float3 VIRIDIS_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.26700, 0.00487, 0.32941),
    float3(0.28233, 0.09495, 0.41733),
    float3(0.25394, 0.26214, 0.52942),
    float3(0.20676, 0.37176, 0.55312),
    float3(0.16386, 0.47113, 0.55815),
    float3(0.12757, 0.56695, 0.55056),
    float3(0.13451, 0.65864, 0.51765),
    float3(0.26694, 0.74875, 0.44142),
    float3(0.47750, 0.82144, 0.31820),
    float3(0.74139, 0.87345, 0.14956),
    float3(0.99325, 0.90616, 0.14394)
};

constant float3 MAGMA_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.00146, 0.00047, 0.01387),
    float3(0.06354, 0.02843, 0.22780),
    float3(0.18892, 0.07113, 0.34727),
    float3(0.30637, 0.09495, 0.45728),
    float3(0.42211, 0.11195, 0.54656),
    float3(0.53333, 0.13227, 0.61566),
    float3(0.64302, 0.15958, 0.65999),
    float3(0.75488, 0.20196, 0.67849),
    float3(0.86501, 0.25394, 0.67237),
    float3(0.96624, 0.32135, 0.63849),
    float3(0.98705, 0.99144, 0.74950)
};

constant float3 INFERNO_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.00146, 0.00047, 0.01387),
    float3(0.07151, 0.01009, 0.26635),
    float3(0.21611, 0.03813, 0.50565),
    float3(0.37778, 0.09112, 0.66951),
    float3(0.52998, 0.15218, 0.78461),
    float3(0.67488, 0.21183, 0.84465),
    float3(0.81636, 0.27339, 0.86441),
    float3(0.93801, 0.42281, 0.84333),
    float3(0.99220, 0.63756, 0.80887),
    float3(0.99734, 0.85918, 0.79621),
    float3(0.98836, 0.99836, 0.64492)
};

constant float3 TURBO_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.18995, 0.07176, 0.23217),
    float3(0.25107, 0.25237, 0.62831),
    float3(0.27628, 0.47672, 0.85988),
    float3(0.20724, 0.68018, 0.85836),
    float3(0.15340, 0.86658, 0.70504),
    float3(0.36921, 0.93261, 0.40213),
    float3(0.67801, 0.86555, 0.15093),
    float3(0.92512, 0.70505, 0.05083),
    float3(0.99673, 0.49712, 0.13476),
    float3(0.98053, 0.28718, 0.38312),
    float3(0.89490, 0.10287, 0.53074)
};

constant float3 COOLWARM_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.22980, 0.29872, 0.75368),
    float3(0.26679, 0.43538, 0.82257),
    float3(0.30659, 0.57122, 0.87068),
    float3(0.34981, 0.70415, 0.89244),
    float3(0.39504, 0.83206, 0.88175),
    float3(0.75475, 0.90930, 0.74860),
    float3(0.88245, 0.77634, 0.51465),
    float3(0.90493, 0.63036, 0.31846),
    float3(0.89376, 0.47136, 0.16899),
    float3(0.84433, 0.30354, 0.07442),
    float3(0.78039, 0.12619, 0.01519)
};

constant float3 BLUEORANGE_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.034, 0.113, 0.345),
    float3(0.076, 0.295, 0.582),
    float3(0.156, 0.479, 0.741),
    float3(0.274, 0.647, 0.818),
    float3(0.445, 0.791, 0.827),
    float3(0.659, 0.859, 0.749),
    float3(0.835, 0.812, 0.588),
    float3(0.938, 0.682, 0.402),
    float3(0.964, 0.482, 0.259),
    float3(0.916, 0.262, 0.170),
    float3(0.800, 0.063, 0.119)
};

constant float3 SEISMIC_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.000, 0.000, 0.300),
    float3(0.000, 0.000, 0.700),
    float3(0.000, 0.400, 1.000),
    float3(0.200, 0.700, 1.000),
    float3(0.500, 0.900, 1.000),
    float3(0.800, 0.800, 0.800),
    float3(1.000, 0.600, 0.600),
    float3(1.000, 0.300, 0.200),
    float3(0.900, 0.000, 0.000),
    float3(0.600, 0.000, 0.000),
    float3(0.300, 0.000, 0.000)
};

constant float3 YLORRD_LUT[COLORMAP_LUT_SIZE] = {
    float3(1.00000, 1.00000, 0.89804),
    float3(1.00000, 0.96863, 0.73725),
    float3(0.99608, 0.89020, 0.56863),
    float3(0.99346, 0.81569, 0.42353),
    float3(0.98824, 0.73333, 0.31078),
    float3(0.98280, 0.65098, 0.20392),
    float3(0.95705, 0.54510, 0.13725),
    float3(0.90588, 0.41569, 0.12157),
    float3(0.85098, 0.28235, 0.09412),
    float3(0.73725, 0.17647, 0.06275),
    float3(0.49804, 0.00000, 0.00000)
};

constant float3 HOT_LUT[COLORMAP_LUT_SIZE] = {
    float3(0.00000, 0.00000, 0.00000),
    float3(0.20000, 0.00000, 0.00000),
    float3(0.40000, 0.00000, 0.00000),
    float3(0.60000, 0.00000, 0.00000),
    float3(0.80000, 0.00000, 0.00000),
    float3(1.00000, 0.20000, 0.00000),
    float3(1.00000, 0.40000, 0.00000),
    float3(1.00000, 0.60000, 0.00000),
    float3(1.00000, 0.80000, 0.00000),
    float3(1.00000, 0.90000, 0.20000),
    float3(1.00000, 1.00000, 1.00000)
};

// Data Structures matching Swift definitions
struct Gaussian3D {
    float3 mean;
    float3x3 covariance;
    float weight;
};

struct PrecalculatedParams {
    float2x2 invCov2D;
    float normConst2D;
    float sigma_n_n;
    float2 meanAdjFactor;
};

struct DynamicParams {
    float2 mean2D;
    float combinedFactor;
};

// Configuration structure (Uniforms)
struct Config {
    uint numDistributions;
    float3x3 rotationMatrix;
   float3 planeNormal;
    float planeOffset;
    float gridMin;
    float gridMax;
};

struct VisualizationConfig {
    uint colormapIndex;
    uint invert;
    uint logScale;
    uint colorLevels;
    float densityMin;
    float densityMax;
    float outlineWidth;
};

// ==============================================================================
// KERNEL 1: PRE-CALCULATION (Orientation Dependent)
// ==============================================================================
// Runs once. Parallelized over N.

kernel void precalculateKernel(constant Config &config [[buffer(0)]],
                               device const Gaussian3D *gaussians [[buffer(1)]],
                               device PrecalculatedParams *precalcParams [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= config.numDistributions) { return; }

    const Gaussian3D G = gaussians[gid];
    const float3x3 R = config.rotationMatrix;

    // 1. Rotate Covariance: C' = R * C * R^T
    float3x3 cov_prime = R * G.covariance * transpose(R);

    // 2. Partitioning
    // Metal matrices (floatNxM) are column-major. M[col][row].
    
    // Top-left 2x2 block (cov_uv_uv). float2x2 constructor takes column vectors.
    float2x2 cov_uv_uv = float2x2(cov_prime[0].xy, cov_prime[1].xy);

    // Cross-terms: Sigma_uv_n.
    float c02 = cov_prime[2][0]; // Col 2, Row 0
    float c12 = cov_prime[2][1]; // Col 2, Row 1
    
    // Variance along normal: Sigma_n_n. C'[2,2].
    float cov_n_n_val = cov_prime[2][2];
    
    precalcParams[gid].sigma_n_n = cov_n_n_val;

    // 3. Calculate Conditional Covariance (Sigma_2D)
    float2x2 cov_2d;
    float cov_n_n_inv;

    if (cov_n_n_val < EPSILON) {
        // Handle near-zero variance along normal: use marginal covariance
        cov_2d = cov_uv_uv;
        cov_n_n_inv = 0.0;
    } else {
        cov_n_n_inv = 1.0 / cov_n_n_val;
        // Adjustment: (C_uv_n * C_n_uv) * C_n_n_inv (Outer product)
        float adj_00 = (c02 * c02) * cov_n_n_inv;
        float adj_01 = (c02 * c12) * cov_n_n_inv;
        float adj_11 = (c12 * c12) * cov_n_n_inv;

        // Apply adjustment (Column-major access)
        cov_2d[0][0] = cov_uv_uv[0][0] - adj_00;
        cov_2d[1][0] = cov_uv_uv[1][0] - adj_01;
        cov_2d[0][1] = cov_uv_uv[0][1] - adj_01; // Symmetric
        cov_2d[1][1] = cov_uv_uv[1][1] - adj_11;
    }

    // 4. Store intermediate adjustment factors (A)
    // A = Sigma_{uv,n} * Sigma_{n,n}^-1
    precalcParams[gid].meanAdjFactor = float2(c02, c12) * cov_n_n_inv;

    // 5. Invert Sigma_2D and calculate normalization constant
    float det_cov = determinant(cov_2d);
    
    if (det_cov > EPSILON) {
        float inv_det = 1.0 / det_cov;
        // Manual 2x2 inverse to avoid relying on unavailable standard library helpers
        float2 column0 = float2(cov_2d[1][1], -cov_2d[1][0]);
        float2 column1 = float2(-cov_2d[0][1], cov_2d[0][0]);
        precalcParams[gid].invCov2D = float2x2(column0 * inv_det, column1 * inv_det);
        // Norm const for 2D PDF: 1 / (2*pi*sqrt(det))
        precalcParams[gid].normConst2D = 1.0 / (TWO_PI * sqrt(det_cov));
    } else {
        // Mark as invalid
        precalcParams[gid].invCov2D = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));
        precalcParams[gid].normConst2D = 0.0;
    }
}

// ==============================================================================
// KERNEL 2: UPDATE PARAMETERS (Offset Dependent)
// ==============================================================================
// Runs every frame. Parallelized over N.

kernel void updateParamsKernel(constant Config &config [[buffer(0)]],
                               device const Gaussian3D *gaussians [[buffer(1)]],
                               device const PrecalculatedParams *precalcParams [[buffer(2)]],
                               device DynamicParams *dynamicParams [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= config.numDistributions) { return; }

    const Gaussian3D G = gaussians[gid];
    const PrecalculatedParams P = precalcParams[gid];
    const float3x3 R = config.rotationMatrix;

    // Optimization: If pre-calculation failed, skip update.
    if (P.normConst2D == 0.0) {
        dynamicParams[gid].combinedFactor = 0.0;
        return;
    }

    // 1. Calculate the current plane point (origin of the slice)
    // Assumes initial plane point is (0,0,0)
    float3 plane_point = config.planeNormal * config.planeOffset;

    // 2. Calculate Rotated Mean relative to the new plane point
    // mean_prime = R * (mean_3d - plane_point)
    float3 mean_shifted = G.mean - plane_point;
    float3 mean_prime = R * mean_shifted;
    float2 mu_uv = mean_prime.xy;
    float mu_n = mean_prime.z;

    // 3. Update Conditional Mean (mu_2D)
    // mu_2D = mu_uv - A * mu_n
    dynamicParams[gid].mean2D = mu_uv - P.meanAdjFactor * mu_n;

    // 4. Calculate Scaling Factor S (Marginal 1D PDF evaluated at 0)
    float S = 0.0;
    float snn = P.sigma_n_n;
    
    if (snn >= EPSILON) {
        float norm_const_1d = 1.0 / (SQRT_2_PI * sqrt(snn));
        float exponent = -0.5 * (mu_n * mu_n) / snn;
        S = norm_const_1d * exp(exponent);
    }
    
    // 5. Calculate Fused Factor (Weight * S * NormConst_2D)
    dynamicParams[gid].combinedFactor = G.weight * S * P.normConst2D;
}

// ==============================================================================
// KERNEL 3: EVALUATION
// ==============================================================================
// Runs every frame. Parallelized over G (Grid Points).

kernel void evaluationKernel(constant Config &config [[buffer(0)]],
                             device const PrecalculatedParams *precalcParams [[buffer(2)]],
                             device const DynamicParams *dynamicParams [[buffer(3)]],
                             texture2d<float, access::write> outputTexture [[texture(0)]],
                             uint2 tid [[thread_position_in_grid]])
{
    uint width = outputTexture.get_width();
    uint height = outputTexture.get_height();

    if (tid.x >= width || tid.y >= height) { return; }

    // 1. Determine 2D grid coordinates (u, v)
    float u_norm = (float)tid.x / (float)(width - 1);
    // Invert V coordinate for visualization (Metal textures typically use top-left origin,
    // but we want (0,0) at bottom-left for the math visualization)
    float v_norm = 1.0 - (float)tid.y / (float)(height - 1);
    
    float gridRange = config.gridMax - config.gridMin;
    float u = config.gridMin + u_norm * gridRange;
    float v = config.gridMin + v_norm * gridRange;
    
    float2 point = float2(u, v);
    float total_density = 0.0;

    // 2. Loop over all distributions
    for (uint i = 0; i < config.numDistributions; ++i) {
        const DynamicParams D = dynamicParams[i];
        
        // Optimization: Skip if factor is negligible
        if (D.combinedFactor <= EPSILON) { continue; }
        
        const PrecalculatedParams P = precalcParams[i];

        // 3. Calculate Mahalanobis distance: (x-mu)^T * InvCov * (x-mu)
        float2 x_mu = point - D.mean2D;
        float2x2 invCov = P.invCov2D;
        
        // Optimized quadratic form for 2x2 symmetric matrix (Manually unrolled)
        // A*u^2 + D*v^2 + 2*B*u*v
        // Use Fused Multiply-Add (FMA) which is efficient on Apple Silicon.
        float m_sq = fma(invCov[0][0], x_mu.x*x_mu.x, 0.0);
        m_sq = fma(invCov[1][1], x_mu.y*x_mu.y, m_sq);
        // invCov[0][1] == invCov[1][0] due to symmetry.
        m_sq = fma(2.0 * invCov[0][1], x_mu.x*x_mu.y, m_sq);

        // 4. PDF calculation: CombinedFactor * exp(-0.5 * m_sq)
        total_density += D.combinedFactor * exp(-0.5 * m_sq);
    }

    // Write the density value to the texture
    outputTexture.write(float4(total_density, 0.0, 0.0, 1.0), tid);
}


// ==============================================================================
// VISUALIZATION SHADERS (Rendering the texture)
// ==============================================================================

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Vertex shader for a full-screen quad
vertex VertexOut vertexShader(uint vid [[vertex_id]]) {
    // Vertices covering the viewport (-1 to 1)
    const float2 vertices[] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2(-1.0,  1.0),
        float2(-1.0,  1.0), float2( 1.0, -1.0), float2( 1.0,  1.0)
    };
    // Corresponding texture coordinates (0 to 1)
    const float2 texCoords[] = {
        float2(0.0, 1.0), float2(1.0, 1.0), float2(0.0, 0.0),
        float2(0.0, 0.0), float2(1.0, 1.0), float2(1.0, 0.0)
    };

    VertexOut out;
    out.position = float4(vertices[vid], 0.0, 1.0);
    out.texCoord = texCoords[vid];
    return out;
}

inline float3 sampleFromLUT(constant float3 *lut, uint size, float t) {
    float clamped = clamp(t, 0.0f, 1.0f);
    float scaled = clamped * (float)(size - 1);
    uint idx0 = (uint)floor(scaled);
    uint idx1 = min(idx0 + 1, size - 1);
    float frac = scaled - (float)idx0;
    return mix(lut[idx0], lut[idx1], frac);
}

inline float3 sampleColormap(uint index, float t) {
    switch (index) {
        case 1:
            return sampleFromLUT(VIRIDIS_LUT, COLORMAP_LUT_SIZE, t);
        case 2:
            return sampleFromLUT(MAGMA_LUT, COLORMAP_LUT_SIZE, t);
        case 3:
            return sampleFromLUT(INFERNO_LUT, COLORMAP_LUT_SIZE, t);
        case 4:
            return sampleFromLUT(TURBO_LUT, COLORMAP_LUT_SIZE, t);
        case 5:
            return sampleFromLUT(COOLWARM_LUT, COLORMAP_LUT_SIZE, t);
        case 6:
            return sampleFromLUT(BLUEORANGE_LUT, COLORMAP_LUT_SIZE, t);
        case 7:
            return sampleFromLUT(SEISMIC_LUT, COLORMAP_LUT_SIZE, t);
        case 8:
            return sampleFromLUT(YLORRD_LUT, COLORMAP_LUT_SIZE, t);
        case 9:
            return sampleFromLUT(HOT_LUT, COLORMAP_LUT_SIZE, t);
        default:
            return sampleFromLUT(PLASMA_LUT, COLORMAP_LUT_SIZE, t);
    }
}

inline float normalizeDensityRaw(float density, constant VisualizationConfig &viz) {
    const float MIN_POSITIVE = 1e-12f;
    float vMin = max(viz.densityMin, MIN_POSITIVE);
    float vMax = max(viz.densityMax, vMin + MIN_POSITIVE);

    if (viz.logScale != 0) {
        float logVMin = log(vMin);
        float logRange = log(vMax) - logVMin;
        float logRangeInv = (fabs(logRange) > MIN_POSITIVE) ? (1.0f / logRange) : 0.0f;
        float safeDensity = max(density, MIN_POSITIVE);
        return (log(safeDensity) - logVMin) * logRangeInv;
    } else {
        float denom = vMax - vMin;
        float inv = (denom > MIN_POSITIVE) ? (1.0f / denom) : 0.0f;
        return (density - vMin) * inv;
    }
}

inline float normalizeDensityValue(float density, constant VisualizationConfig &viz) {
    return clamp(normalizeDensityRaw(density, viz), 0.0f, 1.0f);
}

inline float quantizeNormalizedValue(float norm, uint levels, thread uint &bandIndex) {
    if (levels <= 1) {
        bandIndex = 0;
        return norm;
    }

    float clamped = clamp(norm, 0.0f, 1.0f - 1e-6f);
    float scaled = clamped * (float)levels;
    float idx = floor(scaled);
    if (idx >= (float)levels) { idx = (float)levels - 1.0f; }
    bandIndex = (uint)idx;

    if (levels == 1) {
        return 0.0f;
    }

    float paletteIndex = floor(idx * 255.0f / ((float)levels - 1.0f));
    return paletteIndex / 255.0f;
}

// Fragment shader applies normalization and color map
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float> densityTexture [[texture(0)]],
                               constant VisualizationConfig &viz [[buffer(0)]])
{
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);

    // 1. Sample density and calculate the raw, unclamped normalized density.
    float density = densityTexture.sample(textureSampler, in.texCoord).r;
    float normRaw = normalizeDensityRaw(density, viz);

    // 2. Calculate screen-space derivatives (gradient) of the normalized density.
    // This must be done on the raw, unclamped value for accuracy, especially near boundaries.
    float dx = dfdx(normRaw);
    float dy = dfdy(normRaw);
    // Calculate the magnitude of the density change per pixel (L2 norm).
    float gradientMagnitude = length(float2(dx, dy));

    uint colorLevels = viz.colorLevels;
    float levels = max((float)colorLevels, 1.0f);

    // 3. Prepare values for color mapping (clamped and quantized).
    float normOriginal = clamp(normRaw, 0.0f, 1.0f);
    uint bandIndex = 0;
    float normForColor = normOriginal;
    if (colorLevels > 0) {
        normForColor = quantizeNormalizedValue(normOriginal, colorLevels, bandIndex);
    }

    if (viz.invert != 0) {
        normForColor = 1.0 - normForColor;
    }

    float3 color = sampleColormap(viz.colormapIndex, normForColor);

    // 4. Robust Contour Outlines (Uniform Screen-Space Width)
    // We only draw outlines if width is positive and there are internal boundaries (L > 1).
    if (viz.outlineWidth > 0.0f && colorLevels > 1) {

        // 5. Masking: Prevent stray contours caused by significant interpolation artifacts (overshoot/undershoot).
        // A small epsilon tolerates minor artifacts near the boundaries.
        if (normRaw >= -1e-5f && normRaw <= 1.0f + 1e-5f) {

            // 6. Calculate distance to the nearest contour line, explicitly excluding Min/Max boundaries.
            float scaled = normOriginal * levels;
            float nearestBoundaryIndex = round(scaled);
            float distanceDataScaled = fabs(scaled - nearestBoundaryIndex);

            // 7. Exclude boundaries at Min (0) and Max (L).
            bool isMinMaxBoundary = (nearestBoundaryIndex < 0.5f) || (nearestBoundaryIndex > levels - 0.5f);
            if (isMinMaxBoundary) {
                distanceDataScaled = 0.5f;
            }

            // 8. Calculate the distance in pixels.
            float gradientMagnitudeScaled = gradientMagnitude * levels;
            float pixelDistance = distanceDataScaled / max(gradientMagnitudeScaled, 1e-6f);

            // 9. Calculate outline factor (alpha) with anti-aliasing (1-pixel falloff).
            float halfWidth = viz.outlineWidth * 0.5f;
            float outlineFactor = smoothstep(halfWidth + 0.5f, halfWidth - 0.5f, pixelDistance);

            color = mix(color, float3(0.0, 0.0, 0.0), outlineFactor);
        }
    }

    return float4(color, 1.0);
}

