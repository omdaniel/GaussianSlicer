const COLORMAP_LUT_SIZE: u32 = 11u;

const PLASMA_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.05098, 0.03137, 0.52941),
    vec3<f32>(0.25490, 0.01569, 0.61569),
    vec3<f32>(0.41569, 0.00000, 0.65882),
    vec3<f32>(0.56078, 0.05098, 0.64314),
    vec3<f32>(0.69412, 0.16471, 0.56471),
    vec3<f32>(0.79608, 0.27451, 0.47451),
    vec3<f32>(0.88235, 0.39216, 0.38431),
    vec3<f32>(0.94902, 0.51765, 0.29412),
    vec3<f32>(0.98824, 0.65098, 0.21176),
    vec3<f32>(0.98824, 0.80784, 0.14510),
    vec3<f32>(0.94118, 0.97647, 0.12941),
);

const VIRIDIS_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.26700, 0.00487, 0.32941),
    vec3<f32>(0.28233, 0.09495, 0.41733),
    vec3<f32>(0.25394, 0.26214, 0.52942),
    vec3<f32>(0.20676, 0.37176, 0.55312),
    vec3<f32>(0.16386, 0.47113, 0.55815),
    vec3<f32>(0.12757, 0.56695, 0.55056),
    vec3<f32>(0.13451, 0.65864, 0.51765),
    vec3<f32>(0.26694, 0.74875, 0.44142),
    vec3<f32>(0.47750, 0.82144, 0.31820),
    vec3<f32>(0.74139, 0.87345, 0.14956),
    vec3<f32>(0.99325, 0.90616, 0.14394),
);

const MAGMA_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.00146, 0.00047, 0.01387),
    vec3<f32>(0.06354, 0.02843, 0.22780),
    vec3<f32>(0.18892, 0.07113, 0.34727),
    vec3<f32>(0.30637, 0.09495, 0.45728),
    vec3<f32>(0.42211, 0.11195, 0.54656),
    vec3<f32>(0.53333, 0.13227, 0.61566),
    vec3<f32>(0.64302, 0.15958, 0.65999),
    vec3<f32>(0.75488, 0.20196, 0.67849),
    vec3<f32>(0.86501, 0.25394, 0.67237),
    vec3<f32>(0.96624, 0.32135, 0.63849),
    vec3<f32>(0.98705, 0.99144, 0.74950),
);

const INFERNO_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.00146, 0.00047, 0.01387),
    vec3<f32>(0.07151, 0.01009, 0.26635),
    vec3<f32>(0.21611, 0.03813, 0.50565),
    vec3<f32>(0.37778, 0.09112, 0.66951),
    vec3<f32>(0.52998, 0.15218, 0.78461),
    vec3<f32>(0.67488, 0.21183, 0.84465),
    vec3<f32>(0.81636, 0.27339, 0.86441),
    vec3<f32>(0.93801, 0.42281, 0.84333),
    vec3<f32>(0.99220, 0.63756, 0.80887),
    vec3<f32>(0.99734, 0.85918, 0.79621),
    vec3<f32>(0.98836, 0.99836, 0.64492),
);

const TURBO_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.18995, 0.07176, 0.23217),
    vec3<f32>(0.25107, 0.25237, 0.62831),
    vec3<f32>(0.27628, 0.47672, 0.85988),
    vec3<f32>(0.20724, 0.68018, 0.85836),
    vec3<f32>(0.15340, 0.86658, 0.70504),
    vec3<f32>(0.36921, 0.93261, 0.40213),
    vec3<f32>(0.67801, 0.86555, 0.15093),
    vec3<f32>(0.92512, 0.70505, 0.05083),
    vec3<f32>(0.99673, 0.49712, 0.13476),
    vec3<f32>(0.98053, 0.28718, 0.38312),
    vec3<f32>(0.89490, 0.10287, 0.53074),
);

const COOLWARM_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.22980, 0.29872, 0.75368),
    vec3<f32>(0.26679, 0.43538, 0.82257),
    vec3<f32>(0.30659, 0.57122, 0.87068),
    vec3<f32>(0.34981, 0.70415, 0.89244),
    vec3<f32>(0.39504, 0.83206, 0.88175),
    vec3<f32>(0.75475, 0.90930, 0.74860),
    vec3<f32>(0.88245, 0.77634, 0.51465),
    vec3<f32>(0.90493, 0.63036, 0.31846),
    vec3<f32>(0.89376, 0.47136, 0.16899),
    vec3<f32>(0.84433, 0.30354, 0.07442),
    vec3<f32>(0.78039, 0.12619, 0.01519),
);

const BLUEORANGE_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.034, 0.113, 0.345),
    vec3<f32>(0.076, 0.295, 0.582),
    vec3<f32>(0.156, 0.479, 0.741),
    vec3<f32>(0.274, 0.647, 0.818),
    vec3<f32>(0.445, 0.791, 0.827),
    vec3<f32>(0.659, 0.859, 0.749),
    vec3<f32>(0.835, 0.812, 0.588),
    vec3<f32>(0.938, 0.682, 0.402),
    vec3<f32>(0.964, 0.482, 0.259),
    vec3<f32>(0.916, 0.262, 0.170),
    vec3<f32>(0.800, 0.063, 0.119),
);

const SEISMIC_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.000, 0.000, 0.300),
    vec3<f32>(0.000, 0.000, 0.700),
    vec3<f32>(0.000, 0.400, 1.000),
    vec3<f32>(0.200, 0.700, 1.000),
    vec3<f32>(0.500, 0.900, 1.000),
    vec3<f32>(0.800, 0.800, 0.800),
    vec3<f32>(1.000, 0.600, 0.600),
    vec3<f32>(1.000, 0.300, 0.200),
    vec3<f32>(0.900, 0.000, 0.000),
    vec3<f32>(0.600, 0.000, 0.000),
    vec3<f32>(0.300, 0.000, 0.000),
);

const YLORRD_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(1.00000, 1.00000, 0.89804),
    vec3<f32>(1.00000, 0.96863, 0.73725),
    vec3<f32>(0.99608, 0.89020, 0.56863),
    vec3<f32>(0.99346, 0.81569, 0.42353),
    vec3<f32>(0.98824, 0.73333, 0.31078),
    vec3<f32>(0.98280, 0.65098, 0.20392),
    vec3<f32>(0.95705, 0.54510, 0.13725),
    vec3<f32>(0.90588, 0.41569, 0.12157),
    vec3<f32>(0.85098, 0.28235, 0.09412),
    vec3<f32>(0.73725, 0.17647, 0.06275),
    vec3<f32>(0.49804, 0.00000, 0.00000),
);

const HOT_LUT: array<vec3<f32>, 11> = array<vec3<f32>, 11>(
    vec3<f32>(0.00000, 0.00000, 0.00000),
    vec3<f32>(0.20000, 0.00000, 0.00000),
    vec3<f32>(0.40000, 0.00000, 0.00000),
    vec3<f32>(0.60000, 0.00000, 0.00000),
    vec3<f32>(0.80000, 0.00000, 0.00000),
    vec3<f32>(1.00000, 0.20000, 0.00000),
    vec3<f32>(1.00000, 0.40000, 0.00000),
    vec3<f32>(1.00000, 0.60000, 0.00000),
    vec3<f32>(1.00000, 0.80000, 0.00000),
    vec3<f32>(1.00000, 0.90000, 0.20000),
    vec3<f32>(1.00000, 1.00000, 1.00000),
);

struct VisualizationConfig {
    colormap_index: u32,
    invert: u32,
    log_scale: u32,
    color_levels: u32,
    density_min: f32,
    density_max: f32,
    outline_width: f32,
    filter_mode: u32,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

struct QuantizeResult {
    value: f32,
    band_index: u32,
};

@vertex
fn fullscreen_vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var tex_coords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.tex_coord = tex_coords[vertex_index];
    return out;
}

@group(0) @binding(0) var density_texture: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var nearest_sampler: sampler;
@group(0) @binding(3) var<uniform> viz: VisualizationConfig;

fn sample_lut(lut_index: u32, idx0: u32, idx1: u32, frac: f32) -> vec3<f32> {
    switch (lut_index) {
        case 1u: {
            return mix(VIRIDIS_LUT[idx0], VIRIDIS_LUT[idx1], frac);
        }
        case 2u: {
            return mix(MAGMA_LUT[idx0], MAGMA_LUT[idx1], frac);
        }
        case 3u: {
            return mix(INFERNO_LUT[idx0], INFERNO_LUT[idx1], frac);
        }
        case 4u: {
            return mix(TURBO_LUT[idx0], TURBO_LUT[idx1], frac);
        }
        case 5u: {
            return mix(COOLWARM_LUT[idx0], COOLWARM_LUT[idx1], frac);
        }
        case 6u: {
            return mix(BLUEORANGE_LUT[idx0], BLUEORANGE_LUT[idx1], frac);
        }
        case 7u: {
            return mix(SEISMIC_LUT[idx0], SEISMIC_LUT[idx1], frac);
        }
        case 8u: {
            return mix(YLORRD_LUT[idx0], YLORRD_LUT[idx1], frac);
        }
        case 9u: {
            return mix(HOT_LUT[idx0], HOT_LUT[idx1], frac);
        }
        default: {
            return mix(PLASMA_LUT[idx0], PLASMA_LUT[idx1], frac);
        }
    }
}

fn sample_colormap(index: u32, t: f32) -> vec3<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let scaled = clamped * f32(COLORMAP_LUT_SIZE - 1u);
    let idx0 = u32(floor(scaled));
    let idx1 = min(idx0 + 1u, COLORMAP_LUT_SIZE - 1u);
    let frac = scaled - f32(idx0);
    return sample_lut(index, idx0, idx1, frac);
}

fn normalize_density_raw(density: f32, cfg: VisualizationConfig) -> f32 {
    let min_positive = 1e-16;
    let v_min = max(cfg.density_min, min_positive);
    let v_max = max(cfg.density_max, v_min + min_positive);

    if (cfg.log_scale != 0u) {
        let log_v_min = log(v_min);
        let log_range = log(v_max) - log_v_min;
        if (abs(log_range) > min_positive) {
            let safe_density = max(density, min_positive);
            return (log(safe_density) - log_v_min) / log_range;
        }
        return 0.0;
    }

    let denom = v_max - v_min;
    if (denom > min_positive) {
        return (density - v_min) / denom;
    }
    return 0.0;
}

fn quantize_normalized_value(norm: f32, levels: u32) -> QuantizeResult {
    if (levels <= 1u) {
        return QuantizeResult(norm, 0u);
    }

    let clamped = clamp(norm, 0.0, 1.0 - 1e-6);
    let scaled = clamped * f32(levels);
    var idx = floor(scaled);
    if (idx >= f32(levels)) {
        idx = f32(levels) - 1.0;
    }

    let band_index = u32(idx);
    if (levels == 1u) {
        return QuantizeResult(0.0, band_index);
    }

    let palette_index = floor(idx * 255.0 / (f32(levels) - 1.0));
    let value = palette_index / 255.0;
    return QuantizeResult(value, band_index);
}

fn sample_density_linear(uv: vec2<f32>) -> f32 {
    return textureSample(density_texture, linear_sampler, uv).r;
}

fn sample_density_nearest(uv: vec2<f32>) -> f32 {
    return textureSample(density_texture, nearest_sampler, uv).r;
}

fn sample_density(uv: vec2<f32>) -> f32 {
    if (viz.filter_mode == 0u) {
        return sample_density_linear(uv);
    }
    return sample_density_nearest(uv);
}

fn band_index_at(uv: vec2<f32>, levels: f32) -> u32 {
    let clamped_uv = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let density = sample_density_nearest(clamped_uv);
    let norm = clamp(normalize_density_raw(density, viz), 0.0, 1.0);
    var scaled = norm * levels;
    if (scaled >= levels) {
        scaled = levels - 1.0;
    }
    return u32(floor(scaled));
}

@fragment
fn visualize_fragment(in: VertexOut) -> @location(0) vec4<f32> {
    let density = sample_density(in.tex_coord);
    let norm_raw = normalize_density_raw(density, viz);

    let dx = dpdx(norm_raw);
    let dy = dpdy(norm_raw);
    let gradient_magnitude = length(vec2<f32>(dx, dy));

    let color_levels = viz.color_levels;
    let levels_f = max(f32(color_levels), 1.0);

    let norm_original = clamp(norm_raw, 0.0, 1.0);

    var norm_for_color = norm_original;
    var band_index = 0u;
    if (color_levels > 0u) {
        let quantized = quantize_normalized_value(norm_original, color_levels);
        norm_for_color = quantized.value;
        band_index = quantized.band_index;
    }

    if (viz.invert != 0u) {
        norm_for_color = 1.0 - norm_for_color;
    }

    var color = sample_colormap(viz.colormap_index, norm_for_color);

    if (viz.outline_width > 0.0 && color_levels > 1u) {
        if (viz.filter_mode == 0u) {
            if (norm_raw >= -1e-5 && norm_raw <= 1.0 + 1e-5) {
                let scaled = norm_original * levels_f;
                let nearest_boundary = round(scaled);
                var distance_data_scaled = abs(scaled - nearest_boundary);

                let is_min_or_max =
                    (nearest_boundary < 0.5) || (nearest_boundary > levels_f - 0.5);
                if (is_min_or_max) {
                    distance_data_scaled = 0.5;
                }

                let gradient_scaled = gradient_magnitude * levels_f;
                let pixel_distance =
                    distance_data_scaled / max(gradient_scaled, 1e-6);

                let half_width = viz.outline_width * 0.5;
                let outline_factor = smoothstep(
                    half_width + 0.5,
                    half_width - 0.5,
                    pixel_distance,
                );

                color = mix(
                    color,
                    vec3<f32>(0.0, 0.0, 0.0),
                    vec3<f32>(outline_factor, outline_factor, outline_factor),
                );
            }
        } else {
            let tex_dims = vec2<f32>(vec2<u32>(textureDimensions(density_texture, 0u)));
            let texel = vec2<f32>(1.0 / tex_dims.x, 1.0 / tex_dims.y);

            let center_band = band_index;
            let left_band = band_index_at(in.tex_coord + vec2<f32>(-texel.x, 0.0), levels_f);
            let right_band = band_index_at(in.tex_coord + vec2<f32>(texel.x, 0.0), levels_f);
            let down_band = band_index_at(in.tex_coord + vec2<f32>(0.0, -texel.y), levels_f);
            let up_band = band_index_at(in.tex_coord + vec2<f32>(0.0, texel.y), levels_f);

            let is_boundary = bool(center_band != left_band)
                || bool(center_band != right_band)
                || bool(center_band != down_band)
                || bool(center_band != up_band);
            let outline_factor = select(0.0, 1.0, is_boundary);
            color = mix(
                color,
                vec3<f32>(0.0, 0.0, 0.0),
                vec3<f32>(outline_factor, outline_factor, outline_factor),
            );
        }
    }

    return vec4<f32>(color, 1.0);
}
