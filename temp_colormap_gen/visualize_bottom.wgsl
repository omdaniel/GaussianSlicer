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
