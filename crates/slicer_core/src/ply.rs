//! Minimal Gaussian splat PLY loader shared by the Swift + Rust frontends.
//!
//! The loader understands the vertex schema emitted by the original 3D Gaussian
//! Splatting project:
//! - `x`, `y`, `z` – component means in world space.
//! - `scale_0..2` (log-space principal axes, exponentiated during load).
//! - `rot_0..3` – quaternion (x, y, z, w) describing the principal-axis frame.
//! - `opacity`/`alpha`/`weight` – optional weight; defaults to `1.0`.
//! Additional vertex properties are ignored so scene-specific data can be
//! preserved in the same file (colors, SH coefficients, etc).

use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use anyhow::{bail, Context, Result};
use glam::{Mat3, Quat, Vec3};

use crate::Gaussian3D;

const MIN_VARIANCE: f32 = 1e-6;
const MAX_LOG_SCALE: f32 = 16.0;

/// Loads Gaussian mixture data from a Gaussian Splat PLY file.
pub fn load_gaussians_from_ply(path: impl AsRef<Path>) -> Result<Vec<Gaussian3D>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let header = PlyHeader::read(&mut reader)?;

    let mut payload = Vec::new();
    reader
        .read_to_end(&mut payload)
        .context("failed to read PLY payload")?;

    let mut gaussians = match header.format {
        PlyFormat::Ascii => {
            parse_ascii_vertices(&payload, &header).context("failed to parse ASCII PLY payload")?
        }
        PlyFormat::BinaryLittleEndian => parse_binary_vertices(&payload, &header)
            .context("failed to parse binary PLY payload")?,
    };

    normalize_weights(&mut gaussians);
    Ok(gaussians)
}

fn normalize_weights(gaussians: &mut [Gaussian3D]) {
    if gaussians.is_empty() {
        return;
    }
    let total_weight: f32 = gaussians.iter().map(|g| g.weight).sum();
    if total_weight > 0.0 {
        let inv = 1.0 / total_weight;
        for gaussian in gaussians {
            gaussian.weight *= inv;
        }
    } else {
        let uniform = 1.0 / gaussians.len() as f32;
        for gaussian in gaussians {
            gaussian.weight = uniform;
        }
    }
}

fn parse_ascii_vertices(payload: &[u8], header: &PlyHeader) -> Result<Vec<Gaussian3D>> {
    let text = std::str::from_utf8(payload).context("ASCII payload is not valid UTF-8")?;

    let mut gaussians = Vec::with_capacity(header.vertex_count);
    for line in text.lines() {
        if gaussians.len() >= header.vertex_count {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() < header.properties.len() {
            bail!(
                "expected {} scalars per vertex, found {}",
                header.properties.len(),
                tokens.len()
            );
        }

        let mut builder = GaussianBuilder::default();
        for (prop, token) in header.properties.iter().zip(tokens.iter()) {
            let value = prop
                .ty
                .parse_ascii(token)
                .with_context(|| format!("invalid value for property '{}'", prop.name))?;
            builder.consume(&prop.name, value as f32);
        }
        gaussians.push(builder.build()?);
    }

    if gaussians.len() != header.vertex_count {
        bail!(
            "file declared {} vertices but parsed {}",
            header.vertex_count,
            gaussians.len()
        );
    }
    Ok(gaussians)
}

fn parse_binary_vertices(payload: &[u8], header: &PlyHeader) -> Result<Vec<Gaussian3D>> {
    let stride: usize = header.properties.iter().map(|p| p.ty.byte_size()).sum();
    let expected = header.vertex_count * stride;
    if payload.len() < expected {
        bail!(
            "binary payload too small: expected at least {} bytes, found {}",
            expected,
            payload.len()
        );
    }

    let mut gaussians = Vec::with_capacity(header.vertex_count);
    let mut offset = 0;
    for _ in 0..header.vertex_count {
        let mut builder = GaussianBuilder::default();
        for prop in &header.properties {
            let (value, consumed) = prop
                .ty
                .read_le(&payload[offset..])
                .with_context(|| format!("failed to read property '{}'", prop.name))?;
            offset += consumed;
            builder.consume(&prop.name, value as f32);
        }
        gaussians.push(builder.build()?);
    }
    Ok(gaussians)
}

#[derive(Debug, Clone)]
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    properties: Vec<Property>,
}

impl PlyHeader {
    fn read(reader: &mut BufReader<File>) -> Result<Self> {
        let mut header_lines = Vec::new();
        loop {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                bail!("unexpected EOF while reading PLY header");
            }
            let trimmed = line.trim_end();
            header_lines.push(trimmed.to_string());
            if trimmed == "end_header" {
                break;
            }
        }
        Self::parse(&header_lines)
    }

    fn parse(lines: &[String]) -> Result<Self> {
        if lines.is_empty() {
            bail!("empty PLY header");
        }
        if !lines[0].trim().eq_ignore_ascii_case("ply") {
            bail!("missing leading 'ply' marker");
        }

        let mut format = None;
        let mut vertex_count = None;
        let mut properties = Vec::new();
        let mut in_vertex_element = false;

        for line in lines.iter().skip(1) {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("comment")
                || trimmed.starts_with("obj_info")
            {
                continue;
            }
            let mut parts = trimmed.split_whitespace();
            match parts.next() {
                Some("format") => {
                    let fmt = parts.next().unwrap_or_default();
                    format = Some(match fmt {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        other => bail!("unsupported PLY format '{other}'"),
                    });
                }
                Some("element") => {
                    let name = parts.next().unwrap_or_default();
                    let count_str = parts.next().unwrap_or_default();
                    let count: usize = count_str
                        .parse()
                        .with_context(|| format!("invalid element count '{count_str}'"))?;
                    if name == "vertex" {
                        vertex_count = Some(count);
                        in_vertex_element = true;
                        properties.clear();
                    } else {
                        in_vertex_element = false;
                    }
                }
                Some("property") => {
                    if !in_vertex_element {
                        continue;
                    }
                    let ty_token = parts.next().unwrap_or_default();
                    if ty_token == "list" {
                        bail!("list properties are not supported for vertex data");
                    }
                    let name = parts.next().unwrap_or_default();
                    properties.push(Property {
                        name: name.to_string(),
                        ty: PlyScalarType::parse(ty_token)?,
                    });
                }
                Some("end_header") => break,
                _ => {}
            }
        }

        let format = format.ok_or_else(|| anyhow::anyhow!("missing format declaration"))?;
        let vertex_count =
            vertex_count.ok_or_else(|| anyhow::anyhow!("missing vertex element declaration"))?;
        if properties.is_empty() {
            bail!("vertex element is missing property definitions");
        }

        Ok(Self {
            format,
            vertex_count,
            properties,
        })
    }
}

#[derive(Debug, Clone)]
struct Property {
    name: String,
    ty: PlyScalarType,
}

#[derive(Debug, Clone, Copy)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Debug, Clone, Copy)]
enum PlyScalarType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyScalarType {
    fn parse(token: &str) -> Result<Self> {
        Ok(match token {
            "char" | "int8" => Self::Char,
            "uchar" | "uint8" => Self::UChar,
            "short" | "int16" => Self::Short,
            "ushort" | "uint16" => Self::UShort,
            "int" | "int32" => Self::Int,
            "uint" | "uint32" => Self::UInt,
            "float" | "float32" => Self::Float,
            "double" | "float64" => Self::Double,
            other => bail!("unsupported PLY scalar type '{other}'"),
        })
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    fn parse_ascii(self, token: &str) -> Result<f64> {
        Ok(match self {
            Self::Char => token.parse::<i8>()? as f64,
            Self::UChar => token.parse::<u8>()? as f64,
            Self::Short => token.parse::<i16>()? as f64,
            Self::UShort => token.parse::<u16>()? as f64,
            Self::Int => token.parse::<i32>()? as f64,
            Self::UInt => token.parse::<u32>()? as f64,
            Self::Float => token.parse::<f32>()? as f64,
            Self::Double => token.parse::<f64>()?,
        })
    }

    fn read_le(self, bytes: &[u8]) -> Result<(f64, usize)> {
        let size = self.byte_size();
        if bytes.len() < size {
            bail!("not enough bytes for scalar");
        }
        let value = match self {
            Self::Char => i8::from_le_bytes([bytes[0]]) as f64,
            Self::UChar => bytes[0] as f64,
            Self::Short => i16::from_le_bytes([bytes[0], bytes[1]]) as f64,
            Self::UShort => u16::from_le_bytes([bytes[0], bytes[1]]) as f64,
            Self::Int => i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            Self::UInt => u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            Self::Float => f32::from_le_bytes(bytes[0..4].try_into().unwrap()) as f64,
            Self::Double => f64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        };
        Ok((value, size))
    }
}

#[derive(Default)]
struct GaussianBuilder {
    mean: [f32; 3],
    scale: [f32; 3],
    rotation: [f32; 4],
    mean_mask: [bool; 3],
    scale_mask: [bool; 3],
    rot_mask: [bool; 4],
    weight: Option<f32>,
}

impl GaussianBuilder {
    fn consume(&mut self, name: &str, value: f32) {
        match name {
            "x" => {
                self.mean[0] = value;
                self.mean_mask[0] = true;
            }
            "y" => {
                self.mean[1] = value;
                self.mean_mask[1] = true;
            }
            "z" => {
                self.mean[2] = value;
                self.mean_mask[2] = true;
            }
            "scale_0" | "scale_x" => {
                self.scale[0] = value;
                self.scale_mask[0] = true;
            }
            "scale_1" | "scale_y" => {
                self.scale[1] = value;
                self.scale_mask[1] = true;
            }
            "scale_2" | "scale_z" => {
                self.scale[2] = value;
                self.scale_mask[2] = true;
            }
            "rot_0" | "rot_x" => {
                self.rotation[0] = value;
                self.rot_mask[0] = true;
            }
            "rot_1" | "rot_y" => {
                self.rotation[1] = value;
                self.rot_mask[1] = true;
            }
            "rot_2" | "rot_z" => {
                self.rotation[2] = value;
                self.rot_mask[2] = true;
            }
            "rot_3" | "rot_w" => {
                self.rotation[3] = value;
                self.rot_mask[3] = true;
            }
            "opacity" | "alpha" | "weight" | "w" => {
                self.weight = Some(value);
            }
            _ => {}
        }
    }

    fn build(self) -> Result<Gaussian3D> {
        if !self.mean_mask.iter().all(|&m| m) {
            bail!("PLY record is missing mean components");
        }
        if !self.scale_mask.iter().all(|&m| m) {
            bail!("PLY record is missing scale components");
        }
        if !self.rot_mask.iter().all(|&m| m) {
            bail!("PLY record is missing rotation components");
        }

        let weight = self.weight.unwrap_or(1.0).max(0.0);
        let mean = Vec3::from_array(self.mean);
        let scales = Vec3::new(
            exp_clamped(self.scale[0]),
            exp_clamped(self.scale[1]),
            exp_clamped(self.scale[2]),
        );
        let variances = Vec3::new(
            (scales.x * scales.x).max(MIN_VARIANCE),
            (scales.y * scales.y).max(MIN_VARIANCE),
            (scales.z * scales.z).max(MIN_VARIANCE),
        );
        let quat = Quat::from_xyzw(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        );
        let rotation = if quat.length_squared() < f32::EPSILON {
            Quat::IDENTITY
        } else {
            quat.normalize()
        };
        let basis = Mat3::from_quat(rotation);
        let covariance = basis * Mat3::from_diagonal(variances) * basis.transpose();

        Ok(Gaussian3D::from_components(mean, covariance, weight))
    }
}

fn exp_clamped(value: f32) -> f32 {
    value.clamp(-MAX_LOG_SCALE, MAX_LOG_SCALE).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    use std::io::Write;

    #[test]
    fn parses_ascii_gaussian_splat_file() {
        let mut rng = thread_rng();
        let temp_path =
            std::env::temp_dir().join(format!("gaussian_splat_test_{}.ply", rng.gen::<u64>()));

        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(
                file,
                "ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
0 0 0 0.5 0.0 0.0 0.0 0 0 0 1
1 2 3 1.0 0.1 0.2 0.3 0 0 0 1
"
            )
            .unwrap();
        }

        let gaussians = load_gaussians_from_ply(&temp_path).unwrap();
        assert_eq!(gaussians.len(), 2);
        assert!((gaussians[0].mean[0] - 0.0).abs() < 1e-6);
        assert!((gaussians[1].mean[0] - 1.0).abs() < 1e-6);
        assert!((gaussians[1].mean[2] - 3.0).abs() < 1e-6);

        let weight_sum: f32 = gaussians.iter().map(|g| g.weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);

        let _ = std::fs::remove_file(temp_path);
    }
}
