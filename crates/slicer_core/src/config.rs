//! Shared configuration types for the Gaussian Slicer domain.

use glam::{Mat3, Vec3};
use serde::{Deserialize, Serialize};

use crate::Scalar;

/// High level renderer settings mirrored from the Swift implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RendererSettings {
    pub num_distributions: u32,
    pub grid_resolution: u32,
    pub grid_min: Scalar,
    pub grid_max: Scalar,
    pub mean_std_dev: Scalar,
    pub covariance_scale: Scalar,
    pub seed: Option<u64>,
    pub plane_normal: Vec3,
    pub plane_offset: Scalar,
    pub density_min: Scalar,
    pub density_max: Scalar,
    pub use_log_scale: bool,
}

impl Default for RendererSettings {
    fn default() -> Self {
        Self {
            num_distributions: 50_000,
            grid_resolution: 256,
            grid_min: -8.0,
            grid_max: 8.0,
            mean_std_dev: 2.5,
            covariance_scale: 0.1,
            seed: None,
            plane_normal: Vec3::new(1.0, 0.5, 0.8).normalize(),
            plane_offset: 0.0,
            density_min: 1e-6,
            density_max: 0.05,
            use_log_scale: true,
        }
    }
}

impl RendererSettings {
    /// Computes the orthonormal basis aligned with the current plane normal.
    pub fn rotation_matrix(&self) -> Mat3 {
        rotation_matrix_for_normal(self.plane_normal)
    }

    /// Converts the high level settings into the raw kernel payload.
    pub fn kernel_config(&self) -> KernelConfig {
        debug_assert_eq!(
            core::mem::size_of::<KernelConfig>(),
            160,
            "KernelConfig must match WGSL layout (160 bytes)"
        );
        let rotation = self.rotation_matrix();
        KernelConfig {
            num_distributions: self.num_distributions,
            _pad0: [0; 3],
            rotation_matrix: mat3_to_padded_columns(rotation),
            plane_normal: [
                self.plane_normal.x,
                self.plane_normal.y,
                self.plane_normal.z,
                0.0,
            ],
            grid_params: [self.plane_offset, self.grid_min, self.grid_max, 0.0],
            _reserved0: [0.0; 4],
            _reserved1: [0.0; 4],
            _reserved2: [0.0; 4],
        }
    }
}

/// GPU configuration payload sent to compute kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KernelConfig {
    pub num_distributions: u32,
    pub _pad0: [u32; 3],
    pub rotation_matrix: [[Scalar; 4]; 4],
    pub plane_normal: [Scalar; 4],
    pub grid_params: [Scalar; 4],
    pub _reserved0: [Scalar; 4],
    pub _reserved1: [Scalar; 4],
    pub _reserved2: [Scalar; 4],
}

/// Visualization settings mirrored between host and GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisualizationConfig {
    pub colormap_index: u32,
    pub invert: u32,
    pub log_scale: u32,
    pub color_levels: u32,
    pub density_min: Scalar,
    pub density_max: Scalar,
    pub outline_width: Scalar,
    pub _pad: Scalar,
}

/// Recreates the Swift `MetalRenderer.getRotationMatrix` helper using glam.
pub fn rotation_matrix_for_normal(n: Vec3) -> Mat3 {
    let n_norm = if n.length_squared() <= f32::EPSILON {
        Vec3::Z
    } else {
        n.normalize()
    };

    // Choose axis with smallest absolute component to avoid degeneracy.
    let abs_n = n_norm.abs();
    let (mut arbitrary, min_index) = if abs_n.x <= abs_n.y && abs_n.x <= abs_n.z {
        (Vec3::X, 0)
    } else if abs_n.y <= abs_n.x && abs_n.y <= abs_n.z {
        (Vec3::Y, 1)
    } else {
        (Vec3::Z, 2)
    };

    let mut u = n_norm.cross(arbitrary);
    if u.length_squared() < 1e-6 {
        // Pick a different axis if we ended up collinear.
        arbitrary = match min_index {
            0 => Vec3::Y,
            1 => Vec3::Z,
            _ => Vec3::X,
        };
        u = n_norm.cross(arbitrary);
    }

    let u = u.normalize();
    let v = n_norm.cross(u).normalize();

    // Columns encode the local basis vectors (slice axes in world space).
    Mat3::from_cols(u, v, n_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_aligns_with_normal() {
        let normal = Vec3::new(0.3, -0.5, 1.2).normalize();
        let basis = rotation_matrix_for_normal(normal);
        let n_col = basis * Vec3::Z;

        assert!(n_col.angle_between(normal) < 1e-4);

        let u = basis * Vec3::X;
        let v = basis * Vec3::Y;
        // Columns must be orthonormal.
        assert!((u.length() - 1.0).abs() < 1e-4);
        assert!((v.length() - 1.0).abs() < 1e-4);
        assert!(u.dot(v).abs() < 1e-4);
    }

    #[test]
    fn kernel_config_matches_defaults() {
        let settings = RendererSettings::default();
        let cfg = settings.kernel_config();

        assert_eq!(cfg.num_distributions, settings.num_distributions);
        assert_eq!(cfg.grid_params[1], settings.grid_min);
        assert_eq!(cfg.grid_params[2], settings.grid_max);
        assert_eq!(cfg.plane_normal[0], settings.plane_normal.x);
        assert_eq!(cfg.plane_normal[1], settings.plane_normal.y);
        assert_eq!(cfg.plane_normal[2], settings.plane_normal.z);
        assert_eq!(cfg._reserved0, [0.0; 4]);
        assert_eq!(cfg._reserved1, [0.0; 4]);
        assert_eq!(cfg._reserved2, [0.0; 4]);
        assert_eq!(cfg.rotation_matrix[3], [0.0, 0.0, 0.0, 1.0]);
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            colormap_index: 0,
            invert: 0,
            log_scale: 1,
            color_levels: 0,
            density_min: 1e-6,
            density_max: 0.05,
            outline_width: 0.0,
            _pad: 0.0,
        }
    }
}

impl VisualizationConfig {
    pub fn from_settings(settings: &RendererSettings) -> Self {
        Self {
            colormap_index: 0,
            invert: 0,
            log_scale: if settings.use_log_scale { 1 } else { 0 },
            color_levels: 0,
            density_min: settings.density_min,
            density_max: settings.density_max,
            outline_width: 0.0,
            _pad: 0.0,
        }
    }
}

fn mat3_to_padded_columns(m: Mat3) -> [[Scalar; 4]; 4] {
    [
        [m.row(0).x, m.row(0).y, m.row(0).z, 0.0],
        [m.row(1).x, m.row(1).y, m.row(1).z, 0.0],
        [m.row(2).x, m.row(2).y, m.row(2).z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
