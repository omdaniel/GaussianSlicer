//! GPU-facing data structures shared between host code and WGSL shaders.

use glam::{Mat3, Vec3};

use crate::Scalar;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gaussian3D {
    pub mean: [Scalar; 3],
    pub _mean_pad: Scalar,
    pub covariance: [[Scalar; 4]; 3],
    pub weight: Scalar,
    pub _pad: [Scalar; 3],
}

const _: () = assert!(core::mem::size_of::<Gaussian3D>() == 80);

impl Gaussian3D {
    pub fn from_components(mean: Vec3, covariance: Mat3, weight: Scalar) -> Self {
        Self {
            mean: mean.to_array(),
            _mean_pad: 0.0,
            covariance: mat3_to_std430_columns(covariance),
            weight,
            _pad: [0.0; 3],
        }
    }

    pub fn mean_vec3(&self) -> Vec3 {
        Vec3::from_array(self.mean)
    }

    pub fn covariance_mat3(&self) -> Mat3 {
        std430_columns_to_mat3(self.covariance)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrecalculatedParams {
    pub inv_cov2d_col0: [Scalar; 2],
    pub inv_cov2d_col1: [Scalar; 2],
    pub norm_const2d: Scalar,
    pub sigma_n_n: Scalar,
    pub mean_adj_factor: [Scalar; 2],
    pub _pad: [Scalar; 2],
}

impl PrecalculatedParams {
    pub fn invalid() -> Self {
        Self {
            inv_cov2d_col0: [0.0; 2],
            inv_cov2d_col1: [0.0; 2],
            norm_const2d: 0.0,
            sigma_n_n: 0.0,
            mean_adj_factor: [0.0; 2],
            _pad: [0.0; 2],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DynamicParams {
    pub mean2d: [Scalar; 2],
    pub combined_factor: Scalar,
    pub _pad: [Scalar; 3],
}

impl DynamicParams {
    pub fn zeroed() -> Self {
        Self {
            mean2d: [0.0; 2],
            combined_factor: 0.0,
            _pad: [0.0; 3],
        }
    }
}

fn mat3_to_std430_columns(m: Mat3) -> [[Scalar; 4]; 3] {
    [
        [m.col(0).x, m.col(0).y, m.col(0).z, 0.0],
        [m.col(1).x, m.col(1).y, m.col(1).z, 0.0],
        [m.col(2).x, m.col(2).y, m.col(2).z, 0.0],
    ]
}

fn std430_columns_to_mat3(columns: [[Scalar; 4]; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(columns[0][0], columns[0][1], columns[0][2]),
        Vec3::new(columns[1][0], columns[1][1], columns[1][2]),
        Vec3::new(columns[2][0], columns[2][1], columns[2][2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_round_trip() {
        let mean = Vec3::new(1.0, 2.0, 3.0);
        let covariance = Mat3::from_diagonal(Vec3::new(1.0, 2.0, 3.0));
        let g = Gaussian3D::from_components(mean, covariance, 0.5);

        assert_eq!(g.mean, [1.0, 2.0, 3.0]);
        assert_eq!(g.weight, 0.5);

        let restored = g.covariance_mat3();
        assert!((restored.col(0) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        assert!((restored.col(1) - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-6);
        assert!((restored.col(2) - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-6);
    }
}
