use crate::cpu::{ScalarVectorCase, ScalarVectorReference};
use glam::Vec3;

/// GPU-facing layout for scalar/vector cases (std430-friendly).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarVectorCaseGpu {
    pub alpha: [f32; 4],
    pub x: [f32; 4],
    pub y: [f32; 4],
}

impl From<&ScalarVectorCase> for ScalarVectorCaseGpu {
    fn from(case: &ScalarVectorCase) -> Self {
        Self {
            alpha: [case.alpha, 0.0, 0.0, 0.0],
            x: case.x.extend(0.0).to_array(),
            y: case.y.extend(0.0).to_array(),
        }
    }
}

/// GPU result payload for the scalar/vector kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScalarVectorResultGpu {
    pub dot_and_length: [f32; 4],
    pub normalized: [f32; 4],
    pub axpy: [f32; 4],
    pub flags: [u32; 4],
}

impl ScalarVectorResultGpu {
    pub fn to_reference(&self) -> ScalarVectorReference {
        ScalarVectorReference {
            dot: self.dot_and_length[0],
            axpy: Vec3::new(self.axpy[0], self.axpy[1], self.axpy[2]),
            length_sq: self.dot_and_length[1],
            normalized: Vec3::new(self.normalized[0], self.normalized[1], self.normalized[2]),
            did_normalize: self.flags[0] != 0,
        }
    }
}
