//! CPU reference implementations for the Parity Lab math kernels.

use glam::Vec3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub mod matrix;
pub mod scalars;
pub mod spd;

pub type Scalar = f32;

pub use matrix::{generate_mat3_cases, Mat3Case, Mat3Reference};
pub use scalars::{axpy3, dot3, length_squared3, normalize3_guarded};
pub use spd::{generate_spd3_cases, Spd3Case, Spd3Reference};

/// Input needed to exercise the scalar/vector kernels.
#[derive(Debug, Clone, Copy)]
pub struct ScalarVectorCase {
    pub alpha: f32,
    pub x: Vec3,
    pub y: Vec3,
}

impl ScalarVectorCase {
    pub fn reference(&self, eps: f32) -> ScalarVectorReference {
        let dot = dot3(self.x, self.y);
        let axpy = axpy3(self.alpha, self.x, self.y);
        let length_sq = length_squared3(self.x);
        let (normalized, did_normalize) = normalize3_guarded(self.x, eps);
        ScalarVectorReference {
            dot,
            axpy,
            length_sq,
            normalized,
            did_normalize,
        }
    }
}

/// Reference results for a [`ScalarVectorCase`].
#[derive(Debug, Clone, Copy)]
pub struct ScalarVectorReference {
    pub dot: f32,
    pub axpy: Vec3,
    pub length_sq: f32,
    pub normalized: Vec3,
    pub did_normalize: bool,
}

/// Deterministic generator for scalar/vector parity cases.
pub fn generate_scalar_vector_cases(count: usize, seed: u64) -> Vec<ScalarVectorCase> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut cases = Vec::with_capacity(count);
    for _ in 0..count {
        let alpha: f32 = rng.gen_range(-10.0..=10.0);
        let x = Vec3::new(
            rng.gen_range(-2.0..=2.0),
            rng.gen_range(-2.0..=2.0),
            rng.gen_range(-2.0..=2.0),
        );
        let y = Vec3::new(
            rng.gen_range(-2.0..=2.0),
            rng.gen_range(-2.0..=2.0),
            rng.gen_range(-2.0..=2.0),
        );
        cases.push(ScalarVectorCase { alpha, x, y });
    }
    cases
}

/// Edge/stress cases that capture near-singular normals, axis-aligned vectors, and large/small magnitudes.
pub fn scalar_vector_stress_cases() -> Vec<ScalarVectorCase> {
    vec![
        ScalarVectorCase {
            alpha: 0.0,
            x: Vec3::ZERO,
            y: Vec3::new(1.0, 0.0, 0.0),
        },
        ScalarVectorCase {
            alpha: 1.0,
            x: Vec3::new(1e-8, -1e-8, 1e-8),
            y: Vec3::new(-1e-8, 1e-8, -1e-8),
        },
        ScalarVectorCase {
            alpha: -5.0,
            x: Vec3::new(10.0, -7.5, 2.5),
            y: Vec3::new(-3.5, 4.25, -5.75),
        },
        ScalarVectorCase {
            alpha: 2.0,
            x: Vec3::new(0.0, 1.0, 0.0),
            y: Vec3::new(0.0, 0.0, 1.0),
        },
        ScalarVectorCase {
            alpha: 0.25,
            x: Vec3::new(-2.0, 3.0, -4.0),
            y: Vec3::new(4.0, -6.0, 8.0),
        },
        ScalarVectorCase {
            alpha: -0.75,
            x: Vec3::new(1.0, 1.0, 1.0),
            y: Vec3::new(-1.0, 2.0, -3.0),
        },
        ScalarVectorCase {
            alpha: 0.5,
            x: Vec3::new(1000.0, -2000.0, 3000.0),
            y: Vec3::new(-4000.0, 5000.0, -6000.0),
        },
    ]
}
