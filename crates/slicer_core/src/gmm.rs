//! Deterministic Gaussian mixture generation utilities mirroring the Swift implementation.

use glam::{Mat3, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::config::RendererSettings;
use crate::gpu::Gaussian3D;

const EPSILON: f32 = f32::MIN_POSITIVE;

pub fn generate(settings: &RendererSettings) -> (Vec<Gaussian3D>, u64) {
    let seed = settings.seed.unwrap_or(0x5EED5EED5EED5EED);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let count = settings.num_distributions as usize;
    let mut gaussians = Vec::with_capacity(count);
    let mut total_weight = 0.0;

    for _ in 0..count {
        let mean = Vec3::new(
            random_normal(&mut rng, 0.0, settings.mean_std_dev),
            random_normal(&mut rng, 0.0, settings.mean_std_dev),
            random_normal(&mut rng, 0.0, settings.mean_std_dev),
        );

        let covariance = generate_spd_matrix(&mut rng, settings.covariance_scale);
        let weight: f32 = rng.gen();
        total_weight += weight;

        gaussians.push(Gaussian3D::from_components(mean, covariance, weight));
    }

    if total_weight > 0.0 {
        let inv_total = 1.0 / total_weight;
        for gaussian in &mut gaussians {
            gaussian.weight *= inv_total;
        }
    }

    (gaussians, seed)
}

fn generate_spd_matrix<R: Rng + ?Sized>(rng: &mut R, scale: f32) -> Mat3 {
    let mut cols = [Vec3::ZERO; 3];
    for col in &mut cols {
        *col = Vec3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        );
    }

    let a = Mat3::from_cols(cols[0], cols[1], cols[2]);
    let c = a.transpose() * a;
    let adjusted_scale = scale.max(1e-4);
    (c + Mat3::IDENTITY * 0.5) * adjusted_scale
}

fn random_normal<R: Rng + ?Sized>(rng: &mut R, mean: f32, std_dev: f32) -> f32 {
    let u1 = rng.gen::<f32>().max(EPSILON);
    let u2 = rng.gen::<f32>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    mean + std_dev * radius * theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_count_and_normalizes_weights() {
        let mut settings = RendererSettings::default();
        settings.num_distributions = 8;
        settings.seed = Some(1234);
        let (gaussians, seed) = generate(&settings);
        assert_eq!(gaussians.len(), 8);
        assert_eq!(seed, 1234);

        let sum: f32 = gaussians.iter().map(|g| g.weight).sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }
}
