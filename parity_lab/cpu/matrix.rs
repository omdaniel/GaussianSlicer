use glam::{Mat3, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct Mat3Case {
    pub a: Mat3,
    pub b: Mat3,
    pub v: Vec3,
}

#[derive(Debug, Clone)]
pub struct Mat3Reference {
    pub a_times_v: Vec3,
    pub a_times_b: Mat3,
    pub a_transpose: Mat3,
}

impl Mat3Case {
    pub fn reference(&self) -> Mat3Reference {
        Mat3Reference {
            a_times_v: self.a * self.v,
            a_times_b: self.a * self.b,
            a_transpose: self.a.transpose(),
        }
    }
}

pub fn generate_mat3_cases(count: usize, seed: u64) -> Vec<Mat3Case> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut cases = Vec::with_capacity(count);
    for _ in 0..count {
        let a = random_mat3(&mut rng);
        let b = random_mat3(&mut rng);
        let v = random_vec3(&mut rng);
        cases.push(Mat3Case { a, b, v });
    }
    cases
}

fn random_vec3<R: Rng + ?Sized>(rng: &mut R) -> Vec3 {
    Vec3::new(
        rng.gen_range(-2.0..=2.0),
        rng.gen_range(-2.0..=2.0),
        rng.gen_range(-2.0..=2.0),
    )
}

fn random_mat3<R: Rng + ?Sized>(rng: &mut R) -> Mat3 {
    Mat3::from_cols(random_vec3(rng), random_vec3(rng), random_vec3(rng))
}
