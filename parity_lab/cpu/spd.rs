use glam::{Mat3, Vec3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct Spd3Case {
    pub a: Mat3,
    pub b: Vec3,
}

#[derive(Debug, Clone)]
pub struct Spd3Reference {
    pub chol_lower: Mat3,
    pub solution: Vec3,
}

impl Spd3Case {
    pub fn reference(&self) -> Spd3Reference {
        self.reference_with_epsilon(1e-12)
    }

    pub fn reference_with_epsilon(&self, epsilon: f32) -> Spd3Reference {
        let chol = cholesky3(self.a, epsilon);
        let y = forward_substitution(&chol, self.b);
        let x = backward_substitution(&chol, y);
        Spd3Reference {
            chol_lower: chol,
            solution: x,
        }
    }
}

pub fn generate_spd3_cases(count: usize, seed: u64) -> Vec<Spd3Case> {
    const EPS: f32 = 1e-6;
    const RESIDUAL_LIMIT: f32 = 5e-7;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut cases = Vec::with_capacity(count);
    while cases.len() < count {
        let base = random_mat3(&mut rng);
        let jitter = rng.gen_range(0.05..=0.5);
        let spd = base.transpose() * base + Mat3::IDENTITY * jitter;
        let b = Vec3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        );
        let candidate = Spd3Case { a: spd, b };
        let reference = candidate.reference_with_epsilon(EPS);
        let residual = (candidate.a * reference.solution - candidate.b)
            .abs()
            .max_element();
        if residual <= RESIDUAL_LIMIT {
            cases.push(candidate);
        }
    }
    cases
}

fn random_mat3<R: Rng + ?Sized>(rng: &mut R) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        ),
        Vec3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        ),
        Vec3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        ),
    )
}

fn cholesky3(a: Mat3, epsilon: f32) -> Mat3 {
    let mut l = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut sum = a.col(j)[i];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.max(epsilon).sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Mat3::from_cols(
        Vec3::new(l[0][0], l[1][0], l[2][0]),
        Vec3::new(l[0][1], l[1][1], l[2][1]),
        Vec3::new(l[0][2], l[1][2], l[2][2]),
    )
}

fn forward_substitution(l: &Mat3, b: Vec3) -> Vec3 {
    let l = *l;
    let l00 = l.col(0).x;
    let l10 = l.col(0).y;
    let l20 = l.col(0).z;
    let l11 = l.col(1).y;
    let l21 = l.col(1).z;
    let l22 = l.col(2).z;

    let y0 = b.x / l00;
    let y1 = (b.y - l10 * y0) / l11;
    let y2 = (b.z - l20 * y0 - l21 * y1) / l22;
    Vec3::new(y0, y1, y2)
}

fn backward_substitution(l: &Mat3, y: Vec3) -> Vec3 {
    let l = *l;
    let l00 = l.col(0).x;
    let l10 = l.col(0).y;
    let l20 = l.col(0).z;
    let l11 = l.col(1).y;
    let l21 = l.col(1).z;
    let l22 = l.col(2).z;

    let x2 = y.z / l22;
    let x1 = (y.y - l21 * x2) / l11;
    let x0 = (y.x - l10 * x1 - l20 * x2) / l00;
    Vec3::new(x0, x1, x2)
}
