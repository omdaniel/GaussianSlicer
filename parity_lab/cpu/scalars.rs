use glam::Vec3;

/// Dot product helper that mirrors the Metal implementation for scalar/vector parity cases.
pub fn dot3(lhs: Vec3, rhs: Vec3) -> f32 {
    lhs.dot(rhs)
}

/// Computes `y = alpha * x + y` and returns the updated vector.
pub fn axpy3(alpha: f32, x: Vec3, y: Vec3) -> Vec3 {
    Vec3::new(
        alpha.mul_add(x.x, y.x),
        alpha.mul_add(x.y, y.y),
        alpha.mul_add(x.z, y.z),
    )
}

/// Returns the squared length with explicit f32 handling (to avoid accidental double promotion).
pub fn length_squared3(v: Vec3) -> f32 {
    v.length_squared()
}

/// Normalizes a vector when its squared length exceeds `eps`; otherwise returns the input unchanged.
pub fn normalize3_guarded(v: Vec3, eps: f32) -> (Vec3, bool) {
    let len_sq = v.length_squared();
    if len_sq <= eps.max(0.0) {
        (v, false)
    } else {
        let inv_len = len_sq.sqrt().recip();
        (v * inv_len, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn dot_matches_manual_sum() {
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        for _ in 0..256 {
            let a = Vec3::new(rng.gen(), rng.gen(), rng.gen());
            let b = Vec3::new(rng.gen(), rng.gen(), rng.gen());
            let expected = a.x * b.x + a.y * b.y + a.z * b.z;
            let got = dot3(a, b);
            assert!(
                (expected - got).abs() <= 1e-6,
                "dot mismatch: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn axpy_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);
        for _ in 0..256 {
            let alpha: f32 = rng.gen_range(-10.0..10.0);
            let x = Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let y = Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );
            let expected = Vec3::new(
                alpha.mul_add(x.x, y.x),
                alpha.mul_add(x.y, y.y),
                alpha.mul_add(x.z, y.z),
            );
            let got = axpy3(alpha, x, y);
            let delta = (expected - got).abs().max_element();
            assert!(
                delta <= 1e-6,
                "axpy mismatch: expected {expected:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn normalize_clamps_with_guard() {
        let eps = 1e-6;
        let tiny = Vec3::splat(1e-8);
        let (result, normalized) = normalize3_guarded(tiny, eps);
        assert!(!normalized);
        assert_eq!(result, tiny);

        let v = Vec3::new(3.0, -4.0, 0.0);
        let (result, normalized) = normalize3_guarded(v, eps);
        assert!(normalized);
        let len = result.length();
        assert!((len - 1.0).abs() <= 1e-6);
        assert!(result.z.abs() <= 1e-6);
        assert!(result.x.is_sign_positive());
    }
}
