use parity_lab::cpu::{axpy3, dot3, length_squared3, normalize3_guarded};
use parity_lab::cpu::{generate_scalar_vector_cases, ScalarVectorCase};

#[test]
fn scalar_vector_references_are_deterministic() {
    let cases = generate_scalar_vector_cases(4, 0xA11CE);
    assert_eq!(cases.len(), 4);
    let again = generate_scalar_vector_cases(4, 0xA11CE);
    assert_eq!(
        cases
            .iter()
            .zip(again.iter())
            .all(|(a, b)| approx_case(a, b)),
        true
    );
}

#[test]
fn scalar_vector_reference_matches_primitives() {
    let cases = generate_scalar_vector_cases(16, 42);
    let eps = 1e-6;
    for case in cases {
        let ref_data = case.reference(eps);
        assert!((ref_data.dot - dot3(case.x, case.y)).abs() <= eps);
        let manual_axpy = axpy3(case.alpha, case.x, case.y);
        let delta = (manual_axpy - ref_data.axpy).abs().max_element();
        assert!(delta <= eps);
        assert!((ref_data.length_sq - length_squared3(case.x)).abs() <= eps);
        let (normalized, did_norm) = normalize3_guarded(case.x, eps);
        assert_eq!(did_norm, ref_data.did_normalize);
        let diff = (normalized - ref_data.normalized).abs().max_element();
        assert!(diff <= eps);
    }
}

fn approx_case(a: &ScalarVectorCase, b: &ScalarVectorCase) -> bool {
    (a.alpha - b.alpha).abs() <= 1e-6
        && (a.x - b.x).abs().max_element() <= 1e-6
        && (a.y - b.y).abs().max_element() <= 1e-6
}
