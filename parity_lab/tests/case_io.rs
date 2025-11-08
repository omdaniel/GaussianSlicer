use std::{env, fs, path::PathBuf};

use parity_lab::{
    case_io::{export_case_sets_to_json, import_case_sets_from_json, CaseSet},
    cpu::{
        generate_mat3_cases, generate_scalar_vector_cases, generate_spd3_cases,
        scalar_vector_stress_cases,
    },
};

#[test]
fn case_sets_round_trip_json() {
    let sets = vec![
        CaseSet::ScalarVector {
            label: "basic_random".into(),
            cases: generate_scalar_vector_cases(4, 0x1234),
        },
        CaseSet::ScalarVector {
            label: "stress_edges".into(),
            cases: scalar_vector_stress_cases(),
        },
        CaseSet::Mat3Ops {
            label: "mat_ops".into(),
            cases: generate_mat3_cases(2, 0xBEEF),
        },
        CaseSet::Spd3 {
            label: "spd3".into(),
            cases: generate_spd3_cases(2, 0xFACE),
        },
    ];
    let path = temp_file_path();
    export_case_sets_to_json(&sets, &path).expect("failed to export case sets");
    let loaded = import_case_sets_from_json(&path).expect("failed to import case sets");
    fs::remove_file(&path).ok();

    assert_eq!(sets.len(), loaded.len());
    for (original, restored) in sets.iter().zip(loaded.iter()) {
        match (original, restored) {
            (
                CaseSet::ScalarVector {
                    label: lo,
                    cases: lhs,
                },
                CaseSet::ScalarVector {
                    label: ro,
                    cases: rhs,
                },
            ) => {
                assert_eq!(lo, ro);
                assert_eq!(lhs.len(), rhs.len());
                for (lhs_case, rhs_case) in lhs.iter().zip(rhs.iter()) {
                    assert!((lhs_case.alpha - rhs_case.alpha).abs() <= 1e-6);
                    assert!((lhs_case.x - rhs_case.x).abs().max_element() <= 1e-6);
                    assert!((lhs_case.y - rhs_case.y).abs().max_element() <= 1e-6);
                }
            }
            (
                CaseSet::Mat3Ops {
                    label: lo,
                    cases: lhs,
                },
                CaseSet::Mat3Ops {
                    label: ro,
                    cases: rhs,
                },
            ) => {
                assert_eq!(lo, ro);
                assert_eq!(lhs.len(), rhs.len());
            }
            (
                CaseSet::Spd3 {
                    label: lo,
                    cases: lhs,
                },
                CaseSet::Spd3 {
                    label: ro,
                    cases: rhs,
                },
            ) => {
                assert_eq!(lo, ro);
                assert_eq!(lhs.len(), rhs.len());
            }
            _ => panic!("unexpected case set kind"),
        }
    }
}

#[test]
fn legacy_array_still_parses() {
    let cases = generate_scalar_vector_cases(3, 0x42);
    let json = serde_like_array(&cases);
    let path = temp_file_path();
    fs::write(&path, json).unwrap();
    let loaded = import_case_sets_from_json(&path).expect("legacy parse failed");
    fs::remove_file(&path).ok();
    assert_eq!(loaded.len(), 1);
    match &loaded[0] {
        CaseSet::ScalarVector {
            cases: restored, ..
        } => assert_eq!(restored.len(), cases.len()),
        _ => panic!("expected scalar legacy set"),
    }
}

fn serde_like_array(cases: &[parity_lab::cpu::ScalarVectorCase]) -> String {
    let mut out = String::from("[");
    for (i, case) in cases.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let x = case.x.to_array();
        let y = case.y.to_array();
        out.push_str(&format!(
            "{{\"alpha\": {}, \"x\": [{}, {}, {}], \"y\": [{}, {}, {}]}}",
            case.alpha, x[0], x[1], x[2], y[0], y[1], y[2]
        ));
    }
    out.push(']');
    out
}

fn temp_file_path() -> PathBuf {
    let mut path = env::temp_dir();
    let unique = format!(
        "parity_lab_cases_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    path.push(unique);
    path
}
