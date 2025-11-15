use std::{env, path::PathBuf};

use anyhow::{bail, Context, Result};
use glam::Mat3;

use parity_lab::{
    case_io::{export_case_sets_to_json, import_case_sets_from_json, CaseSet},
    cpu::{
        generate_mat3_cases, generate_scalar_vector_cases, generate_spd3_cases,
        scalar_vector_stress_cases, Mat3Case, ScalarVectorCase, Spd3Case,
    },
    wgpu_matrix_runner::run_gpu_mat_cases,
    wgpu_scalar_runner::run_gpu_cases,
    wgpu_spd_runner::run_gpu_spd_cases,
};

struct CliOptions {
    count: usize,
    seed: u64,
    epsilon: f32,
    input_json: Option<PathBuf>,
    export_json: Option<PathBuf>,
    skip_dispatch: bool,
}

fn main() -> Result<()> {
    let options = parse_options()?;

    let case_sets = if let Some(ref path) = options.input_json {
        println!("Loading parity case sets from {}", path.display());
        import_case_sets_from_json(path)?
    } else {
        println!(
            "Generating parity cases (random count={}, seed=0x{:X})",
            options.count, options.seed
        );
        build_case_sets(options.count, options.seed)
    };

    if let Some(ref path) = options.export_json {
        export_case_sets_to_json(&case_sets, path)?;
        println!(
            "Exported {} case sets ({}) to {}",
            case_sets.len(),
            case_sets
                .iter()
                .map(|set| set.label())
                .collect::<Vec<_>>()
                .join(", "),
            path.display()
        );
    }

    if options.skip_dispatch {
        println!("Skip-dispatch flag set; exiting after case preparation.");
        return Ok(());
    }

    for set in &case_sets {
        match set {
            CaseSet::ScalarVector { label, cases } => {
                println!(
                    "Running scalar/vector set '{}' ({} cases)",
                    label,
                    cases.len()
                );
                run_scalar_set(label, cases, options.epsilon)?;
            }
            CaseSet::Mat3Ops { label, cases } => {
                println!("Running mat3 set '{}' ({} cases)", label, cases.len());
                run_mat_set(label, cases)?;
            }
            CaseSet::Spd3 { label, cases } => {
                println!("Running SPD set '{}' ({} cases)", label, cases.len());
                run_spd_set(label, cases, options.epsilon)?;
            }
        }
    }

    Ok(())
}

fn parse_options() -> Result<CliOptions> {
    let mut opts = CliOptions {
        count: 128,
        seed: 0xA11CE,
        epsilon: 1e-6,
        input_json: None,
        export_json: None,
        skip_dispatch: false,
    };

    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--count=") {
            opts.count = value.parse().context("invalid --count value")?;
        } else if let Some(value) = arg.strip_prefix("--seed=") {
            opts.seed = parse_seed(value).context("invalid --seed value")?;
        } else if let Some(value) = arg.strip_prefix("--epsilon=") {
            opts.epsilon = value.parse().context("invalid --epsilon value")?;
        } else if let Some(value) = arg.strip_prefix("--cases-json=") {
            opts.input_json = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--export-json=") {
            opts.export_json = Some(PathBuf::from(value));
        } else if arg == "--skip-dispatch" {
            opts.skip_dispatch = true;
        } else {
            bail!("unrecognized argument: {arg}");
        }
    }

    Ok(opts)
}

fn run_scalar_set(label: &str, cases: &[ScalarVectorCase], epsilon: f32) -> Result<()> {
    let gpu_refs = run_gpu_cases(cases, epsilon)?;
    let mut max_dot_err = 0.0f32;
    let mut max_axpy_err = 0.0f32;
    let mut max_length_err = 0.0f32;
    let mut max_norm_err = 0.0f32;
    let mut mismatched_flags = 0usize;

    for (case, gpu_ref) in cases.iter().zip(gpu_refs.iter()) {
        let cpu_ref = case.reference(epsilon);
        max_dot_err = max_dot_err.max((cpu_ref.dot - gpu_ref.dot).abs());
        max_length_err = max_length_err.max((cpu_ref.length_sq - gpu_ref.length_sq).abs());
        max_axpy_err = max_axpy_err.max((cpu_ref.axpy - gpu_ref.axpy).abs().max_element());
        max_norm_err = max_norm_err.max(
            (cpu_ref.normalized - gpu_ref.normalized)
                .abs()
                .max_element(),
        );
        if cpu_ref.did_normalize != gpu_ref.did_normalize {
            mismatched_flags += 1;
        }
    }

    println!("Set '{label}' residuals (epsilon={epsilon}):");
    println!("  max |dot| error      : {max_dot_err:.3e}");
    println!("  max |length²| error : {max_length_err:.3e}");
    println!("  max |axpy| error     : {max_axpy_err:.3e}");
    println!("  max |normalize| error: {max_norm_err:.3e}");
    println!("  normalize flag mismatches: {mismatched_flags}");

    let tolerance = 1e-6;
    if max_dot_err > tolerance
        || max_length_err > tolerance
        || max_axpy_err > tolerance
        || max_norm_err > tolerance
        || mismatched_flags > 0
    {
        bail!("GPU results diverged beyond tolerance");
    }

    Ok(())
}

fn run_mat_set(label: &str, cases: &[Mat3Case]) -> Result<()> {
    let gpu_refs = run_gpu_mat_cases(cases)?;
    let mut max_vec_err = 0.0f32;
    let mut max_mat_err = 0.0f32;
    for (case, gpu_ref) in cases.iter().zip(gpu_refs.iter()) {
        let cpu_ref = case.reference();
        max_vec_err = max_vec_err.max((cpu_ref.a_times_v - gpu_ref.a_times_v).abs().max_element());
        max_mat_err = max_mat_err.max(max_mat_component_delta(
            &cpu_ref.a_times_b,
            &gpu_ref.a_times_b,
        ));
        max_mat_err = max_mat_err.max(max_mat_component_delta(
            &cpu_ref.a_transpose,
            &gpu_ref.a_transpose,
        ));
    }
    println!(
        "Set '{}' residuals (mat3): max |a*v|={:.3e}, max |a*b|/|a^T| component diff={:.3e}",
        label, max_vec_err, max_mat_err
    );
    if max_vec_err > 1e-6 || max_mat_err > 1e-6 {
        bail!("Mat3 GPU results diverged beyond tolerance");
    }
    Ok(())
}

fn run_spd_set(label: &str, cases: &[Spd3Case], epsilon: f32) -> Result<()> {
    let gpu_refs = run_gpu_spd_cases(cases, epsilon)?;
    let mut max_chol_err = 0.0f32;
    let mut max_solution_err = 0.0f32;
    let mut worst_solution: Option<(usize, [f32; 3], [f32; 3])> = None;
    let mut worst_chol: Option<(usize, Mat3, Mat3)> = None;
    for (idx, (case, gpu_ref)) in cases.iter().zip(gpu_refs.iter()).enumerate() {
        let residual = case.a * gpu_ref.solution - case.b;
        let sol_err = residual.abs().max_element();
        if sol_err > max_solution_err {
            max_solution_err = sol_err;
            worst_solution = Some((idx, residual.to_array(), gpu_ref.solution.to_array()));
        }
        let cpu_ref = case.reference_with_epsilon(epsilon);
        let chol_err = max_mat_component_delta(&cpu_ref.chol_lower, &gpu_ref.chol_lower);
        if chol_err > max_chol_err {
            max_chol_err = chol_err;
            worst_chol = Some((idx, cpu_ref.chol_lower, gpu_ref.chol_lower));
        }
    }
    println!(
        "Set '{}' residuals (SPD, epsilon={}): max |A·x - b|={:.3e}, max |chol| comp diff={:.3e}",
        label, epsilon, max_solution_err, max_chol_err
    );
    if max_solution_err > 1e-6 || max_chol_err > 1e-6 {
        if let Some((idx, residual, gpu_sol)) = worst_solution {
            println!(
                "  worst solution case #{idx}: gpu solution={gpu_sol:?}, residual={residual:?}"
            );
        }
        if let Some((idx, cpu_chol, gpu_chol)) = worst_chol {
            println!("  worst chol case #{idx}:");
            println!("    cpu={:?}", cpu_chol);
            println!("    gpu={:?}", gpu_chol);
        }
        bail!("SPD GPU results diverged beyond tolerance");
    }
    Ok(())
}

fn build_case_sets(count: usize, seed: u64) -> Vec<CaseSet> {
    vec![
        CaseSet::ScalarVector {
            label: format!("basic_random_0x{seed:X}"),
            cases: generate_scalar_vector_cases(count, seed),
        },
        CaseSet::ScalarVector {
            label: "stress_edges".into(),
            cases: scalar_vector_stress_cases(),
        },
        CaseSet::Mat3Ops {
            label: format!("mat3_ops_0x{seed:X}"),
            cases: generate_mat3_cases(count.max(16), seed ^ 0xDEADBEEF),
        },
        CaseSet::Spd3 {
            label: format!("spd3_0x{seed:X}"),
            cases: generate_spd3_cases(count.max(16), seed ^ 0xBADCAFE),
        },
    ]
}

fn max_mat_component_delta(a: &Mat3, b: &Mat3) -> f32 {
    a.to_cols_array()
        .iter()
        .zip(b.to_cols_array().iter())
        .fold(0.0f32, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()))
}

fn parse_seed(value: &str) -> Result<u64> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).context("expected hex literal")
    } else {
        value.parse().context("expected integer seed")
    }
}
