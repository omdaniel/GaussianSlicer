//! Parity Lab crate: shared math references + harness glue for Metal⇄WGSL parity.

#[path = "../cpu/mod.rs"]
pub mod cpu;

pub mod case_io;
pub mod scalar_vector_gpu;
pub mod wgpu_matrix_runner;
pub mod wgpu_scalar_runner;
pub mod wgpu_spd_runner;
pub mod wgsl;

/// Lightweight entry point invoked by the binary to keep `main.rs` minimal.
pub fn run_cli() {
    println!("Parity Lab harness is under construction. Run `cargo test -p parity_lab` to execute the CPU reference cases.");
}
