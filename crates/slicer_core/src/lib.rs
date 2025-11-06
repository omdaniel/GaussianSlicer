//! Core Gaussian slicer domain logic that remains independent of GPU backends or UI shells.
//!
//! This crate will eventually host:
//! - configuration structs shared between the UI and GPU layer
//! - deterministic Gaussian mixture generation utilities
//! - CPU-side validation helpers for comparing against GPU outputs

pub mod config;
pub mod gmm;
pub mod gpu;
pub mod metrics;

/// Convenience re-export for the scalar type used across the slicer.
pub type Scalar = f32;

/// Small placeholder to keep the crate compiling while the actual logic is ported.
pub fn version_string() -> &'static str {
    "slicer_core 0.1.0 (placeholder)"
}

pub use gpu::{DynamicParams, Gaussian3D, PrecalculatedParams};
