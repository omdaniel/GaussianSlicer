//! Centralized storage for WGSL shader sources and shader-related helpers.

/// Placeholder WGSL module that will be replaced with the translated kernels.
pub mod compute {
    pub const PRECALCULATE: &str = include_str!("kernels/precalculate.wgsl");
    pub const UPDATE_PARAMS: &str = include_str!("kernels/update_params.wgsl");
    pub const EVALUATE: &str = include_str!("kernels/evaluation.wgsl");
}

pub mod render {
    pub const VISUALIZE: &str = include_str!("kernels/visualize.wgsl");
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    fn validate_wgsl(label: &str, source: &str) {
        let module =
            naga::front::wgsl::parse_str(source).unwrap_or_else(|err| panic!("{label}: {err:?}"));
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .unwrap_or_else(|err| panic!("{label}: {err:?}"));
    }

    #[test]
    fn compute_shaders_validate() {
        validate_wgsl("precalculate", compute::PRECALCULATE);
        validate_wgsl("update_params", compute::UPDATE_PARAMS);
        validate_wgsl("evaluation", compute::EVALUATE);
    }

    #[test]
    fn render_shader_validates() {
        validate_wgsl("visualize", render::VISUALIZE);
    }
}
