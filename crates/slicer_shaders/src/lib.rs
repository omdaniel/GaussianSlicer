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
    use naga::{
        front::wgsl,
        proc::{GlobalCtx, Layouter},
        valid::{Capabilities, ValidationFlags, Validator},
        TypeInner,
    };

    fn validate_wgsl(label: &str, source: &str) {
        let module = wgsl::parse_str(source).unwrap_or_else(|err| panic!("{label}: {err:?}"));
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

    #[test]
    fn layout_snapshot() {
        let module =
            wgsl::parse_str(compute::UPDATE_PARAMS).expect("parse update_params.wgsl for layout");
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        validator
            .validate(&module)
            .expect("update_params.wgsl validation");

        let mut layouter = Layouter::default();
        layouter
            .update(GlobalCtx {
                types: &module.types,
                constants: &module.constants,
                overrides: &module.overrides,
                global_expressions: &module.global_expressions,
            })
            .expect("layout update_params.wgsl");

        assert_struct_layout(
            &module,
            &layouter,
            "Gaussian3D",
            &[
                ("mean", 0),
                ("covariance_col0", 16),
                ("covariance_col1", 32),
                ("covariance_col2", 48),
                ("weight_pad", 64),
            ],
            80,
        );

        assert_struct_layout(
            &module,
            &layouter,
            "PrecalculatedParams",
            &[
                ("inv_cov2d_col0", 0),
                ("inv_cov2d_col1", 8),
                ("norm_const2d", 16),
                ("sigma_n_n", 20),
                ("mean_adj_factor", 24),
                ("_pad", 32),
            ],
            40,
        );

        assert_struct_layout(
            &module,
            &layouter,
            "DynamicParams",
            &[
                ("mean2d", 0),
                ("combined_factor", 8),
                ("_padding0", 12),
                ("_pad", 16),
            ],
            32,
        );
    }

    fn assert_struct_layout(
        module: &naga::Module,
        layouter: &Layouter,
        struct_name: &str,
        expected_members: &[(&str, u32)],
        expected_size: u32,
    ) {
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some(struct_name))
            .unwrap_or_else(|| panic!("struct `{struct_name}` not found in WGSL module"));

        let TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("`{struct_name}` is not a struct");
        };
        assert_eq!(
            members.len(),
            expected_members.len(),
            "`{struct_name}` member count mismatch"
        );

        for (member, (expected_name, expected_offset)) in members.iter().zip(expected_members) {
            assert_eq!(
                member.name.as_deref(),
                Some(*expected_name),
                "`{struct_name}` member name mismatch"
            );
            assert_eq!(
                member.offset, *expected_offset,
                "`{struct_name}`.{expected_name} offset mismatch"
            );
        }

        let TypeInner::Struct { .. } = &ty.inner else {
            unreachable!();
        };
        assert_eq!(
            layouter[handle].size, expected_size,
            "`{struct_name}` size mismatch"
        );
    }
}
