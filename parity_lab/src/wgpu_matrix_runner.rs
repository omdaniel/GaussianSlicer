use std::mem::size_of;

use anyhow::{Context, Result};
use glam::{Mat3, Vec3};
use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::{
    cpu::{
        matrix::{Mat3Case, Mat3Reference},
        Scalar,
    },
    wgsl,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat3ColumnsGpu {
    c0: [Scalar; 4],
    c1: [Scalar; 4],
    c2: [Scalar; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat3CaseGpu {
    a: Mat3ColumnsGpu,
    b: Mat3ColumnsGpu,
    v: [Scalar; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat3ResultGpu {
    a_times_v: [Scalar; 4],
    ab_col0: [Scalar; 4],
    ab_col1: [Scalar; 4],
    ab_col2: [Scalar; 4],
    at_col0: [Scalar; 4],
    at_col1: [Scalar; 4],
    at_col2: [Scalar; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    count: u32,
    _pad0: [u32; 3],
    _pad1: [u32; 4],
}

pub fn run_gpu_mat_cases(cases: &[Mat3Case]) -> Result<Vec<Mat3Reference>> {
    block_on(run_gpu_mat_cases_async(cases))
}

async fn run_gpu_mat_cases_async(cases: &[Mat3Case]) -> Result<Vec<Mat3Reference>> {
    if cases.is_empty() {
        return Ok(Vec::new());
    }

    let payload: Vec<Mat3CaseGpu> = cases.iter().map(Mat3CaseGpu::from).collect();
    let config = Config {
        count: cases.len() as u32,
        _pad0: [0; 3],
        _pad1: [0; 4],
    };

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .context("no compatible GPU adapter found for mat3 parity")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .context("failed to request wgpu device")?;

    let case_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mat3Cases"),
        contents: bytemuck::cast_slice(&payload),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_buffer_size = (payload.len() * size_of::<Mat3ResultGpu>()) as wgpu::BufferAddress;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mat3Results"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mat3Staging"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Mat3Config"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mat3BindGroupLayout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mat3BindGroup"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: case_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: config_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mat3PipelineLayout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mat3Shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl::MAT3_OPS.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mat3Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("mat3_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mat3Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (cases.len() as u32 + 63) / 64;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_buffer_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::MaintainBase::Wait);
    let data = buffer_slice.get_mapped_range();
    let gpu_outputs: &[Mat3ResultGpu] = bytemuck::cast_slice(&data);
    let refs = gpu_outputs
        .iter()
        .map(Mat3ResultGpu::to_reference)
        .collect();
    drop(data);
    staging_buffer.unmap();
    Ok(refs)
}

impl From<&Mat3Case> for Mat3CaseGpu {
    fn from(case: &Mat3Case) -> Self {
        Self {
            a: Mat3ColumnsGpu::from(case.a),
            b: Mat3ColumnsGpu::from(case.b),
            v: [case.v.x, case.v.y, case.v.z, 0.0],
        }
    }
}

impl Mat3ColumnsGpu {
    fn from(mat: Mat3) -> Self {
        Self {
            c0: [mat.col(0).x, mat.col(0).y, mat.col(0).z, 0.0],
            c1: [mat.col(1).x, mat.col(1).y, mat.col(1).z, 0.0],
            c2: [mat.col(2).x, mat.col(2).y, mat.col(2).z, 0.0],
        }
    }
}

impl Mat3ResultGpu {
    fn to_reference(&self) -> Mat3Reference {
        Mat3Reference {
            a_times_v: Vec3::new(self.a_times_v[0], self.a_times_v[1], self.a_times_v[2]),
            a_times_b: Mat3::from_cols(
                Vec3::new(self.ab_col0[0], self.ab_col0[1], self.ab_col0[2]),
                Vec3::new(self.ab_col1[0], self.ab_col1[1], self.ab_col1[2]),
                Vec3::new(self.ab_col2[0], self.ab_col2[1], self.ab_col2[2]),
            ),
            a_transpose: Mat3::from_cols(
                Vec3::new(self.at_col0[0], self.at_col0[1], self.at_col0[2]),
                Vec3::new(self.at_col1[0], self.at_col1[1], self.at_col1[2]),
                Vec3::new(self.at_col2[0], self.at_col2[1], self.at_col2[2]),
            ),
        }
    }
}
