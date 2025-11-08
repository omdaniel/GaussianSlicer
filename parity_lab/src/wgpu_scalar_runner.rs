use anyhow::{Context, Result};
use wgpu::util::DeviceExt;

use crate::cpu::{ScalarVectorCase, ScalarVectorReference};
use crate::scalar_vector_gpu::{ScalarVectorCaseGpu, ScalarVectorResultGpu};
use crate::wgsl;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarVectorConfig {
    count: [u32; 4],
    epsilon: [f32; 4],
}

pub async fn run_gpu_cases_async(
    cases: &[ScalarVectorCase],
    epsilon: f32,
) -> Result<Vec<ScalarVectorReference>> {
    if cases.is_empty() {
        return Ok(Vec::new());
    }

    let case_payload: Vec<ScalarVectorCaseGpu> =
        cases.iter().map(ScalarVectorCaseGpu::from).collect();
    let config = ScalarVectorConfig {
        count: [cases.len() as u32, 0, 0, 0],
        epsilon: [epsilon, 0.0, 0.0, 0.0],
    };

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .context("no compatible GPU adapter found for parity lab")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("parity_lab_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .context("failed to request wgpu device")?;

    let case_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ScalarVectorCases"),
        contents: bytemuck::cast_slice(&case_payload),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let result_buffer_size =
        (case_payload.len() * std::mem::size_of::<ScalarVectorResultGpu>()) as wgpu::BufferAddress;
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ScalarVectorResults"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ScalarVectorStaging"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ScalarVectorConfig"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ScalarVectorBindGroupLayout"),
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
        label: Some("ScalarVectorBindGroup"),
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
        label: Some("ScalarVectorPipelineLayout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ScalarVectorShader"),
        source: wgpu::ShaderSource::Wgsl(wgsl::SCALAR_VECTOR.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ScalarVectorPipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("scalar_vector_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ScalarVectorEncoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((cases.len() as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_buffer_size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::MaintainBase::Wait);

    let data = buffer_slice.get_mapped_range();
    let gpu_outputs: &[ScalarVectorResultGpu] = bytemuck::cast_slice(&data);
    let refs = gpu_outputs
        .iter()
        .map(ScalarVectorResultGpu::to_reference)
        .collect();
    drop(data);
    staging_buffer.unmap();

    Ok(refs)
}

pub fn run_gpu_cases(
    cases: &[ScalarVectorCase],
    epsilon: f32,
) -> Result<Vec<ScalarVectorReference>> {
    pollster::block_on(run_gpu_cases_async(cases, epsilon))
}
