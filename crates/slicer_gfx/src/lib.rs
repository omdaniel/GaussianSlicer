//! GPU abstraction layer over `wgpu` responsible for compute passes and presentation.

use std::{borrow::Cow, mem::size_of};

use anyhow::{Context, Result};
use bytemuck::{bytes_of, cast_slice};
use slicer_core::{
    config::{KernelConfig, RendererSettings, VisualizationConfig},
    DynamicParams, Gaussian3D, PrecalculatedParams,
};
use slicer_shaders::{compute, render};
use wgpu::{
    util::DeviceExt, Adapter, Backends, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, FragmentState, Instance, InstanceDescriptor,
    Limits, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PowerPreference,
    PresentMode, PrimitiveState, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderStages, StoreOp, Surface, SurfaceConfiguration, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::window::Window;

pub struct GpuContext<'window> {
    pub instance: Instance,
    pub surface: Surface<'window>,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_config: SurfaceConfiguration,
}

pub struct ShaderModules {
    pub precalc: wgpu::ShaderModule,
    pub update: wgpu::ShaderModule,
    pub evaluate: wgpu::ShaderModule,
    pub visualize: wgpu::ShaderModule,
}

const CONFIG_UNIFORM_SIZE: u64 = 256;
const COV_DEBUG_STRIDE: u64 = 240;

/// Entry point for creating a GPU context and loading shader modules.
pub async fn init(window: &Window) -> Result<(GpuContext<'_>, ShaderModules)> {
    let instance_desc = InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    };
    let instance = Instance::new(&instance_desc);

    let surface = instance
        .create_surface(window)
        .context("failed to create wgpu surface")?;

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .context("no compatible GPU adapter found")?;

    let device_desc = DeviceDescriptor {
        label: Some("GaussianSlicer Device"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    };

    let (device, queue) = adapter
        .request_device(&device_desc)
        .await
        .context("failed to request wgpu device")?;

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = preferred_surface_format(&surface_caps.formats);
    let size = window.inner_size();

    let alpha_mode = surface_caps
        .alpha_modes
        .first()
        .copied()
        .unwrap_or(wgpu::CompositeAlphaMode::Auto);

    let surface_config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: PresentMode::Fifo,
        alpha_mode,
        desired_maximum_frame_latency: 2,
        view_formats: vec![],
    };

    surface.configure(&device, &surface_config);

    let shaders = ShaderModules {
        precalc: create_module(&device, "precalculate.wgsl", compute::PRECALCULATE),
        update: create_module(&device, "update_params.wgsl", compute::UPDATE_PARAMS),
        evaluate: create_module(&device, "evaluation.wgsl", compute::EVALUATE),
        visualize: create_module(&device, "visualize.wgsl", render::VISUALIZE),
    };

    let context = GpuContext {
        instance,
        surface,
        adapter,
        device,
        queue,
        surface_config,
    };

    Ok((context, shaders))
}

fn preferred_surface_format(formats: &[TextureFormat]) -> TextureFormat {
    formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or_else(|| formats[0])
}

fn create_module(device: &Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}

/// Stub for pipeline creation. This will expand once resource bindings solidify.
pub fn create_pipeline_layouts(device: &Device) -> PipelineLayouts {
    let buffer_entry = |binding, ty: BufferBindingType| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let compute_precalc = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("precalc_bind_group_layout"),
        entries: &[
            buffer_entry(0, BufferBindingType::Uniform),
            buffer_entry(1, BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            buffer_entry(3, BufferBindingType::Storage { read_only: false }),
        ],
    });

    let compute_update = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("update_bind_group_layout"),
        entries: &[
            buffer_entry(0, BufferBindingType::Uniform),
            buffer_entry(1, BufferBindingType::Storage { read_only: true }),
            buffer_entry(2, BufferBindingType::Storage { read_only: true }),
            buffer_entry(3, BufferBindingType::Storage { read_only: false }),
        ],
    });

    let compute_evaluate = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("evaluation_bind_group_layout"),
        entries: &[
            buffer_entry(0, BufferBindingType::Uniform),
            buffer_entry(2, BufferBindingType::Storage { read_only: true }),
            buffer_entry(3, BufferBindingType::Storage { read_only: true }),
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let render_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("visualize_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    PipelineLayouts {
        precalc: compute_precalc,
        update: compute_update,
        evaluate: compute_evaluate,
        visualize: render_layout,
    }
}

#[derive(Debug)]
pub struct PipelineLayouts {
    pub precalc: wgpu::BindGroupLayout,
    pub update: wgpu::BindGroupLayout,
    pub evaluate: wgpu::BindGroupLayout,
    pub visualize: wgpu::BindGroupLayout,
}

#[derive(Debug)]
pub struct RendererResources {
    pub device: Device,
    pub queue: Queue,
    pub layouts: PipelineLayouts,
    pub pipelines: Pipelines,
    pub bind_groups: BindGroups,
    pub buffers: Buffers,
    pub density_texture: Texture,
    pub density_view: TextureView,
    pub sampler: wgpu::Sampler,
    pub visualization_config: VisualizationConfig,
    grid_resolution: u32,
    gaussian_buffer_size: u64,
    precalc_buffer_size: u64,
    dynamic_buffer_size: u64,
    precalc_debug_buffer_size: u64,
    dispatch: DispatchCounts,
}

#[derive(Debug)]
pub struct Pipelines {
    pub precalc: ComputePipeline,
    pub update: ComputePipeline,
    pub evaluate: ComputePipeline,
    pub visualize: RenderPipeline,
}

#[derive(Debug)]
pub struct BindGroups {
    pub precalc: wgpu::BindGroup,
    pub update: wgpu::BindGroup,
    pub evaluate: wgpu::BindGroup,
    pub visualize: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct Buffers {
    pub kernel_config: wgpu::Buffer,
    pub gaussians: wgpu::Buffer,
    pub precalc: wgpu::Buffer,
    pub dynamic: wgpu::Buffer,
    pub precalc_debug: wgpu::Buffer,
    pub visualization: wgpu::Buffer,
}

#[derive(Debug, Default)]
struct DispatchCounts {
    precalc: u32,
    update: u32,
    eval_x: u32,
    eval_y: u32,
}

impl RendererResources {
    pub fn new(
        context: &GpuContext<'_>,
        shaders: &ShaderModules,
        settings: &RendererSettings,
        gaussians: &[Gaussian3D],
    ) -> Result<Self> {
        let device = context.device.clone();
        let queue = context.queue.clone();
        let layouts = create_pipeline_layouts(&device);

        let pipelines = create_pipelines(&device, &layouts, shaders, context.surface_config.format);

        let num_distributions = gaussians.len() as u32;
        let kernel_config = settings.kernel_config();
        let visualization_config = VisualizationConfig::from_settings(settings);
        let grid_resolution = settings.grid_resolution;

        let kernel_config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KernelConfig Uniform"),
            size: CONFIG_UNIFORM_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&kernel_config_buffer, 0, bytes_of(&kernel_config));

        let buffers = Buffers {
            kernel_config: kernel_config_buffer,
            gaussians: create_storage_buffer(
                &device,
                "Gaussian Buffer",
                gaussians,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            precalc: create_storage_buffer(
                &device,
                "Precalculated Params Buffer",
                &vec![PrecalculatedParams::invalid(); gaussians.len()],
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            dynamic: create_storage_buffer(
                &device,
                "Dynamic Params Buffer",
                &vec![DynamicParams::zeroed(); gaussians.len()],
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ),
            precalc_debug: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Precalc Debug Buffer"),
                size: (gaussians.len().max(1) as u64) * COV_DEBUG_STRIDE,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            visualization: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VisualizationConfig Uniform"),
                contents: bytes_of(&visualization_config),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }),
        };
        let element_count = gaussians.len().max(1);
        let gaussian_buffer_size = (element_count * size_of::<Gaussian3D>()) as u64;
        let precalc_buffer_size = (element_count * size_of::<PrecalculatedParams>()) as u64;
        let dynamic_buffer_size = (element_count * size_of::<DynamicParams>()) as u64;
        let precalc_debug_buffer_size = element_count as u64 * COV_DEBUG_STRIDE;

        let density_texture = create_density_texture(&device, settings.grid_resolution);
        let density_view = density_texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Density Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let visualize_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Visualize Bind Group"),
            layout: &layouts.visualize,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&density_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buffers.visualization.as_entire_binding(),
                },
            ],
        });

        let bind_groups = BindGroups {
            precalc: device.create_bind_group(&BindGroupDescriptor {
                label: Some("Precalc Bind Group"),
                layout: &layouts.precalc,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffers.kernel_config.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffers.gaussians.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffers.precalc.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffers.precalc_debug.as_entire_binding(),
                    },
                ],
            }),
            update: device.create_bind_group(&BindGroupDescriptor {
                label: Some("Update Bind Group"),
                layout: &layouts.update,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffers.kernel_config.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffers.gaussians.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffers.precalc.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffers.dynamic.as_entire_binding(),
                    },
                ],
            }),
            evaluate: device.create_bind_group(&BindGroupDescriptor {
                label: Some("Evaluate Bind Group"),
                layout: &layouts.evaluate,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffers.kernel_config.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffers.precalc.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffers.dynamic.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&density_view),
                    },
                ],
            }),
            visualize: visualize_bind_group,
        };

        let dispatch = DispatchCounts {
            precalc: div_ceil(num_distributions, 64),
            update: div_ceil(num_distributions, 64),
            eval_x: div_ceil(settings.grid_resolution, 8),
            eval_y: div_ceil(settings.grid_resolution, 8),
        };

        Ok(Self {
            device,
            queue,
            layouts,
            pipelines,
            bind_groups,
            buffers,
            density_texture,
            density_view,
            sampler,
            visualization_config,
            grid_resolution,
            gaussian_buffer_size,
            precalc_buffer_size,
            dynamic_buffer_size,
            precalc_debug_buffer_size,
            dispatch,
        })
    }

    pub fn update_kernel_config(&self, config: &KernelConfig) {
        self.queue
            .write_buffer(&self.buffers.kernel_config, 0, bytes_of(config));
    }

    pub fn update_visualization_config(&mut self, viz: &VisualizationConfig) {
        self.queue
            .write_buffer(&self.buffers.visualization, 0, bytes_of(viz));
        self.visualization_config = *viz;
    }

    pub fn write_gaussians(&self, gaussians: &[Gaussian3D]) {
        if gaussians.is_empty() {
            return;
        }
        self.queue
            .write_buffer(&self.buffers.gaussians, 0, cast_slice(gaussians));
    }

    pub fn run_precalc(&self) -> Result<()> {
        if self.dispatch.precalc == 0 {
            return Ok(());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Precalc Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("K1 Precalculate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.precalc);
            pass.set_bind_group(0, &self.bind_groups.precalc, &[]);
            pass.dispatch_workgroups(self.dispatch.precalc, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn run_update_and_evaluate(&self) -> Result<()> {
        if self.dispatch.update == 0 && (self.dispatch.eval_x == 0 || self.dispatch.eval_y == 0) {
            return Ok(());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Update/Evaluate Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("K2+K3 Compute Pass"),
                timestamp_writes: None,
            });

            if self.dispatch.update > 0 {
                pass.set_pipeline(&self.pipelines.update);
                pass.set_bind_group(0, &self.bind_groups.update, &[]);
                pass.dispatch_workgroups(self.dispatch.update, 1, 1);
            }

            if self.dispatch.eval_x > 0 && self.dispatch.eval_y > 0 {
                pass.set_pipeline(&self.pipelines.evaluate);
                pass.set_bind_group(0, &self.bind_groups.evaluate, &[]);
                pass.dispatch_workgroups(self.dispatch.eval_x, self.dispatch.eval_y, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn density_texture_view(&self) -> &TextureView {
        &self.density_view
    }

    pub fn density_texture(&self) -> &Texture {
        &self.density_texture
    }

    pub fn gaussians_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.gaussians
    }

    pub fn gaussians_buffer_size(&self) -> u64 {
        self.gaussian_buffer_size
    }

    pub fn precalc_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.precalc
    }

    pub fn precalc_buffer_size(&self) -> u64 {
        self.precalc_buffer_size
    }

    pub fn dynamic_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.dynamic
    }

    pub fn dynamic_buffer_size(&self) -> u64 {
        self.dynamic_buffer_size
    }

    pub fn precalc_debug_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.precalc_debug
    }

    pub fn precalc_debug_buffer_size(&self) -> u64 {
        self.precalc_debug_buffer_size
    }

    pub fn kernel_config_buffer(&self) -> &wgpu::Buffer {
        &self.buffers.kernel_config
    }

    pub fn kernel_config_buffer_size(&self) -> u64 {
        CONFIG_UNIFORM_SIZE
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    pub fn grid_resolution(&self) -> u32 {
        self.grid_resolution
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn encode_visualization_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Visualization Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipelines.visualize);
        pass.set_bind_group(0, &self.bind_groups.visualize, &[]);
        pass.draw(0..6, 0..1);
    }
}

fn create_pipelines(
    device: &Device,
    layouts: &PipelineLayouts,
    shaders: &ShaderModules,
    surface_format: TextureFormat,
) -> Pipelines {
    let precalc_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Precalc Pipeline Layout"),
        bind_group_layouts: &[&layouts.precalc],
        push_constant_ranges: &[],
    });
    let update_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Update Pipeline Layout"),
        bind_group_layouts: &[&layouts.update],
        push_constant_ranges: &[],
    });
    let evaluate_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Evaluate Pipeline Layout"),
        bind_group_layouts: &[&layouts.evaluate],
        push_constant_ranges: &[],
    });
    let visualize_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Visualize Pipeline Layout"),
        bind_group_layouts: &[&layouts.visualize],
        push_constant_ranges: &[],
    });

    let precalc = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Precalc Pipeline"),
        layout: Some(&precalc_pipeline_layout),
        module: &shaders.precalc,
        entry_point: Some("precalculate_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let update = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Update Pipeline"),
        layout: Some(&update_pipeline_layout),
        module: &shaders.update,
        entry_point: Some("update_params_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let evaluate = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Evaluate Pipeline"),
        layout: Some(&evaluate_pipeline_layout),
        module: &shaders.evaluate,
        entry_point: Some("evaluation_kernel"),
        compilation_options: Default::default(),
        cache: None,
    });

    let visualize = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Visualize Pipeline"),
        layout: Some(&visualize_pipeline_layout),
        vertex: VertexState {
            module: &shaders.visualize,
            entry_point: Some("fullscreen_vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shaders.visualize,
            entry_point: Some("visualize_fragment"),
            compilation_options: Default::default(),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    });

    Pipelines {
        precalc,
        update,
        evaluate,
        visualize,
    }
}

fn create_storage_buffer<T: bytemuck::Pod>(
    device: &Device,
    label: &str,
    data: &[T],
    usage: BufferUsages,
) -> wgpu::Buffer {
    if data.is_empty() {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_of::<T>().max(1) as u64,
            usage,
            mapped_at_creation: false,
        })
    } else {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: cast_slice(data),
            usage,
        })
    }
}

fn create_density_texture(device: &Device, resolution: u32) -> Texture {
    let size = resolution.max(1);
    device.create_texture(&TextureDescriptor {
        label: Some("Density Texture"),
        size: Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

const fn div_ceil(value: u32, divisor: u32) -> u32 {
    if divisor == 0 {
        0
    } else if value == 0 {
        0
    } else {
        ((value - 1) / divisor) + 1
    }
}
