//! Winit + egui application shell for the Gaussian Slicer wgpu port.

use std::{
    env, fs, mem,
    path::{Path, PathBuf},
    rc::Rc,
    sync::mpsc,
    time::Instant,
};

use anyhow::{Context as _, Result};
use bytemuck::cast_slice;
use egui::{Context as EguiContext, ViewportId};
use egui_wgpu::{wgpu, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use glam::Vec3;
use image::ImageBuffer;
use pollster::block_on;
use slicer_core::{
    config::{RendererSettings, VisualizationConfig},
    gmm, load_gaussians_from_ply, Gaussian3D,
};
use slicer_gfx::{RendererResources, ShaderModules};
use tracer::init_tracing;
use wgpu::SurfaceError;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
compile_error!("wasm32 builds are out of scope for the Gaussian Slicer port.");

fn main() -> Result<()> {
    init_tracing();

    let mut exit_after_ms: Option<u64> = None;
    let mut gaussian_ply_path: Option<PathBuf> = None;
    let mut capture_frame_path: Option<PathBuf> = None;
    let mut dump_gaussians_path: Option<PathBuf> = None;
    let mut dump_precalc_path: Option<PathBuf> = None;
    let mut dump_dynamic_path: Option<PathBuf> = None;
    let mut dump_kernel_config_path: Option<PathBuf> = None;
    let mut dump_precalc_debug_path: Option<PathBuf> = None;
    let mut dump_density_path: Option<PathBuf> = None;
    let mut export_volume_path: Option<PathBuf> = None;
    let mut export_log_normalized = false;
    let mut num_distributions_override: Option<u32> = None;
    let mut grid_resolution_override: Option<u32> = None;
    let mut seed_override: Option<u64> = None;
    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--exit-after-ms=") {
            let millis: u64 = value
                .parse()
                .context("invalid value for --exit-after-ms (expected integer milliseconds)")?;
            exit_after_ms = Some(millis);
        } else if let Some(value) = arg.strip_prefix("--gaussian-ply=") {
            gaussian_ply_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--num-distributions=") {
            let parsed: u32 = value
                .parse()
                .context("invalid value for --num-distributions (expected u32)")?;
            num_distributions_override = Some(parsed);
        } else if let Some(value) = arg.strip_prefix("--grid-resolution=") {
            let parsed: u32 = value
                .parse()
                .context("invalid value for --grid-resolution (expected u32)")?;
            grid_resolution_override = Some(parsed.max(1));
        } else if let Some(value) = arg.strip_prefix("--seed=") {
            let parsed: u64 = value
                .parse()
                .context("invalid value for --seed (expected u64)")?;
            seed_override = Some(parsed);
        } else if let Some(value) = arg.strip_prefix("--capture-frame=") {
            capture_frame_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-density-raw=") {
            dump_density_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-gaussians-raw=") {
            dump_gaussians_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-precalc-raw=") {
            dump_precalc_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-dynamic-raw=") {
            dump_dynamic_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-kernel-config-raw=") {
            dump_kernel_config_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--dump-precalc-debug-raw=") {
            dump_precalc_debug_path = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--export-volume=") {
            export_volume_path = Some(PathBuf::from(value));
        } else if arg == "--export-log-normalized" {
            export_log_normalized = true;
        }
    }

    let event_loop = EventLoop::new().context("failed to create event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);

    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_title("Gaussian Slicer (wgpu)")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
        )
        .context("failed to create window")?;

    let window = Rc::new(window);

    let (mut gpu_context, shaders) = block_on(slicer_gfx::init(window.as_ref()))?;

    let mut settings = RendererSettings::default();
    if let Some(num) = num_distributions_override {
        settings.num_distributions = num;
    }
    if let Some(resolution) = grid_resolution_override {
        settings.grid_resolution = resolution;
    }
    if let Some(seed) = seed_override {
        settings.seed = Some(seed);
    }
    let mut gaussians = if let Some(path) = gaussian_ply_path.as_ref() {
        let dataset = load_gaussians_from_ply(path)
            .with_context(|| format!("failed to load Gaussian splat file {}", path.display()))?;
        settings.num_distributions = dataset.len() as u32;
        dataset
    } else {
        gmm::generate(&settings).0
    };
    tracing::info!(
        "Initialized {} gaussians (sample weight {:.3e})",
        gaussians.len(),
        gaussians.first().map(|g| g.weight).unwrap_or(0.0)
    );
    tracing::debug!("Sample gaussian: {:?}", gaussians.first());

    let mut renderer = RendererResources::new(&gpu_context, &shaders, &settings, &gaussians)
        .context("failed to create renderer resources")?;
    renderer.write_gaussians(&gaussians);
    renderer.run_precalc().context("failed to run K1")?;
    renderer
        .run_update_and_evaluate()
        .context("failed to run K2/K3")?;

    let mut visualization_config = VisualizationConfig::from_settings(&settings);
    renderer.update_visualization_config(&visualization_config);

    if let Some(export_path) = export_volume_path {
        export_volume_cli(
            &mut renderer,
            &mut settings,
            &visualization_config,
            &export_path,
            export_log_normalized,
        )?;
        return Ok(());
    }

    let egui_ctx = EguiContext::default();
    let mut egui_state = EguiWinitState::new(
        egui_ctx.clone(),
        ViewportId::ROOT,
        window.as_ref(),
        Some(window.scale_factor() as f32),
        window.theme(),
        Some(gpu_context.device.limits().max_texture_dimension_2d as usize),
    );
    let mut egui_renderer = egui_wgpu::Renderer::new(
        renderer.device(),
        gpu_context.surface_config.format,
        None,
        1,
        false,
    );

    let mut last_frame_time = Instant::now();
    let app_start_time = Instant::now();

    let window_handle = Rc::clone(&window);

    #[allow(deprecated)]
    event_loop
        .run(move |event, target| {
            let window = window_handle.as_ref();
            match event {
                Event::WindowEvent {
                    window_id,
                    ref event,
                } if window_id == window.id() => {
                    let egui_response = egui_state.on_window_event(&window, event);
                    if egui_response.repaint {
                        window.request_redraw();
                    }
                    if egui_response.consumed {
                        return;
                    }

                    match event {
                        WindowEvent::CloseRequested => target.exit(),
                        WindowEvent::Resized(new_size) => {
                            if new_size.width > 0 && new_size.height > 0 {
                                gpu_context.surface_config.width = new_size.width;
                                gpu_context.surface_config.height = new_size.height;
                                gpu_context
                                    .surface
                                    .configure(&gpu_context.device, &gpu_context.surface_config);
                            }
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {
                            let new_size = window.inner_size();
                            if new_size.width > 0 && new_size.height > 0 {
                                gpu_context.surface_config.width = new_size.width;
                                gpu_context.surface_config.height = new_size.height;
                                gpu_context
                                    .surface
                                    .configure(&gpu_context.device, &gpu_context.surface_config);
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            match render_frame(
                                &window,
                                &egui_ctx,
                                &mut egui_state,
                                &mut egui_renderer,
                                &mut gpu_context,
                                &shaders,
                                &mut renderer,
                                &mut settings,
                                &mut visualization_config,
                                &mut gaussians,
                                gaussian_ply_path.as_deref(),
                                &mut capture_frame_path,
                                &mut dump_density_path,
                                &mut dump_gaussians_path,
                                &mut dump_precalc_path,
                                &mut dump_dynamic_path,
                                &mut dump_kernel_config_path,
                                &mut dump_precalc_debug_path,
                                last_frame_time,
                            ) {
                                Ok(captured) => {
                                    if captured {
                                        target.exit();
                                    }
                                }
                                Err(err) => {
                                    eprintln!("frame render error: {err:?}");
                                }
                            }
                            last_frame_time = Instant::now();
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    if let Some(limit_ms) = exit_after_ms {
                        if app_start_time.elapsed().as_millis() as u64 >= limit_ms {
                            target.exit();
                            return;
                        }
                    }
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .map_err(Into::into)
}

fn render_frame(
    window: &Window,
    egui_ctx: &EguiContext,
    egui_state: &mut EguiWinitState,
    egui_renderer: &mut egui_wgpu::Renderer,
    gpu_context: &mut slicer_gfx::GpuContext<'_>,
    shaders: &ShaderModules,
    renderer: &mut RendererResources,
    settings: &mut RendererSettings,
    viz_config: &mut VisualizationConfig,
    gaussians: &mut Vec<Gaussian3D>,
    gaussian_ply: Option<&Path>,
    capture_frame: &mut Option<PathBuf>,
    dump_density: &mut Option<PathBuf>,
    dump_gaussians: &mut Option<PathBuf>,
    dump_precalc: &mut Option<PathBuf>,
    dump_dynamic: &mut Option<PathBuf>,
    dump_kernel_config: &mut Option<PathBuf>,
    dump_precalc_debug: &mut Option<PathBuf>,
    previous_frame_time: Instant,
) -> Result<bool> {
    let raw_input = egui_state.take_egui_input(window);
    let frame_time_ms = previous_frame_time.elapsed().as_secs_f64() * 1000.0;
    let previous_settings = settings.clone();
    let pending_capture = capture_frame.as_ref().cloned();
    let pending_density_dump = dump_density.as_ref().cloned();
    let pending_gaussian_dump = dump_gaussians.as_ref().cloned();
    let pending_precalc_dump = dump_precalc.as_ref().cloned();
    let pending_dynamic_dump = dump_dynamic.as_ref().cloned();
    let pending_kernel_config_dump = dump_kernel_config.as_ref().cloned();
    let pending_precalc_debug_dump = dump_precalc_debug.as_ref().cloned();

    let full_output = egui_ctx.run(raw_input, |ctx| {
        egui::TopBottomPanel::top("top_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Distributions: {}    Frame time: {:>5.2} ms",
                        settings.num_distributions, frame_time_ms
                    ));
                    if let Some(path) = gaussian_ply {
                        ui.label(format!("Gaussian source: {}", path.display()));
                    }
                });
            });

        egui::SidePanel::right("sidebar")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| {
                ui.heading("Visualization");
                let mut log_scale = viz_config.log_scale != 0;
                if ui.checkbox(&mut log_scale, "Log scale").changed() {
                    viz_config.log_scale = if log_scale { 1 } else { 0 };
                }
                let mut invert = viz_config.invert != 0;
                if ui.checkbox(&mut invert, "Invert colormap").changed() {
                    viz_config.invert = if invert { 1 } else { 0 };
                }
                ui.add(egui::Slider::new(&mut viz_config.colormap_index, 0..=9).text("Colormap"));
                ui.add(egui::Slider::new(&mut viz_config.color_levels, 0..=32).text("Levels"));

                let mut density_min = viz_config.density_min;
                if ui
                    .add(
                        egui::Slider::new(
                            &mut density_min,
                            1e-9..=viz_config.density_max.max(1e-8) - 1e-8,
                        )
                        .logarithmic(true)
                        .text("Density min"),
                    )
                    .changed()
                {
                    viz_config.density_min = density_min.max(1e-9);
                }

                let mut density_max = viz_config.density_max;
                if ui
                    .add(
                        egui::Slider::new(&mut density_max, viz_config.density_min + 1e-8..=1.0)
                            .logarithmic(true)
                            .text("Density max"),
                    )
                    .changed()
                {
                    viz_config.density_max = density_max.max(viz_config.density_min * 1.01);
                }

                ui.add(
                    egui::Slider::new(&mut viz_config.outline_width, 0.0..=4.0)
                        .text("Outline width"),
                );
                ui.label("Plane offset");
                ui.add(
                    egui::Slider::new(
                        &mut settings.plane_offset,
                        settings.grid_min..=settings.grid_max,
                    )
                    .clamping(egui::SliderClamping::Always)
                    .text(""),
                );

                ui.separator();
                ui.heading("Generator");
                ui.collapsing("Parameters", |ui| {
                    let mut distributions = settings.num_distributions as i64;
                    if ui
                        .add(
                            egui::DragValue::new(&mut distributions)
                                .speed(1000.0)
                                .prefix("Distributions: "),
                        )
                        .changed()
                    {
                        settings.num_distributions = distributions.clamp(1, 200_000) as u32;
                    }

                    let mut grid_resolution = settings.grid_resolution as i64;
                    if ui
                        .add(
                            egui::DragValue::new(&mut grid_resolution)
                                .speed(16.0)
                                .prefix("Grid resolution: "),
                        )
                        .changed()
                    {
                        settings.grid_resolution = grid_resolution.clamp(16, 1024) as u32;
                    }

                    let mut grid_min = settings.grid_min;
                    let mut grid_max = settings.grid_max;
                    ui.add(
                        egui::DragValue::new(&mut grid_min)
                            .speed(0.5)
                            .prefix("Grid min: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut grid_max)
                            .speed(0.5)
                            .prefix("Grid max: "),
                    );
                    if grid_max <= grid_min {
                        grid_max = grid_min + 0.5;
                    }
                    settings.grid_min = grid_min;
                    settings.grid_max = grid_max;

                    let mut mean = settings.mean_std_dev;
                    if ui
                        .add(
                            egui::Slider::new(&mut mean, 1e-3..=10.0)
                                .logarithmic(true)
                                .text("Mean std dev"),
                        )
                        .changed()
                    {
                        settings.mean_std_dev = mean.max(1e-3);
                    }

                    let mut cov_scale = settings.covariance_scale;
                    if ui
                        .add(
                            egui::Slider::new(&mut cov_scale, 1e-4..=10.0)
                                .logarithmic(true)
                                .text("Covariance scale"),
                        )
                        .changed()
                    {
                        settings.covariance_scale = cov_scale.max(1e-4);
                    }

                    let mut seed_enabled = settings.seed.is_some();
                    if ui.checkbox(&mut seed_enabled, "Use fixed seed").changed() && !seed_enabled {
                        settings.seed = None;
                    }
                    if seed_enabled {
                        let mut seed_value = settings.seed.unwrap_or(0);
                        if ui
                            .add(
                                egui::DragValue::new(&mut seed_value)
                                    .speed(1.0)
                                    .prefix("Seed: "),
                            )
                            .changed()
                        {
                            settings.seed = Some(seed_value);
                        }
                    }

                    let mut plane = [
                        settings.plane_normal.x,
                        settings.plane_normal.y,
                        settings.plane_normal.z,
                    ];
                    ui.horizontal(|ui| {
                        ui.label("Plane normal:");
                        for component in &mut plane {
                            ui.add(egui::DragValue::new(component).speed(0.05));
                        }
                    });
                    let plane_vec = Vec3::new(plane[0], plane[1], plane[2]);
                    settings.plane_normal = if plane_vec.length_squared() < 1e-6 {
                        Vec3::X
                    } else {
                        plane_vec.normalize()
                    };
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                ui.allocate_space(ui.available_size());
            });
    });

    if gaussian_ply.is_some() {
        settings.num_distributions = gaussians.len().min(u32::MAX as usize) as u32;
    }

    settings.use_log_scale = viz_config.log_scale != 0;
    settings.density_min = viz_config.density_min;
    settings.density_max = viz_config.density_max;

    egui_state.handle_platform_output(window, full_output.platform_output);

    let gaussian_params_changed = gaussian_ply.is_none()
        && (settings.num_distributions != previous_settings.num_distributions
            || (settings.mean_std_dev - previous_settings.mean_std_dev).abs() > f32::EPSILON
            || (settings.covariance_scale - previous_settings.covariance_scale).abs()
                > f32::EPSILON
            || settings.seed != previous_settings.seed);

    let renderer_reset_required = gaussian_params_changed
        || settings.grid_resolution != previous_settings.grid_resolution
        || (settings.grid_min - previous_settings.grid_min).abs() > f32::EPSILON
        || (settings.grid_max - previous_settings.grid_max).abs() > f32::EPSILON;

    if renderer_reset_required {
        if gaussian_params_changed {
            let (new_gaussians, _) = gmm::generate(settings);
            *gaussians = new_gaussians;
        }
        *renderer = RendererResources::new(gpu_context, shaders, settings, &gaussians)
            .context("failed to recreate renderer resources")?;
        renderer.write_gaussians(&gaussians);
        renderer.update_visualization_config(viz_config);
        renderer.update_kernel_config(&settings.kernel_config());
        renderer
            .run_precalc()
            .context("failed to execute pre-calculation pass")?;
    } else {
        renderer.update_kernel_config(&settings.kernel_config());
        if (settings.plane_normal - previous_settings.plane_normal).length_squared() > f32::EPSILON
        {
            renderer
                .run_precalc()
                .context("failed to execute pre-calculation pass")?;
        }
    }

    renderer.update_visualization_config(viz_config);
    renderer
        .run_update_and_evaluate()
        .context("failed to execute compute passes")?;

    for (id, image_delta) in &full_output.textures_delta.set {
        egui_renderer.update_texture(renderer.device(), renderer.queue(), *id, image_delta);
    }
    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }

    let surface_texture = match gpu_context.surface.get_current_texture() {
        Ok(surface_texture) => surface_texture,
        Err(SurfaceError::Outdated) | Err(SurfaceError::Lost) => {
            gpu_context
                .surface
                .configure(&gpu_context.device, &gpu_context.surface_config);
            return Ok(false);
        }
        Err(SurfaceError::Timeout) | Err(SurfaceError::Other) => return Ok(false),
        Err(SurfaceError::OutOfMemory) => {
            eprintln!("wgpu surface out of memory, shutting down");
            return Ok(false);
        }
    };

    let surface_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = gpu_context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GaussianSlicer Encoder"),
        });

    let capture_job = pending_capture.map(|path| {
        let width = settings.grid_resolution.max(1);
        let height = settings.grid_resolution.max(1);
        let bytes_per_pixel = 4u32;
        let bytes_per_row = align_to(
            width * bytes_per_pixel,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u32,
        );
        let buffer_size = bytes_per_row as u64 * height as u64;
        let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu_context.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        CaptureJob {
            texture,
            view,
            buffer,
            path,
            width,
            height,
            bytes_per_row,
        }
    });

    let density_dump = pending_density_dump.map(|path| {
        let width = settings.grid_resolution.max(1);
        let height = settings.grid_resolution.max(1);
        let bytes_per_pixel = 4u32;
        let bytes_per_row = align_to(
            width * bytes_per_pixel,
            wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u32,
        );
        let buffer_size = bytes_per_row as u64 * height as u64;
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Dump Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        DensityDumpJob {
            buffer,
            path,
            width,
            height,
            bytes_per_row,
        }
    });

    let gaussian_dump = pending_gaussian_dump.map(|path| {
        let size = renderer.gaussians_buffer_size();
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gaussian Dump Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        BufferDumpJob { buffer, path, size }
    });

    let precalc_dump = pending_precalc_dump.map(|path| {
        let size = renderer.precalc_buffer_size();
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Precalc Dump Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        BufferDumpJob { buffer, path, size }
    });

    let dynamic_dump = pending_dynamic_dump.map(|path| {
        let size = renderer.dynamic_buffer_size();
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic Dump Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        BufferDumpJob { buffer, path, size }
    });

    let kernel_config_dump = pending_kernel_config_dump.map(|path| {
        let size = renderer.kernel_config_buffer_size();
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Kernel Config Dump Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        BufferDumpJob { buffer, path, size }
    });

    let precalc_debug_dump = pending_precalc_debug_dump.map(|path| {
        let size = renderer.precalc_debug_buffer_size();
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Precalc Debug Dump Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        BufferDumpJob { buffer, path, size }
    });

    if let Some(job) = capture_job.as_ref() {
        renderer.encode_visualization_pass(&mut encoder, &job.view);
    }

    if let Some(job) = density_dump.as_ref() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: renderer.density_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &job.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(job.bytes_per_row),
                    rows_per_image: Some(job.height),
                },
            },
            wgpu::Extent3d {
                width: job.width,
                height: job.height,
                depth_or_array_layers: 1,
            },
        );
    }

    if let Some(job) = gaussian_dump.as_ref() {
        encoder.copy_buffer_to_buffer(renderer.gaussians_buffer(), 0, &job.buffer, 0, job.size);
    }

    if let Some(job) = precalc_dump.as_ref() {
        encoder.copy_buffer_to_buffer(renderer.precalc_buffer(), 0, &job.buffer, 0, job.size);
    }

    if let Some(job) = dynamic_dump.as_ref() {
        encoder.copy_buffer_to_buffer(renderer.dynamic_buffer(), 0, &job.buffer, 0, job.size);
    }

    if let Some(job) = kernel_config_dump.as_ref() {
        encoder.copy_buffer_to_buffer(renderer.kernel_config_buffer(), 0, &job.buffer, 0, job.size);
    }

    if let Some(job) = precalc_debug_dump.as_ref() {
        encoder.copy_buffer_to_buffer(renderer.precalc_debug_buffer(), 0, &job.buffer, 0, job.size);
    }

    renderer.encode_visualization_pass(&mut encoder, &surface_view);

    let pixels_per_point = window.scale_factor() as f32;
    let paint_jobs = egui_ctx.tessellate(full_output.shapes, pixels_per_point);
    let screen_descriptor = ScreenDescriptor {
        size_in_pixels: [
            gpu_context.surface_config.width,
            gpu_context.surface_config.height,
        ],
        pixels_per_point,
    };

    let egui_cmd_buffers = egui_renderer.update_buffers(
        renderer.device(),
        renderer.queue(),
        &mut encoder,
        &paint_jobs,
        &screen_descriptor,
    );

    {
        let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let render_pass: &mut wgpu::RenderPass<'static> = unsafe { mem::transmute(&mut egui_pass) };
        egui_renderer.render(render_pass, &paint_jobs, &screen_descriptor);
    }

    if let Some(job) = capture_job.as_ref() {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &job.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &job.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(job.bytes_per_row),
                    rows_per_image: Some(job.height),
                },
            },
            wgpu::Extent3d {
                width: job.width,
                height: job.height,
                depth_or_array_layers: 1,
            },
        );
    }

    let mut submissions = egui_cmd_buffers;
    submissions.push(encoder.finish());

    renderer.queue().submit(submissions);
    let _ = renderer.device().poll(wgpu::PollType::Poll);

    let mut capture_done = false;
    if let Some(job) = capture_job {
        let buffer_slice = job.buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = renderer.device().poll(wgpu::PollType::Wait);
        receiver
            .recv()
            .expect("capture callback dropped")
            .context("failed to map capture buffer")?;
        {
            let data = buffer_slice.get_mapped_range();
            let row_pitch = (job.width * 4) as usize;
            let mut pixels = vec![0u8; row_pitch * job.height as usize];
            for (row, chunk) in pixels.chunks_exact_mut(row_pitch).enumerate() {
                let offset = row * job.bytes_per_row as usize;
                chunk.copy_from_slice(&data[offset..offset + row_pitch]);
            }
            drop(data);
            job.buffer.unmap();

            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }

            let image = ImageBuffer::<image::Rgba<u8>, _>::from_raw(job.width, job.height, pixels)
                .context("failed to create capture image buffer")?;
            image
                .save(&job.path)
                .with_context(|| format!("failed to save {}", job.path.display()))?;
            capture_done = true;
        }
    }

    let mut density_dumped = false;
    if let Some(job) = density_dump {
        let buffer_slice = job.buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = renderer.device().poll(wgpu::PollType::Wait);
        receiver
            .recv()
            .expect("density dump callback dropped")
            .context("failed to map density buffer")?;
        let floats = {
            let data = buffer_slice.get_mapped_range();
            let row_pitch = (job.width * std::mem::size_of::<f32>() as u32) as usize;
            let mut values = vec![0f32; (job.width * job.height) as usize];
            for (row, chunk) in values.chunks_exact_mut(job.width as usize).enumerate() {
                let offset = row * job.bytes_per_row as usize;
                let row_bytes = &data[offset..offset + row_pitch];
                let row_values: &[f32] = cast_slice(row_bytes);
                chunk.copy_from_slice(row_values);
            }
            values
        };
        job.buffer.unmap();

        let (min_val, max_val) = floats.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min_v, max_v), &value| (min_v.min(value), max_v.max(value)),
        );

        fs::write(&job.path, cast_slice(&floats))
            .with_context(|| format!("failed to write {}", job.path.display()))?;
        println!(
            "Dumped density texture to {} (min={:.3e}, max={:.3e})",
            job.path.display(),
            min_val,
            max_val
        );
        density_dumped = true;
    }

    let mut gaussian_dumped = false;
    let mut precalc_dumped = false;
    let mut dynamic_dumped = false;
    let mut kernel_config_dumped = false;
    let mut precalc_debug_dumped = false;
    {
        let process_buffer_dump = |job_opt: Option<BufferDumpJob>,
                                   label: &str,
                                   flag: &mut Option<PathBuf>,
                                   dumped: &mut bool|
         -> Result<()> {
            if let Some(job) = job_opt {
                let buffer_slice = job.buffer.slice(..job.size);
                let (sender, receiver) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
                let _ = renderer.device().poll(wgpu::PollType::Wait);
                receiver
                    .recv()
                    .unwrap_or_else(|_| panic!("{label} dump callback dropped"))
                    .with_context(|| format!("failed to map {label} buffer"))?;
                let data = buffer_slice.get_mapped_range();
                let bytes = data.to_vec();
                drop(data);
                job.buffer.unmap();

                fs::write(&job.path, &bytes)
                    .with_context(|| format!("failed to write {}", job.path.display()))?;
                println!(
                    "Dumped {label} buffer to {} ({} bytes)",
                    job.path.display(),
                    bytes.len()
                );
                *dumped = true;
                flag.take();
            }
            Ok(())
        };

        process_buffer_dump(
            gaussian_dump,
            "gaussian",
            dump_gaussians,
            &mut gaussian_dumped,
        )?;
        process_buffer_dump(precalc_dump, "precalc", dump_precalc, &mut precalc_dumped)?;
        process_buffer_dump(dynamic_dump, "dynamic", dump_dynamic, &mut dynamic_dumped)?;
        process_buffer_dump(
            kernel_config_dump,
            "kernel config",
            dump_kernel_config,
            &mut kernel_config_dumped,
        )?;
        process_buffer_dump(
            precalc_debug_dump,
            "precalc debug",
            dump_precalc_debug,
            &mut precalc_debug_dumped,
        )?;
    }

    surface_texture.present();

    if capture_done {
        capture_frame.take();
    }
    if density_dumped {
        dump_density.take();
    }
    Ok(capture_done
        || density_dumped
        || gaussian_dumped
        || precalc_dumped
        || dynamic_dumped
        || kernel_config_dumped
        || precalc_debug_dumped)
}

fn export_volume_cli(
    renderer: &mut RendererResources,
    settings: &mut RendererSettings,
    viz: &VisualizationConfig,
    export_path: &Path,
    log_normalized: bool,
) -> Result<()> {
    let resolution = settings.grid_resolution.max(1);
    let slice_elements = (resolution * resolution) as usize;
    let mut volume = vec![0f32; slice_elements * resolution as usize];
    let mut kernel_config = settings.kernel_config();
    let range = settings.grid_max - settings.grid_min;
    let denom = (resolution - 1).max(1);

    for zi in 0..resolution {
        let t = zi as f32 / denom as f32;
        let offset = settings.grid_min + t * range;
        settings.plane_offset = offset;
        kernel_config.grid_params[0] = offset;
        renderer.update_kernel_config(&kernel_config);
        renderer
            .run_update_and_evaluate()
            .with_context(|| format!("failed to evaluate slice {}", zi))?;

        let mut slice = read_density_slice(renderer, resolution)?;
        if log_normalized {
            normalize_slice_log(&mut slice, viz);
        }
        let base = zi as usize * slice_elements;
        volume[base..base + slice_elements].copy_from_slice(&slice);
    }

    let (mhd_path, raw_path) = write_volume_files(&volume, resolution, settings, export_path)?;
    println!(
        "Exported volume ({}^3) to {} + {}",
        resolution,
        mhd_path.display(),
        raw_path.display()
    );
    Ok(())
}

fn read_density_slice(renderer: &RendererResources, resolution: u32) -> Result<Vec<f32>> {
    let width_bytes = resolution * mem::size_of::<f32>() as u32;
    let bytes_per_row = align_to(width_bytes, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = bytes_per_row as u64 * resolution as u64;
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Density Readback Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = renderer
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Density Readback Encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: renderer.density_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(resolution),
            },
        },
        wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: 1,
        },
    );
    renderer.queue().submit(Some(encoder.finish()));

    let buffer_slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = renderer.device().poll(wgpu::PollType::Wait);
    receiver
        .recv()
        .expect("density readback callback dropped")
        .context("failed to map density buffer for export")?;

    let mut values = vec![0f32; (resolution * resolution) as usize];
    {
        let data = buffer_slice.get_mapped_range();
        for (row, chunk) in values.chunks_exact_mut(resolution as usize).enumerate() {
            let offset = row * bytes_per_row as usize;
            let row_bytes = &data[offset..offset + width_bytes as usize];
            let row_values: &[f32] = cast_slice(row_bytes);
            chunk.copy_from_slice(row_values);
        }
    }
    buffer.unmap();
    Ok(values)
}

fn normalize_slice_log(slice: &mut [f32], viz: &VisualizationConfig) {
    let min_pos = 1e-12f32;
    let v_min = viz.density_min.max(min_pos);
    let v_max = viz.density_max.max(v_min + min_pos);
    let log_v_min = v_min.ln();
    let log_den = (v_max.ln() - log_v_min).max(min_pos);
    let inv_log_den = 1.0 / log_den;
    let invert = viz.invert != 0;
    for value in slice.iter_mut() {
        let clamped = value.max(min_pos);
        let mut t = (clamped.ln() - log_v_min) * inv_log_den;
        if invert {
            t = 1.0 - t;
        }
        *value = t.clamp(0.0, 1.0);
    }
}

fn write_volume_files(
    volume: &[f32],
    resolution: u32,
    settings: &RendererSettings,
    export_path: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let mhd_path;
    let raw_path;
    match export_path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("mhd") => {
            mhd_path = export_path.to_path_buf();
            raw_path = export_path.with_extension("raw");
        }
        Some(ext) if ext.eq_ignore_ascii_case("raw") => {
            raw_path = export_path.to_path_buf();
            mhd_path = export_path.with_extension("mhd");
        }
        _ => {
            mhd_path = export_path.with_extension("mhd");
            raw_path = export_path.with_extension("raw");
        }
    }

    fs::write(&raw_path, cast_slice(volume))
        .with_context(|| format!("failed to write RAW volume {}", raw_path.display()))?;

    let denom = (resolution - 1).max(1);
    let spacing = f64::from(settings.grid_max - settings.grid_min) / f64::from(denom);
    let header = format!(
        "ObjectType = Image\nNDims = 3\nDimSize = {0} {0} {0}\nElementType = MET_FLOAT\nElementSpacing = {1} {1} {1}\nElementByteOrderMSB = False\nElementDataFile = {2}\n",
        resolution,
        spacing,
        raw_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("density.raw")
    );
    fs::write(&mhd_path, header)
        .with_context(|| format!("failed to write MHD header {}", mhd_path.display()))?;

    Ok((mhd_path, raw_path))
}

fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}

struct CaptureJob {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    path: PathBuf,
    width: u32,
    height: u32,
    bytes_per_row: u32,
}

struct DensityDumpJob {
    buffer: wgpu::Buffer,
    path: PathBuf,
    width: u32,
    height: u32,
    bytes_per_row: u32,
}

struct BufferDumpJob {
    buffer: wgpu::Buffer,
    path: PathBuf,
    size: u64,
}

mod tracer {
    pub fn init_tracing() {
        let _ = tracing_subscriber::fmt().try_init();
    }
}
