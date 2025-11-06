//! Winit + egui application shell for the Gaussian Slicer wgpu port.

use std::{env, mem, rc::Rc, time::Instant};

use anyhow::{Context as _, Result};
use egui::{Context as EguiContext, ViewportId};
use egui_wgpu::{wgpu, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use glam::Vec3;
use pollster::block_on;
use slicer_core::{
    config::{RendererSettings, VisualizationConfig},
    gmm,
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
    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--exit-after-ms=") {
            let millis: u64 = value
                .parse()
                .context("invalid value for --exit-after-ms (expected integer milliseconds)")?;
            exit_after_ms = Some(millis);
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
    let (gaussians, _seed) = gmm::generate(&settings);

    let mut renderer = RendererResources::new(&gpu_context, &shaders, &settings, &gaussians)
        .context("failed to create renderer resources")?;
    renderer.run_precalc().context("failed to run K1")?;
    renderer
        .run_update_and_evaluate()
        .context("failed to run K2/K3")?;

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

    let mut visualization_config = VisualizationConfig::from_settings(&settings);
    renderer.update_visualization_config(&visualization_config);

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
                            if let Err(err) = render_frame(
                                &window,
                                &egui_ctx,
                                &mut egui_state,
                                &mut egui_renderer,
                                &mut gpu_context,
                                &shaders,
                                &mut renderer,
                                &mut settings,
                                &mut visualization_config,
                                last_frame_time,
                            ) {
                                eprintln!("frame render error: {err:?}");
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
    previous_frame_time: Instant,
) -> Result<()> {
    let raw_input = egui_state.take_egui_input(window);
    let frame_time_ms = previous_frame_time.elapsed().as_secs_f64() * 1000.0;
    let previous_settings = settings.clone();

    let full_output = egui_ctx.run(raw_input, |ctx| {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.label(format!(
                "Distributions: {}    Frame time: {:>5.2} ms",
                settings.num_distributions, frame_time_ms
            ));

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
                egui::Slider::new(&mut viz_config.outline_width, 0.0..=4.0).text("Outline width"),
            );
            ui.add(
                egui::Slider::new(
                    &mut settings.plane_offset,
                    settings.grid_min..=settings.grid_max,
                )
                .text("Plane offset"),
            );
            ui.collapsing("Generator Settings", |ui| {
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
    });

    settings.use_log_scale = viz_config.log_scale != 0;
    settings.density_min = viz_config.density_min;
    settings.density_max = viz_config.density_max;

    egui_state.handle_platform_output(window, full_output.platform_output);

    let data_regen_required = settings.num_distributions != previous_settings.num_distributions
        || settings.grid_resolution != previous_settings.grid_resolution
        || (settings.mean_std_dev - previous_settings.mean_std_dev).abs() > f32::EPSILON
        || (settings.covariance_scale - previous_settings.covariance_scale).abs() > f32::EPSILON
        || settings.seed != previous_settings.seed
        || (settings.grid_min - previous_settings.grid_min).abs() > f32::EPSILON
        || (settings.grid_max - previous_settings.grid_max).abs() > f32::EPSILON;

    if data_regen_required {
        let (gaussians, _) = gmm::generate(settings);
        *renderer = RendererResources::new(gpu_context, shaders, settings, &gaussians)
            .context("failed to recreate renderer resources")?;
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
            return Ok(());
        }
        Err(SurfaceError::Timeout) | Err(SurfaceError::Other) => return Ok(()),
        Err(SurfaceError::OutOfMemory) => {
            eprintln!("wgpu surface out of memory, shutting down");
            return Ok(());
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

    let mut submissions = egui_cmd_buffers;
    submissions.push(encoder.finish());

    renderer.queue().submit(submissions);
    surface_texture.present();

    Ok(())
}

mod tracer {
    pub fn init_tracing() {
        let _ = tracing_subscriber::fmt().try_init();
    }
}
