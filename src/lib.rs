pub mod gpu;
pub mod input;
pub mod physics;
pub mod ui;

#[cfg(target_arch = "wasm32")]
mod web_app {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::Duration;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use web_time::Instant;
    use winit::application::ApplicationHandler;
    use winit::dpi::LogicalSize;
    use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
    use winit::keyboard::PhysicalKey;
    use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
    use winit::window::{CursorGrabMode, Window, WindowId};

    use crate::gpu::context::GpuContext;
    use crate::gpu::pipeline::ComputeRenderPipeline;
    use crate::gpu::uniforms::FrameUniforms;
    use crate::input::InputState;
    use crate::physics::observer::ObserverState;
    use crate::ui::hud::{draw_hud, SimParams};

    struct AppRuntime {
        window: Arc<Window>,
        gpu: GpuContext,
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
        pipeline: ComputeRenderPipeline,
        egui_ctx: egui::Context,
        egui_state: egui_winit::State,
        egui_renderer: egui_wgpu::Renderer,
        input: InputState,
        observer: ObserverState,
        sim_params: SimParams,
        applied_render_scale: f32,
        last_cursor_pos: Option<(f64, f64)>,
        last_frame_time: Instant,
        snapshot_notice: Rc<RefCell<Option<(String, Instant)>>>,
        fps_smooth: f32,
        sim_time: f32,
        planet_time: f32,
        frame_index: u32,
    }

    fn scaled_dims(surface_w: u32, surface_h: u32, scale: f32) -> (u32, u32) {
        let s = scale.clamp(0.35, 1.00);
        let rw = ((surface_w as f32 * s).round() as u32).clamp(64, 3840);
        let rh = ((surface_h as f32 * s).round() as u32).clamp(64, 2160);
        (rw, rh)
    }

    impl AppRuntime {
        async fn new(window: Arc<Window>) -> Result<Self, String> {
            let (gpu, surface) = GpuContext::new_with_window(window.clone()).await?;
            let caps = surface.get_capabilities(&gpu.adapter);
            let surface_format = caps
                .formats
                .iter()
                .copied()
                .find(|f| {
                    matches!(
                        f,
                        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                    )
                })
                .unwrap_or(caps.formats[0]);

            let size = window.inner_size();
            let sw = size.width.clamp(64, 4096);
            let sh = size.height.clamp(64, 4096);
            let sim_params = SimParams::default();
            let (rw, rh) = scaled_dims(sw, sh, sim_params.render_scale);

            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: sw,
                height: sh,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&gpu.device, &surface_config);

            let pipeline =
                ComputeRenderPipeline::new(&gpu, rw, rh, Some(surface_format)).await?;

            let egui_ctx = egui::Context::default();
            let egui_state = egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                &window,
                Some(window.scale_factor() as f32),
                None,
                None,
            );
            let egui_renderer =
                egui_wgpu::Renderer::new(&gpu.device, surface_format, None, 1, false);

            Ok(Self {
                window,
                gpu,
                surface,
                surface_config,
                pipeline,
                egui_ctx,
                egui_state,
                egui_renderer,
                input: InputState::new(),
                observer: ObserverState::new_hero(),
                sim_params,
                applied_render_scale: sim_params.render_scale,
                last_cursor_pos: None,
                last_frame_time: Instant::now(),
                snapshot_notice: Rc::new(RefCell::new(None)),
                fps_smooth: 60.0,
                sim_time: 0.0,
                planet_time: 0.0,
                frame_index: 0,
            })
        }

        fn set_cursor_lock(&mut self, lock: bool) {
            self.input.cursor_locked = lock;
            if lock {
                self.window.set_cursor_visible(false);
                if self.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                    let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
                }
            } else {
                let _ = self.window.set_cursor_grab(CursorGrabMode::None);
                self.window.set_cursor_visible(true);
            }
        }

        fn resize(&mut self, width: u32, height: u32) {
            self.input.on_resize();
            self.last_cursor_pos = None;
            if width == 0 || height == 0 {
                return;
            }
            let sw = width.clamp(64, 4096);
            let sh = height.clamp(64, 4096);
            self.surface_config.width = sw;
            self.surface_config.height = sh;
            self.surface.configure(&self.gpu.device, &self.surface_config);
            let (rw, rh) = scaled_dims(sw, sh, self.sim_params.render_scale);
            self.pipeline.resize(&self.gpu, rw, rh);
            self.applied_render_scale = self.sim_params.render_scale;
        }

        fn sync_render_scale(&mut self) {
            if (self.sim_params.render_scale - self.applied_render_scale).abs() > 0.005 {
                let (rw, rh) = scaled_dims(
                    self.surface_config.width,
                    self.surface_config.height,
                    self.sim_params.render_scale,
                );
                self.pipeline.resize(&self.gpu, rw, rh);
                self.applied_render_scale = self.sim_params.render_scale;
            }
        }

        fn trigger_browser_snapshot(&self) {
            let notice_slot = Rc::clone(&self.snapshot_notice);
            self.pipeline
                .capture_display_png_async(&self.gpu, move |res| match res {
                    Ok(png_bytes) => match download_png_bytes("gargantua_snapshot.png", &png_bytes) {
                        Ok(()) => {
                            *notice_slot.borrow_mut() =
                                Some(("Downloaded snapshot PNG".to_string(), Instant::now()));
                        }
                        Err(err) => {
                            log::warn!("Browser download failed: {}", err);
                            *notice_slot.borrow_mut() = Some((err, Instant::now()));
                        }
                    },
                    Err(err) => {
                        log::warn!("Snapshot capture failed: {}", err);
                        *notice_slot.borrow_mut() = Some((err, Instant::now()));
                    }
                });
        }

        fn render_frame(&mut self) {
            self.sync_render_scale();

            let now = Instant::now();
            let dt = (now - self.last_frame_time).as_secs_f32().clamp(1e-4, 0.1);
            self.last_frame_time = now;
            let inst_fps = 1.0 / dt;
            self.fps_smooth = self.fps_smooth * 0.9 + inst_fps * 0.1;
            self.sim_time += dt;
            if !self.sim_params.planet_paused {
                self.planet_time += dt;
            }
            self.frame_index = self.frame_index.wrapping_add(1);

            let cmd = self.input.take_command();
            if cmd.toggle_cursor_lock_requested {
                let next_lock = !self.input.cursor_locked;
                self.set_cursor_lock(next_lock);
            }
            self.observer.apply_command(&cmd, dt);

            let (w, h) = self.pipeline.size();
            let uniforms = FrameUniforms::from_observer_and_params(
                &self.observer,
                w,
                h,
                self.frame_index,
                self.sim_time,
                self.planet_time,
                &self.sim_params,
            );
            self.pipeline.render_compute_frame(&self.gpu, &uniforms);

            let mut manual_snapshot = cmd.snapshot_requested;
            let active_notice = {
                let mut guard = self.snapshot_notice.borrow_mut();
                match guard.as_ref() {
                    Some((msg, when)) if when.elapsed() < Duration::from_secs(3) => {
                        Some(msg.clone())
                    }
                    _ => {
                        *guard = None;
                        None
                    }
                }
            };

            let raw_input = self.egui_state.take_egui_input(&self.window);
            let full_output = self.egui_ctx.run(raw_input, |ctx| {
                if draw_hud(
                    ctx,
                    &mut self.observer,
                    &mut self.sim_params,
                    self.fps_smooth,
                    (w, h),
                    active_notice.as_deref(),
                ) {
                    manual_snapshot = true;
                }
            });
            self.egui_state
                .handle_platform_output(&self.window, full_output.platform_output);

            if manual_snapshot {
                self.trigger_browser_snapshot();
            }

            let frame = match self.surface.get_current_texture() {
                Ok(f) => f,
                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                    self.surface.configure(&self.gpu.device, &self.surface_config);
                    return;
                }
                Err(e) => {
                    log::warn!("Surface acquire error: {:?}", e);
                    return;
                }
            };
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            let tris = self
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);
            let screen_desc = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.surface_config.width, self.surface_config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            let mut encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("present_encoder"),
                    });

            for (id, delta) in &full_output.textures_delta.set {
                self.egui_renderer
                    .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
            self.egui_renderer.update_buffers(
                &self.gpu.device,
                &self.gpu.queue,
                &mut encoder,
                &tris,
                &screen_desc,
            );

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blit_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                self.pipeline.encode_blit(&mut rpass);
            }

            {
                let mut rpass = encoder
                    .begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui_hud_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    })
                    .forget_lifetime();
                self.egui_renderer.render(&mut rpass, &tris, &screen_desc);
            }

            self.gpu.queue.submit(std::iter::once(encoder.finish()));
            frame.present();

            for id in &full_output.textures_delta.free {
                self.egui_renderer.free_texture(id);
            }
        }
    }

    enum AppEvent {
        Initialized(Result<Box<AppRuntime>, String>),
        ViewportResized(f64, f64),
    }

    struct WebSimulatorApp {
        proxy: EventLoopProxy<AppEvent>,
        window: Option<Arc<Window>>,
        runtime: Option<AppRuntime>,
        init_started: bool,
    }

    impl ApplicationHandler<AppEvent> for WebSimulatorApp {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.init_started {
                return;
            }
            self.init_started = true;

            let web_win = web_sys::window().expect("Missing browser window");
            let doc = web_win.document().expect("Missing browser document");
            let canvas = doc
                .get_element_by_id("black-hole-canvas")
                .expect("Missing #black-hole-canvas")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#black-hole-canvas is not an HtmlCanvasElement");

            let css_w = web_win
                .inner_width()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(1280.0)
                .max(320.0);
            let css_h = web_win
                .inner_height()
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(720.0)
                .max(240.0);

            let attrs = Window::default_attributes()
                .with_title("Gargantua — Relativistic Black Hole Simulator (WebGPU)")
                .with_inner_size(LogicalSize::new(css_w, css_h))
                .with_canvas(Some(canvas.clone()))
                .with_prevent_default(true);

            let window = Arc::new(
                event_loop
                    .create_window(attrs)
                    .expect("Failed to attach winit window to canvas"),
            );
            self.window = Some(window.clone());

            // Keep canvas synced with browser window resizes
            let resize_proxy = self.proxy.clone();
            let web_win_cb = web_win.clone();
            let closure = Closure::<dyn FnMut()>::new(move || {
                let w = web_win_cb
                    .inner_width()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1280.0)
                    .max(320.0);
                let h = web_win_cb
                    .inner_height()
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(720.0)
                    .max(240.0);
                let _ = resize_proxy.send_event(AppEvent::ViewportResized(w, h));
            });
            let _ = web_win
                .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref());
            closure.forget();

            let _ = canvas.focus();

            let proxy = self.proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let res = AppRuntime::new(window).await.map(Box::new);
                let _ = proxy.send_event(AppEvent::Initialized(res));
            });
        }

        fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
            match event {
                AppEvent::Initialized(Ok(rt)) => {
                    hide_overlay("loading-overlay");
                    rt.window.request_redraw();
                    self.runtime = Some(*rt);
                }
                AppEvent::Initialized(Err(err)) => {
                    log::error!("WebGPU init failed: {}", err);
                    hide_overlay("loading-overlay");
                    show_error(&err);
                }
                AppEvent::ViewportResized(css_w, css_h) => {
                    if let Some(ref win) = self.window {
                        let _ = win.request_inner_size(LogicalSize::new(css_w, css_h));
                    }
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            let Some(rt) = self.runtime.as_mut() else {
                return;
            };

            let egui_resp = rt.egui_state.on_window_event(&rt.window, &event);
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => rt.resize(size.width, size.height),
                WindowEvent::Focused(focused) => {
                    if focused {
                        rt.input.on_focus_gained();
                    } else {
                        rt.input.on_focus_lost();
                        rt.set_cursor_lock(false);
                        rt.last_cursor_pos = None;
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left && state == ElementState::Pressed {
                        rt.last_cursor_pos = None;
                        focus_canvas();
                    }
                    rt.input.on_mouse_button(button, state, egui_resp.consumed);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if !rt.input.cursor_locked {
                        if let Some((px, py)) = rt.last_cursor_pos {
                            let dx = (position.x - px) as f32;
                            let dy = (position.y - py) as f32;
                            rt.input.on_mouse_motion(dx, dy);
                        }
                        rt.last_cursor_pos = Some((position.x, position.y));
                    }
                }
                WindowEvent::CursorLeft { .. } => {
                    rt.last_cursor_pos = None;
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        rt.input.on_key(
                            code,
                            event.state,
                            event.repeat,
                            rt.egui_ctx.wants_keyboard_input(),
                        );
                    }
                }
                WindowEvent::RedrawRequested => {
                    rt.render_frame();
                }
                _ => {}
            }
        }

        fn device_event(
            &mut self,
            _event_loop: &ActiveEventLoop,
            _device_id: DeviceId,
            event: DeviceEvent,
        ) {
            let Some(rt) = self.runtime.as_mut() else {
                return;
            };
            if rt.input.cursor_locked {
                if let DeviceEvent::MouseMotion { delta } = event {
                    rt.input.on_mouse_motion(delta.0 as f32, delta.1 as f32);
                }
            }
        }

        fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
            if let Some(ref rt) = self.runtime {
                rt.window.request_redraw();
            }
        }
    }

    fn focus_canvas() {
        if let Some(win) = web_sys::window() {
            if let Some(doc) = win.document() {
                if let Some(el) = doc.get_element_by_id("black-hole-canvas") {
                    if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                        let _ = html_el.focus();
                    }
                }
            }
        }
    }

    fn hide_overlay(id: &str) {
        if let Some(win) = web_sys::window() {
            if let Some(doc) = win.document() {
                if let Some(el) = doc.get_element_by_id(id) {
                    if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                        let _ = html_el.style().set_property("display", "none");
                    }
                }
            }
        }
    }

    fn show_error(message: &str) {
        if let Some(win) = web_sys::window() {
            if let Some(doc) = win.document() {
                if let Some(text_el) = doc.get_element_by_id("error-message") {
                    text_el.set_text_content(Some(message));
                }
                if let Some(el) = doc.get_element_by_id("error-overlay") {
                    if let Ok(html_el) = el.dyn_into::<web_sys::HtmlElement>() {
                        let _ = html_el.style().set_property("display", "flex");
                    }
                }
            }
        }
    }

    fn download_png_bytes(filename: &str, bytes: &[u8]) -> Result<(), String> {
        let win = web_sys::window().ok_or("No browser window")?;
        let doc = win.document().ok_or("No browser document")?;
        let uint8 = js_sys::Uint8Array::from(bytes);
        let parts = js_sys::Array::new();
        parts.push(&uint8.buffer());

        let props = web_sys::BlobPropertyBag::new();
        props.set_type("image/png");
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &props)
            .map_err(|e| format!("Blob error: {:?}", e))?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)
            .map_err(|e| format!("ObjectURL error: {:?}", e))?;

        let anchor = doc
            .create_element("a")
            .map_err(|e| format!("Anchor create failed: {:?}", e))?
            .dyn_into::<web_sys::HtmlAnchorElement>()
            .map_err(|_| "Anchor cast failed")?;
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
        let _ = web_sys::Url::revoke_object_url(&url);
        Ok(())
    }

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);

        let event_loop = EventLoop::<AppEvent>::with_user_event()
            .build()
            .expect("Failed to create winit EventLoop");
        let proxy = event_loop.create_proxy();
        let app = WebSimulatorApp {
            proxy,
            window: None,
            runtime: None,
            init_started: false,
        };
        event_loop.spawn_app(app);
    }
}
