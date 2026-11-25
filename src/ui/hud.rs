use crate::physics::observer::ObserverState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimParams {
    pub bh_spin: f32,
    pub exposure: f32,
    pub bloom_intensity: f32,
    pub star_brightness: f32,
    pub render_scale: f32,
    pub photon_ssaa: bool,
    pub planet_enabled: bool,
    pub planet_paused: bool,
    pub disk_enabled: bool,
    pub doppler_enabled: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            bh_spin: 0.80,
            exposure: 1.0,
            bloom_intensity: 0.42,
            star_brightness: 1.0,
            render_scale: 0.65,
            photon_ssaa: true,
            planet_enabled: true,
            planet_paused: false,
            disk_enabled: true,
            doppler_enabled: true,
        }
    }
}

pub fn draw_hud(
    ctx: &egui::Context,
    observer: &mut ObserverState,
    params: &mut SimParams,
    fps: f32,
    render_size: (u32, u32),
    snapshot_notice: Option<&str>,
) -> bool {
    if !observer.hud_visible {
        return false;
    }

    let ms = if fps > 1e-3 { 1000.0 / fps } else { 0.0 };
    let mut snapshot_clicked = false;

    egui::Area::new(egui::Id::new("awe_hud_overlay"))
        .fixed_pos(egui::pos2(12.0, 12.0))
        .show(ctx, |ui| {
            ui.set_max_width(252.0);
            egui::Frame::NONE
                .fill(egui::Color32::from_rgba_unmultiplied(10, 12, 18, 186))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(9))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("Gargantua — WebGPU")
                            .strong()
                            .color(egui::Color32::from_rgb(255, 215, 160)),
                    );
                    ui.label(format!(
                        "{}x{} · {:.1} FPS ({:.1} ms)",
                        render_size.0, render_size.1, fps, ms
                    ));
                    let r_cam = observer.pos.length();
                    let r_plus = 1.0 + (1.0 - params.bh_spin * params.bh_spin).max(0.0).sqrt();
                    ui.label(format!(
                        "Pos: ({:.1}, {:.1}, {:.1}) r={:.2}M",
                        observer.pos.x, observer.pos.y, observer.pos.z, r_cam
                    ));
                    ui.label(format!(
                        "Speed: {:.2} M/s (r_+={:.2}M)",
                        observer.effective_move_speed(),
                        r_plus
                    ));
                    if r_cam < r_plus {
                        ui.label(
                            egui::RichText::new("Inside Event Horizon")
                                .small()
                                .strong()
                                .color(egui::Color32::from_rgb(255, 130, 110)),
                        );
                    }
                    ui.separator();

                    ui.add(
                        egui::Slider::new(&mut params.bh_spin, -0.99..=0.99)
                            .text("Spin a_*")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.exposure, 0.2..=4.0)
                            .text("Exposure")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.bloom_intensity, 0.0..=1.5)
                            .text("Bloom")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.star_brightness, 0.1..=4.0)
                            .text("Stars")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut observer.move_speed, 0.5..=25.0)
                            .logarithmic(true)
                            .text("Move Speed")
                            .fixed_decimals(1),
                    );
                    ui.add(
                        egui::Slider::new(&mut params.render_scale, 0.35..=1.00)
                            .text("Scale")
                            .fixed_decimals(2),
                    );

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut params.planet_enabled, "Planet");
                        ui.checkbox(&mut params.planet_paused, "Pause Orbit");
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut params.disk_enabled, "Disk");
                        ui.checkbox(&mut params.doppler_enabled, "Doppler");
                    });
                    ui.checkbox(&mut params.photon_ssaa, "Photon Ring 3x SSAA");

                    ui.horizontal(|ui| {
                        if ui.button("Reset [R]").clicked() {
                            observer.reset_to_hero();
                        }
                        if ui.button("Snapshot [P]").clicked() {
                            snapshot_clicked = true;
                        }
                        ui.label(
                            egui::RichText::new("[H]")
                                .small()
                                .color(egui::Color32::from_rgb(175, 180, 195)),
                        );
                    });

                    ui.separator();
                    ui.label(
                        egui::RichText::new("WASD+Q/E Move · Drag/L Look · Shift+WASDQE Rotate · Ctrl/Space Speed")
                            .small()
                            .color(egui::Color32::from_rgb(165, 172, 188)),
                    );

                    if let Some(msg) = snapshot_notice {
                        ui.label(
                            egui::RichText::new(msg)
                                .small()
                                .color(egui::Color32::from_rgb(145, 235, 170)),
                        );
                    }
                });
        });

    snapshot_clicked
}
