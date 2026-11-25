use bytemuck::{Pod, Zeroable};

use crate::physics::metric::KerrSchild;
use crate::physics::observer::ObserverState;
use crate::physics::planet::{PlanetOrbit, HERO_PLANET_POS, HERO_PLANET_RADIUS};
use crate::ui::hud::SimParams;

pub const FLAG_ENABLE_DISK: u32 = 1 << 0;
pub const FLAG_ENABLE_PLANET: u32 = 1 << 1;
pub const FLAG_ENABLE_STARS: u32 = 1 << 2;
pub const FLAG_ENABLE_DOPPLER: u32 = 1 << 3;
pub const FLAG_ENABLE_BLOOM: u32 = 1 << 4;
pub const FLAG_ENABLE_PHOTON_SSAA: u32 = 1 << 5;
pub const FLAG_ALL: u32 = FLAG_ENABLE_DISK
    | FLAG_ENABLE_PLANET
    | FLAG_ENABLE_STARS
    | FLAG_ENABLE_DOPPLER
    | FLAG_ENABLE_BLOOM
    | FLAG_ENABLE_PHOTON_SSAA;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    pub cam_pos: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub resolution: [u32; 2],
    pub frame_index: u32,
    pub flags: u32,
    pub bh_spin: f32,
    pub bh_mass: f32,
    pub fov_y_rad: f32,
    pub sim_time: f32,
    pub disk_r_in: f32,
    pub disk_r_out: f32,
    pub disk_half_height: f32,
    pub disk_temp_peak_k: f32,
    pub planet_pos_radius: [f32; 4],
    pub planet_params: [f32; 4],
    pub render_params: [f32; 4],
}

impl FrameUniforms {
    pub fn from_observer(
        obs: &ObserverState,
        width: u32,
        height: u32,
        frame_index: u32,
        sim_time: f32,
    ) -> Self {
        Self {
            cam_pos: [obs.pos.x, obs.pos.y, obs.pos.z, 1.0],
            cam_fwd: [obs.fwd.x, obs.fwd.y, obs.fwd.z, 0.0],
            cam_right: [obs.right.x, obs.right.y, obs.right.z, 0.0],
            cam_up: [obs.up.x, obs.up.y, obs.up.z, 0.0],
            resolution: [width.max(1), height.max(1)],
            frame_index,
            flags: FLAG_ALL,
            bh_spin: 0.80,
            bh_mass: 1.0,
            fov_y_rad: obs.fov_y_deg.to_radians(),
            sim_time,
            disk_r_in: 4.55,
            disk_r_out: 21.5,
            disk_half_height: 0.34,
            disk_temp_peak_k: 6800.0,
            planet_pos_radius: [
                HERO_PLANET_POS.x,
                HERO_PLANET_POS.y,
                HERO_PLANET_POS.z,
                HERO_PLANET_RADIUS,
            ],
            planet_params: [0.65, 1.8, 0.35, 0.0],
            render_params: [1.0, 0.42, 1.0, 1.0],
        }
    }

    pub fn from_observer_and_params(
        obs: &ObserverState,
        width: u32,
        height: u32,
        frame_index: u32,
        sim_time: f32,
        planet_time: f32,
        params: &SimParams,
    ) -> Self {
        let mut u = Self::from_observer(obs, width, height, frame_index, sim_time);
        let bh_spin = params.bh_spin.clamp(-0.99, 0.99);
        u.bh_spin = bh_spin;
        let r_isco = KerrSchild::new(u.bh_mass as f64, bh_spin as f64).isco_radius() as f32;
        u.disk_r_in = (r_isco * 0.92).max(4.55).min(9.0);
        let orbit = PlanetOrbit::default();
        let (pos, phase) = orbit.position_at(planet_time, u.bh_mass, bh_spin);
        u.planet_pos_radius = [pos.x, pos.y, pos.z, orbit.radius];
        u.planet_params[3] = phase;
        u.render_params[0] = params.exposure;
        u.render_params[1] = params.bloom_intensity;
        u.render_params[3] = params.star_brightness;

        let mut flags = FLAG_ENABLE_STARS | FLAG_ENABLE_BLOOM;
        if params.disk_enabled {
            flags |= FLAG_ENABLE_DISK;
        }
        if params.planet_enabled {
            flags |= FLAG_ENABLE_PLANET;
        }
        if params.doppler_enabled {
            flags |= FLAG_ENABLE_DOPPLER;
        }
        if params.photon_ssaa {
            flags |= FLAG_ENABLE_PHOTON_SSAA;
        }
        u.flags = flags;
        u
    }
}
