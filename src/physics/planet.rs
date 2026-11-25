use glam::Vec3;

pub const HERO_PLANET_POS: Vec3 = Vec3::new(-10.2, -8.2, 4.6);
pub const HERO_PLANET_RADIUS: f32 = 0.28;
pub const ORBIT_TIME_SCALE: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
pub struct PlanetOrbit {
    pub initial_pos: Vec3,
    pub radius: f32,
    pub time_scale: f32,
}

impl Default for PlanetOrbit {
    fn default() -> Self {
        Self {
            initial_pos: HERO_PLANET_POS,
            radius: HERO_PLANET_RADIUS,
            time_scale: ORBIT_TIME_SCALE,
        }
    }
}

impl PlanetOrbit {
    pub fn orbital_radius(&self) -> f32 {
        self.initial_pos.length()
    }

    pub fn omega_kepler(r_orb: f32, bh_mass: f32, bh_spin: f32) -> f32 {
        let m = bh_mass.max(1e-4);
        let r = r_orb.max(0.5 * m);
        let a_s = bh_spin.clamp(-0.999, 0.999);
        let sqrt_m = m.sqrt();
        sqrt_m / (r.powf(1.5) + a_s * m * sqrt_m).max(1e-4)
    }

    /// Returns `(world_position, orbit_phase_rad)` at `planet_time` in a prograde inclined Kerr Keplerian orbit.
    pub fn position_at(&self, planet_time: f32, bh_mass: f32, bh_spin: f32) -> (Vec3, f32) {
        let r_orb = self.initial_pos.length();
        if r_orb <= 1e-5 {
            return (Vec3::ZERO, 0.0);
        }
        let u_orb = self.initial_pos / r_orb;
        let v_raw = Vec3::Z.cross(u_orb);
        let v_orb = if v_raw.length_squared() > 1e-8 {
            v_raw.normalize()
        } else {
            Vec3::Y
        };

        let omega = Self::omega_kepler(r_orb, bh_mass, bh_spin);
        let phase = (omega * self.time_scale * planet_time).rem_euclid(std::f32::consts::TAU);
        let pos = (u_orb * phase.cos() + v_orb * phase.sin()) * r_orb;
        (pos, phase)
    }
}
